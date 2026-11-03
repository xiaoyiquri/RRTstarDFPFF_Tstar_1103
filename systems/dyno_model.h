#pragma once
#include "common/sampling_region.h"
#include "common/timer.h"
#include "common/types.h"
#include "df/paper_r2_df.h"
#include "df/quad3d_df.h"
#include "df/unicycle_df.h"
#include "dynobench/data_adapter.h"
#include "pff/pff.h"
#include "simd/discrete_validator.h"
#include "simd/simd_backend.h"
#include "systems/batch_config.h"
#include "systems/planar_search.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace rrtpff {
struct RobotValidationConfig {
    double max_step{0.005};
    std::size_t max_samples{20000};
    double endpoint_tolerance{1e-6};
    bool terminal_velocity_hint{false};
    std::string df_mode{"full_df"};
};

// The branch policy is a system-specific concern; the generic planner only
// transports Metadata and asks the model to check endpoint semantics.
template <class DF> struct EdgeSemantics {
    using Metadata = typename DF::FlatTrajectory::MetadataType;
    using State = typename DF::FlatState;
    using Trajectory = typename DF::FlatTrajectory;
    static Metadata initial(double, const typename DF::Config &) { return {}; }
    static bool prepare(const State &parent, const Metadata &, Trajectory &edge,
                        const typename DF::Config &) {
        return edge.start.values.isApprox(parent.values, 1e-8);
    }
    static Metadata terminal(const Trajectory &, const typename DF::Config &) { return {}; }
    static bool matches(const Trajectory &, const Metadata &, const typename DF::Config &) {
        return true;
    }
};

template <> struct EdgeSemantics<UnicycleDF> {
    using Metadata = UnicycleTrajectoryMetadata;
    using State = UnicycleDF::FlatState;
    using Trajectory = UnicycleDF::FlatTrajectory;
    static Metadata initial(double velocity, const UnicycleDF::Config &c) {
        Metadata m;
        if (!UnicycleDF::makeMetadata(velocity, m, c))
            throw std::invalid_argument("Unicycle requires explicit nonzero signed velocity");
        return m;
    }
    static bool prepare(const State &parent, const Metadata &m, Trajectory &edge,
                        const UnicycleDF::Config &c) {
        if (!UnicycleDF::validMetadata(m, c) || !edge.start.values.isApprox(parent.values, 1e-8))
            return false;
        const double speed = parent.derivative(1).norm();
        if (speed <= c.velocity_epsilon ||
            std::abs(speed - std::abs(m.signed_start_velocity)) > c.start_tolerance * (1 + speed))
            return false;
        edge.metadata = m;
        return true;
    }
    static Metadata terminal(const Trajectory &edge, const UnicycleDF::Config &c) {
        Metadata m;
        const double sign = edge.metadata.branch == VelocityBranch::Reverse ? -1 : 1;
        UnicycleDF::makeMetadata(sign * edge.end.derivative(1).norm(), m, c);
        return m;
    }
    static bool matches(const Trajectory &edge, const Metadata &m, const UnicycleDF::Config &c) {
        auto end = terminal(edge, c);
        return UnicycleDF::validMetadata(end, c) && UnicycleDF::validMetadata(m, c) &&
               end.branch == m.branch &&
               std::abs(end.signed_start_velocity - m.signed_start_velocity) <=
                   c.start_tolerance * (1 + std::abs(m.signed_start_velocity));
    }
};

template <class DF, class Adapter = PaperDataAdapter> class DynoSystemModel {
  public:
    using FlatState = typename DF::FlatState;
    using FlatOutput = typename FlatState::Output;
    using FlatTrajectory = typename DF::FlatTrajectory;
    using Metadata = typename FlatTrajectory::MetadataType;
    using PhysicalState = typename DF::PhysicalState;
    using PhysicalControl = typename DF::PhysicalControl;
    static constexpr int FlatDim = FlatState::OutputDim, FlatOrder = FlatState::IntegratorOrder;
    using Constraint = TerminalConstraint<FlatDim, FlatOrder>;
    using Region = SamplingRegion<FlatDim>;
    using Semantics = EdgeSemantics<DF>;

    DynoSystemModel(Adapter adapter, const typename DF::LiftData &initial_lift,
                    const typename DF::LiftData &final_lift, typename DF::Config df_config,
                    RobotValidationConfig validation, double signed_velocity = 0,
                    double yaw_lower = 0, double yaw_upper = 0)
        : adapter_(std::move(adapter)), df_config_(std::move(df_config)), validation_(validation) {
        const auto &p = adapter_.problem();
        if ((validation_.df_mode == "paper_r2" && FlatOrder != 2) ||
            (validation_.df_mode == "full_df" && FlatOrder != 4 &&
             !std::is_same_v<DF, UnicycleDF>) ||
            (validation_.df_mode != "paper_r2" && validation_.df_mode != "full_df"))
            throw std::invalid_argument("DF type and boundary mode do not agree");
        if (p.start.size() != PhysicalState::RowsAtCompileTime ||
            p.goal.size() != PhysicalState::RowsAtCompileTime)
            throw std::invalid_argument("Physical state layout does not match selected system");
        if (!(std::isfinite(validation_.max_step) && validation_.max_step > 0) ||
            validation_.max_samples < 2 || !std::isfinite(validation_.endpoint_tolerance) ||
            validation_.endpoint_tolerance <= 0)
            throw std::invalid_argument("Invalid discrete validation parameters");
        if (!simdRuntimeAVX2())
            throw std::runtime_error(
                "This robot pipeline requires AVX2 float8; no scalar validator");
        batch_config_ = makeBatchConfig<DF>(df_config_, p);
        PhysicalState physical_start = p.start, physical_goal = p.goal;
        if (!DF::stateToFlat(physical_start, initial_lift, start_, df_config_) ||
            !DF::stateToFlat(physical_goal, final_lift, goal_, df_config_))
            throw std::invalid_argument(
                "Explicit LiftData is missing or inconsistent with DynoBench endpoints");
        initial_metadata_ = Semantics::initial(signed_velocity, df_config_);
        output_ = goal_.output();
        lower_.setZero();
        upper_.setZero();
        const auto pdim = p.lower_position.size();
        constexpr int PositionDim = FlatDim == 4 ? 3 : 2;
        if (pdim != PositionDim || p.upper_position.size() != pdim ||
            !p.lower_position.allFinite() || !p.upper_position.allFinite() ||
            (p.lower_position.array() >= p.upper_position.array()).any())
            throw std::invalid_argument("Invalid position bounds");
        lower_.head(pdim) = p.lower_position;
        upper_.head(pdim) = p.upper_position;
        if constexpr (FlatDim == 4) {
            if (!std::isfinite(yaw_lower) || !std::isfinite(yaw_upper) || yaw_lower > yaw_upper)
                throw std::invalid_argument("Invalid yaw sampling interval");
            lower_(3) = yaw_lower;
            upper_(3) = yaw_upper;
        }
    }
    const FlatState &start() const { return start_; }
    const FlatOutput &goalOutput() const { return output_; }
    const FlatOutput &lowerOutput() const { return lower_; }
    const FlatOutput &upperOutput() const { return upper_; }
    const std::vector<Region> &samplingRegions() const { return regions_; }
    double guidanceProbability() const { return guidance_probability_; }
    void setPlanarSearch(PlanarSearchConfig config) {
        if constexpr (std::is_base_of_v<Quad3DDF, DF>) {
            if (config.turn_limited_sampling || config.rest_hint || config.terminal_speed != 0 ||
                !config.tilt_hints.empty() || !config.duration_factors.empty())
                throw std::invalid_argument("Quad3D regression baseline excludes planar policies");
        }
        if (!std::isfinite(config.terminal_speed) || config.terminal_speed < 0)
            throw std::invalid_argument("Invalid terminal speed hint");
        if (!std::isfinite(config.turn_safety_factor) || config.turn_safety_factor <= 0 ||
            config.turn_safety_factor > 1 || config.duration_factors.size() > 16)
            throw std::invalid_argument("Invalid turn/time hint configuration");
        for (double factor : config.duration_factors)
            if (!std::isfinite(factor) || factor <= 0)
                throw std::invalid_argument("Duration factors must be finite and positive");
        if constexpr (std::is_same_v<DF, UnicycleDF>) {
            if (config.terminal_speed > 0) {
                auto signed_speed = initial_metadata_.branch == VelocityBranch::Reverse
                                        ? -config.terminal_speed
                                        : config.terminal_speed;
                Metadata check;
                if (!DF::makeMetadata(signed_speed, check, df_config_))
                    throw std::invalid_argument(
                        "Terminal speed hint outside signed control bounds");
            }
        }
        for (const auto &hint : config.tilt_hints) {
            if (!hint.region.lower.allFinite() || !hint.region.upper.allFinite() ||
                (hint.region.lower.array() > hint.region.upper.array()).any() ||
                (hint.region.lower.array() < lower_.template head<2>().array()).any() ||
                (hint.region.upper.array() > upper_.template head<2>().array()).any() ||
                !hint.velocity.allFinite() || !std::isfinite(hint.theta) ||
                !std::isfinite(hint.force_ratio) || hint.force_ratio <= 0)
                throw std::invalid_argument("Invalid Quad2D terminal hint");
        }
        planar_search_ = std::move(config);
    }
    void setSamplingGuidance(std::vector<Region> regions, double probability) {
        if (!std::isfinite(probability) || probability < 0 || probability >= 1)
            throw std::invalid_argument(
                "Guidance probability must be in [0,1); uniform exploration is retained");
        if (probability > 0 && regions.empty())
            throw std::invalid_argument("Guidance regions are required");
        for (const auto &r : regions)
            if (!r.lower.allFinite() || !r.upper.allFinite() ||
                (r.lower.array() > r.upper.array()).any() ||
                (r.lower.array() < lower_.array()).any() ||
                (r.upper.array() > upper_.array()).any())
                throw std::invalid_argument("Sampling region outside flat output bounds");
        regions_ = std::move(regions);
        guidance_probability_ = probability;
    }
    Metadata initialMetadata() const { return initial_metadata_; }
    bool prepareEdge(const FlatState &s, const Metadata &m, FlatTrajectory &edge) const {
        return Semantics::prepare(s, m, edge, df_config_);
    }
    Metadata terminalMetadata(const FlatTrajectory &edge) const {
        return Semantics::terminal(edge, df_config_);
    }
    bool terminalMetadataMatches(const FlatTrajectory &edge, const Metadata &m) const {
        return Semantics::matches(edge, m, df_config_);
    }
    // Only the explicitly sampled goal gets full derivative constraints. Other
    // expansions keep derivatives free; rewiring is always full-state fixed.
    Constraint expansionConstraint(const FlatOutput &target) const {
        if ((target - output_).squaredNorm() < 1e-20)
            return Constraint::full(goal_);
        return Constraint::output(target);
    }
    Constraint goalConstraint() const { return Constraint::full(goal_); }
    std::optional<Constraint> alternativeExpansionConstraint(const FlatState &parent,
                                                             const Metadata &metadata,
                                                             const FlatOutput &target) const {
        // Optional extra candidate, alongside the ordinary free-terminal PFF.
        // The flat velocity points along motion, while edge metadata alone
        // determines forward/reverse physical interpretation.
        if constexpr (std::is_same_v<DF, UnicycleDF>) {
            if (!validation_.terminal_velocity_hint || (target - output_).squaredNorm() < 1e-20)
                return std::nullopt;
            FlatOutput delta = target - parent.output();
            if (delta.norm() < 1e-10)
                return std::nullopt;
            FlatState terminal = parent;
            terminal.setDerivative(0, target);
            terminal.setDerivative(1,
                                   delta.normalized() * std::abs(metadata.signed_start_velocity));
            return Constraint::full(terminal);
        }
        return std::nullopt;
    }
    std::vector<Constraint> additionalExpansionConstraints(const FlatState &parent,
                                                           const Metadata &,
                                                           const FlatOutput &target) const {
        std::vector<Constraint> result;
        if ((target - output_).squaredNorm() < 1e-20)
            return result; // Actual goals always retain the original full lifted state.
        if constexpr (std::is_same_v<DF, UnicycleDF>) {
            if (!planar_search_.turn_limited_sampling)
                return result;
            const Eigen::Vector2d delta = target - parent.output();
            const Eigen::Vector2d velocity = parent.derivative(1);
            if (delta.norm() < 1e-10 || velocity.norm() <= df_config_.velocity_epsilon)
                return result;
            const double phi = std::atan2(velocity.y(), velocity.x());
            const double bearing = std::atan2(delta.y(), delta.x());
            const double terminal_angle =
                phi + 2 * std::remainder(bearing - phi, 2 * std::acos(-1.));
            const double speed =
                planar_search_.terminal_speed > 0 ? planar_search_.terminal_speed : velocity.norm();
            FlatState terminal = parent;
            terminal.setDerivative(0, target);
            terminal.setDerivative(
                1, speed * Eigen::Vector2d(std::cos(terminal_angle), std::sin(terminal_angle)));
            result.push_back(Constraint::full(terminal));
        } else if constexpr (std::is_base_of_v<Quad2DDF, DF>) {
            if (planar_search_.rest_hint) {
                // A deliberate hover boundary candidate. The unconstrained
                // free-derivative candidate remains in the same parent batch.
                FlatState terminal;
                terminal.setDerivative(0, target);
                terminal.setDerivative(1, Eigen::Vector2d::Zero());
                if constexpr (FlatOrder > 2) {
                    terminal.setDerivative(2, Eigen::Vector2d::Zero());
                    terminal.setDerivative(3, Eigen::Vector2d::Zero());
                }
                result.push_back(Constraint::full(terminal));
            }
            if constexpr (FlatOrder > 2) for (const auto &hint : planar_search_.tilt_hints) {
                if ((target.array() < hint.region.lower.array()).any() ||
                    (target.array() > hint.region.upper.array()).any())
                    continue;
                FlatState terminal;
                terminal.setDerivative(0, target);
                terminal.setDerivative(1, hint.velocity);
                const double force = df_config_.gravity * hint.force_ratio;
                terminal.setDerivative(
                    2, Eigen::Vector2d(-force * std::sin(hint.theta),
                                       force * std::cos(hint.theta) - df_config_.gravity));
                terminal.setDerivative(3, Eigen::Vector2d::Zero()); // Explicit zero angular rate.
                result.push_back(Constraint::full(terminal));
            }
        }
        return result;
    }
    FlatOutput metricWeights() const { return FlatOutput::Ones(); }
    std::vector<double> additionalCandidateTimes(const FlatState &parent,
                                                 const Constraint &target) const {
        std::vector<double> times;
        if constexpr (std::is_same_v<DF, UnicycleDF>) {
            if (!planar_search_.turn_limited_sampling || planar_search_.duration_factors.empty())
                return times;
            const Eigen::Vector2d delta = target.target.output() - parent.output();
            const Eigen::Vector2d velocity = parent.derivative(1);
            if (delta.norm() <= 1e-10 || velocity.norm() <= df_config_.velocity_epsilon)
                return times;
            double end_speed =
                planar_search_.terminal_speed > 0 ? planar_search_.terminal_speed : velocity.norm();
            if (target.fixed[2] && target.fixed[3])
                end_speed = target.target.derivative(1).norm();
            const double mean_speed = .5 * (velocity.norm() + end_speed);
            const double turn =
                2 * std::abs(std::remainder(std::atan2(delta.y(), delta.x()) -
                                                std::atan2(velocity.y(), velocity.x()),
                                            2 * std::acos(-1.)));
            double length = delta.norm();
            if (turn > 1e-8 && turn < std::acos(-1.))
                length *= turn / (2 * std::sin(turn / 2));
            for (double factor : planar_search_.duration_factors)
                times.push_back(factor * length / mean_speed);
        }
        return times;
    }
    FlatOutput metricPeriods() const { return FlatOutput::Zero(); }
    double distanceToOutput(const FlatState &s, const FlatOutput &q) const {
        return (s.output() - q).norm();
    }
    FlatOutput limitExtension(const FlatState &s, const FlatOutput &q, double range) const {
        FlatOutput delta = q - s.output();
        const double distance = delta.norm();
        if constexpr (std::is_same_v<DF, UnicycleDF>) {
            const auto velocity = s.derivative(1);
            if (planar_search_.turn_limited_sampling && distance > 1e-10 &&
                velocity.norm() > df_config_.velocity_epsilon) {
                const double end_speed = planar_search_.terminal_speed > 0
                                             ? planar_search_.terminal_speed
                                             : velocity.norm();
                const double speed = .5 * (velocity.norm() + end_speed);
                const double duration = std::min(range, distance) / speed;
                const double phi = std::atan2(velocity.y(), velocity.x());
                const double bearing = std::atan2(delta.y(), delta.x());
                const double lower = std::max(-df_config_.max_omega, df_config_.omega_lower);
                const double upper = std::min(df_config_.max_omega, df_config_.omega_upper);
                const double turn =
                    std::clamp(2 * std::remainder(bearing - phi, 2 * std::acos(-1.)),
                               planar_search_.turn_safety_factor * lower * duration,
                               planar_search_.turn_safety_factor * upper * duration);
                // A reachable-direction OUTPUT proposal, not a dynamics-valid
                // edge. PFF and the SIMD physical validator still decide.
                if (std::abs(turn) < 1e-9)
                    return s.output() + speed * duration * velocity.normalized();
                return s.output() + (speed * duration / turn) *
                                        Eigen::Vector2d(std::sin(phi + turn) - std::sin(phi),
                                                        std::cos(phi) - std::cos(phi + turn));
            }
        }
        return distance > range ? (s.output() + delta * (range / distance)).eval() : q;
    }
    bool endpointStateValid(const Eigen::VectorXd &state) const {
        PhysicalBatch batch;
        for (int k = 0; k < batch_config_.state_dim; ++k)
            batch.states[k][0] = float(state(k));
        stateBatch(batch_config_, 1, adapter_.geometry(), batch);
        return batch.valid_mask == 1;
    }
    bool initialStateValid() const { return endpointStateValid(adapter_.problem().start); }
    bool goalOutputValid() const { return endpointStateValid(adapter_.problem().goal); }
    bool isGoal(const FlatState &s, double tolerance) const {
        // Match every stored graph coordinate: y/dy in paper_r2, y..jerk in full_df.
        // Physical endpoint consistency is checked separately during path export.
        return (s.values - goal_.values).norm() <= tolerance;
    }
    double solutionLowerBound(const FlatState &) const { return 0; }

    bool reconstruct(const FlatTrajectory &edge, DynoSamples &samples, PlanResult &stats) const {
        return discrete(edge, stats, &samples);
    }
    bool trajectoryValid(const FlatTrajectory &edge, PlanResult &stats) const {
        return discrete(edge, stats, nullptr);
    }
    DynoValidation validatePath(const std::vector<FlatTrajectory> &path, DynoSamples &all,
                                PlanResult &stats) const {
        all = {};
        stats.df_mode = validation_.df_mode;
        stats.pseudo_control_order = FlatOrder;
        stats.reduced_order_result = validation_.df_mode == "paper_r2";
        stats.boundary_contract = stats.reduced_order_result ? "reduced_flat_A_prime" : "full_physical";
        stats.boundary_diagnostics.clear();
        stats.physical_boundary_match = false;
        if (path.empty())
            return {false, 0, 0, 0, 0, "empty_path"};
        const double tol = validation_.endpoint_tolerance;
        if ((path.front().start.values - start_.values).norm() > tol ||
            (path.back().end.values - goal_.values).norm() > tol)
            return {false, 0, 0, 0, 0, "flat_endpoint_mismatch"};
        double elapsed = 0;
        Eigen::VectorXd previous_end;
        bool physical_match = true;
        for (std::size_t e = 0; e < path.size(); ++e) {
            const auto &edge = path[e];
            // These are flat graph/metadata contracts, not scalar physical validation.
            if (e && ((path[e - 1].end.values - edge.start.values).norm() > tol ||
                      !Semantics::matches(path[e - 1], edge.metadata, df_config_)))
                return {false, 0, 0, 0, 0, "flat_edge_stitching"};
            DynoSamples samples;
            if (!reconstruct(edge, samples, stats))
                return {false, 0, 0, 0, 0, "simd_discrete_rejection"};
            BoundaryDiagnostic diagnostic;
            diagnostic.kind = e ? 1 : 0;
            diagnostic.edge_index = e;
            diagnostic.time = elapsed;
            const auto &reference = e ? previous_end : adapter_.problem().start;
            diagnostic.physical = physicalBoundaryError(reference, samples.states.front());
            const FlatState &flat_reference = e ? path[e - 1].end : start_;
            diagnostic.flat_position_error = (flat_reference.output() - edge.start.output()).norm();
            diagnostic.flat_velocity_error = (flat_reference.derivative(1) - edge.start.derivative(1)).norm();
            stats.boundary_diagnostics.push_back(diagnostic);
            physical_match = physical_match && diagnostic.physical.maximum() <= tol;
            previous_end = samples.states.back();
            for (std::size_t i = e ? 1 : 0; i < samples.states.size(); ++i) {
                auto x = samples.states[i];
                if constexpr (std::is_base_of_v<Quad3DDF, DF>) {
                    // Choose an equivalent quaternion sign for continuous exported coordinates.
                    if (!all.states.empty() &&
                        all.states.back().template segment<4>(3).dot(x.template segment<4>(3)) < 0)
                        x.template segment<4>(3) *= -1;
                }
                all.states.push_back(std::move(x));
                all.times.push_back(elapsed + samples.times[i]);
            }
            all.controls.insert(all.controls.end(), samples.controls.begin(),
                                samples.controls.end());
            elapsed += edge.T;
        }
        BoundaryDiagnostic goal_diagnostic;
        goal_diagnostic.kind = 2;
        goal_diagnostic.edge_index = path.size() - 1;
        goal_diagnostic.time = elapsed;
        goal_diagnostic.physical = physicalBoundaryError(previous_end, adapter_.problem().goal);
        goal_diagnostic.flat_position_error = (path.back().end.output() - goal_.output()).norm();
        goal_diagnostic.flat_velocity_error = (path.back().end.derivative(1) - goal_.derivative(1)).norm();
        stats.boundary_diagnostics.push_back(goal_diagnostic);
        stats.physical_boundary_match = physical_match && goal_diagnostic.physical.maximum() <= tol;
        if (!stats.reduced_order_result && !stats.physical_boundary_match)
            return {false, 0, 0, 0, 0, "full_physical_boundary_mismatch"};
        return {true, 0, 0, 0, 0, {}};
    }
    const Adapter &adapter() const { return adapter_; }
    double sampleStep() const { return validation_.max_step; }

  private:
    bool discrete(const FlatTrajectory &edge, PlanResult &stats, DynoSamples *samples) const {
        int branch = 1;
        if constexpr (std::is_same_v<DF, UnicycleDF>) {
            if (!DF::validMetadata(edge.metadata, df_config_) ||
                std::abs(edge.start.derivative(1).norm() -
                         std::abs(edge.metadata.signed_start_velocity)) >
                    df_config_.start_tolerance *
                        (1 + std::abs(edge.metadata.signed_start_velocity))) {
                ++stats.df_rejected;
                return false;
            }
            branch = edge.metadata.branch == VelocityBranch::Reverse ? -1 : 1;
        }
        return validateDiscrete(edge, batch_config_, branch, adapter_.geometry(),
                                validation_.max_step, validation_.max_samples, stats, samples);
    }
    PhysicalBatchConfig batch_config_;
    Adapter adapter_;
    typename DF::Config df_config_;
    RobotValidationConfig validation_;
    FlatState start_, goal_;
    FlatOutput output_, lower_, upper_;
    Metadata initial_metadata_{};
    std::vector<Region> regions_;
    double guidance_probability_{0};
    PlanarSearchConfig planar_search_;
};
using PaperUnicycleModel = DynoSystemModel<UnicycleDF>;
using PaperQuad2DModel = DynoSystemModel<Quad2DDF>;
using PaperQuad3DModel = DynoSystemModel<Quad3DDF>;
using PaperQuad2DR2Model = DynoSystemModel<Quad2DR2DF>;
using PaperQuad3DR2Model = DynoSystemModel<Quad3DR2DF>;
} // namespace rrtpff
