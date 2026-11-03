#include "common/sampling_region.h"
#include "core/rrtstar.h"
#include "df/quad2d_df.h"
#include "df/quad3d_df.h"
#include "df/unicycle_df.h"
#include "systems/flat_bounds.h"
#include "systems/linear_model.h"
#include "tests/test_support.h"

using namespace rrtpff;

class GoalModel : public LinearModel {
  public:
    explicit GoalModel(Config c) : LinearModel(c) { goal_.setDerivative(0, c.goal); }
    TerminalConstraint<2, 2> goalConstraint() const {
        return TerminalConstraint<2, 2>::full(goal_);
    }
    bool isGoal(const FlatState &state, double tolerance) const {
        return (state.values - goal_.values).norm() <= tolerance;
    }

  private:
    FlatState goal_;
};

struct GuidedModel {
    static constexpr int FlatDim = 2;
    using FlatOutput = Eigen::Vector2d;
    std::vector<SamplingRegion<2>> regions{{Eigen::Vector2d(2, 2), Eigen::Vector2d(2, 2)}};
    FlatOutput goalOutput() const { return {9, 9}; }
    FlatOutput lowerOutput() const { return {0, 0}; }
    FlatOutput upperOutput() const { return {10, 10}; }
    const auto &samplingRegions() const { return regions; }
    double guidanceProbability() const { return .25; }
};

int main() {
    return runTest([] {
        // An ordinary extension still obeys range. A node entering the Goal region
        // triggers a separate validated PFF connection to the full Goal state.
        LinearModel::Config mc;
        mc.start.values << 1, 1, 0, 0;
        mc.goal << 3, 1;
        mc.upper << 5, 5;
        PlannerConfig pc;
        pc.timeout = 5;
        pc.range = .1;
        pc.goal_bias = 1;
        pc.max_iterations = 1;
        pc.explicit_goal_connection = true;
        pc.rewire_before_first_solution = false;
        pc.goal_tolerance = 1e-7;
        for (bool simd : {false, true}) {
            pc.simd_enabled = simd;

            // One ordinary range-limited extension ends about 1.9 units from Goal.
            pc.goal_connect_radius = 0.5;
            RRTstar<GoalModel> outside(GoalModel(mc), pc);
            auto outside_result = outside.solve();
            require(!outside_result.success && outside_result.node_count == 2 &&
                        outside_result.goal_connection_queries == 0,
                    "no exact Goal connection when new node is outside region");

            // A new node enters the larger region; use a separate full-state PFF
            // edge. Neither sampling nor regular expansion bypasses range.
            pc.goal_connect_radius = 2.0;
            RRTstar<GoalModel> planner(GoalModel(mc), pc);
            auto result = planner.solve();
            require(result.success && result.node_count == 3 &&
                        result.goal_connection_queries == 1 &&
                        result.goal_connection_valid == 1,
                    "region entry triggers one validated exact PFF Goal connection");
            require(planner.path().size() == 2,
                    "normal extension then separate exact Goal edge");
            require((planner.path().front().end.output() - mc.start.output()).norm() <
                        pc.range + 1e-7,
                    "ordinary extension respects range even with Goal bias");
            require(planner.path().back().end.derivative(1).norm() < 1e-8,
                    "exact Goal connection fixes terminal velocity");

            pc.explicit_goal_connection = false;
            RRTstar<GoalModel> limited(GoalModel(mc), pc);
            auto blocked = limited.solve();
            require(!blocked.success && blocked.goal_connection_queries == 0,
                    "disabling explicit Goal connection retains range limit");
            require(blocked.failure_reason == "max_iterations_no_goal", "termination reason");
            pc.explicit_goal_connection = true;
        }
        auto blocked_config = mc;
        blocked_config.circles.push_back({Eigen::Vector2d(2, 1), .2});
        RRTstar<GoalModel> blocked(GoalModel(blocked_config), pc);
        require(!blocked.solve().success, "goal connector cannot bypass physical validation");

        // Free-time stationary roots and interval ends miss a physically feasible
        // duration in this constrained-speed example; extra times recover it.
        using Solver = PFFSolver<2, 2, UnicycleTrajectoryMetadata>;
        using Constraint = Solver::Constraint;
        Solver::State start;
        start.setDerivative(1, Eigen::Vector2d(.3, 0));
        auto target = Constraint::output(Eigen::Vector2d(.7, 0));
        UnicycleDF::Config uni;
        uni.velocity_lower = .25;
        uni.velocity_upper = .5;
        uni.omega_lower = -.25;
        uni.omega_upper = .5;
        UnicycleTrajectoryMetadata metadata;
        require(UnicycleDF::makeMetadata(.3, metadata, uni), "wall lift metadata");
        PFFConfig original;
        original.min_time = .1;
        original.max_time = 8;
        auto any_feasible = [&](std::vector<Solver::Result> candidates) {
            for (auto &c : candidates) {
                c.trajectory.metadata = metadata;
                if (UnicycleDF::feasible(c.trajectory, uni))
                    return true;
            }
            return false;
        };
        require(
            !any_feasible(Solver(Solver::Weight::Identity(), original).candidates(start, target)),
            "roots and bounds alone miss this constrained connection");
        auto sampled = original;
        sampled.additional_time_samples = 12;
        Solver scalar(Solver::Weight::Identity(), sampled);
        auto reference = scalar.candidates(start, target);
        require(any_feasible(reference),
                "additional duration candidates recover valid speed profile");
        std::array<Solver::State, 8> starts;
        starts.fill(start);
        std::array<Constraint, 8> targets;
        targets.fill(target);
        auto batch = SIMD_PFF<2, 2, UnicycleTrajectoryMetadata, 8>(scalar).candidatesBatch(
            starts, targets, 8, true);
        for (const auto &lane : batch) {
            require(lane.size() == reference.size(),
                    "scalar/SIMD retain identical extra durations");
            for (std::size_t k = 0; k < lane.size(); ++k) {
                close(lane[k].T, reference[k].T, 1e-7, "extra time parity");
                close(lane[k].cost, reference[k].cost, 1e-7, "extra cost parity");
            }
        }
        UnicycleDF::FlatControl acceleration;
        acceleration.values << 0, -.12;
        UnicycleDF::PhysicalState x;
        UnicycleDF::PhysicalControl u;
        FeasibilityFailure reason;
        require(!UnicycleDF::sampleFeasible(start, acceleration, metadata, x, u, uni, &reason) &&
                    reason == FeasibilityFailure::Omega,
                "asymmetric negative angular velocity bound");
        acceleration.values << 0, .12;
        require(UnicycleDF::sampleFeasible(start, acceleration, metadata, x, u, uni, &reason),
                "positive angular velocity remains allowed");
        auto reverse = uni;
        reverse.allow_reverse = true;
        reverse.velocity_lower = -.5;
        reverse.velocity_upper = -.25;
        require(UnicycleDF::makeMetadata(-.3, metadata, reverse), "reverse metadata");
        require(UnicycleDF::sampleFeasible(start, acceleration, metadata, x, u, reverse, &reason) &&
                    u(0) < 0,
                "reverse bound keeps signed velocity and heading semantics");
        close(u(0) * std::cos(x(2)), .3, 1e-12, "reverse reconstruction preserves world motion");
        auto slow = start;
        slow.setDerivative(1, Eigen::Vector2d(.2, 0));
        require(!UnicycleDF::sampleFeasible(slow, acceleration, metadata, x, u, reverse, &reason) &&
                    reason == FeasibilityFailure::SpeedMin,
                "reverse minimum magnitude bound");
        // Endpoints legal, hidden interior minimum illegal, even with only one
        // coarse interval. Exact stationary-point speed check must reject it.
        UnicycleDF::FlatTrajectory dip;
        dip.T = 1;
        dip.coefficients(0, 1) = .3;
        dip.coefficients(0, 2) = -.2;
        dip.coefficients(0, 3) = .4 / 3;
        require(UnicycleDF::makeMetadata(.3, dip.metadata, uni), "dip metadata");
        FeasibilityConfig coarse;
        coarse.sample_count = 1;
        require(!UnicycleDF::feasible(dip, uni, coarse, &reason) &&
                    reason == FeasibilityFailure::SpeedMin,
                "continuous minimum-speed violation cannot hide between samples");

        FlatTrajectory<2, 2> curve;
        curve.T = 1;
        curve.coefficients.col(0) << 1, 1;
        curve.coefficients(0, 1) = 8;
        curve.coefficients(0, 2) = -8;
        PlanResult screen_stats;
        for (bool simd : {false, true})
            require(!flatBoundsScreen(curve, Eigen::Vector2d(0, 0), Eigen::Vector2d(2, 2), 2, 100,
                                      100, 16, simd, screen_stats),
                    "flat screening catches intermediate position overshoot");
        curve.coefficients(0, 1) = .2;
        curve.coefficients(0, 2) = 0;
        require(flatBoundsScreen(curve, Eigen::Vector2d(0, 0), Eigen::Vector2d(2, 2), 2, 100, 100,
                                 16, true, screen_stats),
                "flat screen retains inside trajectory");
        Quad3DDF::FlatState flat;
        Quad3DDF::FlatControl snap;
        Quad3DDF::PhysicalState qx;
        Quad3DDF::PhysicalControl qu;
        FlatState<4, 4>::Output a;
        a << 0, 0, 5, 0;
        flat.setDerivative(2, a);
        require(!Quad3DDF::sampleFeasible(flat, snap, {}, qx, qu, {}, &reason) &&
                    reason == FeasibilityFailure::Thrust,
                "quad thrust diagnosis");
        GuidedModel guided;
        Sampler<GuidedModel> sampler(guided, 40);
        std::size_t hints = 0, uniform = 0;
        for (int i = 0; i < 1000; ++i) {
            auto p = sampler.sampleFlatOutput(0);
            if ((p - Eigen::Vector2d(2, 2)).norm() == 0)
                ++hints;
            else
                ++uniform;
        }
        require(hints > 100 && uniform > 500, "guidance retains global uniform exploration");
    });
}
