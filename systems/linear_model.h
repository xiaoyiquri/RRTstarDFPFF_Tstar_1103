#pragma once
#include "common/flat_trajectory.h"
#include "common/physical_trajectory.h"
#include "simd/simd_collision.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace rrtpff {
// Phase 1 double-integrator baseline. This is NOT a DynoBench robot adapter.
class LinearModel {
  public:
    static constexpr int FlatDim = 2, FlatOrder = 2;
    using FlatState = rrtpff::FlatState<FlatDim, FlatOrder>;
    using FlatOutput = FlatState::Output;
    using Trajectory = rrtpff::FlatTrajectory<FlatDim, FlatOrder>;
    using FlatTrajectory = Trajectory;
    using Metadata = typename Trajectory::MetadataType;
    using PhysicalState = FlatState;
    using PhysicalControl = FlatOutput;
    using PhysicalTrajectory = rrtpff::PhysicalTrajectory<PhysicalState, PhysicalControl>;
    struct Circle {
        FlatOutput center;
        double radius;
    };
    struct Config {
        FlatState start{};
        FlatOutput goal = FlatOutput::Zero(), lower = FlatOutput::Zero(),
                   upper = FlatOutput::Constant(20);
        double max_velocity{10}, max_control{8}, obstacle_margin{0};
        std::vector<Circle> circles;
    };
    explicit LinearModel(Config config) : config_(std::move(config)) {
        if (!config_.start.finite() || !config_.goal.allFinite() || !config_.lower.allFinite() ||
            !config_.upper.allFinite() || (config_.lower.array() >= config_.upper.array()).any() ||
            !std::isfinite(config_.max_velocity) || config_.max_velocity <= 0 ||
            !std::isfinite(config_.max_control) || config_.max_control <= 0 ||
            !std::isfinite(config_.obstacle_margin) || config_.obstacle_margin < 0)
            throw std::invalid_argument("Invalid linear model configuration");
        for (const auto &c : config_.circles)
            if (!c.center.allFinite() || !std::isfinite(c.radius) || c.radius <= 0)
                throw std::invalid_argument("Invalid circle obstacle");
        screen_limits_.lower = config_.lower;
        screen_limits_.upper = config_.upper;
        screen_limits_.velocity = config_.max_velocity;
        screen_limits_.control = config_.max_control;
        for (const auto &c : config_.circles)
            screen_obstacles_.push_back({c.center, c.radius + config_.obstacle_margin});
    }
    FlatOutput metricWeights() const {
        return FlatOutput::Ones();
    }
    FlatOutput metricPeriods() const {
        return FlatOutput::Zero();
    }
    bool screenTrajectory(const Trajectory &tr, std::size_t samples, bool interleaved,
                          PlanResult &stats) const {
        return screen_.screen(tr, screen_limits_, screen_obstacles_, samples, interleaved, true,
                              &stats.simd_flat_samples);
    }
    const FlatState &start() const {
        return config_.start;
    }
    Metadata initialMetadata() const {
        return {};
    }
    bool prepareEdge(const FlatState &, const Metadata &, Trajectory &) const {
        return true;
    }
    Metadata terminalMetadata(const Trajectory &trajectory) const {
        return trajectory.metadata;
    }
    const FlatOutput &goalOutput() const {
        return config_.goal;
    }
    const FlatOutput &lowerOutput() const {
        return config_.lower;
    }
    const FlatOutput &upperOutput() const {
        return config_.upper;
    }
    double distanceToOutput(const FlatState &state, const FlatOutput &output) const {
        return (state.output() - output).norm();
    }
    double flatDistance(const FlatState &a, const FlatState &b) const {
        return (a.values - b.values).norm();
    }
    FlatOutput limitExtension(const FlatState &start, const FlatOutput &target,
                              double range) const {
        const FlatOutput delta = target - start.output();
        const double d = delta.norm();
        if (range > 0 && d > range)
            return start.output() + delta * (range / d);
        return target;
    }
    bool initialStateValid() const {
        return positionValid(config_.start.output()) &&
               (config_.start.derivative(1).array().abs() <= config_.max_velocity).all();
    }
    bool goalOutputValid() const {
        return positionValid(config_.goal);
    }
    bool isGoal(const FlatState &state, double tolerance) const {
        return distanceToOutput(state, config_.goal) <= tolerance;
    }
    double solutionLowerBound(const FlatState &via) const {
        return ((via.output() - config_.start.output()).norm() +
                (config_.goal - via.output()).norm()) /
               (std::sqrt(2.0) * config_.max_velocity);
    }
    bool trajectoryValid(const Trajectory &tr, PlanResult &stats) const {
        if (!std::isfinite(tr.T) || tr.T <= 0 || !tr.coefficients.allFinite())
            return false;
        // Check scalar polynomial extrema, rather than a fixed sparse sample grid.
        for (int axis = 0; axis < 2; ++axis)
            for (int derivative = 0; derivative <= 2; ++derivative) {
                std::vector<long double> polynomial(4 - derivative);
                for (int k = derivative; k < 4; ++k)
                    polynomial[k - derivative] =
                        tr.coefficients(axis, k) * factorial(k) / factorial(k - derivative);
                const auto range = polynomialRange(polynomial, tr.T);
                const double lower = derivative == 0 ? config_.lower(axis)
                                                     : -(derivative == 1 ? config_.max_velocity
                                                                         : config_.max_control);
                const double upper = derivative == 0 ? config_.upper(axis)
                                                     : (derivative == 1 ? config_.max_velocity
                                                                        : config_.max_control);
                if (range.first < lower - 1e-8 || range.second > upper + 1e-8)
                    return false;
            }
        for (const auto &obstacle : config_.circles) {
            ++stats.collision_check_count;
            std::vector<long double> squared(7, 0);
            for (int axis = 0; axis < 2; ++axis) {
                std::array<long double, 4> p{};
                for (int k = 0; k < 4; ++k)
                    p[k] = tr.coefficients(axis, k);
                p[0] -= obstacle.center(axis);
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        squared[i + j] += p[i] * p[j];
            }
            const double r = obstacle.radius + config_.obstacle_margin;
            if (polynomialRange(squared, tr.T).first <= r * r + 1e-10)
                return false;
        }
        return true;
    }

  private:
    static std::pair<double, double> polynomialRange(const std::vector<long double> &p, double T) {
        std::vector<long double> dp;
        for (std::size_t i = 1; i < p.size(); ++i)
            dp.push_back(i * p[i]);
        auto times = realRootsInInterval(dp, 0, T);
        times.push_back(0);
        times.push_back(T);
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (double t : times) {
            long double v = 0;
            for (auto it = p.rbegin(); it != p.rend(); ++it)
                v = v * t + *it;
            if (!std::isfinite(static_cast<double>(v)))
                return {-std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity()};
            lo = std::min(lo, static_cast<double>(v));
            hi = std::max(hi, static_cast<double>(v));
        }
        return {lo, hi};
    }
    bool positionValid(const FlatOutput &p) const {
        if (!p.allFinite() || (p.array() < config_.lower.array()).any() ||
            (p.array() > config_.upper.array()).any())
            return false;
        for (const auto &c : config_.circles)
            if ((p - c.center).norm() <= c.radius + config_.obstacle_margin)
                return false;
        return true;
    }
    using Screen = SIMDCollisionScreen<FlatDim, FlatOrder>;
    Screen screen_;
    typename Screen::Limits screen_limits_;
    std::vector<typename Screen::Sphere> screen_obstacles_;
    Config config_;
};
} // namespace rrtpff
