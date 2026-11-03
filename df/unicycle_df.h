#pragma once

#include "common/math_utils.h"
#include "df/df_types.h"
#include "df/differential_flatness.h"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace rrtpff {

struct UnicycleDFConfig {
    double velocity_epsilon = 1e-6;
    bool allow_reverse = false;
    double max_velocity = .5;
    double max_omega = .5;
    double start_tolerance = 1e-8;
    // Actual signed actuator intervals, intersected with the legacy magnitude
    // limits. Infinite defaults preserve existing DF reference configurations.
    double velocity_lower = -std::numeric_limits<double>::infinity();
    double velocity_upper = std::numeric_limits<double>::infinity();
    double omega_lower = -std::numeric_limits<double>::infinity();
    double omega_upper = std::numeric_limits<double>::infinity();
};

struct UnicycleDF {
    using FlatState = rrtpff::FlatState<2, 2>;
    using FlatControl = rrtpff::FlatControl<2>;
    using FlatTrajectory = rrtpff::UnicycleFlatTrajectory;
    using PhysicalState = Eigen::Matrix<double, 3, 1>;
    using PhysicalControl = Eigen::Matrix<double, 2, 1>;
    using LiftData = UnicycleLift;

    using Config = UnicycleDFConfig;

    static bool stateToFlat(const PhysicalState &state, const LiftData &lift, FlatState &flat,
                            Config config = {}) {
        if (!validConfig(config) || !state.allFinite() || !std::isfinite(lift.signed_velocity) ||
            std::abs(lift.signed_velocity) <= config.velocity_epsilon ||
            std::abs(lift.signed_velocity) > config.max_velocity + 1e-12 ||
            lift.signed_velocity < config.velocity_lower ||
            lift.signed_velocity > config.velocity_upper ||
            (!config.allow_reverse && lift.signed_velocity < 0.)) {
            return false;
        }
        flat.values.setZero();
        FlatState::Output position;
        position << state(0), state(1);
        flat.setDerivative(0, position);

        FlatState::Output velocity;
        velocity << lift.signed_velocity * std::cos(state(2)),
            lift.signed_velocity * std::sin(state(2));
        flat.setDerivative(1, velocity);
        return flat.finite();
    }

    static bool makeMetadata(double signed_velocity, UnicycleTrajectoryMetadata &metadata,
                             Config config = {}) {
        if (!validConfig(config) || !std::isfinite(signed_velocity) ||
            std::abs(signed_velocity) <= config.velocity_epsilon ||
            std::abs(signed_velocity) > config.max_velocity + 1e-12 ||
            signed_velocity < config.velocity_lower || signed_velocity > config.velocity_upper ||
            (signed_velocity < 0. && !config.allow_reverse)) {
            return false;
        }
        metadata.signed_start_velocity = signed_velocity;
        metadata.branch = signed_velocity > 0. ? VelocityBranch::Forward : VelocityBranch::Reverse;
        return true;
    }

    static bool validMetadata(const UnicycleTrajectoryMetadata &metadata, Config config = {}) {
        if (!validConfig(config) || !validBranch(metadata.branch) ||
            !std::isfinite(metadata.signed_start_velocity) ||
            std::abs(metadata.signed_start_velocity) <= config.velocity_epsilon ||
            (metadata.branch == VelocityBranch::Reverse && !config.allow_reverse)) {
            return false;
        }
        const bool reverse = metadata.signed_start_velocity < 0.;
        return reverse == (metadata.branch == VelocityBranch::Reverse);
    }

    static bool flatToPhysicalState(const FlatState &flat, VelocityBranch branch,
                                    PhysicalState &state, Config config = {}) {
        if (!validConfig(config) || !validBranch(branch) || !flat.finite()) {
            return false;
        }
        const auto velocity = flat.derivative(1);
        const double speed = velocity.norm();
        if (!std::isfinite(speed) || speed <= config.velocity_epsilon ||
            (branch == VelocityBranch::Reverse && !config.allow_reverse)) {
            return false;
        }

        const double direction = std::atan2(velocity(1), velocity(0));
        const double heading = branch == VelocityBranch::Reverse
                                   ? direction + 3.141592653589793238462643383279502884
                                   : direction;
        state << flat.output(), wrapAngle(heading);
        return state.allFinite();
    }

    static bool flatToPhysicalState(const FlatState &flat,
                                    const UnicycleTrajectoryMetadata &metadata,
                                    PhysicalState &state, Config config = {}) {
        return validMetadata(metadata, config) &&
               flatToPhysicalState(flat, metadata.branch, state, config);
    }

    static bool flatToPhysicalState(const FlatState &flat, const FlatControl &flat_control,
                                    const UnicycleTrajectoryMetadata &metadata,
                                    PhysicalState &state, Config config = {}) {
        return flat_control.finite() && flatToPhysicalState(flat, metadata, state, config);
    }

    static bool flatToPhysicalControl(const FlatState &flat, const FlatControl &flat_control,
                                      VelocityBranch branch, PhysicalControl &control,
                                      Config config = {}) {
        if (!validConfig(config) || !validBranch(branch) || !flat.finite() ||
            !flat_control.finite() ||
            (branch == VelocityBranch::Reverse && !config.allow_reverse)) {
            return false;
        }
        const auto velocity = flat.derivative(1);
        const double denominator = velocity.squaredNorm();
        if (!std::isfinite(denominator) ||
            denominator <= config.velocity_epsilon * config.velocity_epsilon) {
            return false;
        }

        const double speed = std::sqrt(denominator);
        const double signed_speed = branch == VelocityBranch::Reverse ? -speed : speed;
        const auto acceleration = flat_control.values;
        const double omega =
            (velocity(0) * acceleration(1) - velocity(1) * acceleration(0)) / denominator;
        control << signed_speed, omega;
        return control.allFinite();
    }

    static bool flatToPhysicalControl(const FlatState &flat, const FlatControl &flat_control,
                                      const UnicycleTrajectoryMetadata &metadata,
                                      PhysicalControl &control, Config config = {}) {
        return validMetadata(metadata, config) &&
               flatToPhysicalControl(flat, flat_control, metadata.branch, control, config);
    }

    static bool trajectoryState(const FlatTrajectory &trajectory, double t, PhysicalState &state,
                                Config config = {}) {
        if (!validTime(trajectory, t) || !validStart(trajectory, config)) {
            return false;
        }
        return flatToPhysicalState(trajectory.evaluateState(t), trajectory.metadata, state, config);
    }

    static bool trajectoryControl(const FlatTrajectory &trajectory, double t,
                                  PhysicalControl &control, Config config = {}) {
        if (!validTime(trajectory, t) || !validStart(trajectory, config)) {
            return false;
        }
        return flatToPhysicalControl(trajectory.evaluateState(t), trajectory.evaluateControl(t),
                                     trajectory.metadata, control, config);
    }

    static bool sampleFeasible(const FlatState &flat, const FlatControl &flat_control,
                               const UnicycleTrajectoryMetadata &metadata, PhysicalState &state,
                               PhysicalControl &control, Config config = {},
                               FeasibilityFailure *reason = nullptr) {
        if (reason)
            *reason = FeasibilityFailure::None;
        if (!flat.finite() || !flat_control.finite() || !validConfig(config))
            return failFeasibility(reason, FeasibilityFailure::Numeric);
        if (!validMetadata(metadata, config))
            return failFeasibility(reason, FeasibilityFailure::Branch);
        const double speed = flat.derivative(1).norm();
        if (speed <= config.velocity_epsilon)
            return failFeasibility(reason, FeasibilityFailure::SpeedZero);
        if (!flatToPhysicalState(flat, flat_control, metadata, state, config) ||
            !flatToPhysicalControl(flat, flat_control, metadata, control, config))
            return failFeasibility(reason, FeasibilityFailure::Numeric);
        const double min_speed = metadata.branch == VelocityBranch::Reverse
                                     ? std::max(0.0, -config.velocity_upper)
                                     : std::max(0.0, config.velocity_lower);
        const double max_speed =
            std::min(config.max_velocity, metadata.branch == VelocityBranch::Reverse
                                              ? -config.velocity_lower
                                              : config.velocity_upper);
        if (speed < min_speed - 1e-10)
            return failFeasibility(reason, FeasibilityFailure::SpeedMin);
        if (speed > max_speed + 1e-10)
            return failFeasibility(reason, FeasibilityFailure::SpeedMax);
        if (control(1) < std::max(-config.max_omega, config.omega_lower) - 1e-10 ||
            control(1) > std::min(config.max_omega, config.omega_upper) + 1e-10)
            return failFeasibility(reason, FeasibilityFailure::Omega);
        return true;
    }

    static bool feasible(const FlatTrajectory &trajectory, Config config = {},
                         FeasibilityConfig sampling = {}, FeasibilityFailure *reason = nullptr) {
        if (reason)
            *reason = FeasibilityFailure::None;
        if (!validTime(trajectory, 0.) || sampling.sample_count == 0 ||
            !std::isfinite(sampling.velocity_epsilon) || sampling.velocity_epsilon <= 0. ||
            !validConfig(config)) {
            return failFeasibility(reason, FeasibilityFailure::Numeric);
        }
        if (!validStart(trajectory, config))
            return failFeasibility(reason, FeasibilityFailure::Branch);
        // A cubic position gives a quartic speed squared. Check every
        // stationary point, so an unsampled stop cannot hide a branch change.
        const double epsilon = std::max(config.velocity_epsilon, sampling.velocity_epsilon);
        std::vector<long double> speed_squared(5, 0.0L);
        for (int axis = 0; axis < 2; ++axis) {
            long double velocity[3];
            for (int k = 0; k < 3; ++k) {
                velocity[k] = (k + 1) *
                              static_cast<long double>(trajectory.coefficients(axis, k + 1)) *
                              std::pow(static_cast<long double>(trajectory.T), k);
            }
            for (int k = 0; k < 3; ++k) {
                for (int j = 0; j < 3; ++j) {
                    speed_squared[k + j] += velocity[k] * velocity[j];
                }
            }
        }
        std::vector<long double> slope(4);
        for (int k = 1; k <= 4; ++k) {
            slope[k - 1] = k * speed_squared[k];
            if (!std::isfinite(slope[k - 1]))
                return failFeasibility(reason, FeasibilityFailure::Numeric);
        }
        auto critical = realRootsInInterval(slope, 0., 1.);
        critical.push_back(0.);
        critical.push_back(1.);
        for (double fraction : critical) {
            const double speed = trajectory.velocity(fraction * trajectory.T).norm();
            if (!std::isfinite(speed))
                return failFeasibility(reason, FeasibilityFailure::Numeric);
            if (speed <= epsilon)
                return failFeasibility(reason, FeasibilityFailure::SpeedZero);
            PhysicalState state;
            PhysicalControl control;
            if (!sampleFeasible(trajectory.evaluateState(fraction * trajectory.T),
                                trajectory.evaluateControl(fraction * trajectory.T),
                                trajectory.metadata, state, control, config, reason))
                return false;
        }
        for (std::size_t i = 0; i <= sampling.sample_count; ++i) {
            const double t =
                trajectory.T * static_cast<double>(i) / static_cast<double>(sampling.sample_count);
            PhysicalState state;
            PhysicalControl control;
            if (!sampleFeasible(trajectory.evaluateState(t), trajectory.evaluateControl(t),
                                trajectory.metadata, state, control, config, reason))
                return false;
        }
        return true;
    }

    // Explicitly attach semantics to a mathematical PFF curve. Never infer a
    // sign from the flat velocity, whose norm has lost that information.
    static bool attachMetadata(const rrtpff::FlatTrajectory<2, 2> &source, const LiftData &lift,
                               FlatTrajectory &trajectory, Config config = {}) {
        FlatTrajectory candidate;
        if (!makeMetadata(lift.signed_velocity, candidate.metadata, config))
            return false;
        candidate.T = source.T;
        candidate.cost = source.cost;
        candidate.coefficients = source.coefficients;
        candidate.start = source.start;
        candidate.end = source.end;
        if (!validTime(candidate, 0.) || !validStart(candidate, config))
            return false;
        trajectory = std::move(candidate);
        return true;
    }

  private:
    static bool validConfig(const Config &config) {
        return std::isfinite(config.velocity_epsilon) && config.velocity_epsilon > 0. &&
               std::isfinite(config.start_tolerance) && config.start_tolerance >= 0. &&
               std::isfinite(config.max_velocity) && config.max_velocity > 0. &&
               std::isfinite(config.max_omega) && config.max_omega >= 0. &&
               !std::isnan(config.velocity_lower) && !std::isnan(config.velocity_upper) &&
               config.velocity_lower < config.velocity_upper && !std::isnan(config.omega_lower) &&
               !std::isnan(config.omega_upper) && config.omega_lower < config.omega_upper;
    }

    static bool validBranch(VelocityBranch branch) {
        return branch == VelocityBranch::Forward || branch == VelocityBranch::Reverse;
    }

    static bool validTime(const FlatTrajectory &trajectory, double t) {
        return std::isfinite(trajectory.T) && trajectory.T > 0. && std::isfinite(t) && t >= 0. &&
               t <= trajectory.T && trajectory.coefficients.allFinite();
    }

    static bool validStart(const FlatTrajectory &trajectory, Config config) {
        if (!validMetadata(trajectory.metadata, config))
            return false;
        const double speed = trajectory.coefficients.col(1).norm();
        const double expected = std::abs(trajectory.metadata.signed_start_velocity);
        return std::isfinite(speed) && speed > config.velocity_epsilon &&
               std::abs(speed - expected) <= config.start_tolerance * (1. + expected);
    }

    static double wrapAngle(double angle) {
        constexpr double pi = 3.141592653589793238462643383279502884;
        constexpr double two_pi = 2. * pi;
        angle = std::fmod(angle + pi, two_pi);
        if (angle < 0.) {
            angle += two_pi;
        }
        return angle - pi;
    }
};

} // namespace rrtpff
