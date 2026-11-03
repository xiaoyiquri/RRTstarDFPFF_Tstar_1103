#pragma once

#include "df/df_types.h"
#include "df/differential_flatness.h"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <variant>

namespace rrtpff {

struct Quad2DDFConfig {
    double mass = 0.034;
    double gravity = 9.81;
    double inertia = 1e-4;
    double arm_length = 0.1;
    double max_motor = 1.3;
    double max_velocity = 4.0;
    double max_acceleration = 25.0;
    double max_jerk = std::numeric_limits<double>::infinity();
    double max_thrust = std::numeric_limits<double>::infinity();
    double max_omega = 8.0;
    double max_angular_acceleration = 25.0;
    double singular_epsilon = 1e-8;
    double lift_tolerance = 1e-6;
};

struct Quad2DDF {
    using FlatState = rrtpff::FlatState<2, 4>;
    using FlatControl = rrtpff::FlatControl<2>;
    using FlatTrajectory = rrtpff::FlatTrajectory<2, 4>;
    using PhysicalState = Eigen::Matrix<double, 6, 1>;
    using PhysicalControl = Eigen::Matrix<double, 2, 1>;
    using LiftData = Quad2DLift;
    using Config = Quad2DDFConfig;

    static bool stateToFlat(const PhysicalState &state, const LiftData &lift, FlatState &flat,
                            Config config = {}) {
        if (!validConfig(config) || !state.allFinite() || !lift.acceleration.allFinite() ||
            !lift.jerk.allFinite()) {
            return false;
        }

        double theta = 0.0;
        double omega = 0.0;
        if (!attitudeRates(lift.acceleration, lift.jerk, theta, omega, config)) {
            return false;
        }
        if (std::abs(wrapAngle(theta - state(2))) > config.lift_tolerance ||
            std::abs(omega - state(5)) > config.lift_tolerance) {
            return false;
        }

        FlatState::Output value;
        value << state(0), state(1);
        flat.setDerivative(0, value);
        value << state(3), state(4);
        flat.setDerivative(1, value);
        flat.setDerivative(2, lift.acceleration);
        flat.setDerivative(3, lift.jerk);
        return flat.finite();
    }

    static bool flatToPhysicalState(const FlatState &flat, const FlatControl &flat_control,
                                    const std::monostate &, PhysicalState &state,
                                    Config config = {}) {
        if (!validConfig(config) || !flat.finite() || !flat_control.finite()) {
            return false;
        }

        double theta = 0.0;
        double omega = 0.0;
        if (!attitudeRates(flat.derivative(2), flat.derivative(3), theta, omega, config)) {
            return false;
        }

        state << flat.output()(0), flat.output()(1), theta, flat.derivative(1)(0),
            flat.derivative(1)(1), omega;
        return state.allFinite();
    }

    static bool flatToPhysicalControl(const FlatState &flat, const FlatControl &flat_control,
                                      const std::monostate &, PhysicalControl &control,
                                      Config config = {}) {
        if (!validConfig(config) || !flat.finite() || !flat_control.finite() ||
            config.mass <= 0.0 || config.gravity <= 0.0 || config.inertia <= 0.0 ||
            config.arm_length <= 0.0) {
            return false;
        }

        const Eigen::Vector2d acceleration = flat.derivative(2);
        const Eigen::Vector2d jerk = flat.derivative(3);
        const Eigen::Vector2d snap = flat_control.values;
        const double ax = acceleration(0);
        const double azg = acceleration(1) + config.gravity;
        const double denominator = ax * ax + azg * azg;
        if (!std::isfinite(denominator) ||
            denominator <= config.singular_epsilon * config.singular_epsilon) {
            return false;
        }

        double angular_acceleration;
        if (!angularAcceleration(acceleration, jerk, snap, angular_acceleration, config))
            return false;
        const double thrust = config.mass * std::sqrt(denominator);
        const double torque = config.inertia * angular_acceleration;
        const double force_difference = torque / config.arm_length;
        const double motor_scale = config.mass * config.gravity / 2.0;

        control(0) = (thrust + force_difference) / (2.0 * motor_scale);
        control(1) = (thrust - force_difference) / (2.0 * motor_scale);
        return control.allFinite() && control.minCoeff() >= -1e-9 &&
               control.maxCoeff() <= config.max_motor + 1e-9;
    }

    static bool trajectoryState(const FlatTrajectory &trajectory, double t, PhysicalState &state,
                                Config config = {}) {
        if (!validTime(trajectory, t)) {
            return false;
        }
        return flatToPhysicalState(trajectory.evaluateState(t), trajectory.evaluateControl(t),
                                   trajectory.metadata, state, config);
    }

    static bool trajectoryControl(const FlatTrajectory &trajectory, double t,
                                  PhysicalControl &control, Config config = {}) {
        if (!validTime(trajectory, t)) {
            return false;
        }
        return flatToPhysicalControl(trajectory.evaluateState(t), trajectory.evaluateControl(t),
                                     trajectory.metadata, control, config);
    }

    static bool sampleFeasible(const FlatState &flat, const FlatControl &input,
                               const std::monostate &metadata, PhysicalState &state,
                               PhysicalControl &control, Config config = {},
                               FeasibilityFailure *reason = nullptr) {
        if (reason)
            *reason = FeasibilityFailure::None;
        if (!validConfig(config) || !flat.finite() || !input.finite())
            return failFeasibility(reason, FeasibilityFailure::Numeric);
        const Eigen::Vector2d force = flat.derivative(2) + Eigen::Vector2d(0, config.gravity);
        if (force.norm() <= config.singular_epsilon)
            return failFeasibility(reason, FeasibilityFailure::Singularity);
        if (flat.derivative(1).norm() > config.max_velocity + 1e-9)
            return failFeasibility(reason, FeasibilityFailure::SpeedMax);
        if (flat.derivative(2).norm() > config.max_acceleration + 1e-9)
            return failFeasibility(reason, FeasibilityFailure::Acceleration);
        if (flat.derivative(3).norm() > config.max_jerk + 1e-9)
            return failFeasibility(reason, FeasibilityFailure::Jerk);
        if (config.mass * force.norm() > config.max_thrust + 1e-9)
            return failFeasibility(reason, FeasibilityFailure::Thrust);
        if (!flatToPhysicalState(flat, input, metadata, state, config))
            return failFeasibility(reason, FeasibilityFailure::Singularity);
        if (std::abs(state(5)) > config.max_omega + 1e-9)
            return failFeasibility(reason, FeasibilityFailure::Omega);
        double alpha = 0;
        if (!angularAcceleration(flat.derivative(2), flat.derivative(3), input.values, alpha,
                                 config))
            return failFeasibility(reason, FeasibilityFailure::Numeric);
        if (std::abs(alpha) > config.max_angular_acceleration + 1e-9)
            return failFeasibility(reason, FeasibilityFailure::AngularAcceleration);
        if (!flatToPhysicalControl(flat, input, metadata, control, config))
            return failFeasibility(reason, FeasibilityFailure::Motor);
        return true;
    }
    static bool feasible(const FlatTrajectory &trajectory, Config config = {},
                         FeasibilityConfig sampling = {}, FeasibilityFailure *reason = nullptr) {
        if (reason)
            *reason = FeasibilityFailure::None;
        if (!validConfig(config) || !validTime(trajectory, 0) || sampling.sample_count == 0 ||
            !std::isfinite(sampling.force_epsilon) || sampling.force_epsilon <= 0)
            return failFeasibility(reason, FeasibilityFailure::Numeric);
        config.singular_epsilon = std::max(config.singular_epsilon, sampling.force_epsilon);
        for (std::size_t i = 0; i <= sampling.sample_count; ++i) {
            const double t = trajectory.T * static_cast<double>(i) / sampling.sample_count;
            PhysicalState state;
            PhysicalControl control;
            if (!sampleFeasible(trajectory.evaluateState(t), trajectory.evaluateControl(t),
                                trajectory.metadata, state, control, config, reason))
                return false;
        }
        return true;
    }

  private:
    static bool validConfig(const Config &c) {
        return std::isfinite(c.mass) && c.mass > 0. && std::isfinite(c.gravity) && c.gravity > 0. &&
               std::isfinite(c.inertia) && c.inertia > 0. && std::isfinite(c.arm_length) &&
               c.arm_length > 0. && std::isfinite(c.max_motor) && c.max_motor > 0. &&
               std::isfinite(c.max_velocity) && c.max_velocity >= 0. &&
               !std::isnan(c.max_acceleration) && c.max_acceleration >= 0. &&
               (std::isfinite(c.max_jerk) ||
                c.max_jerk == std::numeric_limits<double>::infinity()) &&
               c.max_jerk >= 0. &&
               (std::isfinite(c.max_thrust) ||
                c.max_thrust == std::numeric_limits<double>::infinity()) &&
               c.max_thrust > 0. && std::isfinite(c.max_omega) && c.max_omega >= 0. &&
               !std::isnan(c.max_angular_acceleration) && c.max_angular_acceleration >= 0. &&
               std::isfinite(c.singular_epsilon) && c.singular_epsilon > 0. &&
               std::isfinite(c.lift_tolerance) && c.lift_tolerance >= 0.;
    }

    static bool attitudeRates(const Eigen::Vector2d &acceleration, const Eigen::Vector2d &jerk,
                              double &theta, double &omega, Config config) {
        if (!acceleration.allFinite() || !jerk.allFinite()) {
            return false;
        }
        const double ax = acceleration(0);
        const double azg = acceleration(1) + config.gravity;
        const double denominator = ax * ax + azg * azg;
        if (!std::isfinite(denominator) ||
            denominator <= config.singular_epsilon * config.singular_epsilon) {
            return false;
        }
        theta = std::atan2(-ax, azg);
        omega = (ax * jerk(1) - azg * jerk(0)) / denominator;
        return std::isfinite(theta) && std::isfinite(omega);
    }

    static bool angularAcceleration(const Eigen::Vector2d &acceleration,
                                    const Eigen::Vector2d &jerk, const Eigen::Vector2d &snap,
                                    double &value, Config config) {
        const double ax = acceleration(0);
        const double azg = acceleration(1) + config.gravity;
        const double denominator = ax * ax + azg * azg;
        if (!acceleration.allFinite() || !jerk.allFinite() || !snap.allFinite() ||
            denominator <= config.singular_epsilon * config.singular_epsilon) {
            return false;
        }
        const double numerator = ax * jerk(1) - azg * jerk(0);
        const double numerator_dot = ax * snap(1) - azg * snap(0);
        const double denominator_dot = 2.0 * (ax * jerk(0) + azg * jerk(1));
        value = (numerator_dot * denominator - numerator * denominator_dot) /
                (denominator * denominator);
        return std::isfinite(value);
    }

    static bool validTime(const FlatTrajectory &trajectory, double t) {
        return std::isfinite(trajectory.T) && trajectory.T > 0.0 && std::isfinite(t) && t >= 0.0 &&
               t <= trajectory.T && trajectory.coefficients.allFinite();
    }

    static double wrapAngle(double angle) {
        constexpr double pi = 3.141592653589793238462643383279502884;
        constexpr double two_pi = 2.0 * pi;
        angle = std::fmod(angle + pi, two_pi);
        if (angle < 0.0) {
            angle += two_pi;
        }
        return angle - pi;
    }
};

} // namespace rrtpff
