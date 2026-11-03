#pragma once

#include "df/df_types.h"
#include "df/differential_flatness.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <variant>

namespace rrtpff {

// The adapter supplies the mapping from physical [thrust, body torque] to
// the external actuator units; DF does not guess external normalizations.
struct WrenchControlMap {
    Eigen::Matrix4d inverse = Eigen::Matrix4d::Identity();
    Eigen::Vector4d lower = Eigen::Vector4d::Zero();
    Eigen::Vector4d upper = Eigen::Vector4d::Ones();
};

struct Quad3DDFConfig {
    double mass = 0.034;
    double gravity = 9.81;
    double arm_length = 0.046;
    double thrust_to_torque = 0.006;
    Eigen::Vector3d inertia{16.571710e-6, 16.655602e-6, 29.261652e-6};
    double max_motor = 1.3;
    double max_velocity = 4.0;
    double max_acceleration = 25.0;
    double max_jerk = std::numeric_limits<double>::infinity();
    double max_omega = 8.0;
    double max_angular_acceleration = 20.0;
    double singular_epsilon = 1e-8;
    double lift_tolerance = 1e-6;
    std::shared_ptr<const WrenchControlMap> control_map;
};

struct Quad3DDF {
    using FlatState = rrtpff::FlatState<4, 4>;
    using FlatControl = rrtpff::FlatControl<4>;
    using FlatTrajectory = rrtpff::FlatTrajectory<4, 4>;
    using PhysicalState = Eigen::Matrix<double, 13, 1>;
    using PhysicalControl = Eigen::Matrix<double, 4, 1>;
    using LiftData = Quad3DLift;

    using Config = Quad3DDFConfig;

    static bool stateToFlat(const PhysicalState &state, const LiftData &lift, FlatState &flat,
                            Config config = {}) {
        if (!validConfig(config) || !state.allFinite() || !lift.acceleration.allFinite() ||
            !lift.jerk.allFinite() || !std::isfinite(lift.yaw_rate) ||
            !std::isfinite(lift.yaw_acceleration) || !std::isfinite(lift.yaw_jerk)) {
            return false;
        }
        Eigen::Quaterniond quaternion(state(6), state(3), state(4), state(5));
        if (!quaternion.coeffs().allFinite() || quaternion.norm() <= config.singular_epsilon) {
            return false;
        }
        quaternion.normalize();
        const Eigen::Matrix3d rotation = quaternion.toRotationMatrix();

        FlatState::Output value = FlatState::Output::Zero();
        value.head<3>() = state.head<3>();
        value(3) = std::atan2(rotation(1, 0), rotation(0, 0));
        flat.setDerivative(0, value);
        value.setZero();
        value.head<3>() = state.segment<3>(7);
        value(3) = lift.yaw_rate;
        flat.setDerivative(1, value);
        value.setZero();
        value.head<3>() = lift.acceleration;
        value(3) = lift.yaw_acceleration;
        flat.setDerivative(2, value);
        value.setZero();
        value.head<3>() = lift.jerk;
        value(3) = lift.yaw_jerk;
        flat.setDerivative(3, value);
        if (!flat.finite())
            return false;
        FlatControl zero_control;
        FrameKinematics frame;
        if (!frameKinematics(flat, zero_control, frame, config))
            return false;
        const Eigen::Matrix3d attitude_error = frame.rotation.transpose() * rotation;
        const Eigen::Vector3d recovered_omega =
            vee(frame.rotation.transpose() * frame.rotation_dot);
        return (attitude_error - Eigen::Matrix3d::Identity()).norm() <= config.lift_tolerance &&
               (recovered_omega - state.tail<3>()).norm() <= config.lift_tolerance;
    }

    static bool flatToPhysicalState(const FlatState &flat, const FlatControl &flat_control,
                                    const std::monostate &, PhysicalState &state,
                                    Config config = {}) {
        FrameKinematics frame;
        if (!frameKinematics(flat, flat_control, frame, config)) {
            return false;
        }
        return stateFromFrame(flat, frame, state, config);
    }

  private:
    template <class Frame>
    static bool stateFromFrame(const FlatState &flat, const Frame &frame, PhysicalState &state,
                               const Config &config) {
        Eigen::Quaterniond quaternion(frame.rotation);
        if (!quaternion.coeffs().allFinite() || quaternion.norm() <= config.singular_epsilon) {
            return false;
        }
        quaternion.normalize();
        state.setZero();
        state.head<3>() = flat.output().head<3>();
        state.segment<4>(3) << quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w();
        state.segment<3>(7) = flat.derivative(1).head<3>();
        state.tail<3>() = vee(frame.rotation.transpose() * frame.rotation_dot);
        return state.allFinite();
    }

  public:
    static bool flatToPhysicalControl(const FlatState &flat, const FlatControl &flat_control,
                                      const std::monostate &, PhysicalControl &control,
                                      Config config = {}) {
        FrameKinematics frame;
        if (!frameKinematics(flat, flat_control, frame, config) || config.mass <= 0.0 ||
            config.gravity <= 0.0 || config.arm_length <= 0.0 || config.thrust_to_torque <= 0.0 ||
            (config.inertia.array() <= 0.0).any()) {
            return false;
        }
        return controlFromFrame(flat, frame, control, config);
    }

  private:
    template <class Frame>
    static bool controlFromFrame(const FlatState &flat, const Frame &frame,
                                 PhysicalControl &control, const Config &config) {
        const Eigen::Vector3d force =
            flat.derivative(2).head<3>() + Eigen::Vector3d(0.0, 0.0, config.gravity);
        const double thrust = config.mass * force.norm();
        const Eigen::Vector3d omega = vee(frame.rotation.transpose() * frame.rotation_dot);
        const Eigen::Matrix3d inertia = config.inertia.asDiagonal();
        const Eigen::Vector3d omega_dot =
            vee(frame.rotation_dot.transpose() * frame.rotation_dot +
                frame.rotation.transpose() * frame.rotation_double_dot);
        const Eigen::Vector3d torque = inertia * omega_dot + omega.cross(inertia * omega);
        Eigen::Vector4d wrench;
        wrench << thrust, torque;
        if (config.control_map) {
            control = config.control_map->inverse * wrench;
            return control.allFinite() &&
                   (control.array() >= config.control_map->lower.array() - 1e-9).all() &&
                   (control.array() <= config.control_map->upper.array() + 1e-9).all();
        }
        const double arm = 0.7071067811865475244 * config.arm_length;
        const double nominal = config.mass * config.gravity / 4.0;
        Eigen::Matrix4d allocation;
        allocation << 1.0, 1.0, 1.0, 1.0, -arm, -arm, arm, arm, -arm, arm, arm, -arm,
            -config.thrust_to_torque, config.thrust_to_torque, -config.thrust_to_torque,
            config.thrust_to_torque;
        allocation *= nominal;
        control = allocation.fullPivLu().solve(wrench);
        return control.allFinite() && control.minCoeff() >= -1e-8 &&
               control.maxCoeff() <= config.max_motor + 1e-8;
    }

  public:
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
        const Eigen::Vector3d force =
            flat.derivative(2).head<3>() + Eigen::Vector3d(0, 0, config.gravity);
        if (force.norm() <= config.singular_epsilon)
            return failFeasibility(reason, FeasibilityFailure::Singularity);
        if (flat.derivative(1).head<3>().norm() > config.max_velocity + 1e-9)
            return failFeasibility(reason, FeasibilityFailure::SpeedMax);
        if (flat.derivative(2).head<3>().norm() > config.max_acceleration + 1e-9)
            return failFeasibility(reason, FeasibilityFailure::Acceleration);
        if (flat.derivative(3).head<3>().norm() > config.max_jerk + 1e-9)
            return failFeasibility(reason, FeasibilityFailure::Jerk);
        if (force.norm() > config.gravity * config.max_motor + 1e-9)
            return failFeasibility(reason, FeasibilityFailure::Thrust);
        FrameKinematics frame;
        if (!frameKinematics(flat, input, frame, config))
            return failFeasibility(reason, FeasibilityFailure::Singularity);
        const Eigen::Vector3d omega = vee(frame.rotation.transpose() * frame.rotation_dot);
        const Eigen::Vector3d alpha = vee(frame.rotation_dot.transpose() * frame.rotation_dot +
                                          frame.rotation.transpose() * frame.rotation_double_dot);
        if (omega.norm() > config.max_omega + 1e-9)
            return failFeasibility(reason, FeasibilityFailure::Omega);
        if (alpha.norm() > config.max_angular_acceleration + 1e-9)
            return failFeasibility(reason, FeasibilityFailure::AngularAcceleration);
        if (!stateFromFrame(flat, frame, state, config))
            return failFeasibility(reason, FeasibilityFailure::Singularity);
        if (!controlFromFrame(flat, frame, control, config))
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

    static bool rotation(const FlatState &flat, Eigen::Matrix3d &output, Config config = {}) {
        FlatControl flat_control;
        FrameKinematics frame;
        if (!frameKinematics(flat, flat_control, frame, config)) {
            return false;
        }
        output = frame.rotation;
        return true;
    }

  private:
    struct FrameKinematics {
        Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d rotation_dot = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d rotation_double_dot = Eigen::Matrix3d::Zero();
    };

    static bool validConfig(const Config &config) {
        if (config.control_map &&
            (!config.control_map->inverse.allFinite() || !config.control_map->lower.allFinite() ||
             !config.control_map->upper.allFinite() ||
             (config.control_map->lower.array() > config.control_map->upper.array()).any()))
            return false;
        return std::isfinite(config.mass) && config.mass > 0.0 && std::isfinite(config.gravity) &&
               config.gravity > 0.0 && std::isfinite(config.arm_length) &&
               config.arm_length > 0.0 && std::isfinite(config.thrust_to_torque) &&
               config.thrust_to_torque > 0.0 && config.inertia.allFinite() &&
               (config.inertia.array() > 0.0).all() && std::isfinite(config.max_motor) &&
               config.max_motor > 0.0 && std::isfinite(config.max_velocity) &&
               config.max_velocity >= 0.0 && !std::isnan(config.max_acceleration) &&
               config.max_acceleration >= 0.0 &&
               (std::isfinite(config.max_jerk) ||
                config.max_jerk == std::numeric_limits<double>::infinity()) &&
               config.max_jerk >= 0.0 && std::isfinite(config.max_omega) &&
               config.max_omega >= 0.0 && !std::isnan(config.max_angular_acceleration) &&
               config.max_angular_acceleration >= 0.0 && std::isfinite(config.singular_epsilon) &&
               config.singular_epsilon > 0.0 && std::isfinite(config.lift_tolerance) &&
               config.lift_tolerance >= 0.0;
    }

    static bool frameKinematics(const FlatState &flat, const FlatControl &flat_control,
                                FrameKinematics &frame, Config config) {
        if (!validConfig(config) || !flat.finite() || !flat_control.finite()) {
            return false;
        }
        const Eigen::Vector3d force =
            flat.derivative(2).head<3>() + Eigen::Vector3d(0.0, 0.0, config.gravity);
        const Eigen::Vector3d force_dot = flat.derivative(3).head<3>();
        const Eigen::Vector3d force_double_dot = flat_control.values.head<3>();
        Eigen::Vector3d b3;
        Eigen::Vector3d b3_dot;
        Eigen::Vector3d b3_double_dot;
        if (!normalizeWithDerivatives(force, force_dot, force_double_dot, b3, b3_dot, b3_double_dot,
                                      config)) {
            return false;
        }

        const double yaw = flat.output()(3);
        const double yaw_rate = flat.derivative(1)(3);
        const double yaw_acceleration = flat.derivative(2)(3);
        const Eigen::Vector3d heading(std::cos(yaw), std::sin(yaw), 0.0);
        const Eigen::Vector3d heading_dot(-std::sin(yaw) * yaw_rate, std::cos(yaw) * yaw_rate, 0.0);
        const Eigen::Vector3d heading_double_dot(
            -std::cos(yaw) * yaw_rate * yaw_rate - std::sin(yaw) * yaw_acceleration,
            -std::sin(yaw) * yaw_rate * yaw_rate + std::cos(yaw) * yaw_acceleration, 0.0);
        const Eigen::Vector3d lateral = b3.cross(heading);
        const Eigen::Vector3d lateral_dot = b3_dot.cross(heading) + b3.cross(heading_dot);
        const Eigen::Vector3d lateral_double_dot = b3_double_dot.cross(heading) +
                                                   2.0 * b3_dot.cross(heading_dot) +
                                                   b3.cross(heading_double_dot);
        Eigen::Vector3d b2;
        Eigen::Vector3d b2_dot;
        Eigen::Vector3d b2_double_dot;
        if (!normalizeWithDerivatives(lateral, lateral_dot, lateral_double_dot, b2, b2_dot,
                                      b2_double_dot, config)) {
            return false;
        }
        const Eigen::Vector3d b1 = b2.cross(b3);
        const Eigen::Vector3d b1_dot = b2_dot.cross(b3) + b2.cross(b3_dot);
        const Eigen::Vector3d b1_double_dot =
            b2_double_dot.cross(b3) + 2.0 * b2_dot.cross(b3_dot) + b2.cross(b3_double_dot);
        frame.rotation.col(0) = b1;
        frame.rotation.col(1) = b2;
        frame.rotation.col(2) = b3;
        frame.rotation_dot.col(0) = b1_dot;
        frame.rotation_dot.col(1) = b2_dot;
        frame.rotation_dot.col(2) = b3_dot;
        frame.rotation_double_dot.col(0) = b1_double_dot;
        frame.rotation_double_dot.col(1) = b2_double_dot;
        frame.rotation_double_dot.col(2) = b3_double_dot;
        return frame.rotation.allFinite() && frame.rotation_dot.allFinite() &&
               frame.rotation_double_dot.allFinite() &&
               std::abs(frame.rotation.determinant() - 1.0) < 1e-6;
    }

    static bool normalizeWithDerivatives(const Eigen::Vector3d &value, const Eigen::Vector3d &first,
                                         const Eigen::Vector3d &second, Eigen::Vector3d &unit,
                                         Eigen::Vector3d &unit_first, Eigen::Vector3d &unit_second,
                                         Config config) {
        if (!value.allFinite() || !first.allFinite() || !second.allFinite()) {
            return false;
        }
        const double norm = value.norm();
        if (!std::isfinite(norm) || norm <= config.singular_epsilon) {
            return false;
        }
        unit = value / norm;
        const Eigen::Matrix3d projector = Eigen::Matrix3d::Identity() - unit * unit.transpose();
        const double norm_first = unit.dot(first);
        unit_first = projector * first / norm;
        unit_second = projector * second / norm - unit * unit_first.squaredNorm() -
                      2.0 * (norm_first / norm) * unit_first;
        return unit.allFinite() && unit_first.allFinite() && unit_second.allFinite();
    }

    static Eigen::Vector3d vee(const Eigen::Matrix3d &matrix) {
        const Eigen::Matrix3d skew = 0.5 * (matrix - matrix.transpose());
        Eigen::Vector3d result;
        result << skew(2, 1), skew(0, 2), skew(1, 0);
        return result;
    }

    static bool validTime(const FlatTrajectory &trajectory, double t) {
        return std::isfinite(trajectory.T) && trajectory.T > 0.0 && std::isfinite(t) && t >= 0.0 &&
               t <= trajectory.T && trajectory.coefficients.allFinite();
    }
};

} // namespace rrtpff
