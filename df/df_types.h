#pragma once

#include "common/flat_control.h"
#include "common/flat_state.h"
#include "common/flat_trajectory.h"
#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <limits>
#include <variant>

namespace rrtpff {

enum class VelocityBranch {
    Forward,
    Reverse
};

struct UnicycleTrajectoryMetadata {
    VelocityBranch branch = VelocityBranch::Forward;
    double signed_start_velocity = std::numeric_limits<double>::quiet_NaN();
};

struct UnicycleLift {
    double signed_velocity = std::numeric_limits<double>::quiet_NaN();
};

struct Quad2DLift {
    Eigen::Vector2d acceleration = Eigen::Vector2d::Constant(
        std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector2d jerk = Eigen::Vector2d::Constant(
        std::numeric_limits<double>::quiet_NaN());
};

struct Quad3DLift {
    Eigen::Vector3d acceleration = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d jerk = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::quiet_NaN());
    double yaw_rate = std::numeric_limits<double>::quiet_NaN();
    double yaw_acceleration = std::numeric_limits<double>::quiet_NaN();
    // A complete FlatState<4,4> also needs the third yaw derivative.
    double yaw_jerk = std::numeric_limits<double>::quiet_NaN();
};

struct FeasibilityConfig {
    std::size_t sample_count = 64;
    double velocity_epsilon = 1e-6;
    double force_epsilon = 1e-8;
    double attitude_epsilon = 1e-8;
};

using UnicycleFlatTrajectory =
    FlatTrajectory<2, 2, UnicycleTrajectoryMetadata>;

}  // namespace rrtpff
