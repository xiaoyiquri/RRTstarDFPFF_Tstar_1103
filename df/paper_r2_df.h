#pragma once
#include "df/quad2d_df.h"
#include "df/quad3d_df.h"
#include <Eigen/Geometry>

namespace rrtpff {
// A': reduced graph boundaries only; physical attitude/omega are reconstructed
// from each edge's cubic derivatives and are diagnosed at junctions.
struct QuadR2Lift {};
struct Quad2DR2DF : Quad2DDF {
    using FlatState = rrtpff::FlatState<2, 2>;
    using FlatTrajectory = rrtpff::FlatTrajectory<2, 2>;
    using LiftData = QuadR2Lift;
    static bool stateToFlat(const PhysicalState &x, const LiftData &, FlatState &z,
                            const Config & = {}) {
        if (!x.allFinite()) return false;
        z.setDerivative(0, x.head<2>());
        z.setDerivative(1, x.segment<2>(3));
        return z.finite();
    }
};
struct Quad3DR2DF : Quad3DDF {
    using FlatState = rrtpff::FlatState<4, 2>;
    using FlatTrajectory = rrtpff::FlatTrajectory<4, 2>;
    using LiftData = QuadR2Lift;
    static bool stateToFlat(const PhysicalState &x, const LiftData &, FlatState &z,
                            const Config &c = {}) {
        if (!x.allFinite()) return false;
        Eigen::Quaterniond q(x(6), x(3), x(4), x(5));
        if (q.norm() <= c.singular_epsilon) return false;
        q.normalize();
        const Eigen::Matrix3d rotation = q.toRotationMatrix();
        const Eigen::Vector3d omega = x.tail<3>();
        Eigen::Matrix3d skew;
        skew << 0, -omega.z(), omega.y(), omega.z(), 0, -omega.x(),
                -omega.y(), omega.x(), 0;
        const Eigen::Matrix3d rotation_dot = rotation * skew;
        const double denominator = rotation(0, 0) * rotation(0, 0) +
                                   rotation(1, 0) * rotation(1, 0);
        // Euler yaw is undefined when the body's x axis is vertical.
        if (denominator <= c.singular_epsilon * c.singular_epsilon) return false;
        FlatState::Output y, dy;
        y.head<3>() = x.head<3>();
        y(3) = std::atan2(rotation(1, 0), rotation(0, 0));
        dy.head<3>() = x.segment<3>(7);
        dy(3) = (rotation(0, 0) * rotation_dot(1, 0) -
                 rotation(1, 0) * rotation_dot(0, 0)) / denominator;
        z.setDerivative(0, y);
        z.setDerivative(1, dy);
        return z.finite();
    }
};
} // namespace rrtpff
