#pragma once
#include "df/paper_r2_df.h"
#include "df/quad3d_df.h"
#include "df/unicycle_df.h"
#include "dynobench/data_adapter.h"
#include <type_traits>
namespace rrtpff {
template <class DF>
PhysicalBatchConfig makeBatchConfig(const typename DF::Config &c, const DynoProblemData &d) {
    PhysicalBatchConfig b;
    b.state_dim = DF::PhysicalState::RowsAtCompileTime;
    b.control_dim = DF::PhysicalControl::RowsAtCompileTime;
    b.position_dim = std::is_base_of_v<Quad3DDF, DF> ? 3 : 2;
    b.max_velocity = float(c.max_velocity);
    b.max_omega = float(c.max_omega);
    for (int i = 0; i < b.state_dim; ++i) {
        b.state_lower[i] = float(d.lower_state(i));
        b.state_upper[i] = float(d.upper_state(i));
    }
    for (int i = 0; i < b.control_dim; ++i) {
        b.control_lower[i] = float(d.lower_control(i));
        b.control_upper[i] = float(d.upper_control(i));
    }
    if constexpr (std::is_same_v<DF, UnicycleDF>) {
        b.robot = BatchRobot::Unicycle;
        b.epsilon = float(c.velocity_epsilon);
    } else {
        b.mass = float(c.mass);
        b.gravity = float(c.gravity);
        b.arm = float(c.arm_length);
        b.epsilon = float(c.singular_epsilon);
        b.max_acceleration = float(c.max_acceleration);
        b.max_jerk = float(c.max_jerk);
        b.max_alpha = float(c.max_angular_acceleration);
        if constexpr (std::is_base_of_v<Quad2DDF, DF>) {
            b.robot = BatchRobot::Quad2D;
            b.inertia2 = float(c.inertia);
            b.max_thrust = float(c.max_thrust);
        } else {
            b.robot = BatchRobot::Quad3D;
            b.max_thrust = float(c.mass * c.gravity * c.max_motor);
            for (int i = 0; i < 3; ++i)
                b.inertia[i] = float(c.inertia(i));
            for (int i = 0; i < 4; ++i)
                for (int k = 0; k < 4; ++k)
                    b.wrench_inverse[i][k] = float(c.control_map->inverse(i, k));
        }
    }
    return b;
}
} // namespace rrtpff
