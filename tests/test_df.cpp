#include "df/quad2d_df.h"
#include "df/quad3d_df.h"
#include "df/unicycle_df.h"
#include "systems/unicycle_model.h"
#include "tests/test_support.h"
#include <Eigen/Geometry>
#include <variant>

using namespace rrtpff;

int main() {
    return runTest([] {
        {
            UnicycleDF::PhysicalState physical;
            physical << 1.0, 2.0, 0.25;
            UnicycleLift lift{-1.0};
            UnicycleDF::FlatState flat;
            UnicycleDF::Config config;
            config.allow_reverse = true;
            config.max_velocity = 2.0;
            require(UnicycleDF::stateToFlat(physical, lift, flat, config),
                    "unicycle reverse lifting");

            UnicycleTrajectoryMetadata metadata;
            require(UnicycleDF::makeMetadata(lift.signed_velocity, metadata,
                                             config),
                    "unicycle metadata");
            require(metadata.branch == VelocityBranch::Reverse,
                    "unicycle reverse branch");

            UnicycleDF::PhysicalState recovered;
            require(UnicycleDF::flatToPhysicalState(
                        flat, metadata, recovered, config),
                    "unicycle reverse inverse");
            close(recovered(0), physical(0), 1e-12,
                  "unicycle x round trip");
            close(recovered(1), physical(1), 1e-12,
                  "unicycle y round trip");
            close(recovered(2), physical(2), 1e-12,
                  "unicycle reverse heading");

            UnicycleDF::FlatControl control;
            control.values << 0.0, 0.0;
            UnicycleDF::PhysicalControl physical_control;
            require(UnicycleDF::flatToPhysicalControl(
                        flat, control, metadata, physical_control, config),
                    "unicycle reverse control");
            close(physical_control(0), -1.0, 1e-12,
                  "unicycle signed velocity");

            UnicycleLift stopped{0.0};
            require(!UnicycleDF::stateToFlat(physical, stopped, flat, config),
                    "unicycle zero speed rejection");

            UnicycleDF::FlatState start;
            start.values.setZero();
            start.setDerivative(1, flat.derivative(1));
            Eigen::Vector2d goal;
            goal << -1.0, 0.0;
            Eigen::Vector2d lower;
            lower << -2.0, -2.0;
            Eigen::Vector2d upper;
            upper << 2.0, 2.0;
            UnicycleModel model(start, goal, lower, upper, UniAdapter{},
                                config, -1.0);
            require(model.initialStateValid(),
                    "unicycle model requires explicit signed start velocity");

            UnicycleFlatTrajectory edge;
            edge.T = 1.0;
            edge.coefficients.setZero();
            edge.coefficients.col(1) = start.derivative(1);
            edge.start = edge.evaluateState(0.0);
            edge.end = edge.evaluateState(edge.T);
            const UnicycleTrajectoryMetadata parent_metadata =
                model.initialMetadata();
            require(model.prepareEdge(start, parent_metadata, edge),
                    "unicycle edge inherits parent branch");
            const UnicycleTrajectoryMetadata terminal_metadata =
                model.terminalMetadata(edge);
            require(terminal_metadata.branch == VelocityBranch::Reverse &&
                        terminal_metadata.signed_start_velocity < 0.0,
                    "unicycle node metadata preserves reverse branch");
        }

        {
            Quad2DDF::FlatState flat;
            flat.values.setZero();
            Quad2DDF::FlatControl flat_control;
            flat_control.values.setZero();
            Quad2DDF::PhysicalState state;
            Quad2DDF::PhysicalControl control;
            require(Quad2DDF::flatToPhysicalState(
                        flat, flat_control, std::monostate{}, state),
                    "quad2d hover state");
            require(Quad2DDF::flatToPhysicalControl(
                        flat, flat_control, std::monostate{}, control),
                    "quad2d hover control");
            close(state(2), 0.0, 1e-12, "quad2d hover attitude");
            close(control(0), 1.0, 1e-12, "quad2d motor one");
            close(control(1), 1.0, 1e-12, "quad2d motor two");

            Quad2DDF::PhysicalState physical;
            physical << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            Quad2DLift lift;
            lift.acceleration << 0.0, 0.0;
            lift.jerk << 0.0, 0.0;
            Quad2DDF::FlatState lifted;
            require(Quad2DDF::stateToFlat(physical, lift, lifted),
                    "quad2d lifting");
        }

        {
            Quad3DDF::FlatState flat;
            flat.values.setZero();
            Quad3DDF::FlatControl flat_control;
            flat_control.values.setZero();
            Quad3DDF::PhysicalState state;
            Quad3DDF::PhysicalControl control;
            require(Quad3DDF::flatToPhysicalState(
                        flat, flat_control, std::monostate{}, state),
                    "quad3d hover state");
            require(Quad3DDF::flatToPhysicalControl(
                        flat, flat_control, std::monostate{}, control),
                    "quad3d hover control");
            Eigen::Quaterniond quaternion(
                state(6), state(3), state(4), state(5));
            close(quaternion.norm(), 1.0, 1e-12,
                  "quad3d quaternion norm");
            close(control(0), 1.0, 1e-12, "quad3d motor one");
            close(control(1), 1.0, 1e-12, "quad3d motor two");
            close(control(2), 1.0, 1e-12, "quad3d motor three");
            close(control(3), 1.0, 1e-12, "quad3d motor four");

            Quad3DDF::PhysicalState physical;
            physical.setZero();
            physical(6) = 1.0;
            Quad3DLift lift;
            lift.acceleration.setZero();
            lift.jerk.setZero();
            lift.yaw_rate = 0.0;
            lift.yaw_acceleration = 0.0;
            lift.yaw_jerk = 0.0;
            Quad3DDF::FlatState lifted;
            require(Quad3DDF::stateToFlat(physical, lift, lifted),
                    "quad3d lifting");

            // A tilted, yawing sample exercises analytic R-dot/R-double-dot
            // recovery and the body-torque allocation path.
            flat.values.setZero();
            Quad3DDF::FlatState::Output acceleration;
            acceleration << 1.0, 0.5, 0.0, 0.0;
            flat.setDerivative(2, acceleration);
            Quad3DDF::FlatState::Output jerk;
            jerk << 0.2, -0.1, 0.3, 0.4;
            flat.setDerivative(3, jerk);
            flat_control.values << 0.1, -0.2, 0.3, 0.0;
            require(Quad3DDF::flatToPhysicalState(
                        flat, flat_control, std::monostate{}, state),
                    "quad3d tilted state");
            require(Quad3DDF::flatToPhysicalControl(
                        flat, flat_control, std::monostate{}, control),
                    "quad3d tilted control");
            require(state.tail<3>().norm() > 1e-8,
                    "quad3d analytic angular velocity");
            require(control.allFinite(), "quad3d tilted finite control");
        }
    });
}
