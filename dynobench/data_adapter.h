#pragma once
#include "common/collision_geometry.h"
#include "common/types.h"
#include "simd/physical_batch.h"
#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

namespace rrtpff {
// This boundary deliberately contains no DynoBench or FCL headers. Physical
// vectors are dynamic here because the external library uses VectorXd; models
// and the planner retain their own compile-time dimensions.
struct DynoProblemData {
    std::string robot_type;
    Eigen::VectorXd start, goal, lower_position, upper_position;
    Eigen::VectorXd lower_state, upper_state, lower_control, upper_control;
    double reference_dt{0};
    // Read from the explicit YAML dynamics profile, mapped using upstream source conventions.
    double mass{0}, gravity{0}, inertia_2d{0}, arm_length{0}, thrust_to_torque{0};
    Eigen::Vector3d inertia_3d = Eigen::Vector3d::Zero();
    double max_velocity{0}, max_omega{0}, max_motor{0};
    bool drag{false}, motor_control{true};
    Eigen::Matrix4d control_to_wrench = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d wrench_to_control = Eigen::Matrix4d::Zero();
    double collision_inner_radius{0};
    std::vector<CollisionAABB> collision_boxes;
};

struct DynoSamples {
    std::vector<Eigen::VectorXd> states;
    std::vector<Eigen::VectorXd> controls;
    std::vector<double> times;
};

struct DynoValidation {
    bool success{false};
    double max_jump{0}, start_error{0}, goal_error{0};
    std::size_t collision_checks{0};
    std::string reason;
};

struct DynoTolerances {
    double dynamics{1e-3};
    double endpoint{1e-3};
};

// YAML-only source adapter. It has NO collision, step, stateValid or checkTrajectory API.
class PaperDataAdapter {
  public:
    PaperDataAdapter(const std::string &environment_file, const std::string &model_file,
                     double robot_radius, double sphere_cell_size, double collision_margin);
    const DynoProblemData &problem() const { return data_; }
    const SphereGeometry &geometry() const { return geometry_; }
    void writeTrajectory(const DynoSamples &, const std::string &, double, double,
                         const PlanResult *metadata = nullptr) const;
    void writeGeometry(const std::string &) const;

  private:
    DynoProblemData data_;
    SphereGeometry geometry_;
};
} // namespace rrtpff
