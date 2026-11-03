#pragma once
#include "systems/dyno_model.h"
#include <yaml-cpp/yaml.h>

namespace rrtpff {
template <int N> Eigen::Matrix<double, N, 1> liftVector(const YAML::Node &node) {
    if (!node.IsSequence() || node.size() != N)
        throw std::invalid_argument("Missing explicit LiftData vector");
    Eigen::Matrix<double, N, 1> result;
    for (int i = 0; i < N; ++i)
        result(i) = node[i].as<double>();
    if (!result.allFinite())
        throw std::invalid_argument("Nonfinite LiftData");
    return result;
}
inline std::string dfMode(const YAML::Node &yaml) {
    const auto mode = yaml["df"] ? yaml["df"]["mode"].as<std::string>("full_df") : "full_df";
    if (mode != "paper_r2" && mode != "full_df")
        throw std::invalid_argument("df.mode must be paper_r2 or full_df");
    return mode;
}
inline void validatePaperFairConfig(const YAML::Node &yaml) {
    const auto sampling = yaml["sampling"];
    if (yaml["profile"].as<std::string>("") == "paper_fair") {
        if ((sampling && sampling["guidance_probability"].as<double>(0) != 0) ||
            (sampling && (sampling["regions"] || sampling["passage_hints"] ||
                          sampling["passage_hint"] || sampling["tilt_hints"] || sampling["tilt_hint"])))
            throw std::invalid_argument("paper_fair excludes artificial map guidance");
        const auto search = yaml["planar_search"];
        if (search && (search["tilt_hints"] || search["tilt_hint"] || search["regions"] ||
                       search["passage_hints"] || search["passage_hint"]))
            throw std::invalid_argument("paper_fair excludes artificial passage/tilt hints");
    }
    if (dfMode(yaml) == "paper_r2" && yaml["system"].as<std::string>("") != "unicycle") {
        for (const char *endpoint : {"start", "goal"}) {
            const auto lift = yaml["lifting"] ? yaml["lifting"][endpoint] : YAML::Node();
            if (lift && (lift["acceleration"] || lift["jerk"] || lift["yaw_acceleration"] ||
                         lift["yaw_jerk"] || lift["yaw_rate"]))
                throw std::invalid_argument("paper_r2 derives yaw rate from physical state and excludes higher-order lifting");
        }
    }
}
template <class Model> Model configureSampling(Model model, const YAML::Node &yaml) {
    validatePaperFairConfig(yaml);
    std::vector<typename Model::Region> regions;
    if (yaml["sampling"] && yaml["sampling"]["regions"])
        for (const auto &region : yaml["sampling"]["regions"])
            regions.push_back({liftVector<Model::FlatDim>(region["lower"]),
                               liftVector<Model::FlatDim>(region["upper"])});
    const double probability =
        yaml["sampling"] ? yaml["sampling"]["guidance_probability"].as<double>(0) : 0;
    model.setSamplingGuidance(std::move(regions), probability);
    PlanarSearchConfig search;
    if (yaml["planar_search"]) {
        const auto n = yaml["planar_search"];
        search.turn_limited_sampling = n["turn_limited_sampling"].as<bool>(false);
        search.terminal_speed = n["terminal_speed"].as<double>(0);
        search.turn_safety_factor = n["turn_safety_factor"].as<double>(.7);
        if (n["duration_factors"])
            search.duration_factors = n["duration_factors"].as<std::vector<double>>();
        search.rest_hint = n["rest_hint"].as<bool>(false);
        if (n["tilt_hints"])
            for (const auto &hint : n["tilt_hints"])
                search.tilt_hints.push_back(
                    {{liftVector<2>(hint["lower"]), liftVector<2>(hint["upper"])},
                     liftVector<2>(hint["velocity"]),
                     hint["theta"].as<double>(),
                     hint["force_ratio"].as<double>()});
    }
    model.setPlanarSearch(std::move(search));
    return model;
}
inline PaperUnicycleModel makeUnicycle(PaperDataAdapter adapter, const YAML::Node &yaml,
                                       RobotValidationConfig v) {
    const auto &d = adapter.problem();
    if (d.robot_type.find("unicycle1_") != 0)
        throw std::invalid_argument("Expected DynoBench unicycle1");
    UnicycleDF::Config c;
    c.allow_reverse = d.lower_control(0) < 0;
    c.max_velocity = d.max_velocity;
    c.max_omega = d.max_omega;
    c.velocity_lower = d.lower_control(0);
    c.velocity_upper = d.upper_control(0);
    c.omega_lower = d.lower_control(1);
    c.omega_upper = d.upper_control(1);
    const UnicycleLift a{yaml["lifting"]["start"]["signed_velocity"].as<double>()};
    const UnicycleLift b{yaml["lifting"]["goal"]["signed_velocity"].as<double>()};
    if (a.signed_velocity * b.signed_velocity <= 0)
        throw std::invalid_argument(
            "This planner preserves one branch; explicit stop/branch switching is not implemented");
    if (a.signed_velocity < d.lower_control(0) || a.signed_velocity > d.upper_control(0) ||
        b.signed_velocity < d.lower_control(0) || b.signed_velocity > d.upper_control(0))
        throw std::invalid_argument("Unicycle lifting velocity outside actual DynoBench bounds");
    return configureSampling(PaperUnicycleModel(std::move(adapter), a, b, c, v, a.signed_velocity),
                             yaml);
}
inline Quad2DLift quad2Lift(const YAML::Node &node) {
    Quad2DLift lift;
    lift.acceleration = liftVector<2>(node["acceleration"]);
    lift.jerk = liftVector<2>(node["jerk"]);
    return lift;
}
template <class DF = Quad2DDF>
inline DynoSystemModel<DF> makeQuad2D(PaperDataAdapter adapter, const YAML::Node &yaml,
                                   RobotValidationConfig v) {
    const auto &d = adapter.problem();
    if (d.robot_type.find("quad2d_") != 0)
        throw std::invalid_argument("Expected DynoBench quad2d");
    Quad2DDF::Config c;
    c.mass = d.mass;
    c.gravity = d.gravity;
    c.inertia = d.inertia_2d;
    c.arm_length = d.arm_length;
    c.max_motor = d.max_motor;
    c.max_thrust = d.mass * d.gravity * d.max_motor;
    c.max_velocity = d.max_velocity;
    c.max_omega = d.max_omega;
    c.max_acceleration = yaml["df_limits"]["max_acceleration"].as<double>();
    c.max_angular_acceleration = yaml["df_limits"]["max_angular_acceleration"].as<double>();
    typename DF::LiftData start_lift{}, goal_lift{};
    if constexpr (DF::FlatState::IntegratorOrder == 4) {
        start_lift = quad2Lift(yaml["lifting"]["start"]);
        goal_lift = quad2Lift(yaml["lifting"]["goal"]);
    }
    return configureSampling(DynoSystemModel<DF>(std::move(adapter), start_lift, goal_lift, c, v), yaml);
}
inline Quad3DLift quad3Lift(const YAML::Node &node) {
    Quad3DLift lift;
    lift.acceleration = liftVector<3>(node["acceleration"]);
    lift.jerk = liftVector<3>(node["jerk"]);
    lift.yaw_rate = node["yaw_rate"].as<double>();
    lift.yaw_acceleration = node["yaw_acceleration"].as<double>();
    lift.yaw_jerk = node["yaw_jerk"].as<double>();
    return lift;
}
template <class DF = Quad3DDF>
inline DynoSystemModel<DF> makeQuad3D(PaperDataAdapter adapter, const YAML::Node &yaml,
                                   RobotValidationConfig v) {
    const auto &d = adapter.problem();
    if (d.robot_type.find("quad3d_") != 0)
        throw std::invalid_argument("Expected DynoBench quad3d");
    Quad3DDF::Config c;
    c.mass = d.mass;
    c.gravity = d.gravity;
    c.inertia = d.inertia_3d;
    auto actuation = std::make_shared<WrenchControlMap>();
    actuation->inverse = d.wrench_to_control;
    actuation->lower = d.lower_control;
    actuation->upper = d.upper_control;
    c.control_map = std::move(actuation);
    c.arm_length = d.arm_length;
    c.thrust_to_torque = d.thrust_to_torque;
    // In wrench mode the first normalized actuator is total thrust / (m*g).
    c.max_motor = d.motor_control
                      ? d.max_motor
                      : d.control_to_wrench(0, 0) * d.upper_control(0) / (d.mass * d.gravity);
    c.max_velocity = d.max_velocity;
    c.max_omega = d.max_omega;
    c.max_acceleration = yaml["df_limits"]["max_acceleration"].as<double>();
    c.max_angular_acceleration = yaml["df_limits"]["max_angular_acceleration"].as<double>();
    typename DF::LiftData start_lift{}, goal_lift{};
    if constexpr (DF::FlatState::IntegratorOrder == 4) {
        start_lift = quad3Lift(yaml["lifting"]["start"]);
        goal_lift = quad3Lift(yaml["lifting"]["goal"]);
    }
    return configureSampling(DynoSystemModel<DF>(std::move(adapter), start_lift, goal_lift, c, v, 0,
                                              yaml["sampling"]["yaw_lower"].as<double>(),
                                              yaml["sampling"]["yaw_upper"].as<double>()),
                             yaml);
}
} // namespace rrtpff
