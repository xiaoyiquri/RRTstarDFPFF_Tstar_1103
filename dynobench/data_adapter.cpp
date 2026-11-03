#include "dynobench/data_adapter.h"
#include "dynobench/conservative_sphere.h"
#include <Eigen/LU>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
namespace rrtpff {
namespace {
Eigen::VectorXd vector(const YAML::Node &n, int size) {
    if (!n.IsSequence() || n.size() != static_cast<std::size_t>(size))
        throw std::invalid_argument("Wrong YAML vector dimension");
    Eigen::VectorXd v(size);
    for (int i = 0; i < size; ++i)
        v(i) = n[i].as<double>();
    if (!v.allFinite())
        throw std::invalid_argument("Nonfinite YAML vector");
    return v;
}
double positive(const YAML::Node &n, const char *key) {
    double v = n[key].as<double>();
    if (!std::isfinite(v) || v <= 0)
        throw std::invalid_argument(key);
    return v;
}
void writeVector(std::ostream &out, const Eigen::VectorXd &v) {
    out << '[';
    for (int i = 0; i < v.size(); ++i)
        out << (i ? ", " : "") << v(i);
    out << ']';
}
} // namespace
PaperDataAdapter::PaperDataAdapter(const std::string &environment_file,
                                   const std::string &model_file, double radius, double cell,
                                   double margin) {
    if (!std::isfinite(radius) || radius <= 0 || !std::isfinite(cell) || cell <= 0 ||
        !std::isfinite(margin) || margin < 0)
        throw std::invalid_argument("Invalid explicit sphere geometry parameters");
    const auto env = YAML::LoadFile(environment_file);
    if (!env["robots"].IsSequence() || env["robots"].size() != 1)
        throw std::invalid_argument("Exactly one robot required");
    const auto robot = env["robots"][0];
    auto &d = data_;
    d.robot_type = robot["type"].as<std::string>();
    if (d.robot_type.find('/') != std::string::npos ||
        d.robot_type.find('\\') != std::string::npos ||
        d.robot_type.find("..") != std::string::npos)
        throw std::invalid_argument("Invalid robot model name");
    if (model_file.empty())
        throw std::invalid_argument("A resolved DynoBench model file is required");
    auto m = YAML::LoadFile(model_file);
    const auto dynamics = m["dynamics"].as<std::string>();
    bool uni = dynamics == "unicycle1", q2 = dynamics == "quad2d", q3 = dynamics == "quad3d";
    if (!uni && !q2 && !q3)
        throw std::invalid_argument("Unsupported dynamics");
    const int nx = uni ? 3 : (q2 ? 6 : 13), nu = q3 ? 4 : 2, pdim = q3 ? 3 : 2;
    d.start = vector(robot["start"], nx);
    d.goal = vector(robot["goal"], nx);
    if (q3 && (std::abs(d.start.segment<4>(3).norm() - 1) > 1e-6 ||
               std::abs(d.goal.segment<4>(3).norm() - 1) > 1e-6))
        throw std::invalid_argument("Quad3D endpoints require unit xyzw quaternions");
    d.lower_position = vector(env["environment"]["min"], pdim);
    d.upper_position = vector(env["environment"]["max"], pdim);
    d.lower_state = Eigen::VectorXd::Constant(nx, -std::numeric_limits<double>::infinity());
    d.upper_state = -d.lower_state;
    d.lower_state.head(pdim) = d.lower_position;
    d.upper_state.head(pdim) = d.upper_position;
    d.reference_dt = positive(m, "dt");
    d.max_velocity = positive(m, "max_vel");
    d.max_omega = positive(m, "max_angular_vel");
    if (uni) {
        d.lower_control.resize(2);
        d.upper_control.resize(2);
        d.lower_control << m["min_vel"].as<double>(), m["min_angular_vel"].as<double>();
        d.upper_control << d.max_velocity, d.max_omega;
    } else {
        d.mass = positive(m, "m");
        d.gravity = m["g"].as<double>(9.81);
        // The audited DynoBench source uses its fixed g=9.81 in both models.
        if (d.gravity != 9.81 || m["drag_against_vel"].as<bool>(false))
            throw std::invalid_argument("Only no-drag g=9.81 profile is supported");
        d.max_motor = positive(m, "max_f");
        d.lower_control = Eigen::VectorXd::Zero(nu);
        d.upper_control = Eigen::VectorXd::Constant(nu, d.max_motor);
        const int vel = q2 ? 3 : 7, angular = q2 ? 5 : 10;
        d.lower_state.segment(vel, pdim).setConstant(-d.max_velocity);
        d.upper_state.segment(vel, pdim).setConstant(d.max_velocity);
        d.lower_state.segment(angular, q2 ? 1 : 3).setConstant(-d.max_omega);
        d.upper_state.segment(angular, q2 ? 1 : 3).setConstant(d.max_omega);
        if (q2) {
            d.inertia_2d = positive(m, "I");
            d.arm_length = positive(m, "l");
        } else {
            d.inertia_3d = vector(m["J_v"], 3);
            d.arm_length = positive(m, "arm_length");
            d.thrust_to_torque = positive(m, "t2t");
            if ((d.inertia_3d.array() <= 0).any())
                throw std::invalid_argument("Nonpositive inertia");
            d.motor_control = m["motor_control"].as<bool>();
            if (d.motor_control) {
                const double a = .707106781 * d.arm_length, k = d.thrust_to_torque;
                d.control_to_wrench << 1, 1, 1, 1, -a, -a, a, a, -a, a, a, -a, -k, k, -k, k;
                d.control_to_wrench *= d.mass * d.gravity / 4;
            } else {
                d.control_to_wrench.diagonal() << d.mass * d.gravity, 20, 20, 20;
                d.lower_control = vector(m["u_lb"], 4);
                d.upper_control = vector(m["u_ub"], 4);
            }
            d.wrench_to_control = d.control_to_wrench.inverse();
        }
    }
    if (!d.lower_control.allFinite() || !d.upper_control.allFinite() ||
        (d.lower_control.array() > d.upper_control.array()).any())
        throw std::invalid_argument("Invalid actuator bounds");
    geometry_.dimension = pdim;
    geometry_.robot_radius =
        std::nextafter(static_cast<float>(radius + margin), std::numeric_limits<float>::infinity());
    for (const auto &obstacle : env["environment"]["obstacles"]) {
        const auto type = obstacle["type"].as<std::string>();
        const auto center = vector(obstacle["center"], pdim);
        if (type == "sphere") {
            std::array<double,3> exact{};
            for(int k=0;k<pdim;++k)exact[k]=center(k);
            const auto s=conservativeSphere(exact,positive(obstacle,"radius"),pdim);
            geometry_.spheres.push_back(s);
            continue;
        }
        if (type != "box" || obstacle["rotation"] || obstacle["quaternion"])
            throw std::invalid_argument(
                "Unsupported obstacle; expected axis-aligned box or sphere");
        auto size = vector(obstacle["size"], pdim);
        if ((size.array() <= 0).any())
            throw std::invalid_argument("Invalid box size");
        std::array<int, 3> count{{1, 1, 1}};
        Eigen::Vector3d step = Eigen::Vector3d::Zero();
        double total = 1;
        for (int k = 0; k < pdim; ++k) {
            double cells = std::ceil(size(k) / cell);
            if (cells > 1000000)
                throw std::invalid_argument("Sphere cover too dense");
            count[k] = int(cells);
            step(k) = size(k) / cells;
            total *= cells;
        }
        if (total + geometry_.spheres.size() > 1000000)
            throw std::invalid_argument("Sphere cover exceeds one million primitives");
        const double r = .5 * step.norm();
        for (int i = 0; i < count[0]; ++i)
            for (int j = 0; j < count[1]; ++j)
                for (int k = 0; k < count[2]; ++k) {
                    int index[3] = {i, j, k};
                    std::array<double,3> exact{};
                    for (int axis = 0; axis < pdim; ++axis) {
                        exact[axis] =
                            center(axis) - size(axis) / 2 + (index[axis] + .5) * step(axis);
                    }
                    const auto s=conservativeSphere(exact,r,pdim);
                    geometry_.spheres.push_back(s);
                }
    }
    geometry_.build();
}
void PaperDataAdapter::writeGeometry(const std::string &file) const {
    std::ofstream out(file);
    if (!out)
        throw std::runtime_error("Cannot export geometry");
    out << std::setprecision(9);
    out << "validation_backend: simd_discrete_float8\ngeometry: sphere_union\nposition_dimension: "
        << geometry_.dimension << "\nrobot_radius_with_margin: " << geometry_.robot_radius
        << "\nobstacles:\n";
    for (const auto &s : geometry_.spheres)
        out << "  - {center: [" << s.center[0] << ", " << s.center[1] << ", " << s.center[2]
            << "], radius: " << s.radius << "}\n";
}
void PaperDataAdapter::writeTrajectory(const DynoSamples &s, const std::string &file, double cost,
                                       double max_step, const PlanResult *metadata) const {
    std::ofstream out(file);
    if (!out)
        throw std::runtime_error("Cannot export trajectory");
    out << std::setprecision(17);
    if (metadata) {
        out << "df_mode: " << metadata->df_mode
            << "\npseudo_control_order: " << metadata->pseudo_control_order
            << "\nboundary_contract: " << metadata->boundary_contract
            << "\nreduced_order_result: " << (metadata->reduced_order_result ? "true" : "false")
            << "\nphysical_boundary_match: " << (metadata->physical_boundary_match ? "true" : "false")
            << "\nnumeric_profile: " << metadata->numeric_profile
            << "\nsolver_precision: " << metadata->solver_precision
            << "\nvalidation_precision: float32\ntime_root_precision: " << metadata->time_root_precision
            << "\njunction_attitude_omega_policy: "
            << (metadata->reduced_order_result ? "diagnostic_only" : "enforced")
            << "\nphysical_continuity_certified: false\n";
    }
    out << "validation_backend: simd_discrete_float8\nsampled_feasible: true\ndynobench_verified: "
           "false\ncontinuous_collision_certified: false\n"
        << "sample_step_upper_bound: " << max_step
        << "\ngeometry_file: geometry.yaml\ncost: " << cost << "\nstart: ";
    writeVector(out, data_.start);
    out << "\ngoal: ";
    writeVector(out, data_.goal);
    out << "\nstates:\n";
    for (const auto &x : s.states) {
        out << "  - ";
        writeVector(out, x);
        out << '\n';
    }
    out << "actions:\n";
    for (const auto &u : s.controls) {
        out << "  - ";
        writeVector(out, u);
        out << '\n';
    }
    out << "times: [";
    for (std::size_t i = 0; i < s.times.size(); ++i)
        out << (i ? ", " : "") << s.times[i];
    out << "]\n";
}
} // namespace rrtpff
