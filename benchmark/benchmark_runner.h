#pragma once
#include "benchmark/benchmark_result.h"
#include "core/rrtstar.h"
#include "systems/linear_model.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h>

namespace rrtpff {
#ifdef RRTPFF_WITH_PAPER_SIMD
int runRobotBenchmark(const std::filesystem::path &config_path);
#endif
template <int N> Eigen::Matrix<double, N, 1> readVector(const YAML::Node &node, const char *name) {
    if (!node.IsSequence() || node.size() != N)
        throw std::invalid_argument(std::string("Wrong vector dimension: ") + name);
    Eigen::Matrix<double, N, 1> out;
    for (int i = 0; i < N; ++i)
        out(i) = node[i].as<double>();
    if (!out.allFinite())
        throw std::invalid_argument(std::string("Nonfinite vector: ") + name);
    return out;
}
inline PlannerConfig readPlanner(const YAML::Node &n) {
    PlannerConfig c;
    c.timeout = n["timeout"].as<double>();
    c.goal_bias = n["goal_bias"].as<double>();
    c.range = n["range"].as<double>();
    c.rewire_factor = n["rewire_factor"].as<double>();
    c.goal_tolerance = n["goal_tolerance"].as<double>();
    c.max_iterations = n["max_iterations"].as<std::size_t>();
    c.max_nodes = n["max_nodes"].as<std::size_t>();
    c.hard_safety_node_cap=n["hard_safety_node_cap"].as<std::size_t>(c.hard_safety_node_cap);
    c.prune = n["prune"].as<bool>();
    c.stop_at_first_solution = n["stop_at_first_solution"].as<bool>();
    c.rewire_before_first_solution =
        n["rewire_before_first_solution"].as<bool>(c.rewire_before_first_solution);
    c.explicit_goal_connection = n["explicit_goal_connection"].as<bool>(false);
    //c.goal_connect_radius = n["goal_connect_radius"].as<double>(1.0);
    c.candidate_precheck = n["candidate_precheck"].as<bool>(false);
    c.cache_goal_connections = n["cache_goal_connections"].as<bool>(false);
    c.model_time_hints = n["model_time_hints"].as<bool>(false);
    c.minimum_parent_candidates = n["minimum_parent_candidates"].as<std::size_t>(0);
    c.validate();
    return c;
}
inline int runBenchmark(const std::filesystem::path &config_path) {
    const auto yaml = YAML::LoadFile(config_path.string());
    if (yaml["system"].as<std::string>() != "linear") {
#ifdef RRTPFF_WITH_PAPER_SIMD
        return runRobotBenchmark(config_path);
#else
        throw std::invalid_argument("This Phase 1 build implements only 'linear'. "
                                    "Build the AVX2 paper validation target.");
#endif
    }

    LinearModel::Config model_config;
    model_config.start.values = readVector<4>(yaml["start"], "start");
    model_config.goal = readVector<2>(yaml["goal"], "goal");
    model_config.lower = readVector<2>(yaml["bounds"]["lower"], "bounds.lower");
    model_config.upper = readVector<2>(yaml["bounds"]["upper"], "bounds.upper");
    model_config.max_velocity = yaml["limits"]["max_velocity"].as<double>();
    model_config.max_control = yaml["limits"]["max_control"].as<double>();
    model_config.obstacle_margin = yaml["limits"]["obstacle_margin"].as<double>();
    for (const auto &obstacle : yaml["obstacles"])
        model_config.circles.push_back(
            {readVector<2>(obstacle["center"], "circle.center"), obstacle["radius"].as<double>()});
    const LinearModel model(model_config);
    PlannerConfig planner_config = readPlanner(yaml["planner"]);
    const auto simd = yaml["simd"];
    planner_config.simd_enabled = simd && simd["enabled"].as<bool>(false);
    if (simd) {
        if (simd["width"].as<std::size_t>(8) != 8)
            throw std::invalid_argument(
                "SIMD candidate batch width must be 8; collision samples use one AVX2 float8");
        planner_config.simd_pff = simd["pff"].as<bool>(true);
        planner_config.simd_nn = simd["nearest_neighbor"].as<bool>(true);
        planner_config.simd_screen = simd["collision_screen"].as<bool>(true);
        planner_config.collision_samples = simd["collision_samples"].as<std::size_t>(33);
        planner_config.interleaved_samples = simd["interleaved_samples"].as<bool>(true);
    }
    planner_config.validate();
    std::cout << "Backend: "
              << (selectSIMDBackend(planner_config.simd_enabled) == SIMDBackend::AVX2
                      ? "AVX2 float8 collision samples / double candidate batch 8"
                      : "scalar")
              << '\n';
    PFFConfig pff_config;
    pff_config.min_time = yaml["pff"]["min_time"].as<double>();
    pff_config.max_time = yaml["pff"]["max_time"].as<double>();
    pff_config.fixed_time = yaml["pff"]["fixed_time"].as<double>();
    pff_config.free_terminal_time = yaml["pff"]["free_terminal_time"].as<bool>();
    pff_config.additional_time_samples = yaml["pff"]["additional_time_samples"].as<std::size_t>(0);
    pff_config.defer_trajectories = yaml["pff"]["defer_trajectories"].as<bool>(false);
    if (!yaml["pff"]["free_terminal_state"].as<bool>())
        throw std::invalid_argument("Phase 1 expansion fixes outputs and optimizes terminal "
                                    "derivatives; rewiring fixes the full state.");
    const Eigen::Matrix2d R =
        readVector<2>(yaml["pff"]["R_diagonal"], "pff.R_diagonal").asDiagonal();
    const PFFSolver<2, 2> pff(R, pff_config);
    const std::size_t runs = yaml["benchmark"]["runs"].as<std::size_t>();
    const unsigned seed = yaml["benchmark"]["seed"].as<unsigned>();
    if (runs == 0 || runs - 1 > std::numeric_limits<unsigned>::max() - seed)
        throw std::invalid_argument("Invalid runs or seed overflow");
    auto output = std::filesystem::path(yaml["benchmark"]["output_directory"].as<std::string>());
    if (output.is_relative())
        output = config_path.parent_path() / output;
    std::filesystem::create_directories(output);
    std::ofstream csv(output / "results.csv");
    if (!csv)
        throw std::runtime_error("Cannot create results.csv");
    csv << std::setprecision(17);
    writeCSVHeader(csv);
    std::size_t successes = 0;
    for (std::size_t run = 0; run < runs; ++run) {
        planner_config.seed = seed + static_cast<unsigned>(run);
        RRTstar<LinearModel> planner(model, planner_config, pff);
        const auto result = planner.solve();
        successes += result.success;
        writeCSVRow(csv, run, planner_config.seed, result);
        csv.flush();
        std::cout << "run=" << run << " seed=" << planner_config.seed
                  << " success=" << result.success << " cost=" << result.cost
                  << " time=" << result.planning_time << " nodes=" << result.node_count << '\n';
        if (result.success) {
            std::ofstream trajectory(output / ("trajectory_" + std::to_string(run) + ".csv"));
            if (!trajectory)
                throw std::runtime_error("Cannot create trajectory CSV");
            trajectory << std::setprecision(17) << "edge,t,x,y,vx,vy,ax,ay\n";
            double elapsed = 0;
            std::size_t index = 0;
            for (const auto &edge : planner.path()) {
                for (int k = 0; k <= 100; ++k) {
                    const double t = edge.T * k / 100;
                    const auto x = edge.evaluate(t);
                    const auto u = edge.control(t);
                    trajectory << index << ',' << elapsed + t;
                    for (int i = 0; i < x.values.size(); ++i)
                        trajectory << ',' << x.values(i);
                    for (int i = 0; i < u.size(); ++i)
                        trajectory << ',' << u(i);
                    trajectory << '\n';
                }
                elapsed += edge.T;
                ++index;
            }
        }
    }
    std::cout << "System: Linear double integrator (no DynoBench validation)\nSuccess Rate: "
              << successes << '/' << runs << '\n';
    return successes == runs ? 0 : 2;
}
} // namespace rrtpff
