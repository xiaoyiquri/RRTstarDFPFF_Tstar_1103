#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>
namespace rrtpff {
template<class Real = double> struct PlannerConfigT {
    double timeout{10};
    Real goal_bias{Real(.1)}, range{4}, rewire_factor{3};
    Real goal_tolerance{Real(.05)};
    std::size_t max_iterations{10000}, max_nodes{4001};
    std::size_t hard_safety_node_cap{1000000};
    unsigned seed{40};
    bool simd_enabled{false};
    bool simd_pff{true}, simd_nn{true}, simd_screen{true};
    bool interleaved_samples{true};
    std::size_t collision_samples{33};
    bool prune{false}, stop_at_first_solution{false};
    bool rewire_before_first_solution{true};
    bool enable_rewire{true};
    bool explicit_goal_connection{false};
    std::size_t goal_parent_limit{16};
    // Optional test/baseline prechecks. Robot models expose one SIMD validator.
    bool candidate_precheck{false}, cache_goal_connections{false};
    bool model_time_hints{false};
    std::size_t minimum_parent_candidates{0};
    void validate() const {
        if (!std::isfinite(timeout) || timeout <= 0 || !std::isfinite(range) || range <= 0 ||
            !std::isfinite(goal_bias) || goal_bias < 0 || goal_bias > 1 ||
            !std::isfinite(rewire_factor) || rewire_factor <= 0 || !std::isfinite(goal_tolerance) ||
            goal_tolerance < 0 || (max_nodes && max_nodes < 2) || !goal_parent_limit ||
            (hard_safety_node_cap && hard_safety_node_cap < 2) ||
            minimum_parent_candidates > 1000000 || collision_samples < 2 ||
            collision_samples > 1000000)
            throw std::invalid_argument("Invalid planner configuration");
    }
};
using PlannerConfig = PlannerConfigT<double>;
} // namespace rrtpff
