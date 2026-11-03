#pragma once
#include "common/types.h"
#include <ostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
namespace rrtpff {
inline void writeBoundaryDiagnostics(const std::filesystem::path &file, const PlanResult &r) {
    std::ofstream out(file);
    if (!out) throw std::runtime_error("Cannot write boundary diagnostics");
    out << std::setprecision(17)
        << "kind,edge_index,time,flat_position_error,flat_velocity_error,physical_position_error,"
           "physical_velocity_error,attitude_error_rad,omega_error,df_mode,boundary_contract\n";
    for (const auto &d : r.boundary_diagnostics) {
        const char *kind = d.kind == 0 ? "start" : d.kind == 1 ? "junction" : "goal";
        out << kind << ',' << d.edge_index << ',' << d.time << ',' << d.flat_position_error
            << ',' << d.flat_velocity_error << ',' << d.physical.position << ','
            << d.physical.velocity << ',' << d.physical.attitude << ',' << d.physical.omega
            << ',' << r.df_mode << ',' << r.boundary_contract << '\n';
    }
}
inline void writeCSVHeader(std::ostream &out) {
    out << "run,seed,success,first_solution_time,planning_time,cost,trajectory_duration,path_"
           "length,nodes,pff_evaluations,simd_evaluations,dynobench_validations,collision_checks,"
           "df_time,pff_time,simd_time,dynobench_time,rewires,iterations,rejected_candidates,pff_"
           "rejected,df_rejected,flat_bound_rejected,state_bound_rejected,control_bound_rejected,"
           "collision_rejected,"
           "simd_active,simd_pff_lanes,simd_flat_samples,simd_nn_lanes,screen_rejected,pff_queries,"
           "nn_time,screen_time,validation_time,failure_reason,dynamics_rejected,goal_queries,goal_"
           "valid,closest_goal_distance";
    for (std::size_t i = 1; i < failureNames.size(); ++i)
        out << ",df_" << failureNames[i] << "_rejected";
    out << ",candidate_prechecks,candidate_precheck_rejected,collision_precheck_queries,"
           "collision_precheck_rejected,goal_cache_hits,goal_cache_exhausted,dense_"
           "reconstructions,pff_materializations,simd_collision_samples,simd_validation_queries,"
           "simd_validation_batches,validation_backend,df_mode,pseudo_control_order,boundary_contract,"
           "reduced_order_result,physical_boundary_match,solver_precision,validation_precision,"
           "time_root_precision,numeric_profile,termination_reason,completed_normally,first_solution_cost,"
           "search_time,shortcut_time,total_time,final_validation_time,export_time,raw_cost,raw_length,"
           "raw_duration,shortcut_cost,shortcut_length,shortcut_duration,shortcut_enabled,shortcut_mode,"
           "shortcut_time_mode,shortcut_attempts,shortcut_accepted,shortcut_fallback,speed_certificate_rejected,"
           "speed_certificate_uncertain\n";
}
inline void writeCSVRow(std::ostream &out, std::size_t run, unsigned seed, const PlanResult &r) {
    out << run << ',' << seed << ',' << r.success << ',' << r.first_solution_time << ','
        << r.planning_time << ',' << r.cost << ',' << r.duration << ',' << r.path_length << ','
        << r.node_count << ',' << r.pff_eval_count << ',' << r.simd_eval_count << ','
        << r.dyno_validation_count << ',' << r.collision_check_count << ',' << r.df_time << ','
        << r.pff_time << ',' << r.simd_time << ',' << r.dyno_validation_time << ','
        << r.rewire_count << ',' << r.iterations << ',' << r.rejected_candidates << ','
        << r.pff_rejected << ',' << r.df_rejected << ',' << r.flat_bound_rejected << ','
        << r.state_bound_rejected << ',' << r.control_bound_rejected << ',' << r.collision_rejected
        << ',' << r.simd_active << ',' << r.simd_pff_lanes << ',' << r.simd_flat_samples << ','
        << r.simd_nn_lanes << ',' << r.screen_rejected << ',' << r.pff_query_count << ','
        << r.nn_time << ',' << r.screen_time << ',' << r.validation_time << ',' << r.failure_reason
        << ',' << r.dynamics_rejected << ',' << r.goal_connection_queries << ','
        << r.goal_connection_valid << ',' << r.closest_goal_distance;
    for (std::size_t i = 1; i < failureNames.size(); ++i)
        out << ',' << r.df_failures[i];
    out << ',' << r.candidate_prechecks << ',' << r.candidate_precheck_rejected << ','
        << r.collision_precheck_queries << ',' << r.collision_precheck_rejected << ','
        << r.goal_cache_hits << ',' << r.goal_cache_exhausted << ',' << r.dense_reconstructions
        << ',' << r.pff_materializations << ',' << r.simd_collision_samples << ','
        << r.simd_validation_queries << ',' << r.simd_validation_batches << ','
        << r.validation_backend << ',' << r.df_mode << ',' << r.pseudo_control_order << ','
        << r.boundary_contract << ',' << r.reduced_order_result << ',' << r.physical_boundary_match
        << ',' << r.solver_precision << ",float32," << r.time_root_precision
        << ',' << r.numeric_profile << ',' << r.termination_reason << ',' << r.completed_normally
        << ',' << r.first_solution_cost << ',' << r.search_time << ',' << r.shortcut_time << ',' << r.total_time
        << ',' << r.final_validation_time << ',' << r.export_time << ',' << r.raw_cost << ',' << r.raw_length
        << ',' << r.raw_duration << ',' << r.shortcut_cost << ',' << r.shortcut_length << ',' << r.shortcut_duration
        << ',' << r.shortcut_enabled << ',' << r.shortcut_mode << ',' << r.shortcut_time_mode
        << ',' << r.shortcut_attempts << ',' << r.shortcut_accepted << ',' << r.shortcut_fallback
        << ',' << r.speed_certificate_rejected << ',' << r.speed_certificate_uncertain << '\n';
}
} // namespace rrtpff
