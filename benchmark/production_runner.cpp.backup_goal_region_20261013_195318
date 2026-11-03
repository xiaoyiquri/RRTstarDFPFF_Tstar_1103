#include "production/model.h"
#include "core/rrtstar.h"
#include "postprocess/shortcut.h"
#include "benchmark/benchmark_result.h"
#include "benchmark/benchmark_runner.h"
#include <fstream>
#include <iomanip>
#include <iostream>
namespace rrtpff {
namespace {
std::filesystem::path relativeToConfig(const std::filesystem::path& file,const std::string& value) {
    const std::filesystem::path p(value);
    return std::filesystem::weakly_canonical(p.is_absolute()?p:file.parent_path()/p);
}
PlannerConfigT<float> floatPlanner(const YAML::Node& n) {
    PlannerConfigT<float> c;
    c.timeout=n["timeout"].as<double>();c.goal_bias=n["goal_bias"].as<float>();
    c.range=n["range"].as<float>();c.rewire_factor=n["rewire_factor"].as<float>();
    c.goal_tolerance=n["goal_tolerance"].as<float>();
    c.max_iterations=n["max_iterations"].as<std::size_t>(0);
    c.max_nodes=n["max_nodes"].as<std::size_t>(0);
    c.hard_safety_node_cap=n["hard_safety_node_cap"].as<std::size_t>(1000000);
    c.seed=0;c.simd_enabled=true;c.simd_screen=false;
    c.stop_at_first_solution=n["stop_at_first_solution"].as<bool>();
    c.rewire_before_first_solution=n["rewire_before_first_solution"].as<bool>(false);
    c.enable_rewire = n["enable_rewire"].as<bool>(true);
    c.prune=n["prune"].as<bool>(false);
    c.explicit_goal_connection=n["explicit_goal_connection"].as<bool>(true);
    c.goal_parent_limit=n["goal_parent_limit"].as<std::size_t>(16);
    c.cache_goal_connections=n["cache_goal_connections"].as<bool>(true);
    c.minimum_parent_candidates=n["minimum_parent_candidates"].as<std::size_t>(8);
    c.validate();return c;
}
void validateStrict(const YAML::Node& y) {
    if(y["df"]["mode"].as<std::string>()!="paper_r2")
        throw std::invalid_argument("float32_production supports paper_r2 only; use legacy_mixed for full_df");
    if(y["numeric"]["profile"].as<std::string>()!="float32_production")
        throw std::invalid_argument("Production numerical profile mismatch");
    const auto s=y["sampling"],p=y["planar_search"],c=y["planner"];
    if(s["guidance_probability"].as<float>(0.f)!=0.f||
       s["regions"]||s["passage_hint"]||s["passage_hints"]||s["tilt_hint"]||s["tilt_hints"])
        throw std::invalid_argument("Strict production excludes artificial guidance");
    if(p && (p["turn_limited_sampling"].as<bool>(false)||p["terminal_speed"].as<float>(0.f)!=0.f||
             (p["duration_factors"]&&p["duration_factors"].size())||
             p["rest_hint"].as<bool>(false)||p["tilt_hints"]||p["regions"]||p["passage_hints"]))
        throw std::invalid_argument("Strict production excludes system-specific sampling/time/velocity hints");
    if(c["model_time_hints"].as<bool>(false)||c["terminal_velocity_hint"].as<bool>(false))
        throw std::invalid_argument("Strict production excludes planner hints");
    if(y["system"].as<std::string>()!="unicycle" && y["lifting"])
        throw std::invalid_argument("paper_r2 quad lifting consists only of y/dy from the physical endpoint");
    if(y["unicycle"] && y["unicycle"]["branch_mode"].as<std::string>("fixed")!="fixed")
        throw std::invalid_argument("This profile preserves a single fixed signed branch");
    if(y["profile"].as<std::string>()=="flask_strict") {
        const double radius=y["system"].as<std::string>()=="unicycle"?.125:
                            (y["system"].as<std::string>()=="quad2d"?.1:.25);
        if(y["geometry"]["robot_radius"].as<double>()!=radius ||
           y["geometry"]["obstacle_sphere_cell"].as<double>()!=.1 ||
           y["geometry"]["collision_margin"].as<double>()!=.00001 ||
           y["validation"]["max_step"].as<double>()!=.005)
            throw std::invalid_argument("Strict geometry/resolution parameters are project-fixed");
        if(c["max_nodes"].as<std::size_t>()||c["max_iterations"].as<std::size_t>())
            throw std::invalid_argument("Formal strict profile requires unlimited normal node/iteration caps");
    }
}
void decorateProduction(PlanResult& r) {
    r.numeric_profile="float32_production";r.solver_precision="float32";r.time_root_precision="float32";
    r.df_mode="paper_r2";r.pseudo_control_order=2;r.boundary_contract="reduced_flat_A_prime";
    r.reduced_order_result=true;r.validation_backend="simd_discrete_float8";
}
void endpointDiagnostics(const PaperDataAdapter& a,const DynoSamples& samples,float tolerance,PlanResult& r) {
    if(samples.states.empty()||r.boundary_diagnostics.empty())return;
    r.boundary_diagnostics.front().physical=physicalBoundaryError(a.problem().start,samples.states.front());
    r.boundary_diagnostics.back().physical=physicalBoundaryError(samples.states.back(),a.problem().goal);
    r.physical_boundary_match=true;
    for(const auto& d:r.boundary_diagnostics)r.physical_boundary_match&=d.physical.maximum()<=double(tolerance);
}
template<class Model> int productionRuns(const PaperDataAdapter& adapter,const YAML::Node& y,
                                        const std::filesystem::path& file) {
    using Solver=typename Model::Solver;using Real=float;
    production::FloatPFFConfig fc;
    fc.min_time=y["pff"]["min_time"].as<float>();fc.max_time=y["pff"]["max_time"].as<float>();
    fc.fixed_time=y["pff"]["fixed_time"].as<float>();fc.free_terminal_time=y["pff"]["free_terminal_time"].as<bool>();
    fc.additional_time_samples=y["pff"]["additional_time_samples"].as<std::size_t>(0);
    fc.defer_trajectories=y["pff"]["defer_trajectories"].as<bool>(true);
    if(!y["pff"]["free_terminal_state"].as<bool>())throw std::invalid_argument("Expansion retains free terminal velocity");
    typename Solver::Weight R=Solver::Weight::Zero();
    auto diagonal=y["pff"]["R_diagonal"];
    if(!diagonal.IsSequence()||diagonal.size()!=Model::FlatDim)throw std::invalid_argument("Invalid R_diagonal");
    for(int k=0;k<Model::FlatDim;++k)R(k,k)=diagonal[k].as<float>();
    Model model(adapter,y);auto config=floatPlanner(y["planner"]);
    const auto simd=y["simd"];
    config.simd_nn=simd["nearest_neighbor"].as<bool>(true);config.simd_pff=simd["pff"].as<bool>(true);
    ShortcutConfig sc;
    if(y["postprocess"]){
        sc.enabled=y["postprocess"]["shortcut"].as<bool>(true);
        sc.mode=y["postprocess"]["mode"].as<std::string>("flask");
        sc.time_mode=y["postprocess"]["time_mode"].as<std::string>("original_duration");
    }
    if((sc.mode!="flask"&&sc.mode!="cost_decreasing")||(sc.time_mode!="original_duration"&&sc.time_mode!="optimal"))
        throw std::invalid_argument("Unknown shortcut configuration");
    const auto out=relativeToConfig(file,y["benchmark"]["output_directory"].as<std::string>());
    std::filesystem::create_directories(out);
    if(std::filesystem::exists(out/"results.csv"))throw std::runtime_error("Output results already exist");
    std::ofstream csv(out/"results.csv"),validation(out/"validation.csv");
    csv<<std::setprecision(17);writeCSVHeader(csv);
    validation<<"run,planner_success,sampled_success,validation_backend,reason\n";
    adapter.writeGeometry((out/"geometry.yaml").string());
    const auto runs=y["benchmark"]["runs"].as<std::size_t>();
    const auto seed=y["benchmark"]["seed"].as<unsigned>();
    if(!runs||runs-1>std::numeric_limits<unsigned>::max()-seed)throw std::invalid_argument("Invalid runs/seed");
    std::size_t successes=0;
    for(std::size_t run=0;run<runs;++run) {
        config.seed=seed+unsigned(run);PlanResult result;DynoSamples raw_samples,final_samples;
        std::vector<typename Model::FlatTrajectory> raw,first,final_path;
        Timer search;
        try {
            Solver solver(R,fc);
            RRTstar<Model> planner(model,config,solver);
            const double setup=search.seconds();
            result=planner.solve();
            result.search_time=search.seconds();
            if(std::isfinite(result.first_solution_time))result.first_solution_time+=setup;
            for(auto& event:result.best_cost_events)event.first+=setup;
            raw=planner.path();first=planner.firstPath();
        }catch(const std::bad_alloc&){
            result.completed_normally=false;result.termination_reason="safety_limit";
            result.failure_reason="allocation_failure";result.search_time=search.seconds();
        }
        decorateProduction(result);result.planning_time=result.search_time;
        result.shortcut_enabled=sc.enabled;result.shortcut_mode=sc.mode;result.shortcut_time_mode=sc.time_mode;
        const bool planned=result.success;
        bool raw_valid=false,final_valid=false;
        std::string reason=result.failure_reason.empty()?"no_solution":result.failure_reason;
        if(planned) {
            Timer checked;raw_valid=model.validatePath(raw,raw_samples,result).success;
            result.final_validation_time+=checked.seconds();
            if(raw_valid) {
                endpointDiagnostics(adapter,raw_samples,y["validation"]["endpoint_tolerance"].as<float>(),result);
                auto raw_stats=result;
                Real raw_cost=0.f,raw_duration=0.f;
                for(const auto& tr:raw){raw_cost+=tr.cost;raw_duration+=tr.T;}
                result.raw_cost=double(raw_cost);result.raw_duration=double(raw_duration);
                result.raw_length=0;
                constexpr int D=Model::FlatDim==4?3:2;
                for(std::size_t k=1;k<raw_samples.states.size();++k)
                    result.raw_length+=(raw_samples.states[k].template head<D>()-raw_samples.states[k-1].template head<D>()).norm();
                Solver solver(R,fc);
                auto short_result=shortcutPath(raw,model,solver,sc,result);
                result.shortcut_time=short_result.seconds;result.shortcut_attempts=short_result.attempts;
                result.shortcut_accepted=short_result.accepted;final_path=std::move(short_result.path);
                Timer final_check;final_valid=model.validatePath(final_path,final_samples,result).success;
                result.final_validation_time+=final_check.seconds();
                if(!final_valid) {
                    reason="shortcut_final_validation_failed_raw_retained";final_path=raw;final_samples=raw_samples;
                    result.boundary_diagnostics=raw_stats.boundary_diagnostics;result.physical_boundary_match=raw_stats.physical_boundary_match;
                    final_valid=true;result.shortcut_fallback=true;
                } else reason="";
                endpointDiagnostics(adapter,final_samples,y["validation"]["endpoint_tolerance"].as<float>(),result);
                Real final_cost=0.f,final_duration=0.f;
                for(const auto& tr:final_path){final_cost+=tr.cost;final_duration+=tr.T;}
                result.cost=result.shortcut_cost=double(final_cost);
                result.duration=result.shortcut_duration=double(final_duration);
                result.path_length=0;
                for(std::size_t k=1;k<final_samples.states.size();++k)
                    result.path_length+=(final_samples.states[k].template head<D>()-final_samples.states[k-1].template head<D>()).norm();
                result.shortcut_length=result.path_length;
                Timer exported;
                adapter.writeTrajectory(raw_samples,(out/("trajectory_raw_"+std::to_string(run)+".yaml")).string(),
                                        result.raw_cost,double(model.sampleStep()),&raw_stats);
                writeBoundaryDiagnostics(out/("boundary_diagnostics_raw_"+std::to_string(run)+".csv"),raw_stats);
                if(sc.enabled) {
                    adapter.writeTrajectory(final_samples,(out/("trajectory_shortcut_"+std::to_string(run)+".yaml")).string(),
                                            result.cost,double(model.sampleStep()),&result);
                    writeBoundaryDiagnostics(out/("boundary_diagnostics_shortcut_"+std::to_string(run)+".csv"),result);
                }
                adapter.writeTrajectory(final_samples,(out/("trajectory_"+std::to_string(run)+".yaml")).string(),
                                        result.cost,double(model.sampleStep()),&result);
                if(!first.empty()) {
                    DynoSamples first_samples;PlanResult first_stats;decorateProduction(first_stats);
                    Timer first_check;
                    const bool first_valid=model.validatePath(first,first_samples,first_stats).success;
                    const double first_check_time=first_check.seconds();
                    result.final_validation_time+=first_check_time;
                    result.export_time-=first_check_time;
                    if(first_valid) {
                        Real first_cost=0.f;for(const auto& edge:first)first_cost+=edge.cost;
                        endpointDiagnostics(adapter,first_samples,y["validation"]["endpoint_tolerance"].as<float>(),first_stats);
                        adapter.writeTrajectory(first_samples,(out/("trajectory_first_"+std::to_string(run)+".yaml")).string(),
                                                double(first_cost),double(model.sampleStep()),&first_stats);
                        writeBoundaryDiagnostics(out/("boundary_diagnostics_first_"+std::to_string(run)+".csv"),first_stats);
                        // Evaluate Algorithm 4 on the stored FIRST solution, not on the
                        // best path after 120 s. Do this after planning to preserve search timing.
                        if (sc.enabled) {
                            PlanResult first_short_stats;
                            decorateProduction(first_short_stats);
                            Solver first_short_solver(R,fc);
                            auto first_short_result = shortcutPath(first,model,first_short_solver,sc,first_short_stats);
                            DynoSamples first_short_samples;
                            const bool first_short_valid = model.validatePath(
                                first_short_result.path,first_short_samples,first_short_stats).success;
                            std::ofstream first_short_metrics(
                                out/("first_shortcut_metrics_"+std::to_string(run)+".csv"));
                            first_short_metrics << std::setprecision(17)
                                << "run,seed,first_cost,first_shortcut_cost,first_shortcut_length,first_shortcut_time,shortcut_attempts,shortcut_accepted,valid\n";
                            if (first_short_valid) {
                                Real first_short_cost = 0.f;
                                for (const auto& edge : first_short_result.path)
                                    first_short_cost += edge.cost;
                                double first_short_length = 0.0;
                                for (std::size_t k = 1; k < first_short_samples.states.size(); ++k)
                                    first_short_length +=
                                        (first_short_samples.states[k].template head<D>() -
                                         first_short_samples.states[k-1].template head<D>()).norm();
                                endpointDiagnostics(adapter,first_short_samples,
                                    y["validation"]["endpoint_tolerance"].as<float>(),first_short_stats);
                                adapter.writeTrajectory(first_short_samples,
                                    (out/("trajectory_first_shortcut_"+std::to_string(run)+".yaml")).string(),
                                    double(first_short_cost),double(model.sampleStep()),&first_short_stats);
                                first_short_metrics << run << ',' << config.seed << ',' << double(first_cost)
                                    << ',' << double(first_short_cost) << ',' << first_short_length
                                    << ',' << first_short_result.seconds << ',' << first_short_result.attempts
                                    << ',' << first_short_result.accepted << ",1\n";
                            } else {
                                first_short_metrics << run << ',' << config.seed << ',' << double(first_cost)
                                    << ",,," << first_short_result.seconds << ',' << first_short_result.attempts
                                    << ',' << first_short_result.accepted << ",0\n";
                                std::cerr << "WARNING: first_shortcut validation failed for run="
                                          << run << std::endl;
                            }
                        }
                    }
                }
                result.export_time+=exported.seconds();
            }else reason="raw_final_validation_failed";
        }
        result.total_time=result.search_time+result.shortcut_time;
        result.success=raw_valid&&final_valid&&result.completed_normally;
        if(!result.completed_normally)reason="safety_limit";
        if(result.success)++successes;
        result.failure_reason=result.success?"":reason;
        writeBoundaryDiagnostics(out/("boundary_diagnostics_"+std::to_string(run)+".csv"),result);
        std::ofstream events(out/("anytime_"+std::to_string(run)+".csv"));events<<std::setprecision(17)<<"search_time,cost\n";
        for(auto e:result.best_cost_events)events<<e.first<<','<<e.second<<'\n';
        writeCSVRow(csv,run,config.seed,result);csv.flush();
        validation<<run<<','<<planned<<','<<result.success<<",simd_discrete_float8,"<<reason<<'\n';validation.flush();
        std::cout<<"run="<<run<<" seed="<<config.seed<<" success="<<result.success
                 <<" termination="<<result.termination_reason<<" search="<<result.search_time
                 <<" shortcut="<<result.shortcut_time<<" reason="<<reason<<std::endl;
    }
    return successes==runs?0:2;
}
}
int runProductionBenchmark(const std::filesystem::path& file) {
    auto y=YAML::LoadFile(file.string());validateStrict(y);
    if(!simdRuntimeAVX2())throw std::runtime_error("Production requires AVX2");
    auto source=relativeToConfig(file,y["dynobench"]["source"].as<std::string>());
    auto environment=source/"envs"/y["dynobench"]["environment_file"].as<std::string>();
    std::filesystem::path model;
    if(y["dynobench"]["model_file"])model=relativeToConfig(file,y["dynobench"]["model_file"].as<std::string>());
    else {
        auto env=YAML::LoadFile(environment.string());
        const auto robot=env["robots"][0]["type"].as<std::string>();
        if(robot.find('/')!=std::string::npos||robot.find('\\')!=std::string::npos||robot.find("..")!=std::string::npos)
            throw std::invalid_argument("Unsafe model filename");
        model=source/"models"/(robot+".yaml");
    }
    PaperDataAdapter adapter(environment.string(),model.string(),y["geometry"]["robot_radius"].as<double>(),
                            y["geometry"]["obstacle_sphere_cell"].as<double>(),y["geometry"]["collision_margin"].as<double>());
    const auto system=y["system"].as<std::string>();
    if(system=="unicycle")return productionRuns<production::FloatModel<BatchRobot::Unicycle>>(adapter,y,file);
    if(system=="quad2d")return productionRuns<production::FloatModel<BatchRobot::Quad2D>>(adapter,y,file);
    if(system=="quad3d")return productionRuns<production::FloatModel<BatchRobot::Quad3D>>(adapter,y,file);
    throw std::invalid_argument("Unknown production system");
}
}
