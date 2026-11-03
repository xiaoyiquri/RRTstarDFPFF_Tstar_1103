#include "benchmark/benchmark_runner.h"
#include "systems/robot_factory.h"

namespace rrtpff {
int runProductionBenchmark(const std::filesystem::path&);
namespace {
std::filesystem::path relativeToConfig(const std::filesystem::path &config,
                                       const std::string &value) {
    std::filesystem::path p(value);
    return std::filesystem::weakly_canonical(p.is_absolute() ? p : config.parent_path() / p);
}
std::filesystem::path sourceModelFile(const std::filesystem::path &environment,
                                      const std::filesystem::path &source) {
    const auto env = YAML::LoadFile(environment.string());
    if (!env["robots"].IsSequence() || env["robots"].size() != 1 ||
        !env["robots"][0]["type"])
        throw std::invalid_argument("DynoBench environment must contain one robot type");
    const auto robot = env["robots"][0]["type"].as<std::string>();
    if (robot.find('/') != std::string::npos || robot.find('\\') != std::string::npos ||
        robot.find("..") != std::string::npos)
        throw std::invalid_argument("Invalid DynoBench robot model name");
    return source / "models" / (robot + ".yaml");
}
template <class Model>
int runRobot(const Model &model, const YAML::Node &yaml, const std::filesystem::path &config,
             PlannerConfig planner_config) {
    using Solver = PFFSolver<Model::FlatDim, Model::FlatOrder, typename Model::Metadata>;
    PFFConfig pff;
    pff.min_time = yaml["pff"]["min_time"].as<double>();
    pff.max_time = yaml["pff"]["max_time"].as<double>();
    pff.free_terminal_time = yaml["pff"]["free_terminal_time"].as<bool>();
    pff.fixed_time = yaml["pff"]["fixed_time"].as<double>();
    pff.additional_time_samples = yaml["pff"]["additional_time_samples"].as<std::size_t>(0);
    pff.defer_trajectories = yaml["pff"]["defer_trajectories"].as<bool>(false);
    if (!yaml["pff"]["free_terminal_state"].as<bool>())
        throw std::invalid_argument(
            "Expansion uses free terminal derivatives except full-state goal/rewire");
    const typename Solver::Weight R =
        readVector<Model::FlatDim>(yaml["pff"]["R_diagonal"], "R_diagonal").asDiagonal();
    const auto runs = yaml["benchmark"]["runs"].as<std::size_t>();
    const auto seed = yaml["benchmark"]["seed"].as<unsigned>();
    if (!runs || runs - 1 > std::numeric_limits<unsigned>::max() - seed)
        throw std::invalid_argument("Invalid run count/seed");
    const auto output =
        relativeToConfig(config, yaml["benchmark"]["output_directory"].as<std::string>());
    std::filesystem::create_directories(output);
    // Re-running must not leave an old success trajectory beside a failed run.
    // Refuse to overwrite a result set; callers choose a fresh output directory.
    if (std::filesystem::exists(output / "results.csv"))
        throw std::runtime_error(
            "Output already contains results.csv; choose a fresh output_directory");
    std::ofstream csv(output / "results.csv"), validation_csv(output / "validation.csv");
    if (!csv || !validation_csv)
        throw std::runtime_error("Cannot create result files");
    csv << std::setprecision(17);
    writeCSVHeader(csv);
    validation_csv << std::setprecision(17)
                   << "run,planner_success,sampled_success,validation_backend,reason\n";
    model.adapter().writeGeometry((output / "geometry.yaml").string());
    std::size_t successes = 0;
    for (std::size_t run = 0; run < runs; ++run) {
        planner_config.seed = seed + static_cast<unsigned>(run);
        Timer search;
        RRTstar<Model> planner(model, planner_config, Solver(R, pff));
        const double setup=search.seconds();
        auto result = planner.solve();
        result.search_time=search.seconds();result.total_time=result.search_time;
        if(std::isfinite(result.first_solution_time))result.first_solution_time+=setup;
        for(auto& event:result.best_cost_events)event.first+=setup;
        result.shortcut_mode=result.shortcut_time_mode="disabled";
        result.validation_backend = "simd_discrete_float8";
        result.df_mode = dfMode(yaml);
        result.pseudo_control_order = Model::FlatOrder;
        result.reduced_order_result = result.df_mode == "paper_r2";
        result.boundary_contract = result.reduced_order_result ? "reduced_flat_A_prime" : "full_physical";
        const bool planned = result.success;
        DynoSamples samples;
        DynoValidation checked;
        checked.reason = result.failure_reason.empty() ? "no_solution" : result.failure_reason;
        if (planned) {
            Timer final_check;
            checked = model.validatePath(planner.path(), samples, result);
            result.final_validation_time=final_check.seconds();
            result.success = checked.success && result.completed_normally;
            if (!checked.success)
                result.failure_reason = "final_validation_failed";
            if (checked.success) {
                Timer exported;
                model.adapter().writeTrajectory(
                    samples, (output / ("trajectory_" + std::to_string(run) + ".yaml")).string(),
                    result.cost, model.sampleStep(), &result);
                result.export_time=exported.seconds();
                if(result.completed_normally)++successes;
            }
        }
        // Report translational arc length on the actual exported sample grid, not yaw distance.
        if (result.success) {
            result.path_length = 0;
            constexpr int D = Model::FlatDim == 4 ? 3 : 2;
            for (std::size_t i = 1; i < samples.states.size(); ++i)
                result.path_length += (samples.states[i].template head<D>() -
                                       samples.states[i - 1].template head<D>())
                                          .norm();
        }
        result.simd_eval_count = result.simd_pff_lanes + result.simd_flat_samples +
                                 result.simd_nn_lanes + result.simd_collision_samples;
        result.raw_cost=result.cost;result.raw_duration=result.duration;result.raw_length=result.path_length;
        if(!result.completed_normally)result.failure_reason="safety_limit";
        writeBoundaryDiagnostics(output / ("boundary_diagnostics_" + std::to_string(run) + ".csv"), result);
        writeCSVRow(csv, run, planner_config.seed, result);
        csv.flush();
        validation_csv << run << ',' << planned << ',' << checked.success
                       << ",simd_discrete_float8," << checked.reason << '\n';
        validation_csv.flush();
        std::cout << "run=" << run << " seed=" << planner_config.seed
                  << " planner_success=" << planned << " sampled_success=" << checked.success
                  << " nodes=" << result.node_count << " cost=" << result.cost
                  << " planning_time=" << result.planning_time << " validation=" << checked.reason
                  << std::endl;
    }
    std::cout << "SIMD sampled success: " << successes << '/' << runs << std::endl;
    return successes == runs ? 0 : 2;
}
} // namespace

int runRobotBenchmark(const std::filesystem::path &config) {
    const auto yaml = YAML::LoadFile(config.string());
    const auto numerical = yaml["numeric"] ? yaml["numeric"]["profile"].as<std::string>("legacy_mixed") : "legacy_mixed";
    if (numerical == "float32_production") return runProductionBenchmark(config);
    if (numerical != "legacy_mixed")
        throw std::invalid_argument("double_reference is test-only; benchmark requires an implemented production profile");
    validatePaperFairConfig(yaml);
    const auto mode = dfMode(yaml);
    auto pc = readPlanner(yaml["planner"]);
    const auto simd = yaml["simd"];
    pc.simd_enabled = true;
    pc.simd_pff = simd["pff"].as<bool>();
    pc.simd_nn = simd["nearest_neighbor"].as<bool>();
    // The robot model owns the paper-style discrete float8 collision screen;
    // generic trajectory screening remains disabled to avoid duplicate work.
    pc.simd_screen = false;
    pc.validate();
    RobotValidationConfig validation;
    validation.df_mode = mode;
    if (!simdRuntimeAVX2())
        throw std::runtime_error("AVX2 float8 validation is mandatory; no scalar fallback");
    validation.max_step = yaml["validation"]["max_step"].as<double>();
    validation.max_samples = yaml["validation"]["max_samples"].as<std::size_t>();
    validation.endpoint_tolerance = yaml["validation"]["endpoint_tolerance"].as<double>();
    validation.terminal_velocity_hint = yaml["planner"]["terminal_velocity_hint"].as<bool>(false);
    const auto source = relativeToConfig(config, yaml["dynobench"]["source"].as<std::string>());
    const auto environment =
        source / "envs" / yaml["dynobench"]["environment_file"].as<std::string>();
    const auto model = yaml["dynobench"]["model_file"]
                           ? relativeToConfig(config,
                                              yaml["dynobench"]["model_file"].as<std::string>())
                           : sourceModelFile(environment, source);
    PaperDataAdapter adapter(environment.string(), model.string(),
                             yaml["geometry"]["robot_radius"].as<double>(),
                             yaml["geometry"]["obstacle_sphere_cell"].as<double>(),
                             yaml["geometry"]["collision_margin"].as<double>());
    std::cout << "Model file: " << model << '\n';
    const auto &data = adapter.problem();
    std::cout << "Actual model: mass=" << data.mass << " inertia_3d=" << data.inertia_3d.transpose()
              << " inertia_2d=" << data.inertia_2d << " velocity_limit=" << data.max_velocity
              << " omega_limit=" << data.max_omega << " motor_control=" << data.motor_control
              << "\ncontrol_lower=" << data.lower_control.transpose()
              << "\ncontrol_upper=" << data.upper_control.transpose() << '\n';
    std::cout << "Validation backend: simd_discrete_float8 / sphere union\nEnvironment: "
              << environment << "\nRobot: " << adapter.problem().robot_type
              << "\nSIMD runtime: " << simdRuntimeAVX2() << std::endl;
    const auto system = yaml["system"].as<std::string>();
    if (system == "unicycle")
        return runRobot(makeUnicycle(adapter, yaml, validation), yaml, config, pc);
    if (system == "quad2d") {
        if (mode == "paper_r2")
            return runRobot(makeQuad2D<Quad2DR2DF>(adapter, yaml, validation), yaml, config, pc);
        return runRobot(makeQuad2D(adapter, yaml, validation), yaml, config, pc);
    }
    if (system == "quad3d") {
        if (mode == "paper_r2")
            return runRobot(makeQuad3D<Quad3DR2DF>(adapter, yaml, validation), yaml, config, pc);
        return runRobot(makeQuad3D(adapter, yaml, validation), yaml, config, pc);
    }
    throw std::invalid_argument("Unknown robot system: " + system);
}
} // namespace rrtpff
