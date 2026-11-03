#!/usr/bin/env python3
"""Add first-solution Shortcut export and an LQMT cost overlay to existing code.

Works with the 2026-10-07 RRTstarDFPFF_paper_r2_float32_strict_20261007 code.
Only modifies benchmark/production_runner.cpp and plot_successful_results_paper.py.
Does not replace locally customized plotting code (e.g. pose drawing).
"""
from __future__ import annotations
import sys
import shutil
from pathlib import Path


def replace_exact(src: str, old: str, new: str, name: str) -> str:
    count = src.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one occurrence of '{name}', found {count}; no files were changed")
    return src.replace(old, new, 1)


def patch_cpp(src: str) -> str:
    if 'first_shortcut_metrics_' in src:
        print('C++ file already supports first_shortcut; skipping')
        return src
    anchor = '                        writeBoundaryDiagnostics(out/("boundary_diagnostics_first_"+std::to_string(run)+".csv"),first_stats);'
    addition = r'''
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
                        }'''
    return replace_exact(src, anchor, anchor + addition, 'boundary_diagnostics_first export')


def patch_plot(src: str) -> str:
    if '"first_shortcut"' in src and 'first_shortcut' in src and 'LQMT J:' in src:
        print('Plot script already supports first_shortcut and LQMT J; skipping')
        return src
    src = replace_exact(
        src,
        'values = {"states": [], "actions": [], "times": []}',
        'values = {"states": [], "actions": [], "times": [], "cost": None}',
        'plot YAML decoder')
    src = replace_exact(
        src,
        '        if line.startswith("times:"):',
        '        if line.startswith("cost:"):\n'
        '            values["cost"] = float(line.split(":", 1)[1].strip())\n'
        '        if line.startswith("times:"):',
        'trajectory cost extraction')
    src = replace_exact(
        src,
        'choices=("final","raw","shortcut","first")',
        'choices=("final","raw","shortcut","first","first_shortcut")',
        '--trajectory choices')
    old = '''        if args.trajectory=="first":
            tr=read_simple_yaml(trajectory_file(case_dir,args.run,args.trajectory))
            dim=3 if len(tr["states"][0])==13 else 2
            row["cost"]=row["first_solution_cost"]
            row["trajectory_duration"]=str(tr["times"][-1]-tr["times"][0])
            row["path_length"]=str(sum(math.dist(a[:dim],b[:dim]) for a,b in zip(tr["states"],tr["states"][1:])))'''
    new = '''        if args.trajectory in ("first", "first_shortcut"):
            tr=read_simple_yaml(trajectory_file(case_dir,args.run,args.trajectory))
            dim=3 if len(tr["states"][0])==13 else 2
            if args.trajectory == "first":
                row["cost"] = row["first_solution_cost"]
            else:
                if tr["cost"] is None:
                    raise ValueError("first_shortcut trajectory lacks cost")
                row["cost"] = str(tr["cost"])
            row["trajectory_duration"]=str(tr["times"][-1]-tr["times"][0])
            row["path_length"]=str(sum(math.dist(a[:dim],b[:dim]) for a,b in zip(tr["states"],tr["states"][1:])))'''
    src = replace_exact(src, old, new, 'first trajectory statistics')
    src = replace_exact(
        src,
        '''        f"path length: {float(row['path_length']):.6f} m"''',
        '''        f"LQMT J: {float(row['cost']):.6f} | "
        f"path length: {float(row['path_length']):.6f} m"''',
        'plot title path length')
    return src


def main():
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python3 enable_three_stage.py /absolute/path/to/project')
    root=Path(sys.argv[1]).expanduser().resolve()
    cpp=root/'benchmark/production_runner.cpp'
    plot=root/'plot_successful_results_paper.py'
    if not cpp.exists() or not plot.exists():
        raise SystemExit(f'Project files not found under {root}')
    # First compute everything so a missing anchor cannot leave a half-patched project.
    cpp_source=cpp.read_text(encoding='utf-8')
    plot_source=plot.read_text(encoding='utf-8')
    cpp_new=patch_cpp(cpp_source)
    plot_new=patch_plot(plot_source)
    for path, original, updated in ((cpp,cpp_source,cpp_new),(plot,plot_source,plot_new)):
        if updated==original:
            print('Unchanged:',path)
            continue
        backup=path.with_name(path.name+'.backup_three_stage')
        if not backup.exists():
            shutil.copy2(path,backup)
        path.write_text(updated,encoding='utf-8')
        print('Updated:',path,'(backup:',backup,')')
    print('Patch complete. Rebuild C++ benchmark before running.')


if __name__=='__main__':
    main()
