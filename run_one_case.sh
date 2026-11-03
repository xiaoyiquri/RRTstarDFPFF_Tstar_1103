#!/usr/bin/env bash
# Run and visualize ONE DynoBench environment, then export full-stage metrics.
# Edit settings below; use ./run_one_case.sh without command-line arguments.
set -euo pipefail

# ==========================================================================
# 1. EDIT PARAMETERS HERE
# ==========================================================================
PROJECT_DIR="/home/cxt/MYFILE/newfile/RRTstar/RRTstarDFPFF_paper_r2_float32_strict_20261007"
DYNOBENCH_DIR="/home/cxt/MYFILE/newfile/dynobench-main"

CASE="uni_bugtrap"
RUNS=100
TIMEOUT=1
SEED=41
PLOT_RUN=0
PLOT_ALL_RUNS=false

DF_MODE="paper_r2"
PROTOCOL="anytime"
SHORTCUT="true"
SHORTCUT_MODE="cost_decreasing"
SHORTCUT_TIME="optimal"
PLOT_MODE="dynobench"
BUILD_JOBS=4
PFF_ARGS=()

# Optional: set to an existing experiment folder to ONLY re-generate metrics.
# This skips compilation, planning, and plotting. Set back to "" for new runs.
SUMMARY_ONLY_EXP_DIR=""
# Example:
# SUMMARY_ONLY_EXP_DIR="$PROJECT_DIR/results/uni_bugtrap_20261025_093030_484801865"
# ==========================================================================

summarize() {
    local exp_dir="$1" case_name="$2" budget="$3"
    echo "========== Summary (all runs, full trajectories) =========="
    if [[ -f "$exp_dir/summary.csv" ]]; then
        cat "$exp_dir/summary.csv"
    fi

    /usr/bin/python3 -s - "$exp_dir" "$case_name" "$budget" <<'PY'
import csv
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
case = sys.argv[2]
budget = sys.argv[3]
folder = root / case
results_file = folder / 'results.csv'
if not results_file.is_file():
    raise SystemExit(f'ERROR: Missing results.csv: {results_file}')

with results_file.open(newline='', encoding='utf-8') as stream:
    results = list(csv.DictReader(stream))
if not results:
    raise SystemExit(f'ERROR: Empty results.csv: {results_file}')

# Uni and Quad2D have 2D positions; Quad3D has 3D positions.
position_dim = 3 if case.startswith('quad3d_') else 2


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def trajectory_metrics(stage, run):
    path = folder / f'trajectory_{stage}_{run}.yaml'
    if not path.is_file():
        return {'cost': None, 'length': None}
    previous = None
    length = 0.0
    count = 0
    cost = None
    in_states = False
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            item = line.strip()
            if item.startswith('cost:'):
                cost = number(item.split(':', 1)[1].strip())
            if item == 'states:':
                in_states = True
                continue
            if not in_states:
                continue
            if item == 'actions:':
                break
            if not (item.startswith('- [') and item.endswith(']')):
                continue
            point = tuple(float(x.strip()) for x in item[3:-1].split(',')[:position_dim])
            if len(point) != position_dim:
                raise ValueError(f'Wrong position dimension in {path}')
            if previous is not None:
                length += math.dist(previous, point)
            previous = point
            count += 1
    return {'cost': cost, 'length': length if count else None}


def first_shortcut_time(run):
    path = folder / f'first_shortcut_metrics_{run}.csv'
    if not path.is_file():
        return None
    with path.open(newline='', encoding='utf-8') as stream:
        data = next(csv.DictReader(stream), {})
    return number(data.get('first_shortcut_time'))


def show(value, width=11, digits=4):
    v = number(value)
    return f'{v:>{width}.{digits}f}' if v is not None else f'{"N/A":>{width}}'

records = []
for result in results:
    run = int(result['run'])
    first = trajectory_metrics('first', run)
    first_sc = trajectory_metrics('first_shortcut', run)
    raw = trajectory_metrics('raw', run)
    final_sc = trajectory_metrics('shortcut', run)
    records.append({
        'run': run,
        'seed': result.get('seed', ''),
        'success': result.get('success', ''),
        'first_solution_time_s': number(result.get('first_solution_time')),
        'J_first': number(result.get('first_solution_cost')),
        'J_first_SC': first_sc['cost'],
        'J_raw': number(result.get('raw_cost')),
        'J_final_SC': number(result.get('shortcut_cost')),
        'length_first': first['length'],
        'length_first_SC': first_sc['length'],
        'length_raw': raw['length'],
        'length_final_SC': final_sc['length'],
        'final_shortcut_time_s': number(result.get('shortcut_time')),
        'first_shortcut_time_s': first_shortcut_time(run),
        'rewires': result.get('rewires', ''),
    })

# Export one complete CSV that can be used directly for plotting and statistics.
csv_path = folder / 'all_run_metrics.csv'
with csv_path.open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

print(f'Case: {case}, {len(records)} runs, {budget} s search per run')
print('\n[Costs and Shortcut time]')
print(f'{"run":>3} {"seed":>5} {"ok":>3} {"first_s":>10} '
      f'{"J_first":>11} {"J_first_SC":>11} {"J_raw":>11} '
      f'{"J_final_SC":>11} {"SC_time_s":>11} {"rewires":>8}')
for r in records:
    print(f'{r["run"]:>3} {str(r["seed"]):>5} {str(r["success"]):>3} '
          f'{show(r["first_solution_time_s"], 10, 5)} '
          f'{show(r["J_first"])} {show(r["J_first_SC"])} '
          f'{show(r["J_raw"])} {show(r["J_final_SC"])} '
          f'{show(r["final_shortcut_time_s"], 11, 6)} '
          f'{str(r["rewires"]):>8}')

print('\n[Sampled trajectory lengths, meters]')
print(f'{"run":>3} {"seed":>5} {"length_first":>14} '
      f'{"length_first_SC":>16} {"length_raw":>14} '
      f'{"length_final_SC":>17}')
for r in records:
    print(f'{r["run"]:>3} {str(r["seed"]):>5} '
          f'{show(r["length_first"], 14, 5)} '
          f'{show(r["length_first_SC"], 16, 5)} '
          f'{show(r["length_raw"], 14, 5)} '
          f'{show(r["length_final_SC"], 17, 5)}')

if any(r['first_shortcut_time_s'] is not None for r in records):
    print('\n[First-solution Shortcut time, if available]')
    for r in records:
        print(f'run={r["run"]}, first_SC_time_s='
              f'{show(r["first_shortcut_time_s"], 10, 6).strip()}')

successful = [r for r in records if r['success'] == '1']
if successful:
    columns = ('first_solution_time_s', 'J_first', 'J_first_SC', 'J_raw',
               'J_final_SC', 'length_first', 'length_first_SC',
               'length_raw', 'length_final_SC', 'final_shortcut_time_s')
    print('\n[Mean over successful runs]')
    for key in columns:
        values = [r[key] for r in successful if r[key] is not None]
        if values:
            print(f'{key:24}: {sum(values)/len(values):.6f} (n={len(values)})')

print(f'\nFull CSV: {csv_path}')
print('Note: final_shortcut_time_s measures Shortcut on the final Raw path, '
      'excluding search, validation, and plotting.')
PY
}

main() {
    local project_dir="$PROJECT_DIR" dynobench_dir="$DYNOBENCH_DIR"
    local build_dir exp_dir run_status=0 kind dest traj plot_run
    local -a plot_runs=()

    case "$CASE" in
        uni_bugtrap|uni_wall|quad2d_bugtrap|quad2d_hole|quad3d_block|quad3d_window) ;;
        *) echo "ERROR: Invalid CASE=$CASE" >&2; return 1 ;;
    esac
    if [[ ! "$RUNS" =~ ^[1-9][0-9]*$ || ! "$SEED" =~ ^[0-9]+$ || ! "$PLOT_RUN" =~ ^[0-9]+$ ]]; then
        echo "ERROR: RUNS must be positive; SEED and PLOT_RUN must be nonnegative integers" >&2
        return 1
    fi
    if (( PLOT_RUN >= RUNS )); then
        echo "ERROR: PLOT_RUN must be between 0 and $((RUNS - 1))" >&2
        return 1
    fi

    # Reanalyze an earlier experiment without running the planner again.
    if [[ -n "$SUMMARY_ONLY_EXP_DIR" ]]; then
        summarize "$SUMMARY_ONLY_EXP_DIR" "$CASE" "$TIMEOUT"
        return $?
    fi

    if [[ ! -d "$project_dir" ]]; then
        echo "ERROR: Missing project directory: $project_dir" >&2
        return 1
    fi
    if [[ ! -d "$dynobench_dir/envs" && -d "$dynobench_dir/dynobench-main/envs" ]]; then
        dynobench_dir="$dynobench_dir/dynobench-main"
    fi
    if [[ ! -d "$dynobench_dir/envs" ]]; then
        echo "ERROR: Missing DynoBench envs: $dynobench_dir/envs" >&2
        return 1
    fi
    if [[ "$PLOT_MODE" == "dynobench" && ! -f "$dynobench_dir/dynobench/utils/viewer/robot_viewer.py" ]]; then
        echo "ERROR: Missing DynoBench viewer" >&2
        return 1
    fi
    if [[ -f "$project_dir/build/CMakeCache.txt" ]]; then
        build_dir="$project_dir/build"
    elif [[ -f "$project_dir/build_float32/CMakeCache.txt" ]]; then
        build_dir="$project_dir/build_float32"
    else
        echo "ERROR: No configured CMake build directory" >&2
        return 1
    fi
    if ! grep -q 'trajectory_first_shortcut_' "$project_dir/benchmark/production_runner.cpp"; then
        echo "ERROR: Three-stage patch not found" >&2
        return 1
    fi

    export PYTHONNOUSERSITE=1
    unset PYTHONPATH PYTHONHOME

    echo "========== Build benchmark =========="
    cmake --build "$build_dir" --target benchmark -j "$BUILD_JOBS"

    exp_dir="$project_dir/results/${CASE}_$(date +%Y%m%d_%H%M%S_%N)"
    echo "========== Run $CASE =========="
    echo "Runs=$RUNS, seed=$SEED..$((SEED+RUNS-1)), timeout=${TIMEOUT}s/run, protocol=$PROTOCOL"
    echo "Results: $exp_dir"

    if /usr/bin/python3 -s "$project_dir/tools/run_six.py" \
        --dynobench-source "$dynobench_dir" \
        --build-dir "$build_dir" \
        --df-mode "$DF_MODE" \
        --protocol "$PROTOCOL" \
        --cases "$CASE" \
        --runs "$RUNS" \
        --seed "$SEED" \
        --timeout "$TIMEOUT" \
        --shortcut "$SHORTCUT" \
        --shortcut-mode "$SHORTCUT_MODE" \
        --shortcut-time "$SHORTCUT_TIME" \
        "${PFF_ARGS[@]}" \
        --output "$exp_dir"; then
        run_status=0
    else
        run_status=$?
        echo "WARNING: Planner returned $run_status; inspect $exp_dir/${CASE}.log" >&2
    fi

    if [[ ! -f "$exp_dir/$CASE/results.csv" ]]; then
        echo "ERROR: results.csv was not created" >&2
        return 1
    fi

    if [[ "$PLOT_ALL_RUNS" == true ]]; then
        for ((plot_run=0; plot_run<RUNS; ++plot_run)); do
            plot_runs+=("$plot_run")
        done
    else
        plot_runs=("$PLOT_RUN")
    fi

    echo "========== Draw three trajectory stages =========="
    cd "$project_dir" || return 1
    for plot_run in "${plot_runs[@]}"; do
        for kind in first first_shortcut shortcut; do
            case "$kind" in
                first) dest="$exp_dir/figures_01_first_raw" ;;
                first_shortcut) dest="$exp_dir/figures_02_first_shortcut" ;;
                shortcut) dest="$exp_dir/figures_03_anytime_shortcut" ;;
            esac
            if [[ "$PLOT_ALL_RUNS" == true ]]; then
                dest="${dest}_run${plot_run}"
            fi
            traj="$exp_dir/$CASE/trajectory_${kind}_${plot_run}.yaml"
            if [[ ! -f "$traj" ]]; then
                echo "SKIP: No ${kind} trajectory for run $plot_run"
                continue
            fi
            if /usr/bin/python3 -s plot_successful_results_paper.py \
                --experiment "$exp_dir" \
                --cases "$CASE" \
                --dynobench-source "$dynobench_dir" \
                --trajectory "$kind" \
                --run "$plot_run" \
                --mode "$PLOT_MODE" \
                --skip-failed \
                --show-tree \
                --output "$dest"; then
                if [[ -f "$dest/index.html" ]]; then
                    echo "Figure: $dest/index.html"
                else
                    echo "SKIP: No HTML generated for $kind, run $plot_run"
                fi
            else
                echo "WARNING: Plot failed for $kind, run $plot_run" >&2
            fi
        done
    done

    summarize "$exp_dir" "$CASE" "$TIMEOUT"
    echo "========== Finished =========="
    echo "Results: $exp_dir"
    echo "Search budget excludes postprocessing, validation and visualization time."
    return "$run_status"
}

main

