#!/usr/bin/env bash
# Run and visualize ONE DynoBench environment.
# All user-adjustable parameters are in the configuration block below.
# Usage: ./run_one_case.sh        (NO command-line arguments)
# Requires the three-stage patch and a configured CMake build directory.
set -euo pipefail

# ============================================================================
# 1. EDIT PARAMETERS HERE (no terminal arguments are necessary)
# ============================================================================
PROJECT_DIR="/home/cxt/MYFILE/newfile/RRTstar/RRTstarDFPFF_paper_r2_float32_strict_20261007"
DYNOBENCH_DIR="/home/cxt/MYFILE/newfile/dynobench-main"

# Allowed: uni_bugtrap, uni_wall, quad2d_bugtrap, quad2d_hole,
#          quad3d_block, quad3d_window
CASE="uni_bugtrap"
RUNS=5               # 5 independent runs (to run once, set 1)
TIMEOUT=10           # Search budget in seconds PER run (e.g., 120)
SEED=0               # First seed; runs use SEED .. SEED+RUNS-1
PLOT_RUN=0           # Which run to draw (0-based; must be < RUNS)
PLOT_ALL_RUNS=false  # true: draw all runs; false: draw only PLOT_RUN

DF_MODE="paper_r2"
PROTOCOL="anytime"                # anytime or latency
SHORTCUT="true"                 # true or false
SHORTCUT_MODE="cost_decreasing" # cost_decreasing or flask
SHORTCUT_TIME="optimal"         # optimal or original_duration
PLOT_MODE="dynobench"           # dynobench, checker_geometry, trajectory_only
BUILD_JOBS=4

# Optional PFF CLI overrides for the older run_six.py. Leave empty to use its
# defaults. Example: PFF_ARGS=(--time-policy candidate_set --additional-time-samples 8)
PFF_ARGS=()
# ============================================================================

main() {
    local project_dir="$PROJECT_DIR" dynobench_dir="$DYNOBENCH_DIR"
    local build_dir exp_dir run_status=0 kind dest traj plot_run
    local -a plot_runs=()

    case "$CASE" in
        uni_bugtrap|uni_wall|quad2d_bugtrap|quad2d_hole|quad3d_block|quad3d_window) ;;
        *) echo "ERROR: Invalid CASE=$CASE" >&2; return 1 ;;
    esac
    if [[ ! "$RUNS" =~ ^[1-9][0-9]*$ || ! "$SEED" =~ ^[0-9]+$ || ! "$PLOT_RUN" =~ ^[0-9]+$ ]]; then
        echo "ERROR: RUNS must be positive, SEED and PLOT_RUN nonnegative integers" >&2
        return 1
    fi
    if (( PLOT_RUN >= RUNS )); then
        echo "ERROR: PLOT_RUN=$PLOT_RUN must be between 0 and $((RUNS - 1))" >&2
        return 1
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
        echo "ERROR: Missing DynoBench viewer: $dynobench_dir/dynobench/utils/viewer/robot_viewer.py" >&2
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
        echo "ERROR: Three-stage patch not found in benchmark/production_runner.cpp" >&2
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

    echo "========== Summary (all runs) =========="
    if [[ -f "$exp_dir/summary.csv" ]]; then
        cat "$exp_dir/summary.csv"
    fi

    /usr/bin/python3 - "$exp_dir" "$CASE" "$TIMEOUT" <<'PY'
import csv
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
case = sys.argv[2]
timeout = sys.argv[3]
directory = root / case
with (directory / "results.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

def fmt(value):
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return "N/A"

def first_shortcut_cost(run):
    f = directory / f"trajectory_first_shortcut_{run}.yaml"
    if not f.is_file():
        return None
    m = re.search(r"(?m)^cost:\s*([^\s#]+)", f.read_text(encoding="utf-8"))
    return m.group(1) if m else None

print(f"Case: {case}, {len(rows)} runs, {timeout} s search per run")
print(f'{"run":>3} {"seed":>5} {"ok":>3} {"first_s":>10} {"J_first":>12} '
      f'{"J_first_SC":>12} {"J_raw":>12} {"J_final_SC":>12} {"rewires":>8}')
for row in rows:
    run = int(row["run"])
    print(f'{run:>3} {row.get("seed", "?"):>5} {row.get("success", "?"):>3} '
          f'{fmt(row.get("first_solution_time")):>10} '
          f'{fmt(row.get("first_solution_cost")):>12} '
          f'{fmt(first_shortcut_cost(run)):>12} '
          f'{fmt(row.get("raw_cost")):>12} '
          f'{fmt(row.get("shortcut_cost")):>12} '
          f'{row.get("rewires", "?"):>8}')
PY
    echo "========== Finished =========="
    echo "Results: $exp_dir"
    echo "Search budget excludes postprocessing, validation and visualization time."
    return "$run_status"
}

main

