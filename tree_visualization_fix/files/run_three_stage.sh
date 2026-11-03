#!/usr/bin/env bash
# One anytime search per case; plot successful runs even after partial failures.
set -o pipefail

main() {
    local script_dir project_dir dynobench_dir build_dir python exp_dir
    local run_status=0 plot_status=0 status kind dest case yaml
    local budget="${SEARCH_SECONDS:-120}" seed="${SEED:-20}"
    local -a cases=(uni_bugtrap uni_wall quad2d_bugtrap quad2d_hole quad3d_block quad3d_window)
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
    project_dir="${PROJECT_DIR:-$script_dir}"
    dynobench_dir="${DYNOBENCH_DIR:-$project_dir/../dynobench-main}"
    python="${PYTHON_BIN:-python3}"
    build_dir="${BUILD_DIR:-}"
    if [[ ! -d "$project_dir" ]]; then
        echo "ERROR: project directory missing: $project_dir"; return 1
    fi
    project_dir="$(cd -- "$project_dir" && pwd)" || return 1
    if [[ ! -d "$dynobench_dir/envs" && -d "$dynobench_dir/dynobench-main/envs" ]]; then
        dynobench_dir="$dynobench_dir/dynobench-main"
    fi
    if [[ ! -d "$dynobench_dir/envs" || ! -f "$dynobench_dir/dynobench/utils/viewer/robot_viewer.py" ]]; then
        echo "ERROR: DYNOBENCH_DIR must contain envs/ and dynobench/utils/viewer/robot_viewer.py"
        return 1
    fi
    dynobench_dir="$(cd -- "$dynobench_dir" && pwd)" || return 1
    if [[ -z "$build_dir" ]]; then
        for build_dir in "$project_dir/build" "$project_dir/build_float32" "$project_dir/build_ubuntu_float32"; do
            [[ -f "$build_dir/CMakeCache.txt" ]] && break
        done
    fi
    if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
        echo "ERROR: configure CMake first, or set BUILD_DIR to a configured build directory"
        return 1
    fi
    build_dir="$(cd -- "$build_dir" && pwd)" || return 1
    command -v "$python" >/dev/null || { echo "ERROR: Python not found: $python"; return 1; }
    if ! grep -q 'search_tree_' "$project_dir/benchmark/production_runner.cpp"; then
        echo "ERROR: Install the search-tree export changes before running this script"; return 1
    fi
    # Current run_six.py loads these presets, even with --protocol anytime.
    for case in "${cases[@]}"; do
        yaml="$project_dir/benchmark/config/$case.yaml"
        if [[ ! -f "$yaml" ]]; then
            echo "ERROR: Missing configuration: $yaml"; return 1
        fi
        echo "PFF preset $case: $(grep 'additional_time_samples:' "$yaml")"
    done
    echo "========== Compile benchmark =========="
    cmake --build "$build_dir" --target benchmark -j 4 || return $?
    exp_dir="${OUTPUT_DIR:-$project_dir/results/three_stage_$(date +%Y%m%d_%H%M%S_%N)}"
    echo "========== Run six cases ($budget seconds each; seed $seed) =========="
    echo "Result directory: $exp_dir"
    if "$python" "$project_dir/tools/run_six.py" \
        --dynobench-source "$dynobench_dir" --build-dir "$build_dir" \
        --df-mode paper_r2 --protocol anytime --cases "${cases[@]}" \
        --runs 1 --seed "$seed" --timeout "$budget" \
        --shortcut true --shortcut-mode cost_decreasing --shortcut-time optimal \
        --output "$exp_dir"; then
        run_status=0
    else
        run_status=$?
        # Respect an explicit interruption rather than starting extra work.
        if (( run_status == 130 || run_status == 143 )); then return "$run_status"; fi
        echo "WARNING: planner exited $run_status; drawing available successful trajectories."
    fi
    if [[ ! -d "$exp_dir" ]]; then
        echo "No experiment directory was created; nothing to draw."
        if (( run_status != 0 )); then return "$run_status"; fi
        return 1
    fi
    echo "========== Draw successful trajectories and final search tree =========="
    for kind in first first_shortcut shortcut; do
        case "$kind" in
            first) dest="$exp_dir/figures_01_first_raw" ;;
            first_shortcut) dest="$exp_dir/figures_02_first_shortcut" ;;
            shortcut) dest="$exp_dir/figures_03_anytime_shortcut" ;;
        esac
        if MPLBACKEND=Agg "$python" "$project_dir/plot_successful_results_paper.py" \
            --experiment "$exp_dir" --dynobench-source "$dynobench_dir" \
            --trajectory "$kind" --run 0 --mode dynobench \
            --skip-failed --show-tree --output "$dest"; then
            if [[ -f "$dest/index.html" ]]; then
                echo "Figure: $dest/index.html"
            else
                echo "No successful trajectories available for $kind."
            fi
        else
            status=$?
            if (( status == 130 || status == 143 )); then return "$status"; fi
            (( plot_status == 0 )) && plot_status=$status
            echo "WARNING: plotting $kind failed ($status); continuing with other versions."
        fi
    done
    echo "========== Summary =========="
    if [[ -f "$exp_dir/summary.csv" ]]; then cat "$exp_dir/summary.csv"; fi
    echo "Results: $exp_dir"
    echo "Search budget: $budget s per case; excludes tree export, shortcut and plotting."
    echo "Planner exit: $run_status; plotting exit: $plot_status"
    # Keep failure visible to callers, but only AFTER all plots were attempted.
    if (( run_status != 0 )); then return "$run_status"; fi
    return "$plot_status"
}

main "$@"
