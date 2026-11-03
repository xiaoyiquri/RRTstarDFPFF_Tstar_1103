#!/usr/bin/env bash
# One 120-second RRT* run per case; export and draw
# (1) first raw, (2) first + shortcut, (3) 120s + shortcut.
# Requires enable_three_stage.py patch and rebuilt benchmark.
set -o pipefail

main() {
    local project_dir="${PROJECT_DIR:-/home/cxt/MYFILE/newfile/RRTstar/RRTstarDFPFF_paper_r2_float32_strict_20261007}"
    local dynobench_dir="${DYNOBENCH_DIR:-/home/cxt/MYFILE/newfile/dynobench-main}"
    local build_dir
    local exp_dir
    local run_status
    local -a cases=(uni_bugtrap uni_wall quad2d_bugtrap quad2d_hole quad3d_block quad3d_window)
    local -a pff_args=()

    if [[ ! -d "$project_dir" ]]; then
        echo "ERROR: project directory missing: $project_dir"
        return 1
    fi
    if [[ ! -d "$dynobench_dir/envs" && -d "$dynobench_dir/dynobench-main/envs" ]]; then
        dynobench_dir="$dynobench_dir/dynobench-main"
    fi
    if [[ ! -d "$dynobench_dir/envs" ]]; then
        echo "ERROR: DynoBench envs directory missing: $dynobench_dir/envs"
        return 1
    fi
    if [[ ! -f "$dynobench_dir/dynobench/utils/viewer/robot_viewer.py" ]]; then
        echo "ERROR: DynoBench viewer missing: $dynobench_dir/dynobench/utils/viewer/robot_viewer.py"
        return 1
    fi
    if [[ -f "$project_dir/build/CMakeCache.txt" ]]; then
        build_dir="$project_dir/build"
    elif [[ -f "$project_dir/build_float32/CMakeCache.txt" ]]; then
        build_dir="$project_dir/build_float32"
    else
        echo "ERROR: Could not find configured CMake build directory"
        return 1
    fi
    if ! grep -q 'trajectory_first_shortcut_' "$project_dir/benchmark/production_runner.cpp"; then
        echo "ERROR: First apply the enable_three_stage.py patch to your project."
        return 1
    fi

    export PYTHONNOUSERSITE=1
    unset PYTHONPATH PYTHONHOME

    echo "========== Compile modified benchmark =========="
    cmake --build "$build_dir" --target benchmark -j 4 || return $?

    # New versions accept CLI PFF options; the user's earlier local version does not.
    if /usr/bin/python3 -s "$project_dir/tools/run_six.py" --help 2>&1 | grep -q -- '--additional-time-samples'; then
        pff_args=(--time-policy candidate_set --additional-time-samples 12)
        echo "PFF duration: candidate_set + 12 (command-line override)"
    else
        echo "PFF duration: old runner; checking the six anytime YAML presets"
        local case yaml
        for case in "${cases[@]}"; do
            yaml="$project_dir/benchmark/config/anytime/$case.yaml"
            if [[ ! -f "$yaml" ]]; then
                echo "ERROR: Missing YAML $yaml"
                return 1
            fi
            if ! grep -Eq '^[[:space:]]+additional_time_samples:[[:space:]]*12([[:space:]]*(#.*)?)?$' "$yaml"; then
                echo "ERROR: $yaml does not specify additional_time_samples: 12"
                return 1
            fi
            if grep -Eq '^[[:space:]]+time_policy:' "$yaml" && \
               ! grep -Eq '^[[:space:]]+time_policy:[[:space:]]*candidate_set([[:space:]]*(#.*)?)?$' "$yaml"; then
                echo "ERROR: $yaml must use time_policy: candidate_set"
                return 1
            fi
        done
    fi

    exp_dir="$project_dir/results/three_stage_$(date +%Y%m%d_%H%M%S_%N)"
    echo "========== Run six cases (one seed per case, 120 s each) =========="
    echo "Result directory: $exp_dir"
    /usr/bin/python3 -s "$project_dir/tools/run_six.py" \
        --dynobench-source "$dynobench_dir" \
        --build-dir "$build_dir" \
        --df-mode paper_r2 \
        --protocol anytime \
        --cases "${cases[@]}" \
        --runs 1 \
        --seed 20 \
        --timeout 120 \
        --shortcut true \
        --shortcut-mode cost_decreasing \
        --shortcut-time optimal \
        "${pff_args[@]}" \
        --output "$exp_dir"
    run_status=$?
    if (( run_status != 0 )); then
        echo "ERROR: Planner returned $run_status. Inspect logs in $exp_dir"
        return "$run_status"
    fi

    echo "========== Draw three trajectories =========="
    cd "$project_dir" || return 1
    local kind dest
    for kind in first first_shortcut shortcut; do
        case "$kind" in
            first) dest="$exp_dir/figures_01_first_raw" ;;
            first_shortcut) dest="$exp_dir/figures_02_first_shortcut" ;;
            shortcut) dest="$exp_dir/figures_03_anytime_shortcut" ;;
        esac
        /usr/bin/python3 -s plot_successful_results_paper.py \
            --experiment "$exp_dir" \
            --dynobench-source "$dynobench_dir" \
            --trajectory "$kind" \
            --run 0 \
            --mode dynobench \
            --skip-failed \
            --output "$dest" || return $?
        echo "Figure: $dest/index.html"
    done

    echo "========== Summary =========="
    cat "$exp_dir/summary.csv"
    /usr/bin/python3 - "$exp_dir" <<'PY'
import csv, re, sys
from pathlib import Path
root=Path(sys.argv[1])
print(f'{"case":18} {"J_first":>13} {"J_first_SC":>13} {"J_120s_SC":>13} {"rewires":>10}')
for name in ('uni_bugtrap','uni_wall','quad2d_bugtrap','quad2d_hole','quad3d_block','quad3d_window'):
    d=root/name
    with (d/'results.csv').open(newline='',encoding='utf-8') as f:
        row=next(csv.DictReader(f))
    fsc=d/'trajectory_first_shortcut_0.yaml'
    if fsc.exists():
        m=re.search(r'(?m)^cost:\s*([^\s#]+)',fsc.read_text(encoding='utf-8'))
        jfsc=float(m.group(1)) if m else float('nan')
    else: jfsc=float('nan')
    print(f'{name:18} {float(row["first_solution_cost"]):13.5f} {jfsc:13.5f} {float(row["shortcut_cost"]):13.5f} {row["rewires"]:>10}')
PY
    echo "Done. Three figure sets under: $exp_dir"
    echo "The 120-second search excludes shortcut and image-generation time."
    return 0
}

main "$@"
