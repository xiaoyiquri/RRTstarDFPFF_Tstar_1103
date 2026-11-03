"""Run reproducible, fixed-iteration scalar/SIMD ablations (standard library only)."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import subprocess


def replace_value(text, key, value):
    text, count = re.subn(rf"(?m)^(  {re.escape(key)}:) .*?$", rf"\g<1> {value}", text)
    if count != 1:
        raise ValueError(f"Expected one YAML key: {key}")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/simd_comparison"))
    args = parser.parse_args()
    if args.iterations <= 0 or args.runs <= 0:
        parser.error("iterations and runs must be positive")
    source = Path(__file__).parent / "config/linear_simd_reference.yaml"
    template = source.read_text(encoding="utf-8")
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    modes = {
        "scalar": (False, False, False, False),
        "nn": (True, False, True, False),
        "nn_screen": (True, False, True, True),
        "all": (True, True, True, True),
    }
    results = {}
    for name, flags in modes.items():
        config = template
        values = dict(timeout=120, max_iterations=args.iterations,
                      max_nodes=args.iterations + 2, runs=args.runs,
                      output_directory=json.dumps((destination / name).as_posix()),
                      enabled=str(flags[0]).lower(), pff=str(flags[1]).lower(),
                      nearest_neighbor=str(flags[2]).lower(),
                      collision_screen=str(flags[3]).lower())
        for key, value in values.items():
            config = replace_value(config, key, value)
        config_path = destination / f"{name}.yaml"
        config_path.write_text(config, encoding="utf-8")
        completed = subprocess.run([str(args.exe.resolve()), "--config", str(config_path)],
                                   text=True, capture_output=True,
                                   timeout=args.runs * 125 + 10)
        (destination / f"{name}.log").write_text(completed.stdout + completed.stderr,
                                                encoding="utf-8")
        # Exit 2 means at least one unsolved run, not a program/configuration failure.
        if completed.returncode not in (0, 2):
            raise RuntimeError(f"{name} failed: {completed.stderr}")
        with (destination / name / "results.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != args.runs or any(int(row["iterations"]) != args.iterations for row in rows):
            raise RuntimeError(f"{name} did not finish the fixed workload")
        results[name] = rows
        print(f"{name}: {statistics.median(float(r['planning_time']) for r in rows):.6f}s median",
              flush=True)
    baseline = results["scalar"]
    summary = {}
    for name, rows in results.items():
        equivalent = True
        for a, b in zip(baseline, rows):
            for key in ("success", "nodes", "rewires", "iterations", "pff_queries"):
                equivalent &= a[key] == b[key]
            for key in ("cost", "trajectory_duration", "path_length"):
                equivalent &= math.isclose(float(a[key]), float(b[key]), rel_tol=2e-6, abs_tol=2e-6)
        times = [float(row["planning_time"]) for row in rows]
        summary[name] = {
            "same_workload_and_result": equivalent,
            "median_planning_seconds": statistics.median(times),
            "paired_median_speedup": statistics.median(
                float(a["planning_time"]) / float(b["planning_time"])
                for a, b in zip(baseline, rows)),
            "successes": sum(int(row["success"]) for row in rows),
            "runs": args.runs,
        }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not all(item["same_workload_and_result"] for item in summary.values()):
        raise SystemExit("Backend parity failed; do not interpret timing as a fair speedup")


if __name__ == "__main__":
    main()
