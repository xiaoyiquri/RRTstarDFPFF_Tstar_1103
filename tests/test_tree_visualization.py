"""Visualization and partial-failure integration tests; fixtures are not robot experiments."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("plot_results", PROJECT / "plot_successful_results_paper.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def tree_fixture(dim):
    # Final topology has two branches, including a node not on the successful path.
    coefficients = [[0, 1, 0, 0], [0, 0, 1, 0]]
    if dim == 3:
        coefficients.append([0, 0, 0, 1])
    return dict(schema_version=1, snapshot="final", position_dim=dim, node_count=3,
                nodes=[dict(id=0, parent_id=None, position=[0]*dim, cost=0),
                       dict(id=1, parent_id=0, position=[1]*dim, cost=2, duration=2,
                            normalized=True, coefficients=coefficients),
                       dict(id=2, parent_id=0, position=[2]*dim, cost=3, duration=1,
                            normalized=True, coefficients=[[0, 2, 0, 0]]*dim)])


def write_case(root, name, dim, success=True):
    folder = root / name
    folder.mkdir(parents=True)
    fields = dict(run=0, seed=20, success=int(success), validation_backend="simd_discrete_float8",
                  numeric_profile="float32_production", first_solution_time=0.1,
                  first_solution_cost=2, planning_time=1, path_length=2,
                  trajectory_duration=2, cost=2)
    with (folder / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerow(fields)
    (folder / "search_tree_0.json").write_text(json.dumps(tree_fixture(dim)))
    state_dim = 13 if dim == 3 else 3
    states = [[0]*state_dim, [1]*dim+[0]*(state_dim-dim)]
    for version in ("first", "first_shortcut", "shortcut"):
        text = "cost: 2\nstates:\n" + "".join(f"  - {s}\n" for s in states) + "times: [0, 2]\n"
        (folder / f"trajectory_{version}_0.yaml").write_text(text)


class TreeVisualization(unittest.TestCase):
    def test_curved_edges_not_straight_chords(self):
        import numpy as np
        for dim in (2, 3):
            node = tree_fixture(dim)["nodes"][1]
            samples = plot.sample_tree_edge(node, 3)
            np.testing.assert_allclose(samples[1], [0.5, 0.25, 0.125][:dim])
            node["normalized"] = False
            samples = plot.sample_tree_edge(node, 3)
            np.testing.assert_allclose(samples[1], [1]*dim)

    def test_all_edges_and_nodes_drawn_in_2d_and_3d(self):
        import matplotlib.pyplot as plt
        for dim in (2, 3):
            fig = plt.figure()
            axis = fig.add_subplot(111, projection="3d" if dim == 3 else None)
            plot.draw_search_tree(axis, tree_fixture(dim), samples=5)
            edges = axis.collections[0]
            segments = edges._segments3d if dim == 3 else edges.get_segments()
            self.assertEqual(len(segments), 2)
            self.assertEqual(len(segments[0]), 5)
            plt.close(fig)

    def test_missing_or_invalid_tree_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tree.json"
            with self.assertRaisesRegex(ValueError, "Rebuild and rerun"):
                plot.read_search_tree(path, 2)
            data = tree_fixture(2)
            data["nodes"][1]["parent_id"] = 99
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Invalid search tree edge"):
                plot.read_search_tree(path, 2)

    def test_successful_cases_survive_failed_and_missing_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_case(root, "uni_bugtrap", 2)
            write_case(root, "uni_wall", 2, success=False)
            write_case(root, "quad3d_block", 3)
            out = root / "figures"
            completed = subprocess.run([sys.executable, str(PROJECT/"plot_successful_results_paper.py"),
                "--experiment", str(root), "--output", str(out), "--mode", "trajectory_only",
                "--trajectory", "first", "--skip-failed", "--show-tree"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((out / "uni_bugtrap.png").is_file())
            self.assertTrue((out / "quad3d_block.png").is_file())
            self.assertFalse((out / "uni_wall.png").exists())
            with (out / "metrics.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
            self.assertIn("Skipped:", completed.stdout)

    def exercise_shell(self, run_status, plot_failure="", build_status=0):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, build, bin_dir = root/"dyno", root/"build", root/"bin"
            (source/"envs").mkdir(parents=True)
            (source/"dynobench/utils/viewer").mkdir(parents=True)
            (source/"dynobench/utils/viewer/robot_viewer.py").touch()
            build.mkdir(); (build/"CMakeCache.txt").touch()
            bin_dir.mkdir()
            cmake = bin_dir/"cmake"
            cmake.write_text(f"#!/bin/sh\nexit {build_status}\n"); cmake.chmod(0o755)
            result = root/"results"; result.mkdir()
            write_case(result, "uni_bugtrap", 2, success=run_status == 2)
            fake = bin_dir/"python-wrapper"
            fake.write_text(f"#!{sys.executable}\n" + '''import os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
if args[0].endswith("run_six.py"):
    sys.exit(int(os.environ["TEST_RUN_STATUS"]))
if args[0].endswith("plot_successful_results_paper.py"):
    kind = args[args.index("--trajectory")+1]
    with open(os.environ["TEST_CALLS"], "a") as f: f.write(kind+"\\n")
    if kind == os.environ.get("TEST_PLOT_FAILURE"): sys.exit(7)
    args[args.index("--mode")+1] = "trajectory_only"
sys.exit(subprocess.run([sys.executable]+args).returncode)
''')
            fake.chmod(0o755)
            calls = root/"calls"
            env = dict(os.environ, PROJECT_DIR=str(PROJECT), DYNOBENCH_DIR=str(source),
                       BUILD_DIR=str(build), OUTPUT_DIR=str(result), PYTHON_BIN=str(fake),
                       PATH=str(bin_dir)+os.pathsep+os.environ["PATH"], TEST_RUN_STATUS=str(run_status),
                       TEST_CALLS=str(calls), TEST_PLOT_FAILURE=plot_failure)
            completed = subprocess.run(["bash", str(PROJECT/"run_three_stage.sh")],
                                       env=env, text=True, capture_output=True)
            if not build_status and run_status != 130:
                self.assertTrue(calls.exists(), completed.stdout + completed.stderr)
            if build_status or run_status == 130:
                self.assertFalse(calls.exists())
            else:
                self.assertEqual(calls.read_text().splitlines(), ["first", "first_shortcut", "shortcut"])
                if run_status == 2:
                    self.assertTrue((result/"figures_03_anytime_shortcut/uni_bugtrap.png").is_file())
            return completed

    def test_shell_partial_failure_still_plots_and_preserves_status(self):
        result = self.exercise_shell(2)
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_shell_plot_failure_does_not_skip_later_versions(self):
        result = self.exercise_shell(0, plot_failure="first_shortcut")
        self.assertEqual(result.returncode, 7, result.stderr)

    def test_shell_all_failed_does_not_claim_figure_exists(self):
        result = self.exercise_shell(1)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertNotIn("Figure:", result.stdout)

    def test_shell_build_failure_and_interrupt_still_abort(self):
        self.assertEqual(self.exercise_shell(0, build_status=9).returncode, 9)
        self.assertEqual(self.exercise_shell(130).returncode, 130)


if __name__ == "__main__":
    unittest.main()
