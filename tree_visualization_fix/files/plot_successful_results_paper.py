#!/usr/bin/env python3
"""Official environment display, checker sphere debug, or trajectory-only display."""
import argparse
import csv
import html
import math
import re
import json
import sys
import importlib
from pathlib import Path


CASES = (
    "uni_bugtrap",
    "uni_wall",
    "quad2d_bugtrap",
    "quad2d_hole",
    "quad3d_block",
    "quad3d_window",
)


def parse_vector(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def read_simple_yaml(path):
    values = {"states": [], "actions": [], "times": [], "cost": None}
    section = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line in ("states:", "actions:"):
            section = line[:-1]
            continue
        if line.startswith("- "):
            if section in ("states", "actions"):
                values[section].append(parse_vector(line[2:].strip("[]")))
            continue
        if line.startswith("cost:"):
            values["cost"] = float(line.split(":", 1)[1].strip())
        if line.startswith("times:"):
            values["times"] = parse_vector(line.split(":", 1)[1].strip().strip("[]"))
    if not values["states"] or not values["times"]:
        raise ValueError(f"trajectory has no states/times: {path}")
    return values


def read_geometry(path):
    spheres = []
    pattern = re.compile(
        r"center:\s*\[([^]]+)\],\s*radius:\s*([-+0-9.eE]+)"
    )
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(raw)
        if match:
            center = parse_vector(match.group(1))
            while len(center) < 3:
                center.append(0.0)
            spheres.append((center, float(match.group(2))))
    return spheres


def read_result(case_dir, run):
    result_file = case_dir / "results.csv"
    with result_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    row = next((item for item in rows if int(item["run"]) == run), None)
    if row is None:
        raise ValueError(f"run {run} is absent from {result_file}")
    if row.get("success") != "1" or row.get("validation_backend") != "simd_discrete_float8":
        raise ValueError(
            f"{case_dir.name} run {run} is not a sampled SIMD success "
            f"(success={row.get('success')}, backend={row.get('validation_backend')})"
        )
    return row


def trajectory_file(case_dir,run,selection):
    name=f"trajectory_{run}.yaml" if selection=="final" else f"trajectory_{selection}_{run}.yaml"
    return case_dir/name

def read_search_tree(path, position_dim):
    if not path.is_file():
        raise ValueError(f"Search tree is absent: {path}. Rebuild and rerun the planner "
                         "to export it, or omit --show-tree for older results.")
    tree = json.loads(path.read_text(encoding="utf-8"))
    if (tree.get("schema_version") != 1 or tree.get("snapshot") != "final"
            or tree.get("position_dim") != position_dim):
        raise ValueError(f"Unsupported search tree schema or dimension: {path}")
    nodes = tree["nodes"]
    ids = {node["id"] for node in nodes}
    if len(nodes) != tree["node_count"] or len(ids) != len(nodes):
        raise ValueError(f"Search tree node count/IDs are inconsistent: {path}")
    roots = 0
    for node in nodes:
        if len(node["position"]) != position_dim or not all(map(math.isfinite, node["position"])):
            raise ValueError(f"Invalid search tree position: {path}")
        if node["parent_id"] is None:
            roots += 1
            continue
        coefficients = node["coefficients"]
        if (node["parent_id"] not in ids or node["parent_id"] == node["id"]
                or not math.isfinite(node["duration"]) or node["duration"] <= 0
                or not isinstance(node["normalized"], bool)
                or len(coefficients) != position_dim
                or any(len(c) != 4 or not all(map(math.isfinite, c)) for c in coefficients)):
            raise ValueError(f"Invalid search tree edge: {path}")
    if nodes and roots != 1:
        raise ValueError(f"Search tree must have one root: {path}")
    return tree


def sample_tree_edge(node, samples):
    import numpy as np
    # Coefficients are ascending powers of s=t/T (or t for non-normalized edges).
    parameter = np.linspace(0, 1 if node["normalized"] else node["duration"], samples)
    coefficients = np.asarray(node["coefficients"], dtype=float)
    points = np.zeros((samples, len(coefficients)))
    for column in coefficients.T[::-1]:
        points = points * parameter[:, None] + column
    return points


def draw_search_tree(axis, tree, samples=33):
    import numpy as np
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    is_3d = tree["position_dim"] == 3
    collection_type = Line3DCollection if is_3d else LineCollection
    segments = []
    first_collection = True

    def flush():
        nonlocal first_collection
        if not segments:
            return
        collection = collection_type(
            list(segments), colors="#7a8695", linewidths=0.55, alpha=0.32, zorder=1,
            label="final search tree" if first_collection else "_nolegend_")
        if is_3d:
            axis.add_collection3d(collection)
        else:
            axis.add_collection(collection)
        first_collection = False
        segments.clear()

    # Batch artists, but never subsample or drop edges of the retained tree.
    for node in tree["nodes"]:
        if node["parent_id"] is not None:
            segments.append(sample_tree_edge(node, samples))
        if len(segments) == 4096:
            flush()
    flush()
    if tree["nodes"]:
        positions = np.array([node["position"] for node in tree["nodes"]])
        axis.scatter(*positions.T, s=2, color="#7a8695", alpha=0.35,
                     zorder=1, label="tree nodes")
    axis.autoscale_view()


def official_scene(axis,case,source,manifest):
    import yaml
    module_name,viewer_name = (("unicycle1_viewer","Unicycle1Viewer") if case.startswith("uni_") else
        (("quad2d_viewer","Quad2dViewer") if case.startswith("quad2d_") else ("quad3d_viewer","Quad3dViewer")))
    folder=source/"dynobench/utils/viewer"
    if not (folder/(module_name+".py")).is_file():
        raise ValueError(f"Official DynoBench viewer source missing: {folder}")
    sys.path.insert(0,str(folder))
    module=importlib.import_module(module_name)
    viewer=getattr(module,viewer_name)()
    env_file=manifest.get("environment_files",{}).get(case)
    if not env_file or not Path(env_file).is_file():
        from tools.run_six import CASES as ENVIRONMENTS
        env_file=source/"envs"/ENVIRONMENTS[case][0]
    with Path(env_file).open(encoding="utf-8") as stream: environment=yaml.safe_load(stream)
    #viewer.view_problem(axis,environment)
    #return str(folder/(module_name+".py"))
    viewer.view_problem(axis, environment)
    return viewer

def draw_case(case_dir, row, run, output_dir,mode="dynobench",selection="final",source=None,manifest=None,
              show_tree=False,tree_samples=33):
    import matplotlib.pyplot as plt

    trajectory = read_simple_yaml(trajectory_file(case_dir,run,selection))
    
        # 读取当前绘制轨迹的真实 LQMT Cost
    traj_path = trajectory_file(case_dir, run, selection)

    cost_text = next(
        (
            line.split(":", 1)[1].strip()
            for line in traj_path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("cost:")
        ),
        None
    )

    if cost_text is None:
        raise ValueError(f"轨迹文件中缺少 cost: {traj_path}")

    trajectory_cost = float(cost_text)

    # 使用实际轨迹状态计算几何长度
    states = trajectory["states"]
    position_dim = 3 if len(states[0]) == 13 else 2

    trajectory_length = sum(
        math.dist(a[:position_dim], b[:position_dim])
        for a, b in zip(states, states[1:])
    )
    
    spheres = read_geometry(case_dir / "geometry.yaml") if mode=="checker_geometry" else []
    state_dim = len(trajectory["states"][0])
    positions = [state[:3] for state in trajectory["states"]]
    is_3d = state_dim == 13
    tree = read_search_tree(case_dir/f"search_tree_{run}.json", 3 if is_3d else 2) if show_tree else None
    figure = plt.figure(figsize=(9, 7))
    if is_3d:
        axis = figure.add_subplot(111, projection="3d")
        if tree is not None:
            # Keep the selected path readable above a dense 3D tree background.
            axis.computed_zorder = False
        #if mode=="dynobench": official_scene(axis,case_dir.name,source,manifest)
        if mode == "dynobench":viewer = official_scene(axis, case_dir.name, source, manifest)
        if tree is not None: draw_search_tree(axis, tree, tree_samples)
        axis.plot(
            [point[0] for point in positions],
            [point[1] for point in positions],
            [point[2] for point in positions],
            color="#1167c4",
            linewidth=2,
            label="sampled SIMD path",
            zorder=3,
        )
        # A sphere union is the authoritative geometry. The center cloud is
        # deliberately used instead of replacing it with the old FCL boxes.
        stride = max(1, len(spheres) // 8000)
        obstacle_points = [sphere[0] for sphere in spheres[::stride]]
        if obstacle_points: axis.scatter(
            [point[0] for point in obstacle_points],
            [point[1] for point in obstacle_points],
            [point[2] for point in obstacle_points],
            s=3,
            alpha=0.18,
            color="#6f7782",
            label="sphere-union geometry",
        )
        axis.set_zlabel("z (m)")
    else:
        axis = figure.add_subplot(111)
        #if mode=="dynobench": official_scene(axis,case_dir.name,source,manifest)
        if mode == "dynobench":viewer = official_scene(axis, case_dir.name, source, manifest)
        if tree is not None: draw_search_tree(axis, tree, tree_samples)
        axis.plot(
            [point[0] for point in positions],
            [point[1] for point in positions],
            color="#1167c4",
            linewidth=2,
            label="sampled SIMD path",
            zorder=3,
        )
        for center, radius in spheres:
            axis.add_patch(
                plt.Circle(
                    (center[0], center[1]), radius, color="#6f7782", alpha=0.18
                )
            )
        axis.set_ylabel("y (m)" if state_dim == 3 else "z (m)")
        axis.set_aspect("equal", adjustable="box")
        
        # 在轨迹上的若干位置绘制机器人真实姿态
    if mode == "dynobench":
        states = trajectory["states"]

        num_poses = min(8, len(states))

        if num_poses == 1:
            indices = [0]
        else:
            indices = sorted(set(
                round(k * (len(states) - 1) / (num_poses - 1))
                for k in range(num_poses)
            ))

        for idx in indices:
            # 使用 DynoBench 原生的机器人绘制接口
            viewer.view_state(axis, states[idx])
    start, goal = positions[0], positions[-1]
    if is_3d:
        axis.scatter(*start, color="#168a45", s=45, label="start", zorder=5)
        axis.scatter(*goal, color="#c83333", marker="*", s=90, label="goal", zorder=5)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    else:
        axis.scatter(start[0], start[1], color="#168a45", s=45, label="start", zorder=5)
        axis.scatter(goal[0], goal[1], color="#c83333", marker="*", s=90, label="goal", zorder=5)
        axis.set_xlabel("x (m)")
    backend = row["validation_backend"]
    axis.set_title(
        f"{case_dir.name} | run {run} | {selection}\n"
        f"{row.get('numeric_profile', 'float32_production')}\n"
        f"First solution: {float(row['first_solution_time']):.6f} s\n"
        f"LQMT Cost J = {trajectory_cost:.6f} | "
        f"Path length = {trajectory_length:.6f} m"
        + (f"\nFinal search tree: {tree['node_count']} nodes "
           f"(not a first-solution snapshot)" if tree is not None else ""),
        fontsize=11,
        pad=15
    )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    image = output_dir / f"{case_dir.name}.png"
    figure.savefig(image, dpi=160)
    plt.close(figure)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode",choices=("dynobench","checker_geometry","trajectory_only"),default="dynobench")
    parser.add_argument("--trajectory",choices=("final","raw","shortcut","first","first_shortcut"),default="final")
    parser.add_argument("--dynobench-source",type=Path)
    parser.add_argument("--skip-failed",action="store_true")
    parser.add_argument("--show-tree",action="store_true",
                        help="Overlay every retained edge/node of the FINAL search tree")
    parser.add_argument("--tree-samples",type=int,default=33,
                        help="Display samples per polynomial tree edge (default: 33; no edges dropped)")
    args = parser.parse_args()
    if not 2 <= args.tree_samples <= 4096:
        parser.error("--tree-samples must be between 2 and 4096")
    manifest_file=args.experiment/"manifest.json"
    manifest=json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    source=args.dynobench_source or Path(manifest.get("dynobench_source","."))
    if args.mode=="dynobench" and not (source/"dynobench/utils/viewer/robot_viewer.py").is_file():
        parser.error("Specify --dynobench-source with the official Python viewer source")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    images = []
    for case in dict.fromkeys(args.cases):
        case_dir = args.experiment / case
        try:
            row = read_result(case_dir, args.run)
            if not trajectory_file(case_dir,args.run,args.trajectory).is_file():
                raise ValueError("Requested trajectory is absent: "+str(trajectory_file(case_dir,args.run,args.trajectory)))
        except (ValueError,FileNotFoundError) as exc:
            if args.skip_failed:
                print("Skipped: "+str(exc));continue
            raise
        if args.trajectory in ("raw","shortcut"):
            for metric,key in (("cost","cost"),("path_length","length"),("trajectory_duration","duration")):
                row[metric]=row.get(args.trajectory+"_"+key,row[metric])
        if args.trajectory in ("first", "first_shortcut"):
            tr=read_simple_yaml(trajectory_file(case_dir,args.run,args.trajectory))
            dim=3 if len(tr["states"][0])==13 else 2
            if args.trajectory == "first":
                row["cost"] = row["first_solution_cost"]
            else:
                if tr["cost"] is None:
                    raise ValueError("first_shortcut trajectory lacks cost")
                row["cost"] = str(tr["cost"])
            row["trajectory_duration"]=str(tr["times"][-1]-tr["times"][0])
            row["path_length"]=str(sum(math.dist(a[:dim],b[:dim]) for a,b in zip(tr["states"],tr["states"][1:])))
        image = draw_case(case_dir,row,args.run,args.output,args.mode,args.trajectory,source,manifest,
                          args.show_tree,args.tree_samples)
        images.append(image.name)
        rows.append(
            {
                "environment": case,
                "run": row["run"],
                "seed": row["seed"],
                "sampled_success": row["success"],
                "validation_backend": row["validation_backend"],
                "first_solution_time_s": row["first_solution_time"],
                "planning_time_s": row["planning_time"],
                "path_length_m": row["path_length"],
                "trajectory_duration_s": row["trajectory_duration"],
                "cost": row["cost"],
                "display_mode":args.mode,"trajectory":args.trajectory,
                "simd_validation_queries": row.get("simd_validation_queries", ""),
                "simd_validation_batches": row.get("simd_validation_batches", ""),
            }
        )
    metrics = args.output / "metrics.csv"
    if not rows:
        print("No successful trajectories available for the requested run");return
    with metrics.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    links = "\n".join(
        f'<p><a href="{html.escape(image)}"><img src="{html.escape(image)}" width="800"></a></p>'
        for image in images
    )
    (args.output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>SIMD sampled results</title>"
        "<h1>SIMD discrete float8 sampled results</h1>" + links,
        encoding="utf-8",
    )
    print(f"Output: {args.output}")
    (args.output/"visualization_manifest.json").write_text(json.dumps(dict(
        display_mode=args.mode,trajectory=args.trajectory,dynobench_source=str(source),
        collision_geometry="sphere_union",continuous_collision_certified=False,
        official_viewer=args.mode=="dynobench",show_tree=args.show_tree,
        tree_snapshot="final" if args.show_tree else None,
        tree_samples_per_edge=args.tree_samples if args.show_tree else None),indent=2),encoding="utf-8")
    for row in rows:
        print(
            f"{row['environment']}: first_solution={row['first_solution_time_s']} s | "
            f"path_length={row['path_length_m']} m | duration={row['trajectory_duration_s']} s"
        )


if __name__ == "__main__":
    main()
