#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KINO-RRT* real best-raw-cost vs SEARCH time, from anytime_N.csv.

Usage:
  python3 plot_kino_rrtstar_cost_time.py --experiment results/three_stage_... --case uni_bugtrap
  python3 plot_kino_rrtstar_cost_time.py --experiment results/three_stage_... --case uni_wall --run 2

Only genuine improvement events from anytime_N.csv are plotted. Shortcuts
are offline postprocessing and are deliberately excluded.
"""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import statistics
import sys


def as_float(value):
    try:
        num = float(value)
        return num if math.isfinite(num) else None
    except (ValueError, TypeError):
        return None


def read_events(path):
    with path.open(newline='', encoding='utf-8-sig') as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames or not {'search_time', 'cost'}.issubset(reader.fieldnames):
            raise ValueError(f'{path}: expected CSV columns search_time,cost; got {reader.fieldnames}')
        pairs = []
        for row in reader:
            t, j = as_float(row.get('search_time')), as_float(row.get('cost'))
            if t is None or j is None or t < 0 or j < 0:
                continue
            pairs.append((t, j))
    pairs.sort(key=lambda e: e[0])
    best, result = math.inf, []
    for t, j in pairs:
        if j < best - 1e-10 * max(1.0, abs(best) if math.isfinite(best) else 1.0):
            best = j
            result.append((t, j))
    return result


def read_runs(case_dir):
    file = case_dir / 'results.csv'
    if not file.exists():
        raise FileNotFoundError(f'Missing {file}; needed to distinguish valid and failed runs')
    with file.open(newline='', encoding='utf-8-sig') as fp:
        rows = list(csv.DictReader(fp))
    return rows


def choose_budget(experiment, rows, supplied):
    if supplied is not None:
        return supplied
    manifest = experiment / 'manifest.json'
    if manifest.exists():
        try:
            raw = json.loads(manifest.read_text(encoding='utf-8'))
            val = as_float(raw.get('timeout'))
            if val is not None and val > 0:
                return val
        except (ValueError, OSError):
            pass
    available = [as_float(r.get('search_time')) for r in rows]
    values = [n for n in available if n is not None and n > 0]
    return max(values) if values else None


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches='tight')
    print('Saved:', path)


def choose_log_left(runs, requested=1e-4):
    """Prefer 10^-4 s; expand leftward if real measured events are earlier."""
    positives = [t for run in runs for t in run['times'] if t > 0]
    if positives and min(positives) < requested:
        return 10.0 ** math.floor(math.log10(min(positives)))
    return requested


def format_axes(ax, case, budget, logx=False, log_min=1e-4):
    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_title(f'KINO-RRT* | {case} | search-stage raw cost')
    ax.set_xlabel('Elapsed search time (s)' + (' [log10 scale]' if logx else ''))
    ax.set_ylabel('Best raw LQMT cost J')
    if logx:
        ax.set_xscale('log', base=10)
        ax.set_xlim(left=log_min, right=budget * 1.025)
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))
    else:
        ax.set_xlim(left=0.0, right=budget * 1.025)
    ax.grid(True, alpha=0.28, which='major')
    if logx:
        ax.grid(True, alpha=0.09, which='minor')


def show_steps(ax, runs, budget, logx=False, log_min=1e-4):
    left = choose_log_left(runs, log_min) if logx else None
    for run in runs:
        times = [t for t, _ in run['events']]
        vals = [j for _, j in run['events']]
        # Hold the last reported incumbent only until THIS RUN's search ended,
        # not falsely until the full budget when search terminated early.
        actual_end = max(times[-1], run['search_end'])
        xs = times + [actual_end]
        ys = vals + [vals[-1]]
        if logx:
            # A logarithmic axis cannot show t=0; keep all positive timestamps exact.
            xs = [max(left, t) for t in xs]
        ax.step(xs, ys, where='post', linewidth=1.5, alpha=0.76,
                label=f"seed {run['seed']} (run {run['run']})")
        ax.plot(xs[0], ys[0], marker='o', markersize=2.8, alpha=0.8)
    format_axes(ax, runs[0]['case'], budget, logx, left if logx else log_min)
    if len(runs) <= 15:
        ax.legend(fontsize=8, loc='best', framealpha=0.8)
    else:
        ax.text(.99, .99, f'{len(runs)} successful runs', va='top', ha='right',
                transform=ax.transAxes, fontsize=9)


def step_checkpoint(run, t):
    times = run['times']
    idx = bisect.bisect_right(times, t) - 1
    if idx < 0:
        return None
    return run['values'][idx]


def export_checkpoints(path, runs, denominator, budget):
    checkpoints = [1e-4, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 30, 60, 120, 180]
    checkpoints = sorted(set(t for t in checkpoints if 0 < t <= budget) | {budget})
    rows = []
    for t in checkpoints:
        costs = [x for run in runs if (x := step_checkpoint(run, t)) is not None]
        rows.append({'time_s': f'{t:.9g}', 'solved': len(costs),
                     'total_runs': denominator, 'success_fraction': f'{len(costs)/denominator:.9g}',
                     'mean_best_J_conditional': f'{statistics.mean(costs):.9g}' if costs else '',
                     'median_best_J_conditional': f'{statistics.median(costs):.9g}' if costs else ''})
    with path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('Saved:', path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True, help='Existing run_six.py experiment root')
    p.add_argument('--case', required=True, help='uni_bugtrap, uni_wall, quad2d_bugtrap, ...')
    p.add_argument('--run', type=int, default=None, help='Optional: select only run index, e.g. 2')
    p.add_argument('--budget', type=float, default=None, help='Plot time horizon; defaults to manifest.json timeout')
    p.add_argument('--output', type=Path, default=None, help='Default: EXP/figures_cost_time')
    p.add_argument('--log-min', type=float, default=1e-4,
                   help='Preferred left edge for log10 time plots (seconds, default 1e-4); '
                        'automatically extends earlier if a real event precedes it')
    args = p.parse_args()
    if args.budget is not None and (not math.isfinite(args.budget) or args.budget <= 0):
        p.error('--budget must be finite and positive')
    experiment = args.experiment.expanduser().resolve()
    case_dir = experiment / args.case
    if not case_dir.is_dir():
        p.error(f'Case folder not found: {case_dir}')
    rows = read_runs(case_dir)
    if args.run is not None:
        rows = [r for r in rows if r.get('run') == str(args.run)]
        if not rows:
            p.error(f'Run {args.run} is absent from {case_dir / "results.csv"}')
    budget = choose_budget(experiment, rows, args.budget)
    if not budget:
        p.error('Cannot infer time budget; pass --budget 120')
    if not math.isfinite(args.log_min) or not 0 < args.log_min < budget:
        p.error('--log-min must be positive, finite and smaller than the budget')
    output = args.output or experiment / 'figures_cost_time'
    output.mkdir(parents=True, exist_ok=True)
    selected = []
    print(f'Experiment: {experiment}\nCase: {args.case}\nBudget: {budget:g} s')
    print('run | seed | success | events | first_s | first_J | last_J | search_end_s')
    for row in rows:
        try:
            rid = int(row['run'])
        except (ValueError, KeyError):
            print('WARNING: ignoring row without numeric run index', file=sys.stderr)
            continue
        seed = row.get('seed', '?')
        success = row.get('success', '').strip().lower() in ('1', 'true')
        if row.get('completed_normally', '1').strip().lower() in ('0', 'false'):
            success = False
        path = case_dir / f'anytime_{rid}.csv'
        events = read_events(path) if path.exists() else []
        print(f'{rid} | {seed} | {int(success)} | {len(events)} | '
              f'{events[0][0]:.6g}' if events else f'{rid} | {seed} | {int(success)} | 0 | -', end='')
        if events:
            print(f' | {events[0][1]:.6g} | {events[-1][1]:.6g}', end='')
        else:
            print(' | - | -', end='')
        print(f' | {row.get("search_time", "-")}')
        if not success or not events:
            if success and not events:
                print(f'WARNING: successful run {rid} lacks improvement events', file=sys.stderr)
            continue
        search_end = as_float(row.get('search_time'))
        if search_end is None or search_end < events[-1][0]:
            search_end = events[-1][0]
        if search_end > budget + max(0.02, 0.002 * budget):
            print(f'WARNING: run {rid} search_end exceeds selected budget', file=sys.stderr)
        selected.append(dict(case=args.case, run=rid, seed=seed, events=events,
                             times=[t for t,_ in events], values=[j for _,j in events],
                             search_end=search_end))
    if not selected:
        p.error('No successful runs with valid anytime_N.csv events. Check protocol anytime / result files.')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for logx in (False, True):
        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
        show_steps(ax, selected, budget, logx, args.log_min)
        save_figure(fig, output / f'kino_{args.case}_raw_cost_time_{"logx" if logx else "linear"}.png')
        plt.close(fig)

    for run in selected:
        for logx in (False, True):
            fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
            show_steps(ax, [run], budget, logx=logx, log_min=args.log_min)
            suffix = '_logx' if logx else ''
            save_figure(fig, output / f'kino_{args.case}_seed_{run["seed"]}_run_{run["run"]}{suffix}.png')
            plt.close(fig)

    # A conditional mean is NOT guaranteed to decrease: its membership changes
    # as new seeds find their first solution. Plot success fraction alongside it.
    times = sorted({t for run in selected for t in run['times']} | {budget})
    denom = len(rows)
    avg, ratio = [], []
    for t in times:
        costs = [c for run in selected if (c := step_checkpoint(run,t)) is not None]
        avg.append(statistics.mean(costs) if costs else math.nan)
        ratio.append(len(costs) / denom)
    for logx in (False, True):
        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
        ax.step(times, avg, where='post', label='Mean best raw J (available solutions)', linewidth=2)
        ax.set_title(f'KINO-RRT* | {args.case} | conditional mean & success')
        ax.set_xlabel('Elapsed search time (s)' + (' [log10 scale]' if logx else ''))
        ax.set_ylabel('Mean best raw LQMT cost J')
        ax2 = ax.twinx()
        if logx:
            left = choose_log_left(selected, args.log_min)
            ax.set_xscale('log', base=10)
            ax.set_xlim(left, budget * 1.025)
            from matplotlib.ticker import LogLocator, LogFormatterMathtext
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
            ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))
            ax2.step([left] + times, [0] + ratio, where='post', linestyle='--', label='Success fraction')
        else:
            ax.set_xlim(0, budget * 1.025)
            ax2.step([0] + times, [0] + ratio, where='post', linestyle='--', label='Success fraction')
        ax.grid(alpha=0.25, which='both')
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel('Success fraction')
        line1, label1 = ax.get_legend_handles_labels()
        line2, label2 = ax2.get_legend_handles_labels()
        ax.legend(line1 + line2, label1 + label2, fontsize=9, loc='best')
        suffix = '_logx' if logx else ''
        save_figure(fig, output / f'kino_{args.case}_mean_cost_success{suffix}.png')
        plt.close(fig)
    export_checkpoints(output / f'kino_{args.case}_checkpoints.csv', selected, denom, budget)
    print(f'Finished: {len(selected)}/{denom} successful runs with event history')
    print('IMPORTANT: curves show RRT* search-phase RAW cost only; shortcuts not mixed in.')
    print('Mean can rise when a previously-unsolved run joins the available set.')


if __name__ == '__main__':
    main()
