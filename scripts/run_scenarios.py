"""
M6 driver — run every scheduler on every diagnostic scenario, validate the
expected_winners table from xpu-rt/scenarios.py, and emit per-(scenario,
scheduler) Gantts + a side-by-side composite per scenario + a master report.

Output layout:
  results/scenarios/
    metrics.csv                                 (one row per (scenario, scheduler))
    report.md                                   (expected-vs-observed table)
    gantts/
      <scenario>_<scheduler>.png                (one per cell)
    side_by_side/
      <scenario>.png                            (composite per scenario)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "xpu-rt"))

from scenarios import SCENARIOS  # noqa: E402
from schedulers import get_scheduler  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from report import SchedulerResult, render_gantt  # noqa: E402
from postprocessing import validate_schedule  # noqa: E402


DEFAULT_SCHEDULERS = "heft,critical_path,edf,fastest_device,fifo,mosek,cpsat"


def _run_cell(scenario_name: str, scheduler_name: str, wl, gantt_dir: Path,
              time_limit: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "scenario": scenario_name,
        "scheduler": scheduler_name,
        "n_ops": len(wl.operations),
    }
    sched = get_scheduler(scheduler_name)
    kwargs: Dict[str, Any] = {}
    if scheduler_name == "mosek":
        kwargs = dict(solver_verbosity=0, time_limit=time_limit,
                      restrict_makespan_to_nonperiodic=False,
                      prune_cross_period_constraints=False)
    elif scheduler_name == "cpsat":
        kwargs = dict(time_limit=time_limit)

    t0 = time.perf_counter()
    try:
        t, alpha, _, _ = sched(wl, **kwargs)
    except Exception as exc:
        out.update(feasible=False, error=str(exc))
        return out
    wall = time.perf_counter() - t0
    if t is None or alpha is None:
        out.update(feasible=False, error="solver_returned_none")
        return out

    try:
        ok, _ = validate_schedule(wl, t, alpha, original_json_data={"dispatches": {}})
    except Exception:
        ok = False

    m = compute_metrics(wl, t, alpha, scheduler_name=scheduler_name, solver_wall_time_s=wall)
    out.update(
        feasible=True,
        valid=bool(ok),
        makespan_us=m["makespan_us"],
        deadline_miss_count=m["deadline_miss_count"],
        total_lateness_us=m["total_lateness_us"],
        cross_device_transitions=m["cross_device_transitions"],
        critical_path_us=m["critical_path_us"],
        solver_wall_time_s=m["solver_wall_time_s"],
    )
    gantt_dir.mkdir(parents=True, exist_ok=True)
    gantt_path = gantt_dir / f"{scenario_name}_{scheduler_name}.png"
    try:
        res = SchedulerResult(scheduler_name=scheduler_name, workload=wl,
                              t=t, alpha=alpha, metrics=m, feasible=True)
        render_gantt(res, str(gantt_path),
                     title=f"{scenario_name} / {scheduler_name} | "
                           f"ms={m['makespan_us']:.0f} miss={m['deadline_miss_count']}")
        out["gantt"] = str(gantt_path)
    except Exception:
        out["gantt"] = None
    return out


def _composite(scenario_name: str, schedulers, gantt_dir: Path, side_dir: Path):
    imgs = []
    for s in schedulers:
        p = gantt_dir / f"{scenario_name}_{s}.png"
        if p.exists():
            imgs.append((s, p))
    if not imgs:
        return None
    side_dir.mkdir(parents=True, exist_ok=True)
    rows = (len(imgs) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 3.5 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.array([axes])
    for idx, (s, p) in enumerate(imgs):
        ax = axes[idx // 2][idx % 2]
        ax.imshow(mpimg.imread(p))
        ax.set_title(s)
        ax.axis("off")
    # Hide empty subplots.
    for k in range(len(imgs), rows * 2):
        axes[k // 2][k % 2].axis("off")
    fig.suptitle(scenario_name, fontsize=14)
    out = side_dir / f"{scenario_name}.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def _evaluate_expectations(rows, scenario_name, expected_winners, expected_failures):
    """For each metric in expected_winners, check that the observed best
    scheduler is in the expected list."""
    scenario_rows = [r for r in rows if r["scenario"] == scenario_name and r.get("feasible")]
    if not scenario_rows:
        return []
    checks = []
    for metric, expected in expected_winners.items():
        vals = {r["scheduler"]: r.get(metric) for r in scenario_rows
                if r.get(metric) is not None}
        if not vals:
            checks.append({"metric": metric, "expected": expected, "observed": None,
                           "status": "skip"})
            continue
        best_val = min(vals.values())
        observed_winners = [s for s, v in vals.items() if v == best_val]
        intersect = set(observed_winners) & set(expected)
        status = "match" if intersect else "miss"
        checks.append({
            "metric": metric, "expected": expected, "observed": observed_winners,
            "best_value": best_val, "status": status,
        })
    # Expected-failures: schedulers that should NOT be best.
    for metric, failures in expected_failures.items():
        vals = {r["scheduler"]: r.get(metric) for r in scenario_rows
                if r.get(metric) is not None}
        if not vals:
            continue
        best_val = min(vals.values())
        observed_winners = [s for s, v in vals.items() if v == best_val]
        bad_overlap = set(observed_winners) & set(failures)
        if bad_overlap:
            checks.append({
                "metric": f"{metric} (negative)",
                "expected": f"NOT {failures}",
                "observed": observed_winners,
                "status": "unexpected_winner",
            })
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all",
                    help=f"Comma-separated subset or 'all' from: {','.join(SCENARIOS)}")
    ap.add_argument("--schedulers", default=DEFAULT_SCHEDULERS)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--out", default=str(REPO / "results" / "scenarios"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    gantt_dir = out_dir / "gantts"
    side_dir = out_dir / "side_by_side"
    out_dir.mkdir(parents=True, exist_ok=True)

    scen_names = list(SCENARIOS) if args.scenario == "all" else args.scenario.split(",")
    schedulers = [s.strip() for s in args.schedulers.split(",") if s.strip()]

    rows: List[Dict[str, Any]] = []
    expectations_by_scenario: Dict[str, Any] = {}
    for name in scen_names:
        if name not in SCENARIOS:
            print(f"[warn] unknown scenario: {name}")
            continue
        print(f"\n=== {name} ===")
        wl, exp_win, exp_fail = SCENARIOS[name]()
        expectations_by_scenario[name] = (exp_win, exp_fail)
        for s in schedulers:
            print(f"  -- {s}")
            row = _run_cell(name, s, wl, gantt_dir, args.time_limit)
            rows.append(row)
            print(f"     ms={row.get('makespan_us', 'n/a')}  "
                  f"miss={row.get('deadline_miss_count', 'n/a')}  "
                  f"xfers={row.get('cross_device_transitions', 'n/a')}  "
                  f"feasible={row.get('feasible')}")
        _composite(name, schedulers, gantt_dir, side_dir)

    # metrics.csv
    csv_path = out_dir / "metrics.csv"
    fields = ["scenario", "scheduler", "n_ops", "feasible", "valid",
              "makespan_us", "deadline_miss_count", "total_lateness_us",
              "cross_device_transitions", "critical_path_us",
              "solver_wall_time_s", "error", "gantt"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nMetrics -> {csv_path}")

    # report.md
    report_path = out_dir / "report.md"
    lines: List[str] = ["# M6 — Diagnostic scenarios", ""]
    lines.append(f"- schedulers: {schedulers}")
    lines.append(f"- scenarios: {scen_names}")
    lines.append("")

    pass_count = 0
    total_count = 0
    for name in scen_names:
        if name not in expectations_by_scenario:
            continue
        exp_win, exp_fail = expectations_by_scenario[name]
        checks = _evaluate_expectations(rows, name, exp_win, exp_fail)
        scenario_rows = [r for r in rows if r["scenario"] == name and r.get("feasible")]
        lines.append(f"## {name}")
        lines.append("")
        composite_rel = (side_dir / f"{name}.png").relative_to(out_dir)
        lines.append(f"![composite]({composite_rel})")
        lines.append("")
        # per-scheduler table for this scenario
        lines.append("| scheduler | makespan_us | misses | total_lateness | cross_device | n_ops |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in scenario_rows:
            lines.append(f"| {r['scheduler']} | "
                         f"{r.get('makespan_us', '-'):.1f} | "
                         f"{r.get('deadline_miss_count', '-')} | "
                         f"{r.get('total_lateness_us', 0):.1f} | "
                         f"{r.get('cross_device_transitions', '-')} | "
                         f"{r.get('n_ops', '-')} |")
        lines.append("")
        if checks:
            lines.append("Expectation checks:")
            for c in checks:
                total_count += 1
                emoji = {"match": "PASS", "miss": "FAIL",
                         "skip": "skip", "unexpected_winner": "WARN"}.get(c["status"], c["status"])
                if c["status"] == "match":
                    pass_count += 1
                lines.append(f"- {emoji}  metric `{c['metric']}` "
                             f"expected best in `{c['expected']}` "
                             f"observed best `{c.get('observed')}`")
        lines.append("")

    lines.insert(2, f"**Summary: {pass_count} / {total_count} expectation checks passed.**\n")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report -> {report_path}")
    print(f"Expectation checks: {pass_count}/{total_count} passed.")


if __name__ == "__main__":
    main()
