"""
M15/M16/M17+M18/M20 unified benchmark driver.

Subcommands:
  --target robustness  M16 — noise sweep across scenarios + real workloads
  --target scaling     M15 — workload sizes 20...1000 × 5 seeds
  --target realtime    M17 + M18 — real-frequency QRB5165 packing of 5 models
  --target literature  M20 — Pegasus DAGs (Montage, CyberShake, Epigenomics)

All four share a common sweep core: build a list of (workload, label) tuples,
sweep schedulers, record metrics + Gantt + summary report. CSVs land under
results/<target>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "xpu-rt"))
sys.path.insert(0, str(REPO / "scripts"))


# ---------------------------------------------------------------------------
# Common sweep core
# ---------------------------------------------------------------------------


def _run_one(workload, scheduler_name: str, time_limit: float) -> Dict[str, Any]:
    from schedulers import get_scheduler
    from metrics import compute_metrics
    from postprocessing import validate_schedule

    sched = get_scheduler(scheduler_name)
    kwargs: Dict[str, Any] = {}
    if scheduler_name == "mosek":
        kwargs = dict(solver_verbosity=0, time_limit=time_limit,
                      restrict_makespan_to_nonperiodic=False,
                      prune_cross_period_constraints=False)
    elif scheduler_name in ("cpsat", "cpsat_memory"):
        kwargs = dict(time_limit=time_limit)

    t0 = time.perf_counter()
    try:
        t, alpha, _, _ = sched(workload, **kwargs)
    except Exception as exc:
        return {"feasible": False, "error": str(exc)}
    wall = time.perf_counter() - t0
    if t is None or alpha is None:
        return {"feasible": False, "error": "no_schedule",
                "solver_wall_time_s": wall}
    try:
        ok, _ = validate_schedule(workload, t, alpha,
                                  original_json_data={"dispatches": {}})
    except Exception:
        ok = False
    m = compute_metrics(workload, t, alpha,
                       scheduler_name=scheduler_name, solver_wall_time_s=wall)
    return {
        "feasible": True, "valid": bool(ok),
        "makespan_us": m["makespan_us"],
        "deadline_miss_count": m["deadline_miss_count"],
        "total_lateness_us": m["total_lateness_us"],
        "cross_device_transitions": m["cross_device_transitions"],
        "critical_path_us": m["critical_path_us"],
        "solver_wall_time_s": m["solver_wall_time_s"],
    }


# ---------------------------------------------------------------------------
# M16 — Robustness sweep
# ---------------------------------------------------------------------------


def run_robustness(args):
    """Multiplicative Gaussian noise sweep over (scenario or real workload)
    × scheduler × sigma × seed."""
    from scenarios import SCENARIOS
    from realistic_workloads import build_model_graph, build_workload_from_graph
    from noise import add_processing_time_noise

    out_dir = REPO / "results" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    schedulers = args.schedulers.split(",")
    sigmas = [float(s) for s in args.sigmas.split(",")]
    n_seeds = args.seeds

    # Build workload list. Mix M6 scenarios + 3 real graphs on chipyard.
    workloads: List[Tuple[str, Any]] = []
    for name, fn in SCENARIOS.items():
        wl, _, _ = fn()
        workloads.append((name, wl))
    for model in ("dronet", "mlp_wide", "yolov8n"):
        try:
            g = build_model_graph(model, "chipyard")
            workloads.append((f"real_{model}_chipyard", build_workload_from_graph(g)))
        except Exception as exc:
            print(f"[warn] couldn't load real {model}: {exc}")

    rows: List[Dict[str, Any]] = []
    print(f"Sweep: {len(workloads)} workloads × {len(sigmas)} sigmas × "
          f"{n_seeds} seeds × {len(schedulers)} schedulers = "
          f"{len(workloads) * len(sigmas) * n_seeds * len(schedulers)} cells")

    rng_global = np.random.default_rng(args.seed)
    for wname, wl in workloads:
        for sigma in sigmas:
            for seed in range(n_seeds):
                # Same noise realization across all schedulers in this cell.
                cell_rng = np.random.default_rng(
                    rng_global.integers(0, 2**31 - 1) + seed * 7919
                )
                noisy = add_processing_time_noise(wl, sigma, cell_rng) if sigma > 0 else wl
                for s in schedulers:
                    r = _run_one(noisy, s, args.time_limit)
                    row = {
                        "workload": wname, "sigma_pct": sigma, "seed": seed,
                        "scheduler": s, "n_ops": len(noisy.operations),
                        **r,
                    }
                    rows.append(row)
            print(f"  {wname:<35s} sigma={sigma:>5.1f}%  done ({len(rows)} rows)")

    # CSV
    csv_path = out_dir / "metrics.csv"
    fields = ["workload", "sigma_pct", "seed", "scheduler", "n_ops",
              "feasible", "valid", "makespan_us", "deadline_miss_count",
              "total_lateness_us", "cross_device_transitions",
              "critical_path_us", "solver_wall_time_s", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nMetrics -> {csv_path}")

    _plot_robustness(rows, out_dir, schedulers, sigmas)
    _write_robustness_report(rows, out_dir, schedulers, sigmas, workloads)


def _plot_robustness(rows, out_dir: Path, schedulers: List[str], sigmas: List[float]):
    # Per scheduler: makespan distribution boxes at each sigma, normalized
    # by the per-(workload, scheduler, sigma=0) median (so we compare relative
    # degradation; absolute scales differ across workloads).
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4 * len(sigmas), 5),
                             constrained_layout=True, sharey=True)
    if len(sigmas) == 1:
        axes = [axes]
    # Normalize: for each (workload, scheduler), compute baseline makespan at sigma=0
    base_by = {}
    for r in rows:
        if r["sigma_pct"] == 0 and r.get("feasible"):
            base_by[(r["workload"], r["scheduler"])] = r["makespan_us"]

    # For each sigma, collect normalized makespans per scheduler
    for ax_idx, sigma in enumerate(sigmas):
        ax = axes[ax_idx]
        data_per_sched = []
        for s in schedulers:
            vals = []
            for r in rows:
                if r["scheduler"] != s or r["sigma_pct"] != sigma or not r.get("feasible"):
                    continue
                base = base_by.get((r["workload"], s))
                if base is None or base <= 0:
                    continue
                vals.append(r["makespan_us"] / base)
            data_per_sched.append(vals if vals else [1.0])
        bp = ax.boxplot(data_per_sched, labels=schedulers, showmeans=True)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(f"σ = {sigma:.0f}%")
        ax.set_ylabel("makespan / baseline (σ=0)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)
    fig.suptitle("M16 — robustness to per-op cost noise (boxplot per scheduler)", fontsize=12)
    fig.savefig(out_dir / "robustness_boxplots.png", dpi=120)
    plt.close(fig)


def _write_robustness_report(rows, out_dir: Path, schedulers, sigmas, workloads):
    lines: List[str] = ["# M16 — Noise robustness sweep", ""]
    lines.append(f"- workloads: {len(workloads)}")
    lines.append(f"- sigmas (% one-sigma multiplicative jitter): {sigmas}")
    lines.append(f"- schedulers: {schedulers}")
    lines.append("")
    lines.append("![boxplots](robustness_boxplots.png)")
    lines.append("")
    lines.append("## Mean / p95 makespan ratio vs σ=0 baseline (lower is more robust)")
    lines.append("")
    lines.append("| scheduler | σ=0 mean | σ=10% mean | σ=25% mean | σ=50% mean | σ=25% p95 | σ=50% p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    base_by = {(r["workload"], r["scheduler"]): r["makespan_us"]
               for r in rows if r["sigma_pct"] == 0 and r.get("feasible")}
    for s in schedulers:
        cells = []
        for sig in (0.0, 10.0, 25.0, 50.0):
            ratios = []
            for r in rows:
                if r["scheduler"] != s or r["sigma_pct"] != sig or not r.get("feasible"):
                    continue
                b = base_by.get((r["workload"], s))
                if b is None or b <= 0:
                    continue
                ratios.append(r["makespan_us"] / b)
            cells.append(f"{np.mean(ratios):.3f}" if ratios else "n/a")
        p95_25 = "n/a"; p95_50 = "n/a"
        for sig, store in ((25.0, "p95_25"), (50.0, "p95_50")):
            ratios = []
            for r in rows:
                if r["scheduler"] != s or r["sigma_pct"] != sig or not r.get("feasible"):
                    continue
                b = base_by.get((r["workload"], s))
                if b is None or b <= 0:
                    continue
                ratios.append(r["makespan_us"] / b)
            if ratios:
                val = f"{np.percentile(ratios, 95):.3f}"
                if store == "p95_25":
                    p95_25 = val
                else:
                    p95_50 = val
        lines.append(f"| {s} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {p95_25} | {p95_50} |")
    with open(out_dir / "report.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Report -> {out_dir / 'report.md'}")


# ---------------------------------------------------------------------------
# M15 — Scaling sweep (stub for now; expanded in next step)
# ---------------------------------------------------------------------------


def run_scaling(args):
    print("[run_scaling] not implemented in this commit; coming next.")


def run_realtime(args):
    print("[run_realtime] not implemented in this commit; coming next.")


def run_literature(args):
    print("[run_literature] not implemented in this commit; coming next.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["robustness", "scaling", "realtime", "literature"],
                    required=True)
    ap.add_argument("--schedulers",
                    default="heft,critical_path,edf,fastest_device,fifo,peft,"
                            "simulated_annealing,cpsat,gnn_placement,rl_policy")
    ap.add_argument("--sigmas", default="0,10,25,50",
                    help="Comma-separated noise percentages (M16)")
    ap.add_argument("--seeds", type=int, default=10, help="Seeds per noise level (M16)")
    ap.add_argument("--time-limit", type=float, default=20.0,
                    help="Solver time limit for cpsat/mosek")
    ap.add_argument("--seed", type=int, default=0, help="Master RNG seed")
    args = ap.parse_args()

    if args.target == "robustness":
        run_robustness(args)
    elif args.target == "scaling":
        run_scaling(args)
    elif args.target == "realtime":
        run_realtime(args)
    elif args.target == "literature":
        run_literature(args)


if __name__ == "__main__":
    main()
