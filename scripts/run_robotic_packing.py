"""
Robotic packing hero benchmark — pack multiple periodic models into the
end-to-end envelope of a dominant model on a real heterogeneous SoC, and
sweep the (model, frequency) frontier each scheduler can sustain.

For each (soc, f_dronet_hz, f_mlp_hz, scheduler) cell:
  1. Pack dronet @ f_dronet_hz + mlp_wide @ f_mlp_hz inside an envelope.
  2. Run the scheduler.
  3. Validate.
  4. Record metrics.
  5. Emit a Gantt PNG.

At the sweep level produces:
  results/robotic_packing/metrics.csv
  results/robotic_packing/packing_frontier_<soc>.png
  results/robotic_packing/report.md
  results/robotic_packing/gantts/<soc>_d<fd>Hz_m<fm>Hz_<scheduler>.png

Caveat — firesim time-scale: the per-dispatch latencies measured under
firesim are simulation-time (yolov8n RVV = 98 s, dronet RVV = 1.5 s). To make
the experiments correspond to production-realistic robotic frequencies, the
loader divides all per-op times by ``--time-scale`` (default 100). The
relative comparison between backends is preserved.

MOSEK is auto-skipped above ``--mosek-max-ops`` because the MILP grows
super-linearly; HEFT/CP-SAT/list-baselines remain in the comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "xpu-rt"))

from realistic_workloads import e2e_envelope, pack_periodic_workload  # noqa: E402
from schedulers import get_scheduler  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from report import SchedulerResult, render_gantt  # noqa: E402
from postprocessing import validate_schedule  # noqa: E402


DEFAULT_DRONET_HZ = [5, 10, 20, 30]
DEFAULT_MLP_HZ = [50, 100, 200]
DEFAULT_SCHEDULERS = "heft,critical_path,edf,fastest_device,fifo,mosek"


def _apply_time_scale(workload, factor: float):
    """In-place divide every operation's processing_times by ``factor``.

    Implements the firesim → production time compression. ``factor=100`` is
    the default (firesim ≈ 1 % of silicon speed).
    """
    if factor == 1.0:
        return
    for op in workload.operations:
        op.processing_times = [p / factor for p in op.processing_times]
        if op.deadline_us is not None:
            # Note: deadlines are set by pack_periodic_workload from period_us,
            # which is f_hz-derived — they're already in "wall-clock" target
            # units. So we DO NOT scale deadlines; we scale only compute time.
            pass


def _run_one_cell(
    soc: str,
    f_dronet: float,
    f_mlp: float,
    scheduler_name: str,
    envelope_us: float,
    time_scale: float,
    mosek_max_ops: int,
    gantt_dir: Path,
    time_limit: float,
    cpsat_max_ops: int = 200,
) -> Dict[str, Any]:
    """Build + schedule + measure + plot one cell. Returns metrics dict."""
    wl = pack_periodic_workload(
        envelope_us=envelope_us,
        instances=[("dronet", f_dronet), ("mlp_wide", f_mlp)],
        soc=soc,
    )
    # time_scale only applies to chipyard (firesim measurements); QRB5165 is
    # already real-silicon timings.
    effective_scale = time_scale if soc == "chipyard" else 1.0
    _apply_time_scale(wl, effective_scale)
    n_ops = len(wl.operations)

    base_row: Dict[str, Any] = {
        "soc": soc,
        "f_dronet_hz": f_dronet,
        "f_mlp_hz": f_mlp,
        "scheduler": scheduler_name,
        "n_ops": n_ops,
        "envelope_us": envelope_us,
        "time_scale": time_scale,
    }

    if scheduler_name == "mosek" and n_ops > mosek_max_ops:
        base_row.update(feasible=False, skipped=True,
                        reason=f"MOSEK skipped: {n_ops} ops > limit {mosek_max_ops}")
        return base_row
    if scheduler_name == "cpsat" and n_ops > cpsat_max_ops:
        base_row.update(feasible=False, skipped=True,
                        reason=f"CP-SAT skipped: {n_ops} ops > limit {cpsat_max_ops}")
        return base_row

    sched = get_scheduler(scheduler_name)
    kwargs: Dict[str, Any] = {}
    if scheduler_name == "mosek":
        kwargs = dict(
            solver_verbosity=0, time_limit=time_limit,
            restrict_makespan_to_nonperiodic=False,
            prune_cross_period_constraints=False,
        )
    elif scheduler_name == "cpsat":
        kwargs = dict(time_limit=time_limit)

    t0 = time.perf_counter()
    try:
        t, alpha, _, _ = sched(wl, **kwargs)
    except Exception as exc:
        base_row.update(feasible=False, error=str(exc))
        return base_row
    wall = time.perf_counter() - t0
    if t is None or alpha is None:
        base_row.update(feasible=False, error="solver_returned_none")
        return base_row

    try:
        ok, _ = validate_schedule(wl, t, alpha, original_json_data={"dispatches": {}})
    except Exception as exc:
        ok = False
        print(f"    validate failed: {exc}")

    m = compute_metrics(wl, t, alpha,
                       scheduler_name=scheduler_name,
                       solver_wall_time_s=wall)

    # Render Gantt.
    gantt_dir.mkdir(parents=True, exist_ok=True)
    gantt_path = gantt_dir / f"{soc}_d{f_dronet:g}Hz_m{f_mlp:g}Hz_{scheduler_name}.png"
    try:
        res = SchedulerResult(scheduler_name=scheduler_name, workload=wl, t=t, alpha=alpha,
                              metrics=m, feasible=True)
        render_gantt(res, str(gantt_path),
                     title=(f"{soc} | dronet {f_dronet:g}Hz + mlp_wide {f_mlp:g}Hz | "
                            f"{scheduler_name} | misses={m['deadline_miss_count']}"))
    except Exception as exc:
        print(f"    gantt failed: {exc}")
        gantt_path = None

    base_row.update(
        feasible=True,
        valid=bool(ok),
        makespan_us=m["makespan_us"],
        deadline_miss_count=m["deadline_miss_count"],
        deadline_miss_ratio=m["deadline_miss_ratio"],
        total_lateness_us=m["total_lateness_us"],
        max_lateness_us=m["max_lateness_us"],
        cross_device_transitions=m["cross_device_transitions"],
        critical_path_us=m["critical_path_us"],
        solver_wall_time_s=m["solver_wall_time_s"],
        per_machine_utilization=m["per_machine_utilization"],
        gantt=str(gantt_path) if gantt_path else None,
    )
    return base_row


def _plot_packing_frontier(rows: List[Dict[str, Any]], soc: str, out_path: Path):
    """Small-multiples heatmap: one panel per scheduler, cells coloured by
    miss-count (green = 0 misses, red = many). Skipped cells shown in grey.
    """
    soc_rows = [r for r in rows if r["soc"] == soc]
    if not soc_rows:
        return
    schedulers = sorted({r["scheduler"] for r in soc_rows})
    fds = sorted({r["f_dronet_hz"] for r in soc_rows})
    fms = sorted({r["f_mlp_hz"] for r in soc_rows})

    # Determine global vmax for the colour scale (per soc).
    misses = [r.get("deadline_miss_count", 0) for r in soc_rows
              if r.get("feasible") and not r.get("skipped")]
    vmax = max(misses) if misses else 1
    if vmax == 0:
        vmax = 1

    n = len(schedulers)
    cols = min(3, n)
    rows_panels = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_panels, cols,
                             figsize=(4.0 * cols, 3.6 * rows_panels),
                             constrained_layout=True, squeeze=False)

    for idx, sched in enumerate(schedulers):
        ax = axes[idx // cols][idx % cols]
        grid = np.full((len(fms), len(fds)), np.nan)
        for r in soc_rows:
            if r["scheduler"] != sched:
                continue
            i = fms.index(r["f_mlp_hz"])
            j = fds.index(r["f_dronet_hz"])
            if r.get("skipped"):
                grid[i, j] = -1  # special marker
            elif r.get("feasible"):
                grid[i, j] = float(r.get("deadline_miss_count", 0))
        # Mask skipped cells separately.
        skip_mask = (grid == -1)
        plot_grid = np.where(skip_mask, np.nan, grid)
        im = ax.imshow(
            plot_grid, origin="lower",
            cmap="RdYlGn_r",
            vmin=0, vmax=vmax,
            aspect="auto",
            extent=[-0.5, len(fds) - 0.5, -0.5, len(fms) - 0.5],
        )
        # Overlay skipped cells in grey.
        for i in range(len(fms)):
            for j in range(len(fds)):
                if skip_mask[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               facecolor="#dddddd", edgecolor="white"))
                    ax.text(j, i, "skip", ha="center", va="center", fontsize=7, color="black")
                else:
                    val = grid[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{int(val)}", ha="center", va="center",
                                fontsize=8, color="black")
        ax.set_xticks(range(len(fds)))
        ax.set_xticklabels([f"{x:g}" for x in fds])
        ax.set_yticks(range(len(fms)))
        ax.set_yticklabels([f"{y:g}" for y in fms])
        ax.set_xlabel("dronet (Hz)")
        ax.set_ylabel("mlp_wide (Hz)")
        ax.set_title(sched)

    # Hide unused subplots.
    for k in range(n, rows_panels * cols):
        axes[k // cols][k % cols].axis("off")

    fig.suptitle(f"Packing-frontier — {soc} (cell value = deadline miss count; "
                 f"green=0 misses, red=many; grey=skipped)")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soc", choices=["chipyard", "qrb5165", "both"], default="both")
    ap.add_argument("--schedulers", default=DEFAULT_SCHEDULERS,
                    help="Comma-separated scheduler names")
    ap.add_argument("--dronet-hz", default=",".join(map(str, DEFAULT_DRONET_HZ)))
    ap.add_argument("--mlp-hz", default=",".join(map(str, DEFAULT_MLP_HZ)))
    ap.add_argument("--envelope-us", type=float, default=None,
                    help="Period envelope in microseconds. If omitted, use "
                         "1e6 us (1 s) for chipyard, 50e3 us (50 ms) for qrb5165.")
    ap.add_argument("--time-scale", type=float, default=100.0,
                    help="Divide per-op times by this factor (firesim "
                         "→ production-realistic scaling). Default 100.")
    ap.add_argument("--time-limit", type=float, default=30.0,
                    help="MOSEK time-limit seconds per cell.")
    ap.add_argument("--mosek-max-ops", type=int, default=80,
                    help="Skip MOSEK on cells with more than this many ops.")
    ap.add_argument("--cpsat-max-ops", type=int, default=200,
                    help="Skip CP-SAT on cells with more than this many ops.")
    ap.add_argument("--out", default=str(REPO / "results" / "robotic_packing"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    gantt_dir = out_dir / "gantts"
    out_dir.mkdir(parents=True, exist_ok=True)

    socs = ["chipyard", "qrb5165"] if args.soc == "both" else [args.soc]
    schedulers = [s.strip() for s in args.schedulers.split(",") if s.strip()]
    dronet_hzs = [float(x) for x in args.dronet_hz.split(",")]
    mlp_hzs = [float(x) for x in args.mlp_hz.split(",")]

    rows: List[Dict[str, Any]] = []
    envelopes: Dict[str, float] = {}

    for soc in socs:
        if args.envelope_us is not None:
            envelope = args.envelope_us
        elif soc == "chipyard":
            envelope = 1_000_000.0  # 1 s (firesim-time; effectively ~10 ms production after /100)
        else:
            envelope = 50_000.0  # 50 ms (QRB5165 is already real-silicon)
        envelopes[soc] = envelope

        for fd in dronet_hzs:
            for fm in mlp_hzs:
                for sched in schedulers:
                    print(f"-- {soc} d{fd:g}Hz m{fm:g}Hz {sched} (env={envelope/1000:.0f}ms)")
                    row = _run_one_cell(
                        soc=soc, f_dronet=fd, f_mlp=fm, scheduler_name=sched,
                        envelope_us=envelope, time_scale=args.time_scale,
                        mosek_max_ops=args.mosek_max_ops, gantt_dir=gantt_dir,
                        time_limit=args.time_limit, cpsat_max_ops=args.cpsat_max_ops,
                    )
                    rows.append(row)
                    summary = (f"feasible={row.get('feasible')} "
                               f"makespan={row.get('makespan_us', 'n/a')} "
                               f"misses={row.get('deadline_miss_count', 'n/a')}")
                    print(f"   {summary}")

    # Write metrics CSV.
    csv_path = out_dir / "metrics.csv"
    if rows:
        fields = sorted({k for r in rows for k in r if not isinstance(r.get(k), dict)})
        # Move common columns first.
        first = ["soc", "f_dronet_hz", "f_mlp_hz", "scheduler", "n_ops",
                 "feasible", "valid", "makespan_us", "deadline_miss_count",
                 "total_lateness_us", "cross_device_transitions",
                 "solver_wall_time_s", "envelope_us", "time_scale",
                 "skipped", "reason", "error", "gantt"]
        ordered = [c for c in first if c in fields] + [c for c in fields if c not in first]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: v for k, v in r.items() if not isinstance(v, dict)})
        print(f"\nMetrics -> {csv_path}")

    # Packing-frontier plots per SoC.
    for soc in socs:
        p = out_dir / f"packing_frontier_{soc}.png"
        _plot_packing_frontier(rows, soc, p)
        print(f"Frontier ({soc}) -> {p}")

    # Markdown report.
    report_path = out_dir / "report.md"
    write_report(rows, schedulers, socs, envelopes, args, report_path, out_dir)
    print(f"Report -> {report_path}")


def write_report(rows, schedulers, socs, envelopes, args, path, out_dir):
    lines: List[str] = []
    lines.append("# — robotic packing stress test")
    lines.append("")
    lines.append(f"- envelope_us per soc: {envelopes}")
    lines.append(f"- time_scale: {args.time_scale} (firesim/silicon compression)")
    lines.append(f"- dronet frequencies: {args.dronet_hz}")
    lines.append(f"- mlp_wide frequencies: {args.mlp_hz}")
    lines.append(f"- schedulers: {schedulers}")
    lines.append("")
    for soc in socs:
        lines.append(f"## {soc}")
        lines.append("")
        lines.append(f"![frontier](packing_frontier_{soc}.png)")
        lines.append("")
        # Per-scheduler summary: at how many (f_d, f_m) cells does it hit 0 misses?
        sched_zero_cells: Dict[str, int] = {s: 0 for s in schedulers}
        sched_skip: Dict[str, int] = {s: 0 for s in schedulers}
        sched_misses_total: Dict[str, int] = {s: 0 for s in schedulers}
        for r in rows:
            if r["soc"] != soc:
                continue
            s = r["scheduler"]
            if r.get("skipped"):
                sched_skip[s] = sched_skip.get(s, 0) + 1
                continue
            if not r.get("feasible"):
                continue
            if r.get("deadline_miss_count", 0) == 0:
                sched_zero_cells[s] = sched_zero_cells.get(s, 0) + 1
            sched_misses_total[s] = sched_misses_total.get(s, 0) + r.get("deadline_miss_count", 0)
        lines.append("| scheduler | feasible cells with 0 misses | total misses (across all cells) | cells skipped |")
        lines.append("|---|---:|---:|---:|")
        for s in schedulers:
            lines.append(f"| {s} | {sched_zero_cells.get(s, 0)} "
                         f"| {sched_misses_total.get(s, 0)} "
                         f"| {sched_skip.get(s, 0)} |")

        # Headline: tightest cell.
        tight = max((r["f_dronet_hz"] * r["f_mlp_hz"] for r in rows if r["soc"] == soc), default=0)
        if tight > 0:
            tightest_cells = [r for r in rows
                              if r["soc"] == soc
                              and r["f_dronet_hz"] * r["f_mlp_hz"] == tight
                              and r.get("feasible")]
            if tightest_cells:
                best = min(tightest_cells, key=lambda r: r.get("deadline_miss_count", 1e9))
                lines.append("")
                lines.append(f"**Tightest cell** ({best['f_dronet_hz']:g} Hz dronet, "
                             f"{best['f_mlp_hz']:g} Hz mlp_wide): "
                             f"best scheduler is `{best['scheduler']}` with "
                             f"{best['deadline_miss_count']} deadline misses.")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
