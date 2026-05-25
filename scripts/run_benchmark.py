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


def _build_scaling_workload(n_ops: int, seed: int, soc: str = "chipyard"):
    """Construct a workload of approximately ``n_ops`` operations by
    chaining replicas of a real graph (dronet has 15 ops; chain K=ceil(N/15)
    copies and trim to N). Pseudo-grounded in real measurements.
    """
    from realistic_workloads import build_model_graph, build_workload_from_graph
    from workload import Operation, Workload

    rng = np.random.default_rng(seed)
    # Use dronet (15 ops) as the unit chain. For very small N, take a
    # subset; for large N, chain replicas + minor cost jitter per replica
    # so the ML/HEFT solvers see realistic-but-distinct sub-problems.
    g = build_model_graph("dronet", soc)
    base_wl = build_workload_from_graph(g)
    base_n = len(base_wl.operations)

    # Convert base_wl 4-machine SoC to the 3-machine convention used by
    # scenarios.py so all benchmarks share the same SoC model.
    from scenarios import MACHINES, COMBOS, TRANSFER
    machines, combos, transfer = MACHINES, COMBOS, TRANSFER
    name_map = {m: i for i, m in enumerate(base_wl.machines)}
    cpu_idxs = [name_map[m] for m in base_wl.machines if m in ("scalar", "rvv", "cpu")]
    gpu_idxs = [name_map[m] for m in base_wl.machines if m in ("opu", "gpu")] or cpu_idxs
    npu_idxs = [name_map[m] for m in base_wl.machines if m in ("gemmini", "npu", "hta")] or gpu_idxs

    def _map_costs(src_op):
        pts = list(src_op.processing_times)
        cpu_cost = min(pts[i] for i in cpu_idxs) if cpu_idxs else pts[0]
        gpu_cost = min(pts[i] for i in gpu_idxs)
        npu_cost = min(pts[i] for i in npu_idxs)
        return [float(cpu_cost), float(gpu_cost), float(npu_cost)]

    # Build replicas with ±20% jitter per replica.
    n_replicas = max(1, (n_ops + base_n - 1) // base_n)
    all_ops: List[Operation] = []
    last_sink: Optional[Operation] = None
    for rep in range(n_replicas):
        jitter = rng.uniform(0.8, 1.2, size=3)
        idx_of = {id(o): i for i, o in enumerate(base_wl.operations)}
        replica_ops: List[Operation] = []
        for src_op in base_wl.operations:
            base_costs = _map_costs(src_op)
            costs = [float(c * jitter[k]) for k, c in enumerate(base_costs)]
            op = Operation(
                processing_times=costs,
                operation_name=f"r{rep}_{src_op.operation_name}",
                infeasible_combinations=set(),
            )
            op.output_bytes = getattr(src_op, "output_bytes", 0)
            op.job_id = rep
            replica_ops.append(op)
        for i, src in enumerate(base_wl.operations):
            for p in src.get_predecessors():
                if id(p) in idx_of:
                    replica_ops[i].add_predecessor(replica_ops[idx_of[id(p)]])
        # Chain replicas: connect prior sink to this replica's source.
        if last_sink is not None and replica_ops:
            replica_ops[0].add_predecessor(last_sink)
        if replica_ops:
            last_sink = replica_ops[-1]
            all_ops.extend(replica_ops)
        if len(all_ops) >= n_ops:
            break

    # Trim to n_ops (or close to it).
    all_ops = all_ops[:n_ops]
    return Workload(all_ops, machines, np.array(transfer),
                    job_names=[f"replica_{r}" for r in range(n_replicas)],
                    machine_combinations=combos)


def run_scaling(args):
    out_dir = REPO / "results" / "scaling"
    out_dir.mkdir(parents=True, exist_ok=True)
    schedulers = args.schedulers.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]
    n_seeds = args.seeds

    rows: List[Dict[str, Any]] = []
    print(f"Scaling sweep: {len(sizes)} sizes × {n_seeds} seeds × "
          f"{len(schedulers)} schedulers = "
          f"{len(sizes) * n_seeds * len(schedulers)} cells")

    for n in sizes:
        for seed in range(n_seeds):
            wl = _build_scaling_workload(n, seed=args.seed * 1000 + seed)
            actual_n = len(wl.operations)
            for s in schedulers:
                # Scale CP-SAT/MOSEK time limit with problem size.
                tl = max(args.time_limit, n * 0.3)
                # Skip MOSEK above 80 ops (existing convention).
                if s == "mosek" and n > 80:
                    rows.append({"n_ops": n, "actual_n": actual_n, "seed": seed,
                                 "scheduler": s, "feasible": False,
                                 "error": "skipped_too_large"})
                    continue
                r = _run_one(wl, s, tl)
                row = {"n_ops": n, "actual_n": actual_n, "seed": seed,
                       "scheduler": s, **r}
                rows.append(row)
            print(f"  N={n:>4d}  seed={seed}  done ({len(rows)} rows)")

    # CSV
    csv_path = out_dir / "metrics.csv"
    fields = ["n_ops", "actual_n", "seed", "scheduler", "feasible", "valid",
              "makespan_us", "deadline_miss_count", "total_lateness_us",
              "cross_device_transitions", "critical_path_us",
              "solver_wall_time_s", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nMetrics -> {csv_path}")

    _plot_scaling(rows, out_dir, schedulers, sizes)
    _write_scaling_report(rows, out_dir, schedulers, sizes)


def _plot_scaling(rows, out_dir: Path, schedulers, sizes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for idx, s in enumerate(schedulers):
        xs, ms_mean, ms_std = [], [], []
        ws_mean, ws_std = [], []
        for n in sizes:
            ms_vals = [r["makespan_us"] for r in rows
                       if r["scheduler"] == s and r["n_ops"] == n
                       and r.get("feasible") and r.get("makespan_us")]
            ws_vals = [r["solver_wall_time_s"] for r in rows
                       if r["scheduler"] == s and r["n_ops"] == n
                       and r.get("feasible") and r.get("solver_wall_time_s") is not None]
            if not ms_vals:
                continue
            xs.append(n)
            ms_mean.append(np.mean(ms_vals)); ms_std.append(np.std(ms_vals))
            ws_mean.append(np.mean(ws_vals)); ws_std.append(np.std(ws_vals))
        if xs:
            color = cmap(idx % 10)
            ax1.errorbar(xs, ms_mean, yerr=ms_std, label=s, marker="o", color=color)
            ax2.errorbar(xs, ws_mean, yerr=ws_std, label=s, marker="o", color=color)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("N ops"); ax1.set_ylabel("makespan (us)")
    ax1.set_title("Makespan vs N")
    ax1.legend(fontsize=8, loc="best"); ax1.grid(True, alpha=0.3)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("N ops"); ax2.set_ylabel("solver wall time (s)")
    ax2.set_title("Solver wall time vs N (log-log)")
    ax2.legend(fontsize=8, loc="best"); ax2.grid(True, alpha=0.3)
    fig.suptitle("M15 — Scaling behavior (dronet-chained workloads with per-replica jitter)")
    fig.savefig(out_dir / "scaling_curves.png", dpi=120)
    plt.close(fig)


def _write_scaling_report(rows, out_dir, schedulers, sizes):
    lines = ["# M15 — Scaling sweep", ""]
    lines.append(f"- sizes: {sizes}")
    lines.append(f"- schedulers: {schedulers}")
    lines.append("")
    lines.append("![scaling curves](scaling_curves.png)")
    lines.append("")
    lines.append("## Solver wall time (s) at each N — median across seeds")
    lines.append("")
    lines.append("| scheduler | " + " | ".join(f"N={n}" for n in sizes) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(sizes)) + "|")
    for s in schedulers:
        cells = []
        for n in sizes:
            ws = [r["solver_wall_time_s"] for r in rows
                  if r["scheduler"] == s and r["n_ops"] == n
                  and r.get("feasible") and r.get("solver_wall_time_s") is not None]
            cells.append(f"{np.median(ws):.3f}" if ws else "skip")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Makespan vs N (median, us)")
    lines.append("")
    lines.append("| scheduler | " + " | ".join(f"N={n}" for n in sizes) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(sizes)) + "|")
    for s in schedulers:
        cells = []
        for n in sizes:
            ms = [r["makespan_us"] for r in rows
                  if r["scheduler"] == s and r["n_ops"] == n
                  and r.get("feasible") and r.get("makespan_us")]
            cells.append(f"{np.median(ms):,.0f}" if ms else "skip")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    with open(out_dir / "report.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Report -> {out_dir / 'report.md'}")


def _build_role_workload(role: str, soc: str = "qrb5165"):
    """Construct a Workload for one robotic role using REAL measured ops.
    role ∈ {camera, imu, control, planning, monitor}.
    """
    from realistic_workloads import build_model_graph, build_workload_from_graph
    from workload import Operation, Workload
    from scenarios import MACHINES, COMBOS, TRANSFER

    machines, combos, transfer = MACHINES, COMBOS, TRANSFER

    def _remap_to_3machine(base_wl):
        name_map = {m: i for i, m in enumerate(base_wl.machines)}
        cpu_idxs = [name_map[m] for m in base_wl.machines if m.lower() in ("cpu", "scalar", "rvv", "cpu_big", "cpu_little")]
        gpu_idxs = [name_map[m] for m in base_wl.machines if m.lower() in ("gpu", "opu")] or cpu_idxs
        npu_idxs = [name_map[m] for m in base_wl.machines if m.lower() in ("npu", "hta", "gemmini")] or gpu_idxs
        out_ops = []
        idx_of = {id(o): i for i, o in enumerate(base_wl.operations)}
        for src in base_wl.operations:
            pts = list(src.processing_times)
            cpu = min(pts[i] for i in cpu_idxs) if cpu_idxs else pts[0]
            gpu = min(pts[i] for i in gpu_idxs) if gpu_idxs else cpu
            npu = min(pts[i] for i in npu_idxs) if npu_idxs else gpu
            op = Operation(
                processing_times=[float(cpu), float(gpu), float(npu)],
                operation_name=src.operation_name,
                infeasible_combinations=set(),
            )
            op.output_bytes = getattr(src, "output_bytes", 0)
            out_ops.append(op)
        for i, src in enumerate(base_wl.operations):
            for p in src.get_predecessors():
                if id(p) in idx_of:
                    out_ops[i].add_predecessor(out_ops[idx_of[id(p)]])
        return out_ops

    if role == "camera":
        # dronet perception chain
        g = build_model_graph("dronet", soc)
        ops = _remap_to_3machine(build_workload_from_graph(g))
    elif role == "imu":
        # tiny MLP — mlp_wide chain
        g = build_model_graph("mlp_wide", soc)
        ops = _remap_to_3machine(build_workload_from_graph(g))
    elif role == "control":
        # synthesized 3-op chain based on real elementwise@CPU costs (~260us per op)
        ops = [
            Operation(processing_times=[260.0, 400.0, 600.0],
                      operation_name="ctrl_in",
                      infeasible_combinations={2}),
            Operation(processing_times=[260.0, 400.0, 600.0],
                      operation_name="ctrl_calc",
                      infeasible_combinations={2}),
            Operation(processing_times=[260.0, 400.0, 600.0],
                      operation_name="ctrl_out",
                      infeasible_combinations={2}),
        ]
        ops[1].add_predecessor(ops[0])
        ops[2].add_predecessor(ops[1])
    elif role == "planning":
        # yolov8n's tail (last 15 ops) treated as planning network
        g = build_model_graph("yolov8n", soc)
        full = _remap_to_3machine(build_workload_from_graph(g))
        # Take the last 15 ops and rewire (drop dangling predecessors)
        ops = full[-15:]
        old_to_new = {id(o): i for i, o in enumerate(ops)}
        for op in ops:
            op.predecessors = [p for p in op.predecessors if id(p) in old_to_new]
    elif role == "monitor":
        # 2 small ops: matmul + elementwise
        ops = [
            Operation(processing_times=[1500.0, 800.0, 300.0],
                      operation_name="monitor_check"),
            Operation(processing_times=[200.0, 250.0, 400.0],
                      operation_name="monitor_log",
                      infeasible_combinations={2}),
        ]
        ops[1].add_predecessor(ops[0])
    else:
        raise ValueError(f"unknown role: {role}")

    return Workload(ops, MACHINES, np.array(TRANSFER),
                    job_names=[role], machine_combinations=COMBOS)


def _pack_periodic(roles_and_hz: List[Tuple[str, float]], envelope_us: float,
                   soc: str = "qrb5165"):
    """Build a packed workload: for each (role, hz), instantiate
    floor(envelope_us / period_us) copies and tag each with release time +
    deadline. Returns the combined Workload."""
    from workload import Operation, Workload
    from scenarios import MACHINES, COMBOS, TRANSFER

    all_ops: List[Operation] = []
    job_names: List[str] = []
    job_id = 0
    for role, hz in roles_and_hz:
        period = 1e6 / hz
        n_inst = max(1, int(np.floor(envelope_us / period)))
        for inst in range(n_inst):
            release = inst * period
            deadline = (inst + 1) * period
            base = _build_role_workload(role, soc=soc)
            # Find the sink (max topological position) for deadline tagging.
            n = len(base.operations)
            op_idx = {id(op): i for i, op in enumerate(base.operations)}
            has_succ = set()
            for op in base.operations:
                for p in op.get_predecessors():
                    pi = op_idx.get(id(p))
                    if pi is not None:
                        has_succ.add(pi)
            sink_ids = [i for i in range(n) if i not in has_succ]

            job_names.append(f"{role}@{hz:g}Hz_inst{inst}")
            inst_ops = []
            for i, src in enumerate(base.operations):
                op = Operation(
                    processing_times=list(src.processing_times),
                    operation_name=f"{role}_{hz:g}Hz_i{inst}_{src.operation_name}",
                    infeasible_combinations=set(src.infeasible_combinations),
                    min_start_t=release,
                    deadline_us=deadline if i in sink_ids else None,
                )
                op.output_bytes = getattr(src, "output_bytes", 0)
                op.job_id = job_id
                inst_ops.append(op)
            for i, src in enumerate(base.operations):
                for p in src.get_predecessors():
                    pi = op_idx.get(id(p))
                    if pi is not None:
                        inst_ops[i].add_predecessor(inst_ops[pi])
            all_ops.extend(inst_ops)
            job_id += 1

    return Workload(all_ops, MACHINES, np.array(TRANSFER),
                    job_names=job_names, machine_combinations=COMBOS)


def run_realtime(args):
    out_dir = REPO / "results" / "realtime_packing"
    out_dir.mkdir(parents=True, exist_ok=True)

    schedulers = args.schedulers.split(",")
    envelope_us = args.envelope_us

    # 5-model packing config: real robotics frequencies.
    mix_specs = [
        ("camera_only",         [("camera", 30)]),
        ("camera_imu",          [("camera", 30), ("imu", 200)]),
        ("camera_imu_control",  [("camera", 30), ("imu", 200), ("control", 100)]),
        ("full_stack",          [("camera", 30), ("imu", 200), ("control", 100),
                                 ("planning", 10), ("monitor", 1)]),
    ]

    rows: List[Dict[str, Any]] = []
    print(f"Realtime packing: SoC=qrb5165, envelope={envelope_us}us (no time scaling)")
    for mix_name, mix in mix_specs:
        print(f"  Building mix={mix_name}: {mix}")
        try:
            wl = _pack_periodic(mix, envelope_us, soc="qrb5165")
        except Exception as exc:
            print(f"  [warn] build failed for {mix_name}: {exc}")
            continue
        n_ops = len(wl.operations)
        print(f"     packed {n_ops} ops")
        for s in schedulers:
            if n_ops > 200 and s in ("mosek", "cpsat"):
                tl = max(args.time_limit, 60.0)
            else:
                tl = args.time_limit
            if n_ops > 80 and s == "mosek":
                rows.append({"mix": mix_name, "scheduler": s, "n_ops": n_ops,
                             "feasible": False, "error": "skipped_too_large"})
                continue
            r = _run_one(wl, s, tl)
            row = {"mix": mix_name, "scheduler": s, "n_ops": n_ops, **r}
            rows.append(row)
            ms = r.get("makespan_us", "n/a")
            miss = r.get("deadline_miss_count", "n/a")
            print(f"     {s:<22s} ms={ms}  misses={miss}")

    # CSV
    csv_path = out_dir / "metrics.csv"
    fields = ["mix", "scheduler", "n_ops", "feasible", "valid",
              "makespan_us", "deadline_miss_count", "total_lateness_us",
              "cross_device_transitions", "critical_path_us",
              "solver_wall_time_s", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nMetrics -> {csv_path}")
    _write_realtime_report(rows, out_dir, schedulers, mix_specs, envelope_us)


def _write_realtime_report(rows, out_dir, schedulers, mix_specs, envelope_us):
    lines = ["# M17+M18 — Real-frequency QRB5165 packing", ""]
    lines.append(f"- SoC: QRB5165 (real silicon, NO time scaling)")
    lines.append(f"- envelope: {envelope_us} us")
    lines.append(f"- mixes:")
    for name, mix in mix_specs:
        lines.append(f"  - {name}: {mix}")
    lines.append(f"- schedulers: {schedulers}")
    lines.append("")
    for mix_name, mix in mix_specs:
        lines.append(f"## {mix_name}")
        lines.append("")
        lines.append("| scheduler | n_ops | makespan_us | misses | total_lateness_us | feasible |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for s in schedulers:
            r = next((r for r in rows if r["mix"] == mix_name and r["scheduler"] == s), None)
            if r is None:
                continue
            if r.get("error"):
                lines.append(f"| {s} | {r.get('n_ops', '-')} | error: {r['error']} | - | - | False |")
                continue
            lines.append(f"| {s} | {r['n_ops']} | {r.get('makespan_us', 0):.0f} | "
                         f"{r.get('deadline_miss_count', '-')} | "
                         f"{r.get('total_lateness_us', 0):.0f} | {r.get('feasible')} |")
        lines.append("")
    with open(out_dir / "report.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Report -> {out_dir / 'report.md'}")


def run_literature(args):
    from pegasus_loader import PEGASUS_DAGS
    from scheduler_ml import _lower_bound_makespan

    out_dir = REPO / "results" / "literature"
    out_dir.mkdir(parents=True, exist_ok=True)
    schedulers = args.schedulers.split(",")

    rows: List[Dict[str, Any]] = []
    print(f"Literature DAGs: {list(PEGASUS_DAGS.keys())}")
    for dag_name, builder in PEGASUS_DAGS.items():
        wl = builder()
        n_ops = len(wl.operations)
        lb = _lower_bound_makespan(wl)
        print(f"\n  {dag_name}: {n_ops} ops, critical_path_lower_bound={lb:.1f}us")
        for s in schedulers:
            r = _run_one(wl, s, args.time_limit)
            r["dag"] = dag_name
            r["n_ops"] = n_ops
            r["lower_bound_us"] = lb
            r["scheduler"] = s
            if r.get("feasible") and r.get("makespan_us"):
                r["ratio_vs_lb"] = r["makespan_us"] / lb
                print(f"    {s:<22s} ms={r['makespan_us']:.1f}us  "
                      f"ratio_vs_LB={r['ratio_vs_lb']:.2f}x  "
                      f"solver={r.get('solver_wall_time_s', 0):.3f}s")
            rows.append(r)

    csv_path = out_dir / "metrics.csv"
    fields = ["dag", "scheduler", "n_ops", "lower_bound_us", "feasible", "valid",
              "makespan_us", "ratio_vs_lb", "deadline_miss_count",
              "cross_device_transitions", "solver_wall_time_s", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nMetrics -> {csv_path}")

    # Report.
    lines = ["# M20 — Pegasus literature DAGs", "",
             "Programmatically constructed Pegasus-shaped workflows ",
             "(no DAX XML required). Compares HEFT-on-Pegasus to the ",
             "published 1.3-1.8x makespan-vs-LB range; lower is better.",
             ""]
    lines.append("| DAG | scheduler | ms (us) | LB (us) | ms/LB | solver_s |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in rows:
        if not r.get("feasible") or not r.get("makespan_us"):
            continue
        lines.append(
            f"| {r['dag']} | {r['scheduler']} | {r['makespan_us']:.0f} | "
            f"{r['lower_bound_us']:.0f} | {r.get('ratio_vs_lb', 0):.2f} | "
            f"{r.get('solver_wall_time_s', 0):.3f} |"
        )
    with open(out_dir / "report.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Report -> {out_dir / 'report.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_stress(args):
    from stress_scenarios import STRESS_SCENARIOS, frequency_sweep_breaking_point

    out_dir = REPO / "results" / "stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    schedulers = args.schedulers.split(",")

    rows: List[Dict[str, Any]] = []
    # 1-4. Standard stress scenarios.
    for sc_name, builder in STRESS_SCENARIOS.items():
        try:
            wl = builder()
        except Exception as exc:
            print(f"[warn] {sc_name} build failed: {exc}")
            continue
        n_ops = len(wl.operations)
        print(f"\n  {sc_name}: {n_ops} ops")
        for s in schedulers:
            # Generous time limits for exact solvers on big workloads.
            if s in ("cpsat", "cpsat_memory"):
                tl = max(args.time_limit, 60.0)
            elif s == "mosek" and n_ops > 100:
                rows.append({"scenario": sc_name, "scheduler": s, "n_ops": n_ops,
                             "feasible": False, "error": "skipped_too_large"})
                continue
            else:
                tl = args.time_limit
            r = _run_one(wl, s, tl)
            row = {"scenario": sc_name, "scheduler": s, "n_ops": n_ops, **r}
            rows.append(row)
            print(f"     {s:<22s} ms={r.get('makespan_us', 'n/a')}  "
                  f"misses={r.get('deadline_miss_count', 'n/a')}  "
                  f"solver_s={r.get('solver_wall_time_s', 0):.3f}")

    # 5. Frequency sweep — find each scheduler's breaking point.
    print(f"\n  frequency_sweep_breaking_point: sweeping mlp_hz...")
    sweep_rows: List[Dict[str, Any]] = []
    sweep_schedulers = [s for s in schedulers if s not in ("mosek",)]  # too slow
    for mlp_hz in (20, 50, 100, 200, 400, 800):
        wl = frequency_sweep_breaking_point(mlp_hz=mlp_hz)
        n_ops = len(wl.operations)
        for s in sweep_schedulers:
            tl = max(args.time_limit, 60.0) if s in ("cpsat", "cpsat_memory") else args.time_limit
            r = _run_one(wl, s, tl)
            sweep_rows.append({
                "mlp_hz": mlp_hz, "scheduler": s, "n_ops": n_ops, **r
            })
        print(f"    mlp_hz={mlp_hz:>4} n_ops={n_ops:>3} "
              f"misses: " + " ".join(
                  f"{s}={[r for r in sweep_rows if r['mlp_hz']==mlp_hz and r['scheduler']==s][0].get('deadline_miss_count', 'fail')}"
                  for s in sweep_schedulers if s in ("heft", "edf", "cpsat", "fastest_device")))

    # Write CSVs
    fields = ["scenario", "scheduler", "n_ops", "feasible", "valid",
              "makespan_us", "deadline_miss_count", "total_lateness_us",
              "cross_device_transitions", "critical_path_us",
              "solver_wall_time_s", "error"]
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nMetrics -> {csv_path}")

    sweep_fields = ["mlp_hz", "scheduler", "n_ops", "feasible",
                    "makespan_us", "deadline_miss_count", "total_lateness_us",
                    "solver_wall_time_s", "error"]
    sweep_csv = out_dir / "frequency_sweep.csv"
    with open(sweep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sweep_fields, extrasaction="ignore")
        w.writeheader()
        for r in sweep_rows:
            w.writerow(r)
    print(f"Frequency-sweep -> {sweep_csv}")

    _plot_breaking_point(sweep_rows, sweep_schedulers, out_dir)
    _write_stress_report(rows, sweep_rows, schedulers, out_dir)


def _plot_breaking_point(sweep_rows, schedulers, out_dir):
    import matplotlib.pyplot as plt
    hzs = sorted({r["mlp_hz"] for r in sweep_rows})
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for idx, s in enumerate(schedulers):
        xs, ys = [], []
        for hz in hzs:
            row = next((r for r in sweep_rows
                        if r["mlp_hz"] == hz and r["scheduler"] == s
                        and r.get("feasible")), None)
            if row is None:
                continue
            xs.append(hz); ys.append(row.get("deadline_miss_count", 0))
        if xs:
            ax.plot(xs, ys, marker="o", label=s, color=cmap(idx % 10))
    ax.set_xlabel("mlp_wide frequency (Hz)")
    ax.set_ylabel("deadline miss count")
    ax.set_xscale("log")
    ax.set_title("M23 — Breaking-point sweep: deadline misses vs mlp_wide frequency")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "breaking_point.png", dpi=120)
    plt.close(fig)


def _write_stress_report(rows, sweep_rows, schedulers, out_dir):
    lines = ["# M23 — Stress-test scenarios", ""]
    by_scenario = {}
    for r in rows:
        by_scenario.setdefault(r["scenario"], []).append(r)
    for sc_name, sc_rows in by_scenario.items():
        lines.append(f"## {sc_name}")
        lines.append("")
        n = sc_rows[0].get("n_ops", "?")
        lines.append(f"Workload size: {n} ops")
        lines.append("")
        lines.append("| scheduler | makespan_us | misses | total_lateness_us | solver_s |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in sc_rows:
            if r.get("error"):
                lines.append(f"| {r['scheduler']} | error: {r['error']} | - | - | - |")
                continue
            lines.append(f"| {r['scheduler']} | {r.get('makespan_us', 0):.0f} | "
                         f"{r.get('deadline_miss_count', '-')} | "
                         f"{r.get('total_lateness_us', 0):.0f} | "
                         f"{r.get('solver_wall_time_s', 0):.3f} |")
        lines.append("")

    lines.append("## frequency_sweep_breaking_point")
    lines.append("")
    lines.append("![breaking point](breaking_point.png)")
    lines.append("")
    hzs = sorted({r["mlp_hz"] for r in sweep_rows})
    lines.append("| scheduler | " + " | ".join(f"mlp={h}Hz" for h in hzs) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(hzs)) + "|")
    for s in schedulers:
        cells = []
        for hz in hzs:
            r = next((r for r in sweep_rows
                      if r["mlp_hz"] == hz and r["scheduler"] == s
                      and r.get("feasible")), None)
            cells.append(f"{r.get('deadline_miss_count', '-')}" if r else "skip")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    with open(out_dir / "report.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Report -> {out_dir / 'report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["robustness", "scaling", "realtime", "literature", "stress"],
                    required=True)
    ap.add_argument("--schedulers",
                    default="heft,critical_path,edf,fastest_device,fifo,peft,"
                            "simulated_annealing,cpsat,gnn_placement,rl_policy")
    ap.add_argument("--sigmas", default="0,10,25,50",
                    help="Comma-separated noise percentages (M16)")
    ap.add_argument("--sizes", default="20,50,100,200,500",
                    help="Comma-separated workload sizes (M15)")
    ap.add_argument("--seeds", type=int, default=10, help="Seeds per cell (M15/M16)")
    ap.add_argument("--envelope-us", type=float, default=100_000.0,
                    help="Period envelope (M17/M18) — default 100ms")
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
    elif args.target == "stress":
        run_stress(args)


if __name__ == "__main__":
    main()
