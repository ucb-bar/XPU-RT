"""Experiment 1: scheduling A/B between greedy, MOSEK MILP, and joint CP-SAT.

Runs a fixed workload battery through three solvers and emits both a
JSONL trace (for downstream plotting) and a one-page Markdown summary.

Usage:
    uv run python scripts/experiments/exp1_scheduling_ab.py [--quick]

``--quick`` caps n_ops at 200 and skips the QNN + transformer_block
workloads so the run finishes in well under a minute. The default
configuration includes the scaling sweep up to n=5000 and the real
PyTorch workloads.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiments._synthetic_dag import (  # noqa: E402
    SyntheticDag,
    chain,
    diamond,
    fan_out,
    random_dag,
    transformer_block,
)
from xpu_rt.solve.schedule_joint_cpsat import solve_schedule_joint  # noqa: E402

OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp1"


@dataclass
class SolverResult:
    workload: str
    solver: str
    n_ops: int
    num_devices: int
    makespan_us: float
    solver_time_ms: float
    feasible: bool
    status: str
    note: str = ""


def _topo_order(pids: list[str], deps: dict[str, list[str]]) -> list[str]:
    visited: set[str] = set()
    order: list[str] = []

    def visit(p: str) -> None:
        if p in visited:
            return
        visited.add(p)
        for d in deps.get(p, []):
            visit(d)
        order.append(p)

    for pid in pids:
        visit(pid)
    return order


def greedy_schedule(dag: SyntheticDag) -> SolverResult:
    """Earliest-finish-time list scheduler over the SyntheticDag."""
    t0 = time.perf_counter()
    topo = _topo_order(dag.partition_ids, dag.dependencies)
    device_avail = [0.0] * dag.num_devices
    end_times: dict[str, float] = {}
    chosen_dev: dict[str, int] = {}
    for pid in topo:
        durations = dag.durations_us_by_device[pid]
        best_end = math.inf
        best_dev = -1
        best_start = 0.0
        for d in range(dag.num_devices):
            dur = durations[d]
            if dur is None or (isinstance(dur, float) and (math.isinf(dur) or math.isnan(dur))):
                continue
            ready = device_avail[d]
            for pred in dag.dependencies.get(pid, []):
                pred_end = end_times[pred]
                pred_dev = chosen_dev[pred]
                ready = max(ready, pred_end + dag.transfer_us[pred_dev][d])
            end = ready + float(dur)
            if end < best_end:
                best_end = end
                best_dev = d
                best_start = ready
        end_times[pid] = best_end
        chosen_dev[pid] = best_dev
        device_avail[best_dev] = best_end
        del best_start  # only used for clarity in traces
    makespan = max(end_times.values()) if end_times else 0.0
    return SolverResult(
        workload=dag.name,
        solver="greedy",
        n_ops=len(dag.partition_ids),
        num_devices=dag.num_devices,
        makespan_us=makespan,
        solver_time_ms=(time.perf_counter() - t0) * 1000,
        feasible=True,
        status="optimal_local",
    )


def cpsat_joint(dag: SyntheticDag, timeout_ms: int) -> SolverResult:
    sol = solve_schedule_joint(
        partition_ids=dag.partition_ids,
        durations_us_by_device=dag.durations_us_by_device,
        dependencies=dag.dependencies,
        num_devices=dag.num_devices,
        transfer_us=dag.transfer_us,
        timeout_ms=timeout_ms,
    )
    return SolverResult(
        workload=dag.name,
        solver="cpsat_joint",
        n_ops=len(dag.partition_ids),
        num_devices=dag.num_devices,
        makespan_us=sol.makespan_us,
        solver_time_ms=sol.solve_time_ms,
        feasible=sol.feasible,
        status=sol.status,
    )


def mosek_milp(dag: SyntheticDag, time_limit_s: float) -> SolverResult:
    """Route a SyntheticDag through the existing CVXPY/MOSEK MILP."""
    import numpy as np
    from xpu_rt.scheduler.scheduler import schedule
    from xpu_rt.scheduler.workload import Operation, Workload

    pids = dag.partition_ids
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    ops: list[Operation] = []
    for i, pid in enumerate(pids):
        proc = [float(x) for x in dag.durations_us_by_device[pid]]
        ops.append(
            Operation(
                processing_times=proc,
                operation_id=i,
                operation_name=pid,
                job_id=0,
            )
        )
    for pid in pids:
        for pred in dag.dependencies.get(pid, []):
            ops[pid_to_idx[pid]].add_predecessor(ops[pid_to_idx[pred]])
    machines = [f"dev_{d}" for d in range(dag.num_devices)]
    transfer = np.asarray(dag.transfer_us, dtype=float)
    workload = Workload(ops, machines, transfer)
    t0 = time.perf_counter()
    silent_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(silent_buf), contextlib.redirect_stderr(silent_buf):
            t_arr, alpha, _, _ = schedule(workload, time_limit=time_limit_s)
    except Exception as exc:
        return SolverResult(
            workload=dag.name,
            solver="mosek_milp",
            n_ops=len(pids),
            num_devices=dag.num_devices,
            makespan_us=float("inf"),
            solver_time_ms=(time.perf_counter() - t0) * 1000,
            feasible=False,
            status="error",
            note=type(exc).__name__,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    state = getattr(workload, "solver_state", {}) or {}
    status_str = str(state.get("problem_status", "unknown"))
    feasible = t_arr is not None and alpha is not None and status_str.lower() in ("optimal", "optimal_inaccurate")
    if not feasible:
        return SolverResult(
            workload=dag.name,
            solver="mosek_milp",
            n_ops=len(pids),
            num_devices=dag.num_devices,
            makespan_us=float("inf"),
            solver_time_ms=elapsed_ms,
            feasible=False,
            status=status_str,
        )
    makespan = float(state.get("makespan") or 0.0)
    if makespan <= 0.0:
        end_times = []
        for i, pid in enumerate(pids):
            k = int(alpha[i].argmax())
            end_times.append(float(t_arr[i]) + float(dag.durations_us_by_device[pid][k]))
        makespan = max(end_times)
    return SolverResult(
        workload=dag.name,
        solver="mosek_milp",
        n_ops=len(pids),
        num_devices=dag.num_devices,
        makespan_us=makespan,
        solver_time_ms=elapsed_ms,
        feasible=True,
        status=status_str,
    )


def build_workload_battery(quick: bool) -> list[tuple[str, Callable[[], SyntheticDag]]]:
    items: list[tuple[str, Callable[[], SyntheticDag]]] = [
        ("chain_10", lambda: chain(10)),
        ("fan_out_8", lambda: fan_out(8)),
        ("diamond_5", lambda: diamond(5)),
        ("transformer_L4", lambda: transformer_block(layers=4)),
    ]
    if not quick:
        items.append(("transformer_L12", lambda: transformer_block(layers=12)))
    sweep_sizes = [10, 50, 200] if quick else [10, 50, 200, 1000, 5000]
    for n in sweep_sizes:
        items.append((f"random_n{n}", lambda n=n: random_dag(n_ops=n)))
    return items


def run_experiment(quick: bool) -> list[SolverResult]:
    cpsat_timeout_ms = 5000 if quick else 30000
    mosek_time_limit_s = 5.0 if quick else 30.0
    mosek_max_n = 60 if quick else 200

    results: list[SolverResult] = []
    battery = build_workload_battery(quick)
    for name, getter in battery:
        dag = getter()
        n = len(dag.partition_ids)
        print(f"[exp1] workload={name} n_ops={n} devices={dag.num_devices}")

        g = greedy_schedule(dag)
        print(f"  greedy      makespan={g.makespan_us:.2f}us  time={g.solver_time_ms:.2f}ms")
        results.append(g)

        c = cpsat_joint(dag, timeout_ms=cpsat_timeout_ms)
        print(
            f"  cpsat_joint makespan={c.makespan_us:.2f}us  time={c.solver_time_ms:.2f}ms  status={c.status}"
        )
        results.append(c)

        if n <= mosek_max_n:
            m = mosek_milp(dag, time_limit_s=mosek_time_limit_s)
            print(
                f"  mosek_milp  makespan={m.makespan_us:.2f}us  time={m.solver_time_ms:.2f}ms  status={m.status}"
            )
            results.append(m)
        else:
            skipped = SolverResult(
                workload=dag.name,
                solver="mosek_milp",
                n_ops=n,
                num_devices=dag.num_devices,
                makespan_us=float("inf"),
                solver_time_ms=0.0,
                feasible=False,
                status="skipped",
                note=f"n_ops>{mosek_max_n}",
            )
            print(f"  mosek_milp  skipped (n_ops>{mosek_max_n})")
            results.append(skipped)
    return results


def write_jsonl(results: list[SolverResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")


def _fmt_us(v: float) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:,.1f}"


def _fmt_ms(v: float) -> str:
    return f"{v:,.2f}"


def write_summary(results: list[SolverResult], path: Path, quick: bool) -> None:
    by_wl: dict[str, dict[str, SolverResult]] = defaultdict(dict)
    for r in results:
        by_wl[r.workload][r.solver] = r

    lines: list[str] = []
    lines.append("# Experiment 1 — scheduling A/B (greedy vs MOSEK MILP vs CP-SAT joint)\n")
    lines.append(f"Mode: {'quick' if quick else 'full'}\n\n")
    lines.append(
        "MOSEK is skipped on workloads above the per-mode `n_ops` cap to keep the run within budget.\n\n"
    )

    lines.append("## Per-workload results\n")
    lines.append("| workload | n_ops | devices | greedy makespan (µs) | MOSEK makespan (µs) | CP-SAT makespan (µs) | greedy ms | MOSEK ms | CP-SAT ms | CP-SAT status |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    workload_order = list(by_wl.keys())
    for wl in workload_order:
        bucket = by_wl[wl]
        g = bucket.get("greedy")
        m = bucket.get("mosek_milp")
        c = bucket.get("cpsat_joint")
        n = (g or m or c).n_ops
        nd = (g or m or c).num_devices
        lines.append(
            "| {wl} | {n} | {nd} | {gms} | {mms} | {cms} | {gt} | {mt} | {ct} | {cstat} |\n".format(
                wl=wl,
                n=n,
                nd=nd,
                gms=_fmt_us(g.makespan_us if g else float("inf")),
                mms=_fmt_us(m.makespan_us if (m and m.feasible) else float("inf")),
                cms=_fmt_us(c.makespan_us if (c and c.feasible) else float("inf")),
                gt=_fmt_ms(g.solver_time_ms if g else 0.0),
                mt=_fmt_ms(m.solver_time_ms if m else 0.0),
                ct=_fmt_ms(c.solver_time_ms if c else 0.0),
                cstat=(c.status if c else "n/a"),
            )
        )

    lines.append("\n## Headline\n\n")
    cpsat_vs_mosek_gaps: list[float] = []
    greedy_vs_best_gaps: list[float] = []
    cpsat_matches = 0
    cpsat_compared = 0
    for wl in workload_order:
        bucket = by_wl[wl]
        g = bucket.get("greedy")
        m = bucket.get("mosek_milp")
        c = bucket.get("cpsat_joint")
        candidates = [r.makespan_us for r in (g, m, c) if r and r.feasible and math.isfinite(r.makespan_us)]
        if not candidates:
            continue
        best = min(candidates)
        if g and g.feasible and best > 0:
            greedy_vs_best_gaps.append((g.makespan_us - best) / best * 100.0)
        if m and m.feasible and c and c.feasible and m.makespan_us > 0:
            gap = (c.makespan_us - m.makespan_us) / m.makespan_us * 100.0
            cpsat_vs_mosek_gaps.append(gap)
            cpsat_compared += 1
            if gap <= 2.0:
                cpsat_matches += 1

    if greedy_vs_best_gaps:
        lines.append(
            f"- Greedy gap to best: mean **{sum(greedy_vs_best_gaps) / len(greedy_vs_best_gaps):.2f}%**, "
            f"max **{max(greedy_vs_best_gaps):.2f}%** across {len(greedy_vs_best_gaps)} workloads.\n"
        )
    if cpsat_vs_mosek_gaps:
        lines.append(
            f"- CP-SAT vs MOSEK makespan gap: mean **{sum(cpsat_vs_mosek_gaps) / len(cpsat_vs_mosek_gaps):.2f}%** "
            f"({cpsat_matches}/{cpsat_compared} workloads matched within 2%).\n"
        )
    lines.append("\n## Solver-time scaling (random_dag sweep)\n\n")
    lines.append("| n_ops | greedy ms | CP-SAT ms | MOSEK ms |\n")
    lines.append("|---:|---:|---:|---:|\n")
    for wl in workload_order:
        if not wl.startswith("random_n"):
            continue
        bucket = by_wl[wl]
        g = bucket.get("greedy")
        m = bucket.get("mosek_milp")
        c = bucket.get("cpsat_joint")
        n = (g or m or c).n_ops
        lines.append(
            "| {n} | {gt} | {ct} | {mt} |\n".format(
                n=n,
                gt=_fmt_ms(g.solver_time_ms if g else 0.0),
                ct=_fmt_ms(c.solver_time_ms if c else 0.0),
                mt=(_fmt_ms(m.solver_time_ms) if m and m.status != "skipped" else "skipped"),
            )
        )

    path.write_text("".join(lines))


def maybe_plot(results: list[SolverResult], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    by_wl: dict[str, dict[str, SolverResult]] = defaultdict(dict)
    for r in results:
        by_wl[r.workload][r.solver] = r

    workloads = list(by_wl.keys())
    solvers = ["greedy", "mosek_milp", "cpsat_joint"]
    width = 0.27
    x = list(range(len(workloads)))
    fig, ax = plt.subplots(figsize=(max(8, len(workloads) * 1.2), 4))
    for i, s in enumerate(solvers):
        ys = []
        for wl in workloads:
            r = by_wl[wl].get(s)
            ys.append(r.makespan_us if (r and r.feasible and math.isfinite(r.makespan_us)) else 0.0)
        ax.bar([xi + (i - 1) * width for xi in x], ys, width=width, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("makespan (µs)")
    ax.set_title("Makespan by solver")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "makespan_by_solver.png", dpi=120)
    plt.close(fig)

    sweep = [(by_wl[wl]) for wl in workloads if wl.startswith("random_n")]
    if sweep:
        ns = []
        gt = []
        ct = []
        mt = []
        for bucket in sweep:
            ref = bucket.get("greedy") or bucket.get("cpsat_joint") or bucket.get("mosek_milp")
            ns.append(ref.n_ops)
            gt.append(bucket.get("greedy").solver_time_ms if bucket.get("greedy") else 0.0)
            ct.append(bucket.get("cpsat_joint").solver_time_ms if bucket.get("cpsat_joint") else 0.0)
            mt.append(
                bucket.get("mosek_milp").solver_time_ms
                if (bucket.get("mosek_milp") and bucket.get("mosek_milp").status != "skipped")
                else float("nan")
            )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ns, gt, marker="o", label="greedy")
        ax.plot(ns, ct, marker="s", label="cpsat_joint")
        ax.plot(ns, mt, marker="^", label="mosek_milp")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n_ops")
        ax.set_ylabel("solver wall time (ms)")
        ax.set_title("Solver time vs n_ops (random_dag)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "solver_time_vs_n_ops.png", dpi=120)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Synthetic-only, n_ops capped at 200.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_experiment(quick=args.quick)
    write_jsonl(results, OUT_DIR / "results.jsonl")
    write_summary(results, OUT_DIR / "summary.md", quick=args.quick)
    maybe_plot(results, OUT_DIR)
    print(f"\n[exp1] wrote {OUT_DIR / 'results.jsonl'} and {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
