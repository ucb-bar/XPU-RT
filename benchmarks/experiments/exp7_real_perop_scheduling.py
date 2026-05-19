"""Experiment 7: per-op multi-backend scheduling on real QNN costs.

Runs greedy, CP-SAT joint placement+ordering, and (when feasible) the
CVXPY/MOSEK MILP on the full per-op DAGs for ``yolov8n`` (273 ops) and
``dronet`` (30 ops), built from the profiled cost matrix at
``xpu-rt/data/profiled/qnn_cost_matrix.json``.

Predicted makespans are compared against the single-backend E2E
baselines in ``xpu-rt/data/profiled/qnn_e2e/measurements.json``. The
chain DAG used here under-estimates available parallelism — see
``xpu_rt.scheduler.qnn_real_workload`` for details.

Usage:
    uv run python scripts/experiments/exp7_real_perop_scheduling.py [--quick]

``--quick`` runs only ``k_lookahead=1``; the default sweeps
``k_lookahead in {1, 4}``.
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
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "xpu-rt" / "python"))

from xpu_rt.scheduler.qnn_real_workload import (  # noqa: E402
    BACKENDS,
    QnnDag,
    load_cost_matrix,
    make_chain_dag,
)
from xpu_rt.solve.schedule_joint_cpsat import solve_schedule_joint  # noqa: E402

COST_MATRIX_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_cost_matrix.json"
E2E_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_e2e" / "measurements.json"
OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp7_real_perop"

# n=60 is the MOSEK ceiling established by Experiment 1.
MOSEK_MAX_N = 60


@dataclass
class Exp7Result:
    """One solver run for one (workload, k_lookahead) cell."""

    workload: str
    k_lookahead: int
    solver: str
    n_ops: int
    num_devices: int
    makespan_us: float
    solver_time_ms: float
    feasible: bool
    status: str
    note: str = ""


def _topo_order(pids: list[str], deps: dict[str, list[str]]) -> list[str]:
    """Topological ordering. Stable for our chain inputs."""
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


def greedy_schedule(dag: QnnDag) -> Exp7Result:
    """Earliest-finish-time list scheduler over a QnnDag.

    Skips backends marked ``None`` (unsupported) for that op.
    """
    t0 = time.perf_counter()
    topo = _topo_order(dag.partition_ids, dag.dependencies)
    device_avail = [0.0] * dag.num_devices
    end_times: dict[str, float] = {}
    chosen_dev: dict[str, int] = {}
    for pid in topo:
        durations = dag.durations_us_by_device[pid]
        best_end = math.inf
        best_dev = -1
        for d in range(dag.num_devices):
            dur = durations[d]
            if dur is None or (
                isinstance(dur, float) and (math.isinf(dur) or math.isnan(dur))
            ):
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
        if best_dev < 0:
            return Exp7Result(
                workload=dag.name,
                k_lookahead=-1,  # filled by caller
                solver="greedy",
                n_ops=len(dag.partition_ids),
                num_devices=dag.num_devices,
                makespan_us=float("inf"),
                solver_time_ms=(time.perf_counter() - t0) * 1000,
                feasible=False,
                status="infeasible",
                note=f"no feasible device for {pid}",
            )
        end_times[pid] = best_end
        chosen_dev[pid] = best_dev
        device_avail[best_dev] = best_end
    makespan = max(end_times.values()) if end_times else 0.0
    return Exp7Result(
        workload=dag.name,
        k_lookahead=-1,
        solver="greedy",
        n_ops=len(dag.partition_ids),
        num_devices=dag.num_devices,
        makespan_us=makespan,
        solver_time_ms=(time.perf_counter() - t0) * 1000,
        feasible=True,
        status="optimal_local",
    )


def cpsat_joint(dag: QnnDag, timeout_ms: int) -> Exp7Result:
    """Joint placement+ordering CP-SAT solve."""
    sol = solve_schedule_joint(
        partition_ids=dag.partition_ids,
        durations_us_by_device=dag.durations_us_by_device,
        dependencies=dag.dependencies,
        num_devices=dag.num_devices,
        transfer_us=dag.transfer_us,
        timeout_ms=timeout_ms,
    )
    return Exp7Result(
        workload=dag.name,
        k_lookahead=-1,
        solver="cpsat_joint",
        n_ops=len(dag.partition_ids),
        num_devices=dag.num_devices,
        makespan_us=sol.makespan_us,
        solver_time_ms=sol.solve_time_ms,
        feasible=sol.feasible,
        status=sol.status,
    )


def mosek_milp(dag: QnnDag, time_limit_s: float) -> Exp7Result:
    """Route a QnnDag through the CVXPY/MOSEK MILP."""
    import numpy as np
    from xpu_rt.scheduler.scheduler import schedule
    from xpu_rt.scheduler.workload import Operation, Workload

    pids = dag.partition_ids
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    ops: list[Operation] = []
    for i, pid in enumerate(pids):
        per_dev = dag.durations_us_by_device[pid]
        # MOSEK formulation needs a concrete cost per machine; mark
        # unsupported cells via infeasible_combinations and use a
        # large-but-finite placeholder cost so the MILP can write a
        # sensible big-M without infinities. The infeasible-combo
        # constraint forces alpha[i,k]=0 anyway, so the placeholder
        # is never realised in the objective.
        proc: list[float] = []
        infeas: list[int] = []
        finite_costs = [float(c) for c in per_dev if c is not None]
        placeholder = (max(finite_costs) if finite_costs else 1.0) * 10.0
        for k, c in enumerate(per_dev):
            if c is None:
                proc.append(placeholder)
                infeas.append(k)
            else:
                proc.append(float(c))
        ops.append(
            Operation(
                processing_times=proc,
                operation_id=i,
                operation_name=pid,
                job_id=0,
                infeasible_combinations=infeas or None,
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
        with contextlib.redirect_stdout(silent_buf), contextlib.redirect_stderr(
            silent_buf
        ):
            t_arr, alpha, _, _ = schedule(workload, time_limit=time_limit_s)
    except Exception as exc:  # noqa: BLE001
        return Exp7Result(
            workload=dag.name,
            k_lookahead=-1,
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
    feasible = (
        t_arr is not None
        and alpha is not None
        and status_str.lower() in ("optimal", "optimal_inaccurate")
    )
    if not feasible:
        return Exp7Result(
            workload=dag.name,
            k_lookahead=-1,
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
        end_times: list[float] = []
        for i, pid in enumerate(pids):
            k = int(alpha[i].argmax())
            c = dag.durations_us_by_device[pid][k]
            end_times.append(float(t_arr[i]) + (float(c) if c is not None else 0.0))
        makespan = max(end_times)
    return Exp7Result(
        workload=dag.name,
        k_lookahead=-1,
        solver="mosek_milp",
        n_ops=len(pids),
        num_devices=dag.num_devices,
        makespan_us=makespan,
        solver_time_ms=elapsed_ms,
        feasible=True,
        status=status_str,
    )


def load_e2e_baselines() -> dict[str, dict[str, float]]:
    """Load the per-network E2E wall times (µs) keyed by workload, backend."""
    with E2E_PATH.open() as fh:
        raw = json.load(fh)
    out: dict[str, dict[str, float]] = {}
    for wl, by_backend in raw.get("matrix", {}).items():
        out[wl] = {}
        for backend, row in by_backend.items():
            if row.get("ok") and row.get("mean_us") is not None:
                out[wl][backend] = float(row["mean_us"])
    return out


def run_experiment(
    *, quick: bool, cpsat_timeout_ms: int, mosek_time_limit_s: float
) -> list[Exp7Result]:
    """Run A1 across (workload, k_lookahead, solver)."""
    matrix = load_cost_matrix(COST_MATRIX_PATH)
    k_values = [1] if quick else [1, 4]
    results: list[Exp7Result] = []

    for wl in ("yolov8n", "dronet"):
        for k in k_values:
            dag = make_chain_dag(wl, matrix, k_lookahead=k)
            n = len(dag.partition_ids)
            print(f"[exp7] workload={wl} k_lookahead={k} n_ops={n}")

            g = greedy_schedule(dag)
            g.k_lookahead = k
            print(
                f"  greedy      makespan={g.makespan_us:>12.1f} µs "
                f"time={g.solver_time_ms:>7.2f} ms status={g.status}"
            )
            results.append(g)

            c = cpsat_joint(dag, timeout_ms=cpsat_timeout_ms)
            c.k_lookahead = k
            print(
                f"  cpsat_joint makespan={c.makespan_us:>12.1f} µs "
                f"time={c.solver_time_ms:>7.2f} ms status={c.status}"
            )
            results.append(c)

            if n <= MOSEK_MAX_N:
                m = mosek_milp(dag, time_limit_s=mosek_time_limit_s)
                m.k_lookahead = k
                print(
                    f"  mosek_milp  makespan={m.makespan_us:>12.1f} µs "
                    f"time={m.solver_time_ms:>7.2f} ms status={m.status}"
                )
                results.append(m)
            else:
                skipped = Exp7Result(
                    workload=dag.name,
                    k_lookahead=k,
                    solver="mosek_milp",
                    n_ops=n,
                    num_devices=dag.num_devices,
                    makespan_us=float("inf"),
                    solver_time_ms=0.0,
                    feasible=False,
                    status="skipped",
                    note=f"n_ops={n} > {MOSEK_MAX_N}",
                )
                print(f"  mosek_milp  skipped (n_ops={n} > {MOSEK_MAX_N})")
                results.append(skipped)
    return results


def write_jsonl(results: list[Exp7Result], path: Path) -> None:
    """Persist results as JSONL for downstream tooling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")


def _fmt_us(v: float) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:,.1f}"


def _fmt_ms(v: float) -> str:
    return f"{v:,.2f}"


def write_summary(
    results: list[Exp7Result],
    path: Path,
    *,
    quick: bool,
    e2e: dict[str, dict[str, float]],
) -> None:
    """Render the one-page Markdown summary for A1."""
    by_cell: dict[tuple[str, int], dict[str, Exp7Result]] = defaultdict(dict)
    for r in results:
        # Strip the "_k{N}" suffix added by make_chain_dag so the
        # rendered workload label matches what the cost matrix uses.
        raw = r.workload
        bare = raw.rsplit("_k", 1)[0] if "_k" in raw else raw
        by_cell[(bare, r.k_lookahead)][r.solver] = r

    lines: list[str] = []
    lines.append("# Experiment 7 — per-op multi-backend scheduling on real QNN costs\n\n")
    lines.append(f"Mode: {'quick' if quick else 'full'}\n\n")
    lines.append(
        "Backends (in device-index order): "
        + ", ".join(BACKENDS)
        + ". Cross-backend transfer penalty: 100 µs (constant).\n\n"
    )

    lines.append("## E2E single-backend baselines (µs)\n\n")
    lines.append("| workload | CPU | GPU | DSP | best |\n|---|---:|---:|---:|---:|\n")
    for wl in ("yolov8n", "dronet"):
        row = e2e.get(wl, {})
        best = min(row.values()) if row else float("inf")
        lines.append(
            "| {wl} | {cpu} | {gpu} | {dsp} | {best} |\n".format(
                wl=wl,
                cpu=_fmt_us(row.get("CPU", float("inf"))),
                gpu=_fmt_us(row.get("GPU", float("inf"))),
                dsp=_fmt_us(row.get("DSP", float("inf"))),
                best=_fmt_us(best),
            )
        )

    lines.append("\n## Per-workload × solver results\n\n")
    lines.append(
        "| workload | k | n_ops | solver | predicted makespan (µs) | solver time (ms) | status |\n"
    )
    lines.append("|---|---:|---:|---|---:|---:|---|\n")
    for (wl, k), bucket in sorted(by_cell.items()):
        for solver in ("greedy", "mosek_milp", "cpsat_joint"):
            r = bucket.get(solver)
            if r is None:
                continue
            lines.append(
                "| {wl} | {k} | {n} | {s} | {ms} | {t} | {st} |\n".format(
                    wl=wl,
                    k=k,
                    n=r.n_ops,
                    s=solver,
                    ms=_fmt_us(r.makespan_us if r.feasible else float("inf")),
                    t=_fmt_ms(r.solver_time_ms),
                    st=r.status + (f" ({r.note})" if r.note else ""),
                )
            )

    lines.append("\n## Headline — did anything beat the single-backend baseline?\n\n")
    for wl in ("yolov8n", "dronet"):
        row = e2e.get(wl, {})
        if not row:
            continue
        best_e2e = min(row.values())
        best_e2e_backend = min(row.items(), key=lambda kv: kv[1])[0]
        best_predicted = float("inf")
        best_predicted_solver = "n/a"
        best_predicted_k = -1
        for (cell_wl, k), bucket in by_cell.items():
            if cell_wl != wl:
                continue
            for solver, r in bucket.items():
                if not r.feasible or not math.isfinite(r.makespan_us):
                    continue
                if r.makespan_us < best_predicted:
                    best_predicted = r.makespan_us
                    best_predicted_solver = solver
                    best_predicted_k = k
        if math.isinf(best_predicted):
            lines.append(f"- **{wl}**: no feasible schedule found.\n")
            continue
        delta = best_predicted - best_e2e
        pct = (delta / best_e2e) * 100.0
        verdict = "BEATS" if delta < 0 else "DOES NOT BEAT"
        lines.append(
            f"- **{wl}**: best E2E baseline is **{best_e2e_backend} = "
            f"{best_e2e:,.0f} µs**. Best predicted makespan is "
            f"**{best_predicted:,.0f} µs** ({best_predicted_solver}, "
            f"k={best_predicted_k}). Schedule **{verdict}** the baseline "
            f"by {pct:+.1f}% (Δ = {delta:+,.0f} µs).\n"
        )

    lines.append(
        "\n### Caveats\n\n"
        "- **Chain-DAG bias.** The cost matrix carries no dependency edges, "
        "so we serialise ops in their QNN execution order. Real YOLOv8 has "
        "residual fan-outs the chain DAG hides — the *upper bound* on "
        "achievable parallelism is therefore higher than what these solvers "
        "are allowed to find. ``k_lookahead=4`` only partially relaxes this.\n"
        "- **Transfer cost is a flat 100 µs** off-diagonal. The QRB5165 "
        "``qrb5165_costs.json`` carries fits in µs-per-element for "
        "quant/dequant transitions, but we lack per-op tensor shapes here. "
        "100 µs is conservative and dominates only at very high K (per-op).\n"
        "- **Per-op sums are far below the E2E baselines** "
        "(e.g., yolov8n CPU sum = 269 ms vs CPU E2E 325 ms): some of the "
        "E2E wall time is QNN runtime overhead the cost matrix does not "
        "include. The schedulers should still rank solutions correctly.\n"
    )

    path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="k_lookahead=1 only; ~30s budget end-to-end.",
    )
    parser.add_argument(
        "--cpsat-timeout-ms",
        type=int,
        default=None,
        help="CP-SAT per-call timeout (default 10000 quick / 30000 full).",
    )
    parser.add_argument(
        "--mosek-time-limit-s",
        type=float,
        default=None,
        help="MOSEK per-call time limit in seconds (default 10 quick / 30 full).",
    )
    args = parser.parse_args()

    cpsat_timeout_ms = args.cpsat_timeout_ms or (10000 if args.quick else 30000)
    mosek_time_limit_s = args.mosek_time_limit_s or (10.0 if args.quick else 30.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_experiment(
        quick=args.quick,
        cpsat_timeout_ms=cpsat_timeout_ms,
        mosek_time_limit_s=mosek_time_limit_s,
    )
    e2e = load_e2e_baselines()
    write_jsonl(results, OUT_DIR / "results.jsonl")
    write_summary(results, OUT_DIR / "summary.md", quick=args.quick, e2e=e2e)
    print(f"\n[exp7] wrote {OUT_DIR / 'results.jsonl'} and {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
