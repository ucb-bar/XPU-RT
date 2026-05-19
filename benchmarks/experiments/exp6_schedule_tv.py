"""Experiment 6: translation validation of scheduler outputs.

Runs CP-SAT (and optionally MOSEK) on a handful of synthetic DAGs,
then translation-validates each solution with both the Z3 path and the
pure-Python path. Two negative-control rows are injected at the end to
prove the harness actually fires.

Usage:
    uv run python scripts/experiments/exp6_schedule_tv.py [--quick]

``--quick`` keeps the workload set small (chain_20, diamond_8,
transformer_L4) and caps MOSEK at ``n_partitions <= 60`` per the
Exp 1 policy. The default mode adds ``random_dag(80)`` and lifts the
MOSEK cap to 200.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiments._synthetic_dag import (  # noqa: E402
    SyntheticDag,
    chain,
    diamond,
    random_dag,
    transformer_block,
)
from xpu_rt.solve.schedule_joint_cpsat import (  # noqa: E402
    JointScheduleSolution,
    solve_schedule_joint,
)
from xpu_rt.solve.schedule_tv import (  # noqa: E402
    ScheduleTVResult,
    translation_validate_schedule,
)

OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp6_schedule_tv"


@dataclass
class TVRow:
    workload: str
    solver: str
    n_partitions: int
    n_deps: int
    n_overlap_pairs: int
    proved: bool
    n_violations: int
    z3_time_ms: float
    python_time_ms: float
    z3_overhead_ratio: float
    makespan_us: float
    note: str = ""


def _solution_to_dict(dag: SyntheticDag, sol: JointScheduleSolution) -> dict[str, Any]:
    return {
        "partition_ids": list(dag.partition_ids),
        "durations_us_by_device": dict(dag.durations_us_by_device),
        "dependencies": {k: list(v) for k, v in dag.dependencies.items()},
        "num_devices": dag.num_devices,
        "start_times": dict(sol.start_times),
        "end_times": dict(sol.end_times),
        "device_assignments": dict(sol.device_assignments),
        "makespan_us": float(sol.makespan_us),
        "transfer_us": [list(row) for row in dag.transfer_us],
    }


def _run_cpsat(dag: SyntheticDag, timeout_ms: int) -> JointScheduleSolution:
    return solve_schedule_joint(
        partition_ids=dag.partition_ids,
        durations_us_by_device=dag.durations_us_by_device,
        dependencies=dag.dependencies,
        num_devices=dag.num_devices,
        transfer_us=dag.transfer_us,
        timeout_ms=timeout_ms,
    )


def _run_mosek(dag: SyntheticDag, time_limit_s: float) -> JointScheduleSolution | None:
    """Route a SyntheticDag through the CVXPY/MOSEK envelope.

    Returns ``None`` if anything in the envelope fails — MOSEK is not
    required for this experiment.
    """
    try:
        import numpy as np

        from xpu_rt.scheduler.scheduler import schedule
        from xpu_rt.scheduler.workload import Operation, Workload
    except Exception:
        return None

    pids = dag.partition_ids
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    ops: list[Any] = []
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
    silent = io.StringIO()
    try:
        with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
            t_arr, alpha, _, _ = schedule(workload, time_limit=time_limit_s)
    except Exception:
        return None
    if t_arr is None or alpha is None:
        return None

    state = getattr(workload, "solver_state", {}) or {}
    status_str = str(state.get("problem_status", "unknown")).lower()
    if status_str not in ("optimal", "optimal_inaccurate"):
        return None

    start_times: dict[str, float] = {}
    end_times: dict[str, float] = {}
    device_assignments: dict[str, int] = {}
    for i, pid in enumerate(pids):
        k = int(alpha[i].argmax())
        start_times[pid] = float(t_arr[i])
        end_times[pid] = float(t_arr[i]) + float(dag.durations_us_by_device[pid][k])
        device_assignments[pid] = k
    makespan = float(state.get("makespan") or max(end_times.values()))
    return JointScheduleSolution(
        start_times=start_times,
        end_times=end_times,
        device_assignments=device_assignments,
        makespan_us=makespan,
        feasible=True,
        status="optimal",
    )


def _validate(
    dag_name: str,
    solver_name: str,
    payload: dict[str, Any],
    *,
    note: str = "",
) -> list[TVRow]:
    rows: list[TVRow] = []
    for use_z3 in (True, False):
        res: ScheduleTVResult = translation_validate_schedule(
            partition_ids=payload["partition_ids"],
            durations_us_by_device=payload["durations_us_by_device"],
            dependencies=payload["dependencies"],
            num_devices=payload["num_devices"],
            start_times=payload["start_times"],
            end_times=payload["end_times"],
            device_assignments=payload["device_assignments"],
            makespan_us=payload["makespan_us"],
            transfer_us=payload["transfer_us"],
            use_z3=use_z3,
        )
        ratio = (
            res.z3_time_ms / res.python_time_ms
            if res.python_time_ms > 0 and use_z3
            else 0.0
        )
        rows.append(
            TVRow(
                workload=dag_name,
                solver=f"{solver_name}+{'z3' if use_z3 else 'python'}",
                n_partitions=len(payload["partition_ids"]),
                n_deps=res.n_deps_checked,
                n_overlap_pairs=res.n_overlap_pairs_checked,
                proved=res.proved,
                n_violations=len(res.violations),
                z3_time_ms=res.z3_time_ms,
                python_time_ms=res.python_time_ms,
                z3_overhead_ratio=ratio,
                makespan_us=payload["makespan_us"],
                note=note,
            )
        )
    return rows


def build_workloads(quick: bool) -> list[tuple[str, Callable[[], SyntheticDag]]]:
    items: list[tuple[str, Callable[[], SyntheticDag]]] = [
        ("chain_20", lambda: chain(20)),
        ("diamond_8", lambda: diamond(8)),
        ("transformer_L4", lambda: transformer_block(layers=4)),
    ]
    if not quick:
        items.append(("random_n80", lambda: random_dag(n_ops=80, seed=0)))
    return items


def run_experiment(quick: bool) -> list[TVRow]:
    cpsat_timeout_ms = 5000 if quick else 30000
    mosek_time_limit_s = 5.0 if quick else 30.0
    mosek_max_n = 60 if quick else 200

    rows: list[TVRow] = []
    workloads = build_workloads(quick)

    # First clean run will be reused for negative controls.
    neg_control_payload: dict[str, Any] | None = None
    neg_control_dag_name = ""

    for name, getter in workloads:
        dag = getter()
        n = len(dag.partition_ids)
        print(f"[exp6] workload={name} n_partitions={n}")

        sol = _run_cpsat(dag, timeout_ms=cpsat_timeout_ms)
        if not sol.feasible:
            print(f"  cpsat infeasible (status={sol.status}); skipping TV")
            continue
        payload = _solution_to_dict(dag, sol)
        v_rows = _validate(name, "cpsat", payload)
        for r in v_rows:
            print(
                f"  cpsat ({r.solver.split('+')[1]:>6})  proved={r.proved}  "
                f"py={r.python_time_ms:.2f}ms  z3={r.z3_time_ms:.2f}ms"
            )
        rows.extend(v_rows)

        if neg_control_payload is None:
            neg_control_payload = copy.deepcopy(payload)
            neg_control_dag_name = name

        if n <= mosek_max_n:
            mosek_sol = _run_mosek(dag, time_limit_s=mosek_time_limit_s)
            if mosek_sol is None:
                print("  mosek    skipped (envelope failure or unavailable)")
                rows.append(
                    TVRow(
                        workload=name,
                        solver="mosek+skipped",
                        n_partitions=n,
                        n_deps=0,
                        n_overlap_pairs=0,
                        proved=False,
                        n_violations=0,
                        z3_time_ms=0.0,
                        python_time_ms=0.0,
                        z3_overhead_ratio=0.0,
                        makespan_us=float("inf"),
                        note="envelope_failed",
                    )
                )
            else:
                mosek_payload = _solution_to_dict(dag, mosek_sol)
                m_rows = _validate(name, "mosek", mosek_payload)
                for r in m_rows:
                    print(
                        f"  mosek ({r.solver.split('+')[1]:>6})  proved={r.proved}  "
                        f"py={r.python_time_ms:.2f}ms  z3={r.z3_time_ms:.2f}ms"
                    )
                rows.extend(m_rows)
        else:
            print(f"  mosek    skipped (n_partitions > {mosek_max_n})")
            rows.append(
                TVRow(
                    workload=name,
                    solver="mosek+skipped",
                    n_partitions=n,
                    n_deps=0,
                    n_overlap_pairs=0,
                    proved=False,
                    n_violations=0,
                    z3_time_ms=0.0,
                    python_time_ms=0.0,
                    z3_overhead_ratio=0.0,
                    makespan_us=float("inf"),
                    note=f"n>{mosek_max_n}",
                )
            )

    # Negative controls.
    if neg_control_payload is not None:
        print(f"[exp6] negative controls on {neg_control_dag_name}")
        nc_rows = _inject_negative_controls(neg_control_dag_name, neg_control_payload)
        for r in nc_rows:
            print(
                f"  neg {r.solver:>20}  proved={r.proved}  violations={r.n_violations}"
            )
        rows.extend(nc_rows)

    return rows


def _inject_negative_controls(dag_name: str, payload: dict[str, Any]) -> list[TVRow]:
    """Build two corrupted variants of a clean schedule and TV them.

    Returns rows for both (with Z3 on) and asserts via ``note`` whether
    the expected violation kind was detected.
    """
    rows: list[TVRow] = []

    # (a) Dependency violation: shift the second partition's start
    # backwards by 50% of its declared duration so it now starts before
    # at least one predecessor's end.
    dep_p = copy.deepcopy(payload)
    pids = dep_p["partition_ids"]
    # Pick the first partition with at least one predecessor.
    target = next((p for p in pids if dep_p["dependencies"].get(p)), None)
    nc_dep_rows: list[TVRow] = []
    if target is not None:
        old_start = dep_p["start_times"][target]
        d = dep_p["device_assignments"][target]
        dur = dep_p["durations_us_by_device"][target][d]
        # Move it 50% earlier; clamp to 0.
        new_start = max(0.0, old_start - dur * 0.5)
        if new_start < old_start:
            dep_p["start_times"][target] = new_start
            dep_p["end_times"][target] = new_start + dur
            dep_p["makespan_us"] = max(dep_p["end_times"].values())
            res = translation_validate_schedule(
                partition_ids=dep_p["partition_ids"],
                durations_us_by_device=dep_p["durations_us_by_device"],
                dependencies=dep_p["dependencies"],
                num_devices=dep_p["num_devices"],
                start_times=dep_p["start_times"],
                end_times=dep_p["end_times"],
                device_assignments=dep_p["device_assignments"],
                makespan_us=dep_p["makespan_us"],
                transfer_us=dep_p["transfer_us"],
                use_z3=True,
            )
            kinds = {v.kind for v in res.violations}
            note = "dep_violated_detected" if "dep_violated" in kinds else "MISS"
            nc_dep_rows.append(
                TVRow(
                    workload=dag_name,
                    solver="neg_dep_violation+z3",
                    n_partitions=len(pids),
                    n_deps=res.n_deps_checked,
                    n_overlap_pairs=res.n_overlap_pairs_checked,
                    proved=res.proved,
                    n_violations=len(res.violations),
                    z3_time_ms=res.z3_time_ms,
                    python_time_ms=res.python_time_ms,
                    z3_overhead_ratio=(res.z3_time_ms / res.python_time_ms) if res.python_time_ms else 0.0,
                    makespan_us=dep_p["makespan_us"],
                    note=note,
                )
            )
    rows.extend(nc_dep_rows)

    # (b) Device overlap: take two partitions on the same device and
    # force them to overlap by moving the later one's start before the
    # earlier one's end.
    ov_p = copy.deepcopy(payload)
    by_dev: dict[int, list[str]] = {}
    for p in ov_p["partition_ids"]:
        d = ov_p["device_assignments"].get(p, -1)
        if d >= 0:
            by_dev.setdefault(d, []).append(p)
    overlap_done = False
    for d, plist in by_dev.items():
        if len(plist) < 2:
            continue
        plist_sorted = sorted(plist, key=lambda x: ov_p["start_times"][x])
        a, b = plist_sorted[0], plist_sorted[1]
        # Set b's start equal to a's start (clear overlap if both
        # durations are positive); preserve b's duration to keep
        # duration-consistency green so we isolate the overlap signal.
        d_b = ov_p["device_assignments"][b]
        dur_b = ov_p["durations_us_by_device"][b][d_b]
        ov_p["start_times"][b] = ov_p["start_times"][a]
        ov_p["end_times"][b] = ov_p["start_times"][a] + dur_b
        ov_p["makespan_us"] = max(ov_p["end_times"].values())
        res = translation_validate_schedule(
            partition_ids=ov_p["partition_ids"],
            durations_us_by_device=ov_p["durations_us_by_device"],
            dependencies=ov_p["dependencies"],
            num_devices=ov_p["num_devices"],
            start_times=ov_p["start_times"],
            end_times=ov_p["end_times"],
            device_assignments=ov_p["device_assignments"],
            makespan_us=ov_p["makespan_us"],
            transfer_us=ov_p["transfer_us"],
            use_z3=True,
        )
        kinds = {v.kind for v in res.violations}
        note = "device_overlap_detected" if "device_overlap" in kinds else "MISS"
        rows.append(
            TVRow(
                workload=dag_name,
                solver="neg_device_overlap+z3",
                n_partitions=len(ov_p["partition_ids"]),
                n_deps=res.n_deps_checked,
                n_overlap_pairs=res.n_overlap_pairs_checked,
                proved=res.proved,
                n_violations=len(res.violations),
                z3_time_ms=res.z3_time_ms,
                python_time_ms=res.python_time_ms,
                z3_overhead_ratio=(res.z3_time_ms / res.python_time_ms) if res.python_time_ms else 0.0,
                makespan_us=ov_p["makespan_us"],
                note=note,
            )
        )
        overlap_done = True
        break
    if not overlap_done:
        rows.append(
            TVRow(
                workload=dag_name,
                solver="neg_device_overlap+z3",
                n_partitions=len(payload["partition_ids"]),
                n_deps=0,
                n_overlap_pairs=0,
                proved=False,
                n_violations=0,
                z3_time_ms=0.0,
                python_time_ms=0.0,
                z3_overhead_ratio=0.0,
                makespan_us=payload["makespan_us"],
                note="no_same_device_pair_available",
            )
        )

    return rows


def write_jsonl(rows: list[TVRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")


def _fmt_us(v: float) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:,.1f}"


def write_summary(rows: list[TVRow], path: Path, quick: bool) -> None:
    lines: list[str] = []
    lines.append("# Experiment 6 — translation validation of scheduler outputs\n\n")
    lines.append(f"Mode: {'quick' if quick else 'full'}\n\n")

    lines.append("## Per-workload results\n\n")
    lines.append(
        "| workload | solver+path | n_parts | n_deps | n_overlap | proved | n_viol | py ms | z3 ms | z3/py | makespan µs | note |\n"
    )
    lines.append("|---|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        lines.append(
            f"| {r.workload} | {r.solver} | {r.n_partitions} | {r.n_deps} | {r.n_overlap_pairs} | "
            f"{'YES' if r.proved else 'no'} | {r.n_violations} | {r.python_time_ms:.2f} | "
            f"{r.z3_time_ms:.2f} | {r.z3_overhead_ratio:.1f} | {_fmt_us(r.makespan_us)} | {r.note} |\n"
        )

    # Z3 vs Python timing aggregate.
    z3_rows = [r for r in rows if r.solver.endswith("+z3") and r.python_time_ms > 0 and r.z3_time_ms > 0]
    if z3_rows:
        mean_ratio = sum(r.z3_overhead_ratio for r in z3_rows) / len(z3_rows)
        max_ratio = max(r.z3_overhead_ratio for r in z3_rows)
        min_ratio = min(r.z3_overhead_ratio for r in z3_rows)
        lines.append("\n## Z3 vs Python timing\n\n")
        lines.append(f"- Rows with both paths timed: **{len(z3_rows)}**\n")
        lines.append(f"- Mean z3/python ratio: **{mean_ratio:.1f}x**\n")
        lines.append(f"- Range: {min_ratio:.1f}x to {max_ratio:.1f}x\n")

    # Negative controls.
    neg_rows = [r for r in rows if r.solver.startswith("neg_")]
    if neg_rows:
        lines.append("\n## Negative controls\n\n")
        lines.append("| injection | proved | n_viol | detection note |\n")
        lines.append("|---|:---:|---:|---|\n")
        for r in neg_rows:
            lines.append(f"| {r.solver} | {'YES' if r.proved else 'no'} | {r.n_violations} | {r.note} |\n")
        dep_ok = any("dep_violated_detected" in r.note for r in neg_rows)
        ov_ok = any("device_overlap_detected" in r.note for r in neg_rows)
        lines.append(
            f"\n- Dep-violation control fired: **{'YES' if dep_ok else 'NO'}**\n"
            f"- Device-overlap control fired: **{'YES' if ov_ok else 'NO'}**\n"
        )

    # Solver verdict.
    cp_proved = [r for r in rows if r.solver.startswith("cpsat") and not r.solver.startswith("neg_")]
    cp_pass = all(r.proved for r in cp_proved) if cp_proved else False
    mosek_real = [
        r for r in rows
        if r.solver.startswith("mosek+")
        and not r.solver.startswith("neg_")
        and r.solver not in ("mosek+skipped",)
    ]
    mosek_exercised = len(mosek_real) > 0
    mosek_pass = all(r.proved for r in mosek_real) if mosek_real else None
    lines.append("\n## Verdict\n\n")
    lines.append(f"- CP-SAT joint solutions translation-validated: **{'all clean' if cp_pass else 'failures observed'}**\n")
    if mosek_exercised:
        lines.append(f"- MOSEK solutions translation-validated: **{'all clean' if mosek_pass else 'failures observed'}**\n")
    else:
        lines.append("- MOSEK was not exercised (envelope skipped per policy / failed); CP-SAT alone covers the TV signal.\n")

    lines.append(
        "\n## What Z3 buys here\n\n"
        "Honestly, on concrete numeric output Z3 adds **no** semantic value over the\n"
        "Python checks — every obligation reduces to integer comparisons that pass or\n"
        "fail by direct evaluation. The Z3 path costs roughly an order of magnitude\n"
        "more wall time (see the ratio table above) because each obligation spins up a\n"
        "fresh solver. The reasons to keep it wired anyway are:\n\n"
        "1. **Uniform counterexample shape.** When TV is invoked by the obligation\n"
        "   harness across many check kinds, a single `ScheduleTVViolation` schema\n"
        "   beats ad-hoc per-check error structs.\n"
        "2. **Forward compatibility.** Once durations or transfer costs become\n"
        "   symbolic (e.g. parametric on workload size, or conditional on a runtime\n"
        "   predicate), the Python path stops working and Z3 takes over verbatim.\n"
        "3. **Drop-in obligation slot.** The existing harness already speaks\n"
        "   `SolverRequest` / `SolverResponse`; the schedule_tv encoder slots into\n"
        "   that pipeline without a new dispatch layer.\n\n"
        "For now we report both timings so the calibration cost is visible.\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="small workloads only, fast run")
    args = p.parse_args(argv)
    t0 = time.perf_counter()
    rows = run_experiment(quick=args.quick)
    write_jsonl(rows, OUT_DIR / "results.jsonl")
    write_summary(rows, OUT_DIR / "summary.md", args.quick)
    print(f"[exp6] {len(rows)} rows -> {OUT_DIR} ({(time.perf_counter() - t0):.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
