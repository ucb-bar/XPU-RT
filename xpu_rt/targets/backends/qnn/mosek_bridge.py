"""MOSEK MILP entry point for QNN workloads with deadlines.

Wraps :func:`xpu_rt.scheduler.scheduler.schedule` for the QNN agentic
loop. The bridge:

1. Calls
   :func:`xpu_rt.solve.backends.mosek_backend.ensure_mosek_license_env`
   so the license at ``<repo>/xpu-rt/mosek.lic`` is auto-discovered.
2. Solves the workload via MOSEK MILP (deadlines + periodic windows
   + infeasibility hard-exclusion are already implemented in
   ``scheduler.schedule``).
3. Returns a ``schedule.json``-shaped dict compatible with the
   existing markdown renderers (`render_gantt_markdown`,
   `render_deltas_markdown`) and the proof writer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xpu_rt.scheduler.scheduler import schedule
from xpu_rt.scheduler.workload import Workload
from xpu_rt.solve.backends.mosek_backend import ensure_mosek_license_env


def solve_qnn_mosek(
    workload: Workload,
    *,
    time_limit: float = 60.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Solve a QNN realtime workload via MOSEK MILP.

    Returns a dict shaped like ``placement.schedule_to_dict``'s output
    plus a ``milp_status`` block carrying the solver state. On
    infeasibility, returns ``{"feasible": False, "status": ..., …}``
    so the agent can surface the failure reason directly to the user.
    """
    license_path = ensure_mosek_license_env()
    # ``restrict_makespan_to_nonperiodic=False`` is critical: every op
    # in the QNN workload carries ``max_end_t`` (the global yolov8n
    # makespan bound), so the default behaviour would exclude all ops
    # from the C_max objective and leave the MILP degenerate.
    try:
        t, alpha, _fused, _fmap = schedule(
            workload,
            fusion_threshold=None,
            verbose=verbose,
            time_limit=time_limit,
            restrict_makespan_to_nonperiodic=False,
        )
    except Exception as exc:  # noqa: BLE001 — propagate as typed result
        return {
            "schema_version": "qnn_mosek_schedule_v1",
            "feasible": False,
            "status": f"solver_error:{type(exc).__name__}",
            "error": str(exc)[:400],
            "license_path": license_path,
            "milp_status": {"problem_status": "solver_error"},
            "machines": list(workload.machines),
        }

    state = getattr(workload, "solver_state", {}) or {}
    status = state.get("problem_status", "unknown")
    feasible = (t is not None and alpha is not None
                and status in ("optimal", "optimal_inaccurate"))

    out: dict[str, Any] = {
        "schema_version": "qnn_mosek_schedule_v1",
        "feasible": bool(feasible),
        "status": status,
        "license_path": license_path,
        "milp_status": {
            "problem_status": status,
            "makespan_us": state.get("makespan"),
            "objective_value": state.get("objective_value"),
            "num_operations": state.get("num_operations"),
            "num_combinations": state.get("num_combinations"),
            "time_limit_s": time_limit,
        },
        "machines": list(workload.machines),
    }
    if not feasible:
        # Carry through skip indicators for the caller's report.
        out["skipped_op_indices"] = list(
            getattr(workload, "skipped_op_indices", [])
        )
        return out

    ops_list: list[dict[str, Any]] = []
    dispatches: dict[str, dict[str, Any]] = {}
    machine_combinations = workload.get_machine_combinations()
    # Each combination is a list of machine names; for the coarse
    # whole-network case combinations are singletons → flatten.
    combo_label = [",".join(c) for c in machine_combinations]
    n_ops = len(workload.operations)
    makespan = 0.0
    for i in range(n_ops):
        op = workload.operations[i]
        row = alpha[i]
        # Pick the machine with the highest assignment weight.
        k = int(np.argmax(row))
        machine = combo_label[k]
        start = float(t[i]) if t[i] is not None else 0.0
        proc = float(op.processing_times[k])
        finish = start + proc
        makespan = max(makespan, finish)
        # Per-instance deadline check.
        dline = op.deadline_us
        deadline_met = (dline is None) or (finish <= float(dline) + 1e-6)
        entry = {
            "name": op.operation_id or f"op_{i}",
            "workload": op.job_id or "unknown",
            "machine": machine,
            "start_us": start,
            "finish_us": finish,
            "predicted_us": proc,
            "deadline_us": dline,
            "deadline_met": deadline_met,
            "min_start_t": op.min_start_t,
            "max_end_t": op.max_end_t,
        }
        ops_list.append(entry)
        dispatches[entry["name"]] = entry

    out["makespan_us"] = makespan
    out["ops"] = sorted(ops_list, key=lambda o: (o["machine"], o["start_us"]))
    out["dispatches"] = dispatches
    out["deadlines_met_count"] = sum(1 for o in ops_list if o["deadline_met"])
    out["deadlines_total"] = sum(1 for o in ops_list if o["deadline_us"] is not None)
    return out
