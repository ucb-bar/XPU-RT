"""Real-time workload builder for the QNN flow.

Translates the paper-figure YAML (workloads, copies, sla_us,
period_us, deadline_us) into a
``xpu_rt.scheduler.workload.Workload`` whose Operations carry the
MOSEK-MILP-recognised deadline / periodic-window attributes.

Two real-time semantics combined per the user's choice:

* **Per-instance deadline.** Each DroNet copy has its own
  ``deadline_us = 40000`` (25 Hz). The MOSEK MILP refuses any
  assignment that violates it (``skip_allowed=False``).
* **Batch bound.** The whole 1×yolov8n + 12×dronet workload's wall
  clock must fit within YOLOv8n's measured single-instance makespan.
  Encoded as ``max_end_t = makespan_bound_us`` on every operation
  (including yolov8n itself).

Latency matrix shape:

    latency_matrix[workload_id][backend] -> mean_us (float | None)

When a (workload, backend) cell is missing or infeasible (``None``),
that cell is added to ``infeasible_combinations`` so the MILP
hard-excludes it via the standard alpha[i,k]=0 path.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from xpu_rt.scheduler.workload import Operation, Workload


@dataclasses.dataclass(frozen=True)
class WorkloadSummary:
    """Bookkeeping that the MCP tool / proof writer surfaces."""

    workload_id: str
    instance_index: int     # 0..copies-1
    operation_id: str       # f"{workload_id}.{instance_index}"
    period_us: float | None
    deadline_us: float | None
    min_start_t: float | None
    max_end_t: float | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _cell_value(cell: Any) -> tuple[float | None, str]:
    """Normalise a latency-matrix cell to (mean_us, provenance).

    Accepts: bare float (provenance "measured" implicit),
    ``None`` (no data → provenance "none"),
    ``{"mean_us": x, "provenance": p, "bound_only": b}``.
    """
    if cell is None:
        return None, "none"
    if isinstance(cell, (int, float)):
        if cell > 0:
            return float(cell), "measured"
        return None, "none"
    if isinstance(cell, Mapping):
        mean = cell.get("mean_us")
        prov = str(cell.get("provenance", "measured"))
        bound_only = bool(cell.get("bound_only", False))
        if mean is None or not isinstance(mean, (int, float)) or mean <= 0:
            return None, "none"
        # An analytical_bound cell is only usable when bound_only=True.
        if prov == "analytical_bound" and not bound_only:
            return float(mean), "analytical_bound_unopted"
        return float(mean), prov
    return None, "none"


def _infeasible_set(
    latencies: Mapping[str, Any],
    machines: list[str],
    *,
    latency_budget_us: float | None = None,
    allow_analytical_bounds: bool = False,
) -> tuple[set[int], list[str]]:
    """Cells the MILP must NOT pick.

    Returns ``(infeasible_indices, rejection_reasons)``. A cell is
    infeasible when:
      - the cell has no measurement, OR
      - the latency exceeds the per-instance budget, OR
      - the cell is analytical_bound and the caller did not opt in.

    ``rejection_reasons`` is a list of human-readable strings, one
    per rejected cell, that the caller can surface in error
    messages / the proof report.
    """
    out: set[int] = set()
    reasons: list[str] = []
    for i, m in enumerate(machines):
        cell = latencies.get(m)
        mean, prov = _cell_value(cell)
        if mean is None:
            out.add(i)
            reasons.append(f"{m}: no measurement")
            continue
        if prov == "analytical_bound_unopted":
            out.add(i)
            reasons.append(
                f"{m}: analytical-bound cell rejected "
                f"(allow_analytical_bounds=False)"
            )
            continue
        if prov == "analytical_bound" and not allow_analytical_bounds:
            out.add(i)
            reasons.append(
                f"{m}: analytical bound (bound_only=True) but caller "
                f"didn't allow analytical bounds"
            )
            continue
        if latency_budget_us is not None and mean > latency_budget_us:
            out.add(i)
            reasons.append(
                f"{m}: latency {mean:.0f}µs exceeds per-instance budget "
                f"{latency_budget_us:.0f}µs"
            )
    return out, reasons


def _processing_times(
    latencies: Mapping[str, Any],
    machines: list[str],
    big: float = 1e9,
) -> list[float]:
    """Replace missing/rejected cells with a sentinel large value.

    The MILP's hard-exclusion constraint zeros these out anyway, but
    feeding a finite value keeps the cost matrix well-conditioned.
    """
    out: list[float] = []
    for m in machines:
        mean, _ = _cell_value(latencies.get(m))
        out.append(float(mean) if mean is not None else big)
    return out


def build_realtime_workload(
    model_yaml: Mapping[str, Any],
    latency_matrix: Mapping[str, Mapping[str, Any]],
    *,
    makespan_bound_us: float | None,
    transfer_us_matrix: np.ndarray | None = None,
    allow_analytical_bounds: bool = False,
) -> tuple[Workload, list[WorkloadSummary]]:
    """Build a Workload + per-instance summary from the YAML + latencies.

    ``model_yaml`` is the parsed YAML dict (``workloads``, ``machines``).
    ``latency_matrix`` is the on-board / cost-table measurements per
    ``(workload, machine)`` — each cell can be either a bare float
    (treated as measured) or a dict ``{"mean_us": x, "provenance":
    "measured" | "analytical_bound", "bound_only": bool}``.

    **Real-only by default.** Cells tagged ``provenance=
    "analytical_bound"`` are *rejected* unless
    ``allow_analytical_bounds=True``. The closed-loop autonomous
    driver leaves this False; only callers that explicitly opt in
    (e.g. a "what-if" planner exploration) may pass True.

    ``makespan_bound_us`` is the global upper bound (typically
    YOLOv8n's measured single-instance makespan).
    """
    machines = list(model_yaml.get("machines") or ["HTA", "GPU", "CPU"])
    workloads_yaml = list(model_yaml.get("workloads") or [])

    ops: list[Operation] = []
    summaries: list[WorkloadSummary] = []
    job_names: list[str] = []

    for spec in workloads_yaml:
        wid = str(spec["id"])
        copies = int(spec.get("copies", 1))
        period_us = spec.get("period_us")          # frequency hint (informational)
        deadline_us = spec.get("deadline_us")      # PER-INSTANCE latency budget
        latencies = latency_matrix.get(wid, {})
        # The deadline_us in the YAML is interpreted as the PER-INSTANCE
        # wall-clock budget: backends on which a single instance would
        # already exceed the budget become infeasible cells. This
        # captures the "DroNet 25 Hz" constraint without staggered
        # release (so all copies remain dispatchable in t∈[0, makespan]).
        proc = _processing_times(latencies, machines)
        infeasible, rejection_reasons = _infeasible_set(
            latencies, machines,
            latency_budget_us=deadline_us,
            allow_analytical_bounds=allow_analytical_bounds,
        )
        # If every cell is rejected the workload is unschedulable —
        # raise loudly rather than letting the MILP try and fail
        # silently.
        if len(infeasible) >= len(machines):
            raise ValueError(
                f"workload {wid!r}: all {len(machines)} backend cells "
                f"rejected (no schedulable choice). Reasons: "
                + " | ".join(rejection_reasons)
            )

        for k in range(copies):
            op_id = f"{wid}.{k}" if copies > 1 else wid
            # All copies start at t=0 (ready immediately); the MILP
            # picks the start within max_end_t = makespan_bound_us.
            min_start_t = 0.0
            # Global batch bound: every op must finish before this.
            max_end_t = makespan_bound_us
            # Hard deadline_us for the MILP is the same as max_end_t —
            # the global makespan bound. The per-instance latency
            # budget is enforced via infeasible_combinations.
            hard_deadline = makespan_bound_us

            op = Operation(
                processing_times=list(proc),
                predecessors=[],
                operation_id=op_id,
                operation_name=op_id,
                job_id=wid,
                min_start_t=min_start_t,
                max_end_t=max_end_t,
                deadline_us=hard_deadline,
                skip_allowed=False,
                infeasible_combinations=infeasible,
            )
            ops.append(op)
            job_names.append(wid)
            summaries.append(WorkloadSummary(
                workload_id=wid,
                instance_index=k,
                operation_id=op_id,
                period_us=float(period_us) if period_us else None,
                deadline_us=float(deadline_us) if deadline_us else None,
                min_start_t=min_start_t,
                max_end_t=max_end_t,
            ))

    if transfer_us_matrix is None:
        n = len(machines)
        # Zero transfer for the coarse whole-network island case
        # (each instance is a single QNN context and doesn't move
        # bytes across backends mid-execution).
        transfer_us_matrix = np.zeros((n, n), dtype=float)

    workload = Workload(
        operations=ops,
        machines=machines,
        transfer_times=transfer_us_matrix,
        job_names=job_names,
    )
    return workload, summaries


def load_workload_yaml(path: Path | str) -> dict[str, Any]:
    """Read the paper-figure YAML."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
