"""Coarse-grained workload placement for the QNN-native flow.

When the user drives the iterative loop with full-network DLCs (the
canonical Qualcomm path), each workload is a single island that must
be placed on exactly one backend. The placement problem is small
(≤ a few workloads × {CPU, GPU, DSP/HTA}); we brute-force every
assignment and pick the one minimising the *concurrent* makespan,
defined as ``max_b sum_w latency[w, b] · 1[w→b]``.

This module is a deliberate simplification of the MOSEK-driven
multi-cluster scheduler in ``xpu_rt.scheduler``. It exists because:

1. The user prefers the QNN-native path (no merlin / iree-compile).
2. With ≤ 12 workloads × 3 backends the exhaustive search is
   <100k assignments — orders of magnitude faster than MOSEK setup,
   and lets us avoid the mosek dependency in the dry/board path.
3. The decision the agent then makes (split / coarsen / keep) maps
   to whole-network re-placement at this granularity, not per-op
   restructuring.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PlacementResult:
    """The result of an exhaustive coarse-grained placement search."""

    assignment: dict[str, str]                 # workload_id -> backend
    backend_load_us: dict[str, float]          # backend -> sum of latencies (µs)
    makespan_us: float                         # max over backend_load_us
    n_assignments_evaluated: int

    def to_dict(self) -> dict:
        return {
            "assignment": dict(self.assignment),
            "backend_load_us": dict(self.backend_load_us),
            "makespan_us": self.makespan_us,
            "n_assignments_evaluated": self.n_assignments_evaluated,
        }


def coarse_place(
    workloads: Sequence[str],
    backends: Sequence[str],
    latency_us: Mapping[str, Mapping[str, float | None]],
    *,
    forbidden: Mapping[str, Sequence[str]] | None = None,
) -> PlacementResult:
    """Exhaustively pick the (workload → backend) map that minimises makespan.

    ``latency_us[w][b]`` must be present (a positive float) for every
    legal (workload, backend) pair the agent is willing to consider.
    ``None`` / missing / non-positive entries are treated as infeasible
    cells. ``forbidden[w]`` is an optional per-workload backend block-list.

    Tie-breaks by total sum-of-loads (favour the more balanced
    schedule among assignments with the same makespan).
    """
    workloads = list(workloads)
    backends = list(backends)
    blocked = {w: set(forbidden.get(w, ())) if forbidden else set() for w in workloads}

    legal: dict[str, list[str]] = {}
    for w in workloads:
        cells = []
        for b in backends:
            if b in blocked[w]:
                continue
            v = latency_us.get(w, {}).get(b)
            if isinstance(v, (int, float)) and v > 0:
                cells.append(b)
        if not cells:
            raise ValueError(
                f"workload {w!r} has no feasible backend "
                f"(measurements were {latency_us.get(w)})"
            )
        legal[w] = cells

    best: PlacementResult | None = None
    evaluated = 0
    for combo in itertools.product(*[legal[w] for w in workloads]):
        evaluated += 1
        assn = dict(zip(workloads, combo))
        load: dict[str, float] = {b: 0.0 for b in backends}
        for w, b in assn.items():
            load[b] += float(latency_us[w][b])
        makespan = max(load.values())
        if (best is None
                or makespan < best.makespan_us
                or (makespan == best.makespan_us
                    and sum(load.values()) < sum(best.backend_load_us.values()))):
            best = PlacementResult(
                assignment=assn, backend_load_us=load,
                makespan_us=makespan, n_assignments_evaluated=evaluated,
            )
    assert best is not None
    return PlacementResult(
        assignment=best.assignment,
        backend_load_us=best.backend_load_us,
        makespan_us=best.makespan_us,
        n_assignments_evaluated=evaluated,
    )


def schedule_to_dict(
    placement: PlacementResult,
    *,
    latency_us: Mapping[str, Mapping[str, float | None]],
) -> dict:
    """Render the placement as a schedule.json-shaped dict.

    Each placed workload becomes a single ``op`` running from 0 to its
    on-backend latency. This lets the existing markdown / Gantt
    renderers consume the output without modification.
    """
    ops = []
    dispatches: dict[str, dict] = {}
    for w, b in placement.assignment.items():
        lat = float(latency_us[w][b])
        # Pack each backend's workloads back-to-back so the
        # backend_load matches what we measured (sum of latencies).
        # Order doesn't matter for max-makespan; pick alphabetical.
        op = {
            "name": w,
            "workload": w,
            "machine": b,
            "start_us": 0.0,    # filled below
            "finish_us": 0.0,
            "predicted_us": lat,
        }
        ops.append(op)
        dispatches[w] = op
    # Lay out per-backend.
    per_backend_cursor: dict[str, float] = {}
    for op in sorted(ops, key=lambda o: o["name"]):
        c = per_backend_cursor.get(op["machine"], 0.0)
        op["start_us"] = c
        op["finish_us"] = c + op["predicted_us"]
        per_backend_cursor[op["machine"]] = op["finish_us"]
    return {
        "schema_version": "qnn_native_schedule_v1",
        "makespan_us": placement.makespan_us,
        "machines": list(placement.backend_load_us.keys()),
        "assignment": dict(placement.assignment),
        "backend_load_us": dict(placement.backend_load_us),
        "ops": ops,
        "dispatches": dispatches,
    }
