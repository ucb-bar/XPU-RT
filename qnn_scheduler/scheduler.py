"""Heterogeneous-island scheduler — bridges into XPU-RT's `scheduler.py`.

The thin layer that:
  * takes a DAG of IslandVariantGroups (from merlin's recognizers)
  * picks one IslandCandidate per group and assigns it to a backend
  * minimises makespan honoring island-DAG vs hardware-DAG parallelism
  * uses on-board-measured execute / transfer costs (no heuristics)

Strategy:
  1. Enumerate variant choices (Cartesian product across groups). For
     YOLOv8 with ~10-20 fusion-toggle groups, this is ≤10⁶ combinations
     at most; greedy enumeration is feasible. For larger workloads,
     a MILP encoding with one indicator per group goes through
     `xpu-rt/scheduler.py:schedule()` instead.
  2. For each configuration, compute per-edge transfer cost via
     TransferModel and bake it into the consumer's effective duration.
  3. Run XPU-RT's transfer-aware greedy on the resulting flat workload.
  4. Pick the configuration with the minimum makespan.

For the demo we exercise a single configuration; the variant-enumeration
loop is structurally trivial to add and lives behind the same API.
"""

from __future__ import annotations

import dataclasses
import itertools
import pathlib
import sys
from typing import Iterable

import numpy as np

# Allow importing xpu-rt's scheduling primitives.
_HERE = pathlib.Path(__file__).resolve()
_XPU_RT_PY = _HERE.parent.parent / "xpu-rt"
if str(_XPU_RT_PY) not in sys.path:
    sys.path.insert(0, str(_XPU_RT_PY))

from .cost_table import CostTable, OpKey, BackendKey, canonical_op_key  # noqa
from .island_dag import IslandCandidate, IslandVariantGroup, TensorSpec  # noqa
from .transfer_model import EdgeSpec, TransferModel  # noqa


@dataclasses.dataclass
class ScheduleResult:
    selected_candidate_id: dict[str, str]   # group_id -> chosen candidate_id
    machine: dict[str, str]                  # candidate_id -> backend
    start_us: dict[str, float]
    finish_us: dict[str, float]
    makespan_us: float


def _parse_op_key(s: str) -> OpKey:
    """Parse "<op_kind>@<shape_signature>@<dtype>" with the shape
    signature potentially containing "@" only via canonical_op_key — but
    in practice it never does, so split on the *last* two delimiters."""
    parts = s.split("@")
    if len(parts) < 3:
        raise ValueError(f"malformed op_key: {s!r}")
    op_kind = parts[0]
    dtype = parts[-1]
    shape_signature = "@".join(parts[1:-1])
    return OpKey(op_kind=op_kind, shape_signature=shape_signature, dtype=dtype)


def _bake_processing_times(
    candidates: list[IslandCandidate],
    cost_table: CostTable,
    machines: list[str],
) -> dict[str, list[float]]:
    """For each candidate, fill a per-machine row. Unsupported machines
    get a sentinel large value so the picker rejects them but the MILP
    stays bounded."""
    big = 1e9
    out: dict[str, list[float]] = {}
    for c in candidates:
        row = []
        for m in machines:
            if c.backend != m:
                row.append(big)
                continue
            try:
                t = cost_table.execute_us(
                    _parse_op_key(c.op_key),
                    BackendKey(backend=c.backend, fused=c.fused_with_next),
                    allow_extrapolation=True,
                )
            except KeyError:
                t = big
            row.append(t)
        out[c.candidate_id] = row
    return out


def schedule_groups(
    groups: list[IslandVariantGroup],
    cost_table: CostTable,
    *,
    machines: tuple[str, ...] = ("HTA", "GPU", "CPU"),
    pick_strategy: str = "first",   # "first" | "enumerate"
    iterations_amortized: float = 100.0,  # for setup-cost amortisation
) -> ScheduleResult:
    """Return the best (candidate, machine) selection + timeline.

    pick_strategy="first" picks the first candidate of each group (used
    by tests). "enumerate" tries every combination and returns the one
    with the minimum makespan. With the variant-group structure, even
    fully exhaustive enumeration on YOLOv8 is small (~10³-10⁶).
    """
    machine_idx = {m: i for i, m in enumerate(machines)}
    transfer = TransferModel(cost_table)

    # Build a flat candidate list per chosen configuration.
    if pick_strategy == "first":
        configs: Iterable[tuple[IslandCandidate, ...]] = (
            tuple(g.alternatives[0] for g in groups),
        )
    elif pick_strategy == "enumerate":
        configs = itertools.product(*(g.alternatives for g in groups))
    else:
        raise ValueError(f"unknown pick_strategy: {pick_strategy}")

    best: ScheduleResult | None = None

    for cfg in configs:
        # group_id -> chosen candidate
        chosen: dict[str, IslandCandidate] = {
            g.group_id: c for g, c in zip(groups, cfg)
        }
        # Topological order over groups, materialised as candidates.
        order = _topo_order(groups)

        machine_free = {m: 0.0 for m in machines}
        machine_seen = {m: False for m in machines}
        finish: dict[str, tuple[float, str, str]] = {}  # group_id -> (finish_us, machine, candidate_id)
        start: dict[str, float] = {}
        for gid in order:
            grp = next(g for g in groups if g.group_id == gid)
            cand = chosen[gid]
            mi = machine_idx[cand.backend]

            ready_t = 0.0
            for up in grp.upstream_group_ids:
                up_finish, up_machine, _ = finish[up]
                up_grp = next(g for g in groups if g.group_id == up)
                up_cand = chosen[up]
                # Edge cost: producer's chosen output tensor to consumer's
                # chosen input tensor. We pair them positionally for now.
                if up_cand.outputs and cand.inputs:
                    edge = EdgeSpec(
                        producer_id=up_cand.candidate_id,
                        consumer_id=cand.candidate_id,
                        producer_out=up_cand.outputs[0],
                        consumer_in=cand.inputs[0],
                        producer_machine=up_machine,
                        consumer_machine=cand.backend,
                    )
                    bridge = transfer.cost_us(edge)
                else:
                    bridge = 0.0
                ready_t = max(ready_t, up_finish + bridge)

            s = max(ready_t, machine_free[cand.backend])
            execute_us = cand.notes_processing_time(cost_table) \
                if hasattr(cand, "notes_processing_time") else None
            if execute_us is None:
                # Fall back to direct lookup if helper absent.
                row = _bake_processing_times([cand], cost_table, list(machines))
                execute_us = row[cand.candidate_id][mi]
            # Setup cost amortised over expected inferences (one-time cost).
            if not machine_seen[cand.backend]:
                machine_seen[cand.backend] = True
                execute_us += cost_table.init_us(cand.backend) / iterations_amortized
            f = s + execute_us
            start[cand.candidate_id] = s
            finish[gid] = (f, cand.backend, cand.candidate_id)
            machine_free[cand.backend] = f

        makespan = max(f for f, _, _ in finish.values())
        if best is None or makespan < best.makespan_us:
            best = ScheduleResult(
                selected_candidate_id={gid: c.candidate_id for gid, c in chosen.items()},
                machine={c.candidate_id: finish[gid][1]
                         for gid, c in chosen.items()},
                start_us=start,
                finish_us={c.candidate_id: finish[gid][0]
                           for gid, c in chosen.items()},
                makespan_us=makespan,
            )

    assert best is not None
    return best


def _topo_order(groups: list[IslandVariantGroup]) -> list[str]:
    by_id = {g.group_id: g for g in groups}
    pending = {g.group_id: list(g.upstream_group_ids) for g in groups}
    order: list[str] = []
    while pending:
        ready = sorted(n for n, d in pending.items() if not d)
        if not ready:
            raise RuntimeError("cycle in island DAG")
        for n in ready:
            order.append(n)
            del pending[n]
        for rem in pending.values():
            for n in ready:
                if n in rem:
                    rem.remove(n)
    return order
