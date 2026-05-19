"""Greedy first-fit memory planner.

Pure deterministic baseline that consumes the same
:class:`~xpu_rt.solve.memory_planner.MemoryPlanInput` and returns a
:class:`~xpu_rt.solve.memory_planner.MemoryPlanSolved` as the MILP
planner. Exists so ``scripts/experiments/exp2_memory_planning_ab.py``
can quantify what the MILP buys over a textbook heuristic on shared
workloads.

Algorithm:

1. Tier assignment by spill cost descending (ties: size descending);
   each buffer goes to the cheapest-weight allowed tier that still
   has live capacity.
2. First-fit-decreasing offset assignment within each tier; offsets
   are alignment-rounded.
3. ``aliases_with`` is unset by default (aliasing is the MILP's
   advantage; we do not bake it into the baseline).

The objective scoring matches the MILP entry point so that
``objective_value`` is directly comparable across the two planners.
"""

from __future__ import annotations

import time

import structlog

from xpu_rt.solve.memory_planner import (
    MEMORY_PLAN_SCHEMA_VERSION,
    BufferAllocation,
    BufferSpec,
    MemoryPlanInput,
    MemoryPlanSolved,
    TierCapacity,
    _align_up,
    _build_formulation,
    _lifetimes_overlap,
    _validate_input,
)
from xpu_rt.solve.solver_types import compute_formulation_hash

__all__ = ["plan_memory_greedy"]


_LOG = structlog.get_logger(__name__)


def plan_memory_greedy(
    problem: MemoryPlanInput,
    *,
    activate_aliases: bool = False,
) -> MemoryPlanSolved:
    """Plan memory via greedy first-fit-decreasing.

    Args:
        problem: Same ``MemoryPlanInput`` the MILP planner consumes.
        activate_aliases: When ``True``, declared
            :class:`~xpu_rt.solve.memory_planner.AliasCandidate` pairs
            with disjoint lifetimes are collapsed to a shared offset.
            Default is ``False`` so the baseline does not steal the
            MILP's aliasing wins.

    Returns:
        ``MemoryPlanSolved`` with ``solver_backend="greedy_first_fit"``.
        Status is ``"optimal"`` for a successful pack (the heuristic
        commits to the layout it finds), ``"infeasible"`` when no
        allowed tier can absorb a buffer's contribution to peak.
    """

    err = _validate_input(problem)
    if err:
        return _infeasible(problem, reason=err)

    t0 = time.perf_counter()
    tier_by_id: dict[str, TierCapacity] = {t.tier_id: t for t in problem.tier_capacities}

    tier_choice = _assign_tiers(problem, tier_by_id)
    if tier_choice is None:
        return _infeasible(problem, reason="no allowed tier fits buffer at peak")

    offsets, aliases = _pack_offsets(
        problem,
        tier_choice,
        activate_aliases=activate_aliases,
    )

    allocations = tuple(
        BufferAllocation(
            buffer_id=b.buffer_id,
            tier=tier_choice[b.buffer_id],
            offset_bytes=offsets[b.buffer_id],
            aliases_with=aliases.get(b.buffer_id),
        )
        for b in problem.buffers
    )

    tier_peak = _compute_tier_peak(problem.buffers, allocations)
    for tier_id, peak in tier_peak.items():
        cap = tier_by_id[tier_id].capacity_bytes
        if peak > cap:
            return _infeasible(problem, reason=f"tier {tier_id} peak {peak} > capacity {cap}")

    objective = _objective_value(problem, allocations, tier_peak, tier_by_id)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _LOG.debug(
        "greedy_memory_plan",
        buffers=len(problem.buffers),
        tiers=len(problem.tier_capacities),
        objective=objective,
        elapsed_ms=elapsed_ms,
    )

    return MemoryPlanSolved(
        schema_version=MEMORY_PLAN_SCHEMA_VERSION,
        solver_backend="greedy_first_fit",
        status="optimal",
        buffers=allocations,
        tier_peak_usage=tier_peak,
        objective_value=objective,
        formulation_hash=compute_formulation_hash(_build_formulation(problem)),
    )


def _assign_tiers(
    problem: MemoryPlanInput,
    tier_by_id: dict[str, TierCapacity],
) -> dict[str, str] | None:
    # Track running peak per tier so a tier we have already filled to
    # capacity is rejected for the next buffer that overlaps in time.
    fixed = problem.fixed_assignments
    ordered = sorted(
        problem.buffers,
        key=lambda b: (-b.spill_cost, -b.size_bytes, b.buffer_id),
    )
    chosen: dict[str, str] = {}
    placed_by_tier: dict[str, list[BufferSpec]] = {}

    for buf in ordered:
        if buf.buffer_id in fixed:
            tier_id = fixed[buf.buffer_id]
            if not _tier_has_room(buf, tier_by_id[tier_id], placed_by_tier.get(tier_id, [])):
                return None
            chosen[buf.buffer_id] = tier_id
            placed_by_tier.setdefault(tier_id, []).append(buf)
            continue

        candidates = sorted(
            (tier_by_id[t] for t in buf.allowed_tiers if t in tier_by_id),
            key=lambda t: (t.weight, t.tier_id),
        )
        chosen_tier: str | None = None
        for tier in candidates:
            if _tier_has_room(buf, tier, placed_by_tier.get(tier.tier_id, [])):
                chosen_tier = tier.tier_id
                break
        if chosen_tier is None:
            return None
        chosen[buf.buffer_id] = chosen_tier
        placed_by_tier.setdefault(chosen_tier, []).append(buf)

    return chosen


def _tier_has_room(buf: BufferSpec, tier: TierCapacity, placed: list[BufferSpec]) -> bool:
    # Sweep events to find peak live-bytes within this tier if `buf`
    # is admitted. Rejecting on peak > capacity is a true sufficient
    # condition; rejecting on a looser sum-of-overlaps would refuse
    # workloads the offset packer can actually fit.
    events: list[tuple[int, int, int]] = []
    for other in (*placed, buf):
        events.append((other.lifetime_start, 0, other.size_bytes))
        events.append((other.lifetime_end + 1, 1, -other.size_bytes))
    events.sort()
    live = 0
    peak = 0
    for _, _, delta in events:
        live += delta
        if live > peak:
            peak = live
    return peak <= tier.capacity_bytes


def _pack_offsets(
    problem: MemoryPlanInput,
    tier_choice: dict[str, str],
    *,
    activate_aliases: bool,
) -> tuple[dict[str, int], dict[str, str | None]]:
    by_tier: dict[str, list[BufferSpec]] = {}
    for buf in problem.buffers:
        by_tier.setdefault(tier_choice[buf.buffer_id], []).append(buf)

    alias_partner: dict[str, str] = {}
    if activate_aliases:
        for ac in problem.alias_candidates:
            if tier_choice.get(ac.buffer_a) != tier_choice.get(ac.buffer_b):
                continue
            alias_partner.setdefault(ac.buffer_a, ac.buffer_b)
            alias_partner.setdefault(ac.buffer_b, ac.buffer_a)

    offsets: dict[str, int] = {}
    aliases: dict[str, str | None] = {}

    for tier_id, bufs in by_tier.items():
        bufs_sorted = sorted(
            bufs,
            key=lambda b: (-b.size_bytes, -b.spill_cost, b.buffer_id),
        )
        placed: list[tuple[int, int, BufferSpec]] = []
        for buf in bufs_sorted:
            if activate_aliases and buf.buffer_id in alias_partner:
                partner = alias_partner[buf.buffer_id]
                if partner in offsets and not _lifetimes_overlap(
                    buf,
                    next(b for b in problem.buffers if b.buffer_id == partner),
                ):
                    offsets[buf.buffer_id] = offsets[partner]
                    aliases[buf.buffer_id] = partner
                    continue
            offset = _first_fit_offset(buf, placed)
            offsets[buf.buffer_id] = offset
            aliases[buf.buffer_id] = None
            placed.append((offset, offset + buf.size_bytes, buf))

    return offsets, aliases


def _first_fit_offset(buf: BufferSpec, placed: list[tuple[int, int, BufferSpec]]) -> int:
    candidate = _align_up(0, buf.alignment)
    overlapping = [(lo, hi) for (lo, hi, other) in placed if _lifetimes_overlap(buf, other)]
    overlapping.sort()
    for lo, hi in overlapping:
        if candidate + buf.size_bytes <= lo:
            return candidate
        candidate = _align_up(max(candidate, hi), buf.alignment)
    return candidate


def _compute_tier_peak(
    buffers: tuple[BufferSpec, ...],
    allocations: tuple[BufferAllocation, ...],
) -> dict[str, int]:
    by_id = {a.buffer_id: a for a in allocations}
    out: dict[str, int] = {}
    for buf in buffers:
        alloc = by_id[buf.buffer_id]
        ceiling = alloc.offset_bytes + buf.size_bytes
        out[alloc.tier] = max(out.get(alloc.tier, 0), ceiling)
    return out


def _objective_value(
    problem: MemoryPlanInput,
    allocations: tuple[BufferAllocation, ...],
    tier_peak: dict[str, int],
    tier_by_id: dict[str, TierCapacity],
) -> float:
    by_id = {b.buffer_id: b for b in problem.buffers}
    spill_term = 0.0
    for alloc in allocations:
        buf = by_id[alloc.buffer_id]
        weight = tier_by_id[alloc.tier].weight
        spill_term += buf.spill_cost * weight
    peak_term = problem.objective_lambda * float(sum(tier_peak.values()))
    return spill_term + peak_term


def _infeasible(problem: MemoryPlanInput, *, reason: str) -> MemoryPlanSolved:
    _LOG.warning("greedy_memory_plan_infeasible", reason=reason)
    return MemoryPlanSolved(
        schema_version=MEMORY_PLAN_SCHEMA_VERSION,
        solver_backend="greedy_first_fit",
        status="infeasible",
        buffers=(),
        tier_peak_usage={},
        objective_value=None,
        formulation_hash=compute_formulation_hash(_build_formulation(problem)),
    )
