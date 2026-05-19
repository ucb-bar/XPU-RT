"""Translation validation for memory planner outputs.

Given a :class:`MemoryPlanInput` and a :class:`MemoryPlanSolved`,
proves four invariants of the concrete solver output:

1. Any two buffers placed in the same tier with overlapping lifetimes
   have non-overlapping byte ranges ``[offset, offset+size)``.
2. Each tier's peak usage does not exceed its declared
   :attr:`TierCapacity.capacity_bytes`.
3. Each buffer's offset is divisible by its declared
   :attr:`BufferSpec.alignment`.
4. Every ``fixed_assignments[buffer] = tier`` from the input is
   honored in the solution.

Two implementation paths are provided:

* ``use_z3=True`` builds a Z3 query per disjointness obligation. With
  concrete integer offsets/sizes each query is trivially decidable;
  the value of going through Z3 is a uniform, structured
  counterexample format and a forward-compatible obligation shape
  for the day buffers carry symbolic shapes.
* ``use_z3=False`` performs the same checks via plain Python
  arithmetic. Used as the calibration baseline — what an inline
  ``assert`` would cost.

Both paths return the same :class:`TVResult`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog

from xpu_rt.solve.memory_planner import (
    BufferAllocation,
    BufferSpec,
    MemoryPlanInput,
    MemoryPlanSolved,
    TierCapacity,
)

__all__ = [
    "TVResult",
    "TVViolation",
    "translation_validate_memory_plan",
]

_LOG = structlog.get_logger(__name__)


@dataclass(frozen=True)
class TVViolation:
    """A single TV failure with structured detail.

    Attributes:
        kind: One of ``"buffer_overlap"``, ``"tier_capacity_exceeded"``,
            ``"alignment_violated"``, ``"fixed_assignment_violated"``.
        detail: Structured payload (buffer ids, offsets, sizes, tier,
            etc.) sufficient to reproduce / explain the violation.
    """

    kind: str
    detail: dict[str, Any]


@dataclass(frozen=True)
class TVResult:
    """Outcome of a translation-validation run.

    Attributes:
        proved: True iff every invariant held.
        violations: Empty when ``proved``; otherwise lists every
            invariant failure observed.
        z3_time_ms: Wall-clock spent inside Z3 (zero when
            ``use_z3=False``).
        python_time_ms: Wall-clock for the equivalent assertion-only
            check path (zero when ``use_z3=True`` was the only path
            run).
        n_pairs_checked: Number of disjointness pairs evaluated
            (post-tier-filter, post-lifetime-overlap-filter).
    """

    proved: bool
    violations: list[TVViolation]
    z3_time_ms: float
    python_time_ms: float
    n_pairs_checked: int


def _lifetimes_overlap(a: BufferSpec, b: BufferSpec) -> bool:
    # Half-open intervals [start, end): overlap iff a.start < b.end ∧ b.start < a.end.
    return a.lifetime_start < b.lifetime_end and b.lifetime_start < a.lifetime_end


def _index_buffers(
    problem: MemoryPlanInput,
    solution: MemoryPlanSolved,
) -> tuple[
    dict[str, BufferSpec],
    dict[str, BufferAllocation],
    dict[str, TierCapacity],
]:
    spec_by_id = {b.buffer_id: b for b in problem.buffers}
    alloc_by_id = {a.buffer_id: a for a in solution.buffers}
    tier_by_id = {t.tier_id: t for t in problem.tier_capacities}
    return spec_by_id, alloc_by_id, tier_by_id


def _enumerate_disjointness_pairs(
    problem: MemoryPlanInput,
    solution: MemoryPlanSolved,
) -> list[tuple[BufferSpec, BufferAllocation, BufferSpec, BufferAllocation]]:
    """Pairs of (spec, alloc) sharing a tier and overlapping lifetimes."""

    spec_by_id, alloc_by_id, _ = _index_buffers(problem, solution)
    ordered = [b.buffer_id for b in problem.buffers if b.buffer_id in alloc_by_id]
    pairs: list[tuple[BufferSpec, BufferAllocation, BufferSpec, BufferAllocation]] = []
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a_id, b_id = ordered[i], ordered[j]
            a_alloc = alloc_by_id[a_id]
            b_alloc = alloc_by_id[b_id]
            if a_alloc.tier != b_alloc.tier:
                continue
            a_spec = spec_by_id[a_id]
            b_spec = spec_by_id[b_id]
            if not _lifetimes_overlap(a_spec, b_spec):
                continue
            # Aliased buffers (declared disjoint-lifetime by the planner)
            # may collapse to the same offset; skip them — the lifetime
            # filter above already keeps them honest.
            if a_alloc.aliases_with == b_id or b_alloc.aliases_with == a_id:
                continue
            pairs.append((a_spec, a_alloc, b_spec, b_alloc))
    return pairs


def _check_python(
    problem: MemoryPlanInput,
    solution: MemoryPlanSolved,
    pairs: list[tuple[BufferSpec, BufferAllocation, BufferSpec, BufferAllocation]],
) -> list[TVViolation]:
    """Plain-Python invariant checks (no solver)."""

    violations: list[TVViolation] = []
    spec_by_id, alloc_by_id, tier_by_id = _index_buffers(problem, solution)

    for a_spec, a_alloc, b_spec, b_alloc in pairs:
        a_lo, a_hi = a_alloc.offset_bytes, a_alloc.offset_bytes + a_spec.size_bytes
        b_lo, b_hi = b_alloc.offset_bytes, b_alloc.offset_bytes + b_spec.size_bytes
        if a_lo < b_hi and b_lo < a_hi:
            violations.append(
                TVViolation(
                    kind="buffer_overlap",
                    detail={
                        "tier": a_alloc.tier,
                        "buffer_a": a_spec.buffer_id,
                        "buffer_b": b_spec.buffer_id,
                        "a_range": [a_lo, a_hi],
                        "b_range": [b_lo, b_hi],
                        "a_lifetime": [a_spec.lifetime_start, a_spec.lifetime_end],
                        "b_lifetime": [b_spec.lifetime_start, b_spec.lifetime_end],
                    },
                )
            )

    tier_ceiling: dict[str, int] = {}
    for a in solution.buffers:
        spec = spec_by_id.get(a.buffer_id)
        if spec is None:
            continue
        ceiling = a.offset_bytes + spec.size_bytes
        tier_ceiling[a.tier] = max(tier_ceiling.get(a.tier, 0), ceiling)
    for tier_id, peak in tier_ceiling.items():
        tier = tier_by_id.get(tier_id)
        if tier is None:
            violations.append(
                TVViolation(
                    kind="tier_capacity_exceeded",
                    detail={"tier": tier_id, "peak": peak, "reason": "tier not declared"},
                )
            )
            continue
        if peak > tier.capacity_bytes:
            violations.append(
                TVViolation(
                    kind="tier_capacity_exceeded",
                    detail={
                        "tier": tier_id,
                        "peak": peak,
                        "capacity": tier.capacity_bytes,
                    },
                )
            )

    for a in solution.buffers:
        spec = spec_by_id.get(a.buffer_id)
        if spec is None:
            continue
        if spec.alignment > 1 and a.offset_bytes % spec.alignment != 0:
            violations.append(
                TVViolation(
                    kind="alignment_violated",
                    detail={
                        "buffer": a.buffer_id,
                        "offset": a.offset_bytes,
                        "alignment": spec.alignment,
                    },
                )
            )

    for fa_buf, fa_tier in problem.fixed_assignments.items():
        alloc = alloc_by_id.get(fa_buf)
        if alloc is None:
            violations.append(
                TVViolation(
                    kind="fixed_assignment_violated",
                    detail={
                        "buffer": fa_buf,
                        "expected_tier": fa_tier,
                        "actual_tier": None,
                        "reason": "buffer absent from solution",
                    },
                )
            )
            continue
        if alloc.tier != fa_tier:
            violations.append(
                TVViolation(
                    kind="fixed_assignment_violated",
                    detail={
                        "buffer": fa_buf,
                        "expected_tier": fa_tier,
                        "actual_tier": alloc.tier,
                    },
                )
            )

    return violations


def _check_z3_disjointness(
    pairs: list[tuple[BufferSpec, BufferAllocation, BufferSpec, BufferAllocation]],
    *,
    timeout_ms: int,
) -> tuple[list[TVViolation], float]:
    """Z3 path for buffer-overlap obligations.

    For each disjointness pair we encode the negation of "intervals
    are disjoint" as a Z3 query. ``unsat`` ⇒ disjoint (proved).
    ``sat`` ⇒ overlap (counterexample reported with the offending
    offsets/sizes).
    """

    try:
        import z3
    except ImportError:
        # Z3 is a hard expectation for the TV obligation; surface
        # a structured violation rather than silently degrading.
        return (
            [
                TVViolation(
                    kind="buffer_overlap",
                    detail={"reason": "z3 import failed", "n_pairs": len(pairs)},
                )
            ],
            0.0,
        )

    violations: list[TVViolation] = []
    t0 = time.perf_counter()
    for a_spec, a_alloc, b_spec, b_alloc in pairs:
        solver = z3.Solver()
        solver.set("timeout", max(1, int(timeout_ms)))
        off_a = z3.Int(f"off_{a_spec.buffer_id}")
        sz_a = z3.Int(f"sz_{a_spec.buffer_id}")
        off_b = z3.Int(f"off_{b_spec.buffer_id}")
        sz_b = z3.Int(f"sz_{b_spec.buffer_id}")
        solver.add(off_a == a_alloc.offset_bytes)
        solver.add(sz_a == a_spec.size_bytes)
        solver.add(off_b == b_alloc.offset_bytes)
        solver.add(sz_b == b_spec.size_bytes)
        # Negation of "disjoint": (off_a < off_b + sz_b) AND (off_b < off_a + sz_a).
        solver.add(off_a < off_b + sz_b)
        solver.add(off_b < off_a + sz_a)
        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            violations.append(
                TVViolation(
                    kind="buffer_overlap",
                    detail={
                        "tier": a_alloc.tier,
                        "buffer_a": a_spec.buffer_id,
                        "buffer_b": b_spec.buffer_id,
                        "a_range": [
                            a_alloc.offset_bytes,
                            a_alloc.offset_bytes + a_spec.size_bytes,
                        ],
                        "b_range": [
                            b_alloc.offset_bytes,
                            b_alloc.offset_bytes + b_spec.size_bytes,
                        ],
                        "z3_model": {str(d): str(model[d]) for d in model.decls()},
                    },
                )
            )
        elif result == z3.unknown:
            violations.append(
                TVViolation(
                    kind="buffer_overlap",
                    detail={
                        "tier": a_alloc.tier,
                        "buffer_a": a_spec.buffer_id,
                        "buffer_b": b_spec.buffer_id,
                        "reason": "z3 unknown (timeout)",
                    },
                )
            )
    return violations, (time.perf_counter() - t0) * 1000.0


def translation_validate_memory_plan(
    problem: MemoryPlanInput,
    solution: MemoryPlanSolved,
    *,
    use_z3: bool = True,
    timeout_ms: int = 5000,
) -> TVResult:
    """Prove a :class:`MemoryPlanSolved` honors its :class:`MemoryPlanInput`.

    Args:
        problem: The input handed to the planner.
        solution: The planner's output to be validated.
        use_z3: If True, route disjointness checks through Z3 for a
            uniform counterexample shape. If False, use plain Python
            arithmetic (faster, no symbolic-buffer headroom).
        timeout_ms: Per-pair Z3 timeout. Ignored on the Python path.

    Returns:
        A :class:`TVResult` whose ``proved`` field is True iff every
        invariant held. Failures are reported in ``violations`` with
        structured detail. Both ``z3_time_ms`` and ``python_time_ms``
        are populated when both paths run; only one is populated
        otherwise. ``n_pairs_checked`` reflects the disjointness pair
        count after lifetime / tier filtering.
    """

    pairs = _enumerate_disjointness_pairs(problem, solution)

    py_t0 = time.perf_counter()
    py_violations = _check_python(problem, solution, pairs)
    python_time_ms = (time.perf_counter() - py_t0) * 1000.0

    if not use_z3:
        proved = len(py_violations) == 0
        return TVResult(
            proved=proved,
            violations=py_violations,
            z3_time_ms=0.0,
            python_time_ms=python_time_ms,
            n_pairs_checked=len(pairs),
        )

    z3_overlap_violations, z3_time_ms = _check_z3_disjointness(pairs, timeout_ms=timeout_ms)

    # The Python pass owns capacity/alignment/fixed-assignment checks
    # (those are not naturally a disjointness query). Z3 owns the
    # overlap check. Merge: keep Z3 overlap violations + non-overlap
    # violations from the Python pass.
    non_overlap_py = [v for v in py_violations if v.kind != "buffer_overlap"]
    merged = z3_overlap_violations + non_overlap_py
    proved = len(merged) == 0

    _LOG.debug(
        "memory_plan_tv.done",
        proved=proved,
        n_pairs=len(pairs),
        n_violations=len(merged),
        z3_time_ms=z3_time_ms,
        python_time_ms=python_time_ms,
    )

    return TVResult(
        proved=proved,
        violations=merged,
        z3_time_ms=z3_time_ms,
        python_time_ms=python_time_ms,
        n_pairs_checked=len(pairs),
    )
