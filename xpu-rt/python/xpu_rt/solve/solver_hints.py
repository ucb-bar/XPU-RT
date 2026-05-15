"""Solver search-space pruning hints.

A solver hint is **best-effort guidance** that narrows the
combinatorial space MOSEK / HiGHS explore — not a correctness
primitive. The MILP still runs and still validates; if the hint
is wrong on a particular buffer, MOSEK simply drops that hint and
falls back to its own branch-and-bound. The premise:

> An LLM (or a deterministic heuristic) doesn't need to be 100%
> correct. Even 65–75% of the assignments fixed pre-solve
> exponentially shrinks the search space.

Five hint kinds, ordered by aggressiveness:

1. **tier_hint** (cheap, low risk): for each buffer, the tier we
   *think* it belongs on. Consumed as a MOSEK MIP starting point
   (``Task.putxxslice``). MOSEK is free to deviate.
2. **offset_warm_start** (cheap, low risk): a candidate byte
   offset per buffer. Same MIP starting-point consumption.
3. **fixed_assignment** (aggressive): the LLM is highly confident.
   Variable fixed via ``boundkey.fx``. If infeasible, MOSEK
   reports it — no silent fallback.
4. **stage_partition**: groups of buffers that can be planned
   independently. The planner solves one MILP per stage instead
   of one giant MILP. For TinyLlama-scale (22 layers), this is
   the biggest win.
5. **symmetry_class**: buffers within a class are interchangeable;
   we add ordering constraints so MOSEK doesn't enumerate
   symmetric solutions.

The hint can come from:

* A **deterministic rule-based heuristic** (always available,
  ``rule_based_memory_hints``).
* An **LLM** via the agent-decision request path (opt-in;
  ``llm_memory_hints`` reads back JSON the LLM produced).
* A **manual operator override** (test-friendly).

The hint provider is consulted upstream; the planner consumes the
typed ``MemoryHints`` dataclass and translates it to MOSEK API
calls. Pruning is invisible to the rest of the system except for
solve-time (and the ``hints_applied`` block in the solver
response).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TierHint",
    "OffsetHint",
    "StageGroup",
    "SymmetryClass",
    "MemoryHints",
    "rule_based_memory_hints",
    "merge_hints",
]


@dataclass(frozen=True)
class TierHint:
    """A best-effort tier assignment for one buffer.

    ``confidence`` ≥ 0.9 elevates the hint to a fixed variable in
    the MILP. Lower confidence drops it to a warm-start only.
    """

    buffer_id: str
    tier_id: str
    confidence: float = 0.7
    reason: str = ""


@dataclass(frozen=True)
class OffsetHint:
    buffer_id: str
    offset_bytes: int
    reason: str = ""


@dataclass(frozen=True)
class StageGroup:
    """A group of buffers that can be planned independently.

    Stage decomposition: instead of solving one MILP over all
    buffers, solve one per group. Works when the groups have
    disjoint lifetimes (the standard case for layered networks).
    """

    stage_id: str
    buffer_ids: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True)
class SymmetryClass:
    """Buffers in this class are interchangeable. The planner adds
    ``offset[b_0] <= offset[b_1] <= ...`` to break symmetry."""

    class_id: str
    buffer_ids: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True)
class MemoryHints:
    """Bundle of hints passed to ``memory_planner.plan_memory``."""

    tier_hints: tuple[TierHint, ...] = ()
    offset_warm_start: tuple[OffsetHint, ...] = ()
    stage_partition: tuple[StageGroup, ...] = ()
    symmetry_classes: tuple[SymmetryClass, ...] = ()
    source: str = "unspecified"  # "rule_based" | "llm" | "manual"
    confidence_summary: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "confidence_summary": dict(self.confidence_summary),
            "tier_hints": [
                {
                    "buffer_id": t.buffer_id, "tier_id": t.tier_id,
                    "confidence": t.confidence, "reason": t.reason,
                }
                for t in self.tier_hints
            ],
            "offset_warm_start": [
                {"buffer_id": o.buffer_id, "offset_bytes": o.offset_bytes, "reason": o.reason}
                for o in self.offset_warm_start
            ],
            "stage_partition": [
                {"stage_id": s.stage_id, "buffer_ids": list(s.buffer_ids), "reason": s.reason}
                for s in self.stage_partition
            ],
            "symmetry_classes": [
                {"class_id": c.class_id, "buffer_ids": list(c.buffer_ids), "reason": c.reason}
                for c in self.symmetry_classes
            ],
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> MemoryHints:
        return cls(
            tier_hints=tuple(
                TierHint(
                    buffer_id=t["buffer_id"], tier_id=t["tier_id"],
                    confidence=float(t.get("confidence", 0.7)),
                    reason=str(t.get("reason", "")),
                )
                for t in body.get("tier_hints", [])
            ),
            offset_warm_start=tuple(
                OffsetHint(
                    buffer_id=o["buffer_id"],
                    offset_bytes=int(o["offset_bytes"]),
                    reason=str(o.get("reason", "")),
                )
                for o in body.get("offset_warm_start", [])
            ),
            stage_partition=tuple(
                StageGroup(
                    stage_id=s["stage_id"],
                    buffer_ids=tuple(s["buffer_ids"]),
                    reason=str(s.get("reason", "")),
                )
                for s in body.get("stage_partition", [])
            ),
            symmetry_classes=tuple(
                SymmetryClass(
                    class_id=c["class_id"],
                    buffer_ids=tuple(c["buffer_ids"]),
                    reason=str(c.get("reason", "")),
                )
                for c in body.get("symmetry_classes", [])
            ),
            source=body.get("source", "unspecified"),
            confidence_summary=dict(body.get("confidence_summary", {})),
        )

    @property
    def is_empty(self) -> bool:
        return not (
            self.tier_hints
            or self.offset_warm_start
            or self.stage_partition
            or self.symmetry_classes
        )


# ---------------------------------------------------------------------------
# Rule-based generator
# ---------------------------------------------------------------------------


def rule_based_memory_hints(plan_input: Any) -> MemoryHints:
    """Deterministic heuristic that mimics what an LLM would say
    for a layered-network memory plan.

    Rules (all defensible, all deterministic):

    1. **Large, long-lived buffers → host tier** (confidence 0.85):
       size > median × 4 AND lifetime > 50% of total tick range.
       Mirrors how a model-aware engineer plans: weights and KV
       caches stay in DRAM; activations cycle through scratchpad.

    2. **Small, short-lived buffers → scratchpad tier**
       (confidence 0.75): size < median AND lifetime < 25% of
       total tick range. Per-layer activations qualify.

    3. **Stage partition by layer prefix**: buffer_ids of the form
       ``layer<N>.<x>`` group into stage ``layer<N>``. If no
       layer naming convention, fall back to one stage.

    4. **Symmetry within a stage**: within each stage, buffers
       sharing the same size are an equivalence class.

    The heuristic only emits a hint when the rule applies — empty
    bundles are honestly empty (no fake hints).
    """

    from compgen.solve.memory_planner import (  # noqa: PLC0415
        BufferSpec,
        MemoryPlanInput,
        TierCapacity,
    )

    if not isinstance(plan_input, MemoryPlanInput):
        return MemoryHints(source="rule_based")

    buffers: tuple[BufferSpec, ...] = plan_input.buffers
    tiers: tuple[TierCapacity, ...] = plan_input.tier_capacities
    if not buffers or not tiers:
        return MemoryHints(source="rule_based")

    sizes = sorted(b.size_bytes for b in buffers)
    median_size = sizes[len(sizes) // 2]
    max_tick = max(b.lifetime_end for b in buffers)
    min_tick = min(b.lifetime_start for b in buffers)
    span = max(max_tick - min_tick, 1)

    # Find "host"-like and "scratchpad"-like tiers by name; fall
    # back to capacity-ordering if the names aren't standard.
    tiers_by_name = {t.tier_id: t for t in tiers}
    host_tier = next(
        (t.tier_id for t in tiers if "host" in t.tier_id.lower() or "dram" in t.tier_id.lower()),
        None,
    )
    fast_tier = next(
        (t.tier_id for t in tiers if "scratch" in t.tier_id.lower() or "sram" in t.tier_id.lower()),
        None,
    )
    if host_tier is None or fast_tier is None:
        # Two-tier fallback by capacity: smallest = fast, largest = host.
        sorted_tiers = sorted(tiers, key=lambda t: t.capacity_bytes)
        if len(sorted_tiers) >= 2:
            fast_tier = fast_tier or sorted_tiers[0].tier_id
            host_tier = host_tier or sorted_tiers[-1].tier_id

    tier_hints: list[TierHint] = []
    for b in buffers:
        size_factor = b.size_bytes / max(median_size, 1)
        life_factor = (b.lifetime_end - b.lifetime_start) / span
        # Rule 1: large + long-lived → host.
        if size_factor >= 4.0 and life_factor >= 0.5 and host_tier in b.allowed_tiers:
            tier_hints.append(TierHint(
                buffer_id=b.buffer_id,
                tier_id=host_tier,
                confidence=0.85,
                reason="large_long_lived (size_factor={:.1f}, life_factor={:.2f})".format(
                    size_factor, life_factor
                ),
            ))
            continue
        # Rule 2: small + short-lived → scratchpad.
        if (
            fast_tier is not None
            and size_factor <= 1.0
            and life_factor <= 0.25
            and fast_tier in b.allowed_tiers
        ):
            tier_hints.append(TierHint(
                buffer_id=b.buffer_id,
                tier_id=fast_tier,
                confidence=0.75,
                reason="small_short_lived",
            ))

    # Rule 3: stage_partition by layer prefix.
    stage_groups: dict[str, list[str]] = {}
    for b in buffers:
        bid = b.buffer_id
        if "." in bid and bid.split(".", 1)[0].startswith("layer"):
            stage_id = bid.split(".", 1)[0]
            stage_groups.setdefault(stage_id, []).append(bid)
        else:
            stage_groups.setdefault("stage_unbatched", []).append(bid)
    stage_partition = tuple(
        StageGroup(stage_id=sid, buffer_ids=tuple(sorted(ids)),
                   reason="layer_prefix_grouping")
        for sid, ids in sorted(stage_groups.items())
    ) if len(stage_groups) > 1 else ()

    # Rule 4: symmetry within stage by size.
    symmetry_classes: list[SymmetryClass] = []
    by_size_per_stage: dict[tuple[str, int], list[str]] = {}
    for b in buffers:
        bid = b.buffer_id
        stage_id = bid.split(".", 1)[0] if "." in bid else "global"
        by_size_per_stage.setdefault((stage_id, b.size_bytes), []).append(bid)
    for (stage_id, size), ids in sorted(by_size_per_stage.items()):
        if len(ids) >= 2:
            symmetry_classes.append(SymmetryClass(
                class_id=f"{stage_id}_size_{size}",
                buffer_ids=tuple(sorted(ids)),
                reason=f"same_size_within_{stage_id}",
            ))

    confidence_summary = {
        "tier_hints_fraction": (len(tier_hints) / max(len(buffers), 1)),
        "high_confidence_fraction": (
            sum(1 for t in tier_hints if t.confidence >= 0.9) / max(len(buffers), 1)
        ),
    }
    return MemoryHints(
        tier_hints=tuple(tier_hints),
        stage_partition=stage_partition,
        symmetry_classes=tuple(symmetry_classes),
        source="rule_based",
        confidence_summary=confidence_summary,
    )


def merge_hints(*sources: MemoryHints) -> MemoryHints:
    """Merge multiple hint bundles, prefer higher-confidence hints
    where they conflict, deduplicate stage partitions by stage_id."""

    by_buffer: dict[str, TierHint] = {}
    for source in sources:
        for h in source.tier_hints:
            existing = by_buffer.get(h.buffer_id)
            if existing is None or h.confidence > existing.confidence:
                by_buffer[h.buffer_id] = h
    offsets: dict[str, OffsetHint] = {}
    for source in sources:
        for o in source.offset_warm_start:
            offsets.setdefault(o.buffer_id, o)
    stages: dict[str, StageGroup] = {}
    for source in sources:
        for s in source.stage_partition:
            stages.setdefault(s.stage_id, s)
    classes: dict[str, SymmetryClass] = {}
    for source in sources:
        for c in source.symmetry_classes:
            classes.setdefault(c.class_id, c)
    return MemoryHints(
        tier_hints=tuple(by_buffer.values()),
        offset_warm_start=tuple(offsets.values()),
        stage_partition=tuple(stages.values()),
        symmetry_classes=tuple(classes.values()),
        source="merged:" + ",".join(s.source for s in sources),
    )
