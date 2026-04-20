"""``assign_streams`` -- allocate stream_ids + async-wrap decisions.

Reconstruction of XLA's ``StreamAttributeAnnotator`` +
``AsyncWrapper`` + hexagon-mlir's ``form-async-threads``. Zero
external references; CompGen owns the rewrite.

Operates on :class:`ExecutionPlan`. Fills in each
``RegionPlacement.stream_id`` and stashes an async-wrap decision
in ``plan.summary["stream_kinds"][region_id]`` (one of
``"sync"`` / ``"async_wrap"`` / ``"async_passthrough"``).

Policy:

- Each unique ``queue`` gets a dedicated stream_id 0..N-1.
- Regions whose producer is on a **different** queue get
  ``"async_wrap"`` (they must fence across streams).
- Regions whose producer is on the **same** queue get
  ``"sync"`` (in-order execution).
- Regions with no dependency edges get ``"sync"`` too.

The stream_id is deterministic: queues are sorted alphabetically
and assigned incrementing ids.

LLM-tool signature:

    tool_name="assign_streams"
    wraps_pass="CompGen:XLAStreamAttributeAnnotator+AsyncWrapper"
    invent_slot="runtime/stream_assignment"
    policy="SyncWithinQueueAsyncAcrossQueues"
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compgen.runtime.execution_plan import (
    ExecutionPlan,
    StreamAnnotation,
)


@dataclass(frozen=True)
class AssignStreamsConfig:
    default_async_kind: str = "async_wrap"  # or "async_passthrough"
    force_sync: bool = False
    overwrite_existing: bool = False


@dataclass
class AssignStreamsStats:
    regions_seen: int = 0
    regions_assigned: int = 0
    sync_regions: int = 0
    async_wrap_regions: int = 0
    queues_bound: set[str] = field(default_factory=set)


def _producers_of(region_id: str, edges: list[tuple[str, str]]) -> list[str]:
    return [frm for frm, to in edges if to == region_id]


def run_assign_streams(
    plan: ExecutionPlan,
    *,
    config: AssignStreamsConfig | None = None,
) -> AssignStreamsStats:
    cfg = config if config is not None else AssignStreamsConfig()
    stats = AssignStreamsStats()

    # Queue -> stream_id (stable ordering).
    unique_queues = sorted({rp.queue for rp in plan.region_placement})
    queue_to_stream: dict[str, int] = {q: i for i, q in enumerate(unique_queues)}

    queue_by_region: dict[str, str] = {rp.region_id: rp.queue for rp in plan.region_placement}
    edges = [(e.from_region, e.to_region) for e in plan.dependency_edges]

    annotations: list[StreamAnnotation] = []
    for rp in plan.region_placement:
        stats.regions_seen += 1
        producers = _producers_of(rp.region_id, edges)
        # Decide sync vs async.
        if cfg.force_sync or not producers:
            kind = "sync"
        else:
            producer_queues = {queue_by_region.get(p) for p in producers}
            if rp.queue in producer_queues and len(producer_queues) == 1:
                kind = "sync"
            else:
                kind = cfg.default_async_kind

        sid = queue_to_stream.get(rp.queue, 0)
        annotations.append(StreamAnnotation(region_id=rp.region_id, stream_id=sid, kind=kind))
        stats.queues_bound.add(rp.queue)
        if kind == "sync":
            stats.sync_regions += 1
        else:
            stats.async_wrap_regions += 1

    # Apply annotations.
    kinds = dict(plan.summary.get("stream_kinds", {}))
    for a in annotations:
        # Respect overwrite gate against stream_id only (kinds are
        # always overwritten because they derive from the dep graph).
        rp = plan.placement_for(a.region_id)
        if not cfg.overwrite_existing and rp.stream_id > 0:
            # Keep the existing stream_id but update kind.
            kinds[a.region_id] = a.kind
            continue
        rp.stream_id = a.stream_id
        kinds[a.region_id] = a.kind
        stats.regions_assigned += 1
    plan.summary["stream_kinds"] = kinds

    return stats


__all__ = [
    "AssignStreamsConfig",
    "AssignStreamsStats",
    "run_assign_streams",
]
