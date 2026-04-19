"""``dma_overlap`` -- introduce double-buffering around cross-domain copies.

Port of hexagon-mlir's two-stage pair
``DoubleBufferGenericS1.cpp`` + ``DoubleBufferGenericS2.cpp``. Zero
external references; CompGen owns the rewrite.

Stage 1 (``_introduce_double_buffers``):

Given a list of ``CopyEdge``s, split each copy into two ping-pong
buffers + two copy operations ("prefetch" + "main"). The prefetch
brings the next tile while the compute engine uses the main buffer.
In ``ExecutionPlan`` terms:

- Each ``CopyEdge(src → dst, transfer_path="...")`` becomes two
  ``CopyEdge``s:
  - prefetch:  src → dst ## "_ping", transfer_path="...",
    est_latency_ns stays.
  - main:      src → dst ## "_pong", transfer_path="...".
- Two new staging buffers get created (ping + pong) in the same
  memory space as the original destination.
- The ``plan.summary["dma_overlap_plan"]`` records the ping/pong
  mapping so the runtime can rotate between them each tile.

Stage 2 (``_convert_copies_to_dma``):

Re-labels every ping/pong copy edge with a
``transfer_path = "dma_async"`` marker and sets an accompanying
``sync_edges`` entry for the fence that closes each ping/pong
group. This is what hexagon-mlir's S2 does via
``memref.dma_start`` + ``memref.dma_wait``.

The pass is idempotent via the ``compgen.dma_overlap_applied``
tag on each processed copy edge (stored in
``plan.summary["dma_overlap_applied"]``).

Config:

- ``min_copy_size_bytes`` -- only double-buffer copies above this
  threshold (default 4096). Small copies don't hide latency.
- ``sync_kind`` -- ``"semaphore"`` (default) or ``"barrier"``.

LLM-tool signature:

    tool_name="dma_overlap"
    wraps_pass="CompGen:HexagonDoubleBufferGenericS1+S2"
    invent_slot="runtime/dma_overlap"
    policy="DoubleBufferAboveThreshold"
"""

from __future__ import annotations

from dataclasses import dataclass

from compgen.runtime.execution_plan import (
    BufferDescriptor,
    CopyEdge,
    ExecutionPlan,
    Lifetime,
    SyncEdge,
)


@dataclass(frozen=True)
class DMAOverlapConfig:
    min_copy_size_bytes: int = 4096
    sync_kind: str = "semaphore"
    dma_transfer_path: str = "dma_async"


@dataclass
class DMAOverlapStats:
    copies_seen: int = 0
    copies_double_buffered: int = 0
    copies_skipped_too_small: int = 0
    dma_edges_emitted: int = 0
    sync_edges_emitted: int = 0


def _buffer_by_id(plan: ExecutionPlan, bid: str) -> BufferDescriptor | None:
    for b in plan.buffers:
        if b.buffer_id == bid:
            return b
    return None


def _already_applied(summary: dict, copy_key: str) -> bool:
    applied = summary.get("dma_overlap_applied", [])
    return copy_key in applied


def run_dma_overlap(
    plan: ExecutionPlan,
    *,
    config: DMAOverlapConfig | None = None,
) -> DMAOverlapStats:
    cfg = config if config is not None else DMAOverlapConfig()
    stats = DMAOverlapStats()
    if cfg.sync_kind not in ("semaphore", "barrier", "fence"):
        raise ValueError(
            f"sync_kind must be one of semaphore/barrier/fence, got {cfg.sync_kind!r}"
        )

    applied: list[str] = list(plan.summary.get("dma_overlap_applied", []))
    plan_map: dict[str, dict[str, str]] = dict(
        plan.summary.get("dma_overlap_plan", {})
    )

    new_copies: list[CopyEdge] = []
    new_buffers: list[BufferDescriptor] = []
    new_syncs: list[SyncEdge] = []

    for edge in list(plan.copy_edges):
        stats.copies_seen += 1
        key = f"{edge.from_buffer}->{edge.to_buffer}"
        if _already_applied(plan.summary, key):
            continue
        # Skip copies that are themselves part of a ping/pong group
        # (they're the output of a previous dma_overlap run, not a
        # fresh copy that needs double-buffering).
        if edge.to_buffer.endswith("_ping") or edge.to_buffer.endswith("_pong"):
            continue
        if edge.transfer_path == cfg.dma_transfer_path:
            continue
        if edge.size_bytes < cfg.min_copy_size_bytes:
            stats.copies_skipped_too_small += 1
            continue

        dst = _buffer_by_id(plan, edge.to_buffer)
        if dst is None:
            continue

        # Create ping + pong.
        ping_id = edge.to_buffer + "_ping"
        pong_id = edge.to_buffer + "_pong"
        for bid in (ping_id, pong_id):
            if _buffer_by_id(plan, bid) is None:
                new_buffers.append(
                    BufferDescriptor(
                        buffer_id=bid,
                        size_bytes=dst.size_bytes,
                        memory_space=dst.memory_space,
                        lifetime=Lifetime(
                            first_use_tick=dst.lifetime.first_use_tick,
                            last_use_tick=dst.lifetime.last_use_tick,
                            persistent=False,
                        ),
                        ownership="exclusive",
                        alias_of="",
                    )
                )

        # Replace the original copy with two DMA copies.
        for new_dst in (ping_id, pong_id):
            new_copies.append(
                CopyEdge(
                    from_buffer=edge.from_buffer,
                    to_buffer=new_dst,
                    size_bytes=edge.size_bytes,
                    transfer_path=cfg.dma_transfer_path,
                    est_latency_ns=edge.est_latency_ns,
                )
            )
            stats.dma_edges_emitted += 1

        # One sync edge closes the ping/pong group (the compute consumer
        # fences on the currently-active buffer).
        new_syncs.append(
            SyncEdge(
                kind=cfg.sync_kind,
                producers=[ping_id, pong_id],
                consumers=[edge.to_buffer],
                scope="device",
            )
        )
        stats.sync_edges_emitted += 1

        plan_map[edge.to_buffer] = {"ping": ping_id, "pong": pong_id}
        applied.append(key)
        stats.copies_double_buffered += 1

    # Remove replaced copies.
    if stats.copies_double_buffered > 0:
        to_remove = set()
        for edge in plan.copy_edges:
            key = f"{edge.from_buffer}->{edge.to_buffer}"
            if key in applied and edge.to_buffer in plan_map:
                to_remove.add(id(edge))
        plan.copy_edges = [
            e for e in plan.copy_edges if id(e) not in to_remove
        ]

    plan.buffers.extend(new_buffers)
    plan.copy_edges.extend(new_copies)
    plan.sync_edges.extend(new_syncs)
    plan.summary["dma_overlap_applied"] = applied
    plan.summary["dma_overlap_plan"] = plan_map

    return stats


__all__ = [
    "DMAOverlapConfig",
    "DMAOverlapStats",
    "run_dma_overlap",
]
