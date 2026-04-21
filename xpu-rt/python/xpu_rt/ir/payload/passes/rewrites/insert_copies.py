"""``insert_copies`` -- emit cross-memory-space ``CopyEdge``s.

Reconstruction of XLA's ``CopyInsertion`` + hexagon-mlir's
``copy-canonicalization``. Zero external references; CompGen owns
the rewrite.

Operates on :class:`ExecutionPlan`. Walks ``dependency_edges``:
when two connected regions produce / consume buffers in different
memory spaces, insert a ``CopyEdge`` between those buffers.

The producer-side buffer is identified via ``DependencyEdge.value_ref``
when populated (the stable contract for "this edge represents this
value"); otherwise we fall back to pairing the producer's outputs
with the consumer's inputs by name.

Config:

- ``default_transfer_path`` -- filled into each CopyEdge when the
  real ``transfer_path`` isn't known at this layer. Replaced by
   pipeline driver when a concrete ``target_resource.v2``
  is available.
- ``estimate_latency_ns`` -- simple model:
  ``size_bytes / bandwidth_gbps``. Default bandwidth 100 GB/s
  (mid-tier DDR4).

LLM-tool signature:

    tool_name="insert_copies"
    wraps_pass="CompGen:XLACopyInsertion+HexagonCopyCanonicalization"
    invent_slot="runtime/copy_insertion"
    policy="InsertOnMemorySpaceBoundary"
"""

from __future__ import annotations

from dataclasses import dataclass

from compgen.runtime.execution_plan import (
    BufferDescriptor,
    CopyEdge,
    ExecutionPlan,
    Lifetime,
)


@dataclass(frozen=True)
class InsertCopiesConfig:
    default_transfer_path: str = "generic_transfer"
    estimated_bandwidth_gbps: float = 100.0
    emit_latency: bool = True
    staging_buffer_suffix: str = "_staging"


@dataclass
class InsertCopiesStats:
    edges_seen: int = 0
    copies_inserted: int = 0
    skipped_same_space: int = 0
    skipped_no_value_ref: int = 0


def _buffer_by_id(plan: ExecutionPlan, bid: str) -> BufferDescriptor | None:
    for b in plan.buffers:
        if b.buffer_id == bid:
            return b
    return None


def _latency_ns(size_bytes: int, bandwidth_gbps: float) -> int:
    if size_bytes <= 0 or bandwidth_gbps <= 0:
        return 0
    bytes_per_ns = bandwidth_gbps  # 1 GB/s ≈ 1 byte/ns.
    return int(size_bytes / bytes_per_ns) if bytes_per_ns > 0 else 0


def run_insert_copies(
    plan: ExecutionPlan,
    *,
    config: InsertCopiesConfig | None = None,
) -> InsertCopiesStats:
    cfg = config if config is not None else InsertCopiesConfig()
    stats = InsertCopiesStats()

    existing_pairs = {(e.from_buffer, e.to_buffer) for e in plan.copy_edges}

    for edge in plan.dependency_edges:
        stats.edges_seen += 1
        if not edge.value_ref:
            stats.skipped_no_value_ref += 1
            continue

        producer_buf = _buffer_by_id(plan, edge.value_ref)
        if producer_buf is None:
            stats.skipped_no_value_ref += 1
            continue

        # Find which region the consumer is and what memory space it
        # reads from. Heuristic: the consumer runs on a specific
        # device; we pick a buffer that's co-located with the
        # consumer's region (or synthesize a staging buffer).
        try:
            consumer_placement = plan.placement_for(edge.to_region)
        except KeyError:
            continue

        # Consumer's target memory space is either explicit (if a
        # buffer exists on that device) or the consumer-device's
        # default. We use the producer buffer's memory space as the
        # signal: if different from the consumer's, we need a copy.
        #
        # For this pass we synthesize a new "staging" buffer in the
        # consumer's memory space (heuristically ``"dram"`` when we
        # can't tell). 's pipeline driver will set the
        # consumer space from target.
        producer_space = producer_buf.memory_space
        consumer_space = plan.summary.get("device_default_space", {}).get(consumer_placement.device, producer_space)

        if producer_space == consumer_space:
            stats.skipped_same_space += 1
            continue

        staging_id = producer_buf.buffer_id + cfg.staging_buffer_suffix
        staging = _buffer_by_id(plan, staging_id)
        if staging is None:
            staging = BufferDescriptor(
                buffer_id=staging_id,
                size_bytes=producer_buf.size_bytes,
                memory_space=consumer_space,
                lifetime=Lifetime(
                    first_use_tick=producer_buf.lifetime.last_use_tick,
                    last_use_tick=producer_buf.lifetime.last_use_tick + 1,
                    persistent=False,
                ),
                ownership="exclusive",
                alias_of="",
            )
            plan.buffers.append(staging)

        copy_pair = (producer_buf.buffer_id, staging_id)
        if copy_pair in existing_pairs:
            continue

        est = _latency_ns(producer_buf.size_bytes, cfg.estimated_bandwidth_gbps) if cfg.emit_latency else 0
        plan.copy_edges.append(
            CopyEdge(
                from_buffer=producer_buf.buffer_id,
                to_buffer=staging_id,
                size_bytes=producer_buf.size_bytes,
                transfer_path=cfg.default_transfer_path,
                est_latency_ns=est,
            )
        )
        existing_pairs.add(copy_pair)
        stats.copies_inserted += 1

    return stats


__all__ = [
    "InsertCopiesConfig",
    "InsertCopiesStats",
    "run_insert_copies",
]
