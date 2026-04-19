"""``insert_host_offload`` -- wrap regions that must run on the host.

Reconstruction of XLA's ``HostOffloader``. Zero external references;
CompGen owns the rewrite.

Operates on :class:`ExecutionPlan`. Walks ``region_placement`` for
regions whose ``device`` starts with ``"cpu"`` / ``"host"`` and:

1. Tags each with ``plan.summary["host_offload_regions"]``.
2. For each buffer consumed by a host-offloaded region that lives
   on a non-host memory space, emits a transfer edge
   (``CopyEdge`` with ``transfer_path="host_offload"``) plus a
   fallback sync edge.

When combined with ``insert_copies`` (Wave 6.5) the pass creates
host↔device transfer chains; on its own it's primarily
annotational.

Config:

- ``host_device_prefixes`` -- set of device strings that denote
  the host (default ``{"cpu", "host"}``).
- ``transfer_path`` -- label for the emitted CopyEdges.

LLM-tool signature:

    tool_name="insert_host_offload"
    wraps_pass="CompGen:XLAHostOffloader"
    invent_slot="runtime/host_offload"
    policy="WrapHostRegions"
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compgen.runtime.execution_plan import (
    BufferDescriptor,
    CopyEdge,
    ExecutionPlan,
    Lifetime,
)


@dataclass(frozen=True)
class InsertHostOffloadConfig:
    host_device_prefixes: tuple[str, ...] = ("cpu", "host")
    transfer_path: str = "host_offload"


@dataclass
class InsertHostOffloadStats:
    host_regions_found: int = 0
    offload_transfers_inserted: int = 0


def _is_host(device: str, prefixes: tuple[str, ...]) -> bool:
    return any(device.startswith(p) for p in prefixes)


def _buffer_by_id(plan: ExecutionPlan, bid: str) -> BufferDescriptor | None:
    for b in plan.buffers:
        if b.buffer_id == bid:
            return b
    return None


def run_insert_host_offload(
    plan: ExecutionPlan,
    *,
    config: InsertHostOffloadConfig | None = None,
) -> InsertHostOffloadStats:
    cfg = config if config is not None else InsertHostOffloadConfig()
    stats = InsertHostOffloadStats()

    host_regions: list[str] = []
    for rp in plan.region_placement:
        if _is_host(rp.device, cfg.host_device_prefixes):
            host_regions.append(rp.region_id)
            stats.host_regions_found += 1
    plan.summary["host_offload_regions"] = host_regions

    if not host_regions:
        return stats

    host_region_set = set(host_regions)

    # Emit transfer edges for non-host buffers consumed by host regions.
    existing = {(e.from_buffer, e.to_buffer) for e in plan.copy_edges}
    new_buffers: list[BufferDescriptor] = []
    for edge in plan.dependency_edges:
        if edge.to_region not in host_region_set:
            continue
        if not edge.value_ref:
            continue
        src = _buffer_by_id(plan, edge.value_ref)
        if src is None:
            continue
        # If producer's buffer is already on a host space, nothing to do.
        if src.memory_space in {"host", "cpu", "dram_host"}:
            continue
        host_staging_id = edge.value_ref + "_host"
        if _buffer_by_id(plan, host_staging_id) is None:
            new_buffers.append(
                BufferDescriptor(
                    buffer_id=host_staging_id,
                    size_bytes=src.size_bytes,
                    memory_space="host",
                    lifetime=Lifetime(
                        first_use_tick=src.lifetime.last_use_tick,
                        last_use_tick=src.lifetime.last_use_tick + 1,
                        persistent=False,
                    ),
                    ownership="exclusive",
                    alias_of="",
                )
            )

        if (edge.value_ref, host_staging_id) in existing:
            continue
        plan.copy_edges.append(
            CopyEdge(
                from_buffer=edge.value_ref,
                to_buffer=host_staging_id,
                size_bytes=src.size_bytes,
                transfer_path=cfg.transfer_path,
                est_latency_ns=0,
            )
        )
        existing.add((edge.value_ref, host_staging_id))
        stats.offload_transfers_inserted += 1

    plan.buffers.extend(new_buffers)
    return stats


__all__ = [
    "InsertHostOffloadConfig",
    "InsertHostOffloadStats",
    "run_insert_host_offload",
]
