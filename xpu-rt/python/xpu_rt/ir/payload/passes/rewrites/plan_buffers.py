"""``plan_buffers`` -- port hexagon-mlir's ``MemoryOffsetsPass``.

Reconstruction of hexagon-mlir
``qcom_hexagon_backend/lib/Transforms/MemoryOffsetsPass.cpp`` as a
Python pass on :class:`ExecutionPlan`. Zero external references;
CompGen owns the rewrite.

Algorithm:

1. For each memory space, collect its buffers.
2. Build an interference graph via
   :func:`compgen.runtime.liveness.compute_interference_graph`
   (lifetime overlap).
3. Greedy-color by degree-desc.
4. For each color class, compute an offset + alignment-padded size.
5. Pool the entire color class into a single logical allocation.
6. Record each buffer's offset on the summary dict:
   ``plan.summary["buffer_offsets"][memory_space][buffer_id] = off``.

Aliased buffers (``ownership=alias``) are left as-is -- they point
at their alias_of target which the memory planner already placed.

Config:

- ``alignment_bytes`` -- byte alignment for each buffer (default
  128, matching hexagon-mlir VTCM alignment).
- ``restrict_to_spaces`` -- only pool buffers in these memory spaces;
  default is all non-empty spaces.

LLM-tool signature:

    tool_name="plan_buffers"
    wraps_pass="CompGen:MemoryOffsetsPass+BufferAssigner"
    invent_slot="runtime/buffer_pooling"
    policy="GreedyColorOffsetPool"
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from compgen.runtime.execution_plan import ExecutionPlan
from compgen.runtime.liveness import (
    compute_interference_graph,
    compute_liveness,
    greedy_color,
)


@dataclass(frozen=True)
class PlanBuffersConfig:
    alignment_bytes: int = 128
    restrict_to_spaces: frozenset[str] = frozenset()


@dataclass
class PlanBuffersStats:
    spaces_planned: int = 0
    buffers_pooled: int = 0
    total_bytes: dict[str, int] = field(default_factory=dict)
    num_colors_per_space: dict[str, int] = field(default_factory=dict)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def run_plan_buffers(
    plan: ExecutionPlan,
    *,
    config: PlanBuffersConfig | None = None,
) -> PlanBuffersStats:
    cfg = config if config is not None else PlanBuffersConfig()
    stats = PlanBuffersStats()

    # Bucket non-alias buffers by memory_space.
    by_space: dict[str, list] = defaultdict(list)
    alias_buffers: list = []
    for buf in plan.buffers:
        if buf.ownership == "alias":
            alias_buffers.append(buf)
            continue
        if not buf.memory_space:
            continue
        if cfg.restrict_to_spaces and buf.memory_space not in cfg.restrict_to_spaces:
            continue
        by_space[buf.memory_space].append(buf)

    # Compute global liveness / interference once, then filter per space.
    liveness = compute_liveness(plan)

    offsets: dict[str, dict[str, int]] = defaultdict(dict)

    for space, buffers in by_space.items():
        stats.spaces_planned += 1
        # Build a restricted interference graph over just this space.
        # We use the global graph (which already respects memory-space
        # separation via ``only_same_memory_space=True``) and read the
        # subgraph.
        global_graph = compute_interference_graph(
            liveness, only_same_memory_space=True
        )
        local_ids = {b.buffer_id for b in buffers}

        # Build per-color sizes. Alignment-pad each buffer.
        coloring = greedy_color(global_graph)
        color_sizes: dict[int, int] = {}
        for buf in buffers:
            c = coloring.get(buf.buffer_id, 0)
            padded = _align_up(buf.size_bytes, cfg.alignment_bytes)
            color_sizes[c] = max(color_sizes.get(c, 0), padded)

        # Lay colors out end-to-end, starting at offset 0.
        color_to_offset: dict[int, int] = {}
        running = 0
        for c in sorted(color_sizes):
            color_to_offset[c] = running
            running += color_sizes[c]

        # Record per-buffer offsets.
        for buf in buffers:
            c = coloring.get(buf.buffer_id, 0)
            offsets[space][buf.buffer_id] = color_to_offset[c]
            stats.buffers_pooled += 1

        stats.total_bytes[space] = running
        stats.num_colors_per_space[space] = len(color_sizes)

    # Alias buffers inherit their owner's offset.
    for buf in alias_buffers:
        owner_id = buf.alias_of
        if not owner_id:
            continue
        owner_space = None
        for other in plan.buffers:
            if other.buffer_id == owner_id:
                owner_space = other.memory_space
                break
        if owner_space is None or owner_space not in offsets:
            continue
        owner_off = offsets[owner_space].get(owner_id)
        if owner_off is None:
            continue
        buf.memory_space = owner_space
        offsets[owner_space][buf.buffer_id] = owner_off

    # Serialize to plan.summary.
    existing = dict(plan.summary.get("buffer_offsets", {}))
    existing.update({k: dict(v) for k, v in offsets.items()})
    plan.summary["buffer_offsets"] = existing
    plan.summary["buffer_pool_alignment_bytes"] = cfg.alignment_bytes
    plan.summary["buffer_pool_total_bytes"] = dict(stats.total_bytes)

    return stats


__all__ = [
    "PlanBuffersConfig",
    "PlanBuffersStats",
    "run_plan_buffers",
]
