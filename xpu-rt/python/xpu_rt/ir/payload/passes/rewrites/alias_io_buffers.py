"""``alias_io_buffers`` -- fold in-place I/O buffer aliasing into the
buffer plan.

Reconstruction of XLA's ``OptimizeInputOutputBufferAlias`` +
hexagon-mlir's ``hexagon-rvo`` (Return Value Optimization). Zero
external references; CompGen owns the rewrite.

Operates on :class:`ExecutionPlan`. When a buffer's lifetime is
exclusively "produced, then immediately consumed as a final
result" (no further users), we can materialize it directly into
the caller-provided output tensor, avoiding one allocation.

Detection: find every "leaf" buffer -- a buffer that appears in
``plan.summary["output_buffers"]`` or is tagged with
``lifetime.persistent=False`` and has only one writer in the
dependency graph. When two leafs share non-overlapping lifetimes
AND the same memory space, we can alias them.

This pass does two things:

1. Alias candidate leaf pairs: set ``ownership="alias"`` and
   ``alias_of`` on the smaller-use-tick buffer.
2. Emit ``plan.summary["alias_decisions"]`` listing the
   ``(owner, alias)`` pairs for the runtime to honor.

Config:

- ``allow_different_sizes`` -- when ``True``, only alias when the
  larger buffer's size ≥ smaller's. Default ``False`` (strict).
- ``restrict_to_spaces`` -- only consider aliasing within these
  memory spaces.

LLM-tool signature:

    tool_name="alias_io_buffers"
    wraps_pass="CompGen:XLAOptimizeInputOutputBufferAlias+HexagonRVO"
    invent_slot="runtime/io_buffer_aliasing"
    policy="AliasLeafsInSameSpace"
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from compgen.runtime.execution_plan import (
    BufferDescriptor,
    ExecutionPlan,
)


@dataclass(frozen=True)
class AliasIOBuffersConfig:
    allow_different_sizes: bool = False
    restrict_to_spaces: frozenset[str] = frozenset()


@dataclass
class AliasIOBuffersStats:
    candidate_leafs: int = 0
    aliases_created: int = 0
    aliases_skipped_lifetime_overlap: int = 0
    aliases_skipped_size_mismatch: int = 0
    aliases_skipped_space_mismatch: int = 0


def _producers(plan: ExecutionPlan, bid: str) -> int:
    """Count edges whose value_ref is this buffer."""
    return sum(1 for e in plan.dependency_edges if e.value_ref == bid)


def _is_leaf(plan: ExecutionPlan, buf: BufferDescriptor) -> bool:
    """Candidate leaf for aliasing.

    Rules:
    - Not persistent / weight-like.
    - Not already an alias.
    - Single producer (at most one region writes it).
    """
    if buf.lifetime.persistent:
        return False
    if buf.ownership != "exclusive":
        return False
    if _producers(plan, buf.buffer_id) > 1:
        return False
    return True


def _lifetimes_overlap(a: BufferDescriptor, b: BufferDescriptor) -> bool:
    if a.lifetime.persistent or b.lifetime.persistent:
        return True
    return not (
        a.lifetime.last_use_tick < b.lifetime.first_use_tick or b.lifetime.last_use_tick < a.lifetime.first_use_tick
    )


def run_alias_io_buffers(
    plan: ExecutionPlan,
    *,
    config: AliasIOBuffersConfig | None = None,
) -> AliasIOBuffersStats:
    cfg = config if config is not None else AliasIOBuffersConfig()
    stats = AliasIOBuffersStats()

    # Bucket leafs by memory space.
    by_space: dict[str, list[BufferDescriptor]] = defaultdict(list)
    for buf in plan.buffers:
        if cfg.restrict_to_spaces and buf.memory_space not in cfg.restrict_to_spaces:
            continue
        if _is_leaf(plan, buf):
            by_space[buf.memory_space].append(buf)
            stats.candidate_leafs += 1

    decisions: list[tuple[str, str]] = list(plan.summary.get("alias_decisions", []))

    for space, leafs in by_space.items():
        # Sort by first_use_tick for deterministic pairing.
        leafs.sort(key=lambda b: (b.lifetime.first_use_tick, b.buffer_id))
        already_aliased: set[str] = set()
        for i, a in enumerate(leafs):
            if a.buffer_id in already_aliased:
                continue
            for b in leafs[i + 1 :]:
                if b.buffer_id in already_aliased:
                    continue
                if a.memory_space != b.memory_space:
                    stats.aliases_skipped_space_mismatch += 1
                    continue
                if _lifetimes_overlap(a, b):
                    stats.aliases_skipped_lifetime_overlap += 1
                    continue
                if not cfg.allow_different_sizes and a.size_bytes != b.size_bytes:
                    stats.aliases_skipped_size_mismatch += 1
                    continue
                # Alias b onto a (a is the earlier producer).
                b.ownership = "alias"
                b.alias_of = a.buffer_id
                already_aliased.add(b.buffer_id)
                decisions.append((a.buffer_id, b.buffer_id))
                stats.aliases_created += 1
                break  # each buffer aliases at most one other

    plan.summary["alias_decisions"] = decisions
    return stats


__all__ = [
    "AliasIOBuffersConfig",
    "AliasIOBuffersStats",
    "run_alias_io_buffers",
]
