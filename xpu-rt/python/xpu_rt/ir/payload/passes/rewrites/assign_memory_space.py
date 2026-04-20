"""``assign_memory_space`` -- place each buffer in a memory domain.

Reconstruction of IREE's HAL memory-space assignment + hexagon-mlir's
``convert-to-hexagonmem``. Zero external references; CompGen owns
the rewrite.

Operates on an :class:`ExecutionPlan` (not xDSL IR): fills in each
``BufferDescriptor.memory_space`` based on the buffer's role + size
+ liveness.

Policy (deterministic, size- + liveness-driven):

- Buffers whose lifetime is persistent (weight / constant) go to
  ``"hbm"`` (or ``"dram"`` on CPU).
- Buffers whose peak live bytes ≤ ``vtcm_bytes`` threshold and
  whose role is activation / intermediate go to ``"vtcm"``
  (hexagon) / ``"shared"`` (cuda) / ``"scratchpad"`` (generic).
- Everything else goes to the configured ``default_memory_space``.

The role is inferred from:

- ``persistent=True`` in the lifetime → weight/constant.
- ``ownership="shared_readonly"`` → weight.
- default → activation / intermediate.

Config:

- ``vtcm_bytes`` -- size threshold for the scratchpad domain.
- ``default_memory_space`` -- fallback space string.
- ``weight_memory_space`` -- space for persistent buffers.
- ``scratch_memory_space`` -- space for small intermediates.

This pass is idempotent: buffers that already have a non-empty
``memory_space`` aren't touched.

LLM-tool signature:

    tool_name="assign_memory_space"
    wraps_pass="CompGen:ConvertToHexagonMem"
    invent_slot="runtime/memory_space_assignment"
    policy="SizeAndLivenessDriven"
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compgen.runtime.execution_plan import BufferDescriptor, ExecutionPlan


@dataclass(frozen=True)
class AssignMemorySpaceConfig:
    default_memory_space: str = "dram"
    weight_memory_space: str = "dram"
    scratch_memory_space: str = "scratchpad"
    vtcm_bytes: int = 0  # 0 means "unused" — everything goes to default.
    overwrite_existing: bool = False


@dataclass
class AssignMemorySpaceStats:
    buffers_seen: int = 0
    buffers_assigned: int = 0
    buffers_already_assigned: int = 0
    placed_by_space: dict[str, int] = field(default_factory=dict)

    def record(self, space: str) -> None:
        self.placed_by_space[space] = self.placed_by_space.get(space, 0) + 1


def _is_weight_like(buf: BufferDescriptor) -> bool:
    return buf.lifetime.persistent or buf.ownership == "shared_readonly"


def _choose_space(
    buf: BufferDescriptor,
    cfg: AssignMemorySpaceConfig,
) -> str:
    if _is_weight_like(buf):
        return cfg.weight_memory_space
    # Small enough to fit in the scratchpad tier?
    if cfg.vtcm_bytes > 0 and buf.size_bytes <= cfg.vtcm_bytes:
        return cfg.scratch_memory_space
    return cfg.default_memory_space


def run_assign_memory_space(
    plan: ExecutionPlan,
    *,
    config: AssignMemorySpaceConfig | None = None,
) -> AssignMemorySpaceStats:
    cfg = config if config is not None else AssignMemorySpaceConfig()
    stats = AssignMemorySpaceStats()

    for buf in plan.buffers:
        stats.buffers_seen += 1
        if buf.memory_space and not cfg.overwrite_existing:
            stats.buffers_already_assigned += 1
            continue
        chosen = _choose_space(buf, cfg)
        buf.memory_space = chosen
        stats.buffers_assigned += 1
        stats.record(chosen)

    return stats


__all__ = [
    "AssignMemorySpaceConfig",
    "AssignMemorySpaceStats",
    "run_assign_memory_space",
]
