"""Phase 5 runtime-contract tool stubs (P15).

Scaffolded per the approved plan's Wave 4 / Phase 5 coverage. Every
tool is a ``_StubPass`` subclass with full metadata + tool-args +
``stub=True``. Real destructive implementations land in follow-up
waves alongside the execution-plan emission and buffer solver.

Phase 5 tools cover what IREE's Stream dialect + XLA's BufferAssigner
do: queue/stream assignment, memory-space placement, buffer planning,
copy insertion, IO aliasing, host-offload insertion, and the XLA
post-layout sub-byte normalizer that runs after layout commit.
"""

from __future__ import annotations

from typing import Any, ClassVar

from xdsl.dialects.builtin import ModuleOp

from compgen.ir.payload.passes.base import PayloadPass
from compgen.llm.registry import AutocompCostImpact, ToolArg


class _RuntimeStub(PayloadPass):
    """Runtime-tool stub: phase=5, stub=True, identity run()."""

    phase: ClassVar[int] = 5
    stub: ClassVar[bool] = True

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        return module


class AssignQueue(_RuntimeStub):
    name: ClassVar[str] = "assign_queue"
    wraps_pass: ClassVar[str] = "XLA:queue_id annotation (Stream)"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "indirect"
    description: ClassVar[str] = (
        "[STUB] assign a region to a queue/stream id for async execution."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg("execution_plan", "plan_ref", "execution plan ref"),
            ToolArg("region", "region_ref", "region"),
            ToolArg("stream_id", "integer", "stream id"),
        )


class AssignMemorySpace(_RuntimeStub):
    name: ClassVar[str] = "assign_memory_space"
    wraps_pass: ClassVar[str] = "IREE:HAL memory-space assignment"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "indirect"
    description: ClassVar[str] = (
        "[STUB] assign a region's buffers to a named memory domain."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg("execution_plan", "plan_ref", "execution plan ref"),
            ToolArg("region", "region_ref", "region"),
            ToolArg("space", "string", "memory domain id"),
        )


class PlanBuffers(_RuntimeStub):
    name: ClassVar[str] = "plan_buffers"
    wraps_pass: ClassVar[str] = "XLA:BufferAssigner"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "indirect"
    description: ClassVar[str] = (
        "[STUB] allocate buffers with liveness + aliasing (feeds solve_memory)."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg("execution_plan", "plan_ref", "execution plan ref"),
            ToolArg("coloring_policy", "enum", "buffer coloring policy",
                    enum=("greedy", "first_fit", "min_peak"),
                    required=False, default="first_fit"),
        )


class InsertCopies(_RuntimeStub):
    name: ClassVar[str] = "insert_copies"
    wraps_pass: ClassVar[str] = "XLA:CopyInsertion"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "indirect"
    description: ClassVar[str] = (
        "[STUB] insert explicit copies where liveness demands."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg("execution_plan", "plan_ref", "execution plan ref"),
            ToolArg("schedule_phase", "enum", "pre or post scheduling",
                    enum=("pre_scheduling", "post_scheduling"),
                    required=False, default="post_scheduling"),
        )


class AliasIoBuffers(_RuntimeStub):
    name: ClassVar[str] = "alias_io_buffers"
    wraps_pass: ClassVar[str] = "XLA:OptimizeInputOutputBufferAlias"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "indirect"
    description: ClassVar[str] = (
        "[STUB] alias compatible I/O buffers to cut memory usage."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()


class InsertHostOffload(_RuntimeStub):
    name: ClassVar[str] = "insert_host_offload"
    wraps_pass: ClassVar[str] = "XLA:HostOffloader"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "indirect"
    description: ClassVar[str] = (
        "[STUB] insert host-offload nodes per policy."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg("region", "region_ref", "region"),
            ToolArg("policy", "enum", "offload policy",
                    enum=("always", "memory_pressure_driven", "never"),
                    required=False, default="memory_pressure_driven"),
        )


class AssignStreams(_RuntimeStub):
    name: ClassVar[str] = "assign_streams"
    wraps_pass: ClassVar[str] = "XLA:StreamAttributeAnnotator+AsyncWrapper"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "indirect"
    description: ClassVar[str] = (
        "[STUB] annotate regions with stream ids for concurrency."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg("execution_plan", "plan_ref", "execution plan ref"),
            ToolArg("stream_count", "integer", "max concurrent streams",
                    required=False, default=1),
        )


class NormalizeSubBytePostLayout(_RuntimeStub):
    name: ClassVar[str] = "normalize_subbyte_post_layout"
    wraps_pass: ClassVar[str] = "XLA:SubByteNormalization (post-layout mode)"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "low"
    description: ClassVar[str] = (
        "[STUB] second-pass sub-byte normalization after layout commit."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()


_RUNTIME_PASSES: list[PayloadPass] = [
    AliasIoBuffers(),
    AssignMemorySpace(),
    AssignQueue(),
    AssignStreams(),
    InsertCopies(),
    InsertHostOffload(),
    NormalizeSubBytePostLayout(),
    PlanBuffers(),
]


def register_runtime_passes() -> None:
    """Register all Phase 5 runtime stubs. Idempotent."""
    for p in _RUNTIME_PASSES:
        p.register()


__all__ = [
    "AliasIoBuffers",
    "AssignMemorySpace",
    "AssignQueue",
    "AssignStreams",
    "InsertCopies",
    "InsertHostOffload",
    "NormalizeSubBytePostLayout",
    "PlanBuffers",
    "register_runtime_passes",
]
