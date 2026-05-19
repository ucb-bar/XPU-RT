"""Observation types for the agent-first compiler environment.

Contains the data classes the agent sees at each step — no IR text, all
info pre-extracted for efficient consumption."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ============================================================================
# Observation: what the agent sees (compact, structured, NOT raw IR text)
# ============================================================================


@dataclass(frozen=True)
class RegionInfo:
    """What the agent knows about one IR region.

    This is the agent-efficient representation of an op. No IR text parsing
    needed — all relevant info is pre-extracted.
    """

    region_id: str
    op_type: str                     # "matmul", "gelu", "add", "transpose", etc.
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    flops: int                       # estimated FLOPs (0 for non-compute ops)
    bytes_in: int                    # input bytes
    bytes_out: int                   # output bytes
    arithmetic_intensity: float      # flops / total_bytes
    estimated_latency_us: float      # from cost model
    device_index: int                # current device assignment (-1 = unassigned)
    is_compute_bound: bool           # True if compute-bound, False if memory-bound
    dtype: str                       # "f32", "f16", etc.
    consumers: tuple[str, ...]       # region_ids that consume this op's output
    producers: tuple[str, ...]       # region_ids that produce this op's inputs


@dataclass(frozen=True)
class VerifiedFactInfo:
    """A formally verified fact about a region.

    Attributes:
        kind: Fact type ("local_mem_fit", "tile_divisible", "fusible",
              "contiguous_layout", "backend_eligible").
        region_id: Which region this fact applies to.
        confidence: "verified" (formal proof) or "estimated".
        detail: Human-readable description.
    """

    kind: str
    region_id: str
    confidence: str = "estimated"
    detail: str = ""


@dataclass(frozen=True)
class VerificationSummary:
    """What the agent knows about the current verification state.

    Attributes:
        tv_passed: Number of translation validations that passed.
        tv_failed: Number of translation validations that failed.
        tv_pending: Number of regions not yet TV-checked.
        last_failure_region: Region ID of the most recent TV failure.
        last_counterexample_summary: One-line counterexample description.
        verified_facts: Formally verified facts about regions.
        verifiable_op_types: Op types with defined semantics (TV-eligible).
    """

    tv_passed: int = 0
    tv_failed: int = 0
    tv_pending: int = 0
    last_failure_region: str = ""
    last_counterexample_summary: str = ""
    verified_facts: tuple[VerifiedFactInfo, ...] = ()
    verifiable_op_types: tuple[str, ...] = (
        "arith.addi", "arith.subi", "arith.muli", "arith.divui",
        "arith.divsi", "arith.remui", "arith.remsi", "arith.cmpi",
        "arith.select", "arith.constant",
    )


@dataclass(frozen=True)
class Observation:
    """Complete observation for the agent. Compact, structured, no IR text.

    This is what the agent receives at every step. It contains everything
    needed to make the next decision without parsing MLIR.
    """

    regions: tuple[RegionInfo, ...]
    total_flops: int
    total_bytes: int
    estimated_total_latency_us: float
    num_devices: int
    device_names: tuple[str, ...]
    device_memory_bytes: tuple[int, ...]
    objective: str                   # "latency", "throughput", "memory", "energy"
    step_count: int
    budget_remaining: int
    best_latency_us: float           # best seen so far
    history_summary: tuple[StepRecord, ...]  # last N steps
    verification: VerificationSummary | None = None
    graph_break_count: int = 0
    guard_count: int = 0
    unsupported_ops: tuple[str, ...] = ()
    analysis_dossier: Any = None
    active_packs: tuple[str, ...] = ()
    sealed_surfaces: tuple[str, ...] = ()
    generation_apertures: tuple[str, ...] = ()
    available_profilers: tuple[str, ...] = ()
    pack_benchmark_targets: tuple[str, ...] = ()
    integration_branch: str = ""


@dataclass(frozen=True)
class StepRecord:
    """Record of one past step, for the agent's history window."""

    step: int
    action_type: str
    action_target: str               # region_id or description
    was_legal: bool
    was_applied: bool
    cost_before_us: float
    cost_after_us: float
    improvement_pct: float
    verification_passed: bool
    error: str
