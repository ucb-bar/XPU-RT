"""KernelContract v3 — sharp-but-correct boundary, granularity-aware.

Design (April 2026, after the v3.1 sharpening pass)
---------------------------------------------------

Audience model — *one* contract, *two read projections*:

* ``contract.kernel_facing()`` returns a ``KernelFacingView`` —
  everything the kernel codegen is *allowed to read*. That includes IO
  (shape, dtype, layout, alignment, attributes, numerics), the
  ``ExecutionEnvelope`` (hardware caps, memory budget, concurrency
  unit, padding policy, latency-vs-throughput priority), the memory
  residency tiers it loads/stores from, the events it must fire on
  completion, and the dispatch model it must implement (one-shot,
  persistent, inlined). It does *not* include compiler-only fields.

* ``contract.compiler_only()`` returns a ``CompilerOnlyView`` —
  fields strictly invisible to the kernel: which events it waits on
  (the dispatcher inserts the wait), whether the launch blocks the
  host, output buffer lifetimes (memory planner only), fusion policy,
  observability hooks.

Only the compiler authors the contract — the kernel never writes any
field. The two views are *read* projections, not write privileges.

Granularity dimension — orthogonal to archetype:

* ``MICRO``  — ukernel-dialect tile primitive. Inlined into the caller's
  body, register/scratchpad-resident, fires no events, no dispatch.
  Example: a 16x16x16 fp16 matmul tile that an attention megakernel
  invokes from inside its inner loop.

* ``NORMAL`` — one logical op = one dispatch. Standard async launch with
  completion event, scratchpad+DRAM IO, the default for everything we
  generate today.

* ``MEGA``   — persistent kernel covering N sub-ops. Carries a ``body``
  of constituent NORMAL/MICRO sub-contracts and an
  ``internal_events`` graph describing the sync edges between them.
  All sub-buffers must be scratchpad/register-resident (the megakernel
  keeps intermediates in fast memory). Example: an attention block
  that fuses matmul→softmax→matmul into one persistent kernel.

Archetype (op family) and granularity (dispatch unit) are orthogonal:
a POINTWISE addf can be MICRO (a tile op) or NORMAL (a full pass), and
a COMPUTE_TILED matmul can be NORMAL (single launch) or part of a
MEGA's body.

This module defines schema only — provider retrofits and the megakernel
codegen pass live in their own modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from compgen.ir.payload.contracts import (
    AutocompCostBudget,
    CostEstimate,
)
from compgen.ir.payload.contracts import (
    KernelContract as KernelContractV2,
)

CONTRACT_VERSION = 3


# ===========================================================================
# Archetypes (op family) and Granularity (dispatch unit)
# ===========================================================================


class KernelArchetype(Enum):
    COMPUTE_TILED = "compute_tiled"
    POINTWISE = "pointwise"
    REDUCE = "reduce"
    MEMORY = "memory"
    ACTIVATION = "activation"
    TYPE_CONV_INDEX = "type_conv_index"


class Granularity(Enum):
    MICRO = "micro"  # ukernel — inlined, register/scratchpad-only, no events
    NORMAL = "normal"  # one op = one dispatch (default)
    MEGA = "mega"  # persistent composite — owns sub-kernels + internal sync


# ===========================================================================
# IO — tensor data + static op-attrs + numerics
# ===========================================================================


class LayoutKind(Enum):
    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"
    BLOCKED = "blocked"
    PACKED_K_MAJOR = "packed_k_major"
    OPAQUE = "opaque"


@dataclass(frozen=True)
class ShapeClass:
    """Per-tensor shape spec. ``dims[i]`` is a positive int (concrete) or
    ``None`` (dynamic; kernel must support any size)."""

    dims: tuple[int | None, ...]
    max_dims: tuple[int | None, ...] | None = None
    divisibility: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        if self.max_dims is not None and len(self.max_dims) != len(self.dims):
            raise ValueError("max_dims must align with dims")
        if self.divisibility is not None and len(self.divisibility) != len(self.dims):
            raise ValueError("divisibility must align with dims")


@dataclass(frozen=True)
class TensorIO:
    """One operand or result of the kernel. All fields are kernel-readable."""

    name: str
    shape: ShapeClass
    dtype_class: tuple[str, ...]
    layout: LayoutKind = LayoutKind.ROW_MAJOR
    strides: tuple[int, ...] | None = None
    alignment_bytes: int = 16
    broadcast_pattern: str | None = None


@dataclass(frozen=True)
class StaticAttr:
    """Compile-time op attribute the kernel reads (axis, kind, base, …)."""

    name: str
    value: Any


@dataclass(frozen=True)
class NumericsSpec:
    """Numeric guarantees the kernel must satisfy."""

    accumulator_dtype: str | None = None
    fast_math: bool = False
    max_relative_error: float = 1e-3
    deterministic: bool = True


@dataclass(frozen=True)
class IOContract:
    inputs: tuple[TensorIO, ...]
    outputs: tuple[TensorIO, ...]
    attributes: tuple[StaticAttr, ...] = ()
    numerics: NumericsSpec = field(default_factory=NumericsSpec)

    def __post_init__(self) -> None:
        if not self.outputs:
            raise ValueError("IOContract must declare at least one output")
        names = [o.name for o in (*self.inputs, *self.outputs)]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate IO operand names in contract: {names}")


# ===========================================================================
# Execution envelope — hardware + budget + concurrency + padding + priority
# ===========================================================================


@dataclass(frozen=True)
class HardwareEnvelope:
    """Summary of target-hardware caps the kernel may rely on at codegen.

    Read-through from the target profile (we don't duplicate the full
    profile — only the fields a kernel needs to make register/tile
    decisions). Kernels generated for one envelope are not portable to
    another without re-compilation.

    ``codegen_hints`` carries target-specific guidance strings the
    kernel codegen (Claude Code) reads as prompt context — think of
    them as the autocomp ``get_hw_config_specific_rules`` equivalent,
    but structured and authored per target. Examples: "use tl.dot with
    bf16 inputs + f32 accumulate for tensor cores", "Hexagon HVX
    vectors are 128B — use vmpyubacc for int8".

    ``mma_shapes`` maps each native dtype to its hardware MMA tile
    (M, N, K) so codegen can align inner loops to native instructions.
    ``peak_compute_per_dtype`` carries TFLOPS per dtype; the tile
    oracle uses both to pick MMA-aligned tile sizes.
    """

    target_name: str
    vector_lanes: int
    scratchpad_bytes: int
    register_bytes: int
    native_dtypes: tuple[str, ...]
    peak_bandwidth_gbps: float = 0.0
    codegen_hints: tuple[str, ...] = ()
    # W2 additions — drive the tile / packing oracle
    mma_shapes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    peak_compute_per_dtype: dict[str, float] = field(default_factory=dict)
    register_quota_per_thread: int = 256  # bytes per-thread soft cap
    max_concurrent_blocks: int = 0  # 0 = unbounded by the contract


class PaddingPolicy(Enum):
    """How to handle shapes that don't divide tile / vector width."""

    NONE = "none"  # shape always divides; no pad logic
    ZERO_FILL = "zero_fill"  # pad with zeros; kernel reads padded region
    KERNEL_HANDLES = "kernel_handles"  # kernel masks the boundary itself
    PLANNER_PADS = "planner_pads"  # planner allocates padded buffer


class ConcurrencyUnit(Enum):
    """Parallelism quantum the kernel runs in."""

    WARP = "warp"
    CTA = "cta"
    BLOCK = "block"
    DSP_SLICE = "dsp_slice"
    VECTOR_LANE_GROUP = "vector_lane_group"
    HOST_THREAD = "host_thread"


class PerformancePriority(Enum):
    LATENCY = "latency"
    BALANCED = "balanced"
    THROUGHPUT = "throughput"


@dataclass(frozen=True)
class ExecutionEnvelope:
    """Hardware + per-launch limits + concurrency the kernel must obey.

    All fields are kernel-readable: they drive tile selection, vector
    width, register pressure, and pad-handling codegen.
    """

    hardware: HardwareEnvelope
    memory_budget_bytes: int = 0  # 0 = unbounded
    concurrency_unit: ConcurrencyUnit = ConcurrencyUnit.CTA
    padding: PaddingPolicy = PaddingPolicy.KERNEL_HANDLES
    priority: PerformancePriority = PerformancePriority.BALANCED


# ===========================================================================
# Sync / Memory / Fusion / Dispatch / Observability
# ===========================================================================


@dataclass(frozen=True)
class EventDecl:
    """One event the kernel fires on completion (kernel-readable)."""

    name: str
    scope: str = "block"  # "block" | "device" | "host"
    wait_count: int = 1


@dataclass(frozen=True)
class AliasPair:
    """An (input_idx, output_idx) pair the planner allows to share storage."""

    input_idx: int
    output_idx: int


@dataclass(frozen=True)
class BufferLifetime:
    """Output-buffer lifetime hint (compiler-only, planner reads)."""

    output_idx: int = 0
    live_after: str = "epoch_end"


@dataclass(frozen=True)
class SyncSpec:
    """Dispatch sync contract.

    ``event_decls``  — kernel-readable (it must fire them).
    ``wait_on``      — compiler-only (the dispatcher inserts the wait).
    ``aliasing``     — kernel-readable (so it knows if in-place is safe).
    ``blocking``     — compiler-only (host-side knob).
    """

    event_decls: tuple[EventDecl, ...] = ()
    wait_on: tuple[str, ...] = ()
    aliasing: tuple[AliasPair, ...] = ()
    blocking: bool = False


class MemoryTier(Enum):
    REGISTER = "register"
    SCRATCHPAD = "scratchpad"
    L2 = "l2"
    DEVICE_DRAM = "device_dram"
    HOST = "host"


@dataclass(frozen=True)
class MemorySpec:
    """IO residency + lifetime contract.

    ``input_tiers`` / ``output_tiers`` / ``in_place_safe`` — kernel-readable
    (they determine load/store address-space codegen).
    ``lifetimes`` — compiler-only (memory planner uses to coalesce).
    """

    input_tiers: tuple[MemoryTier, ...] = ()
    output_tiers: tuple[MemoryTier, ...] = ()
    lifetimes: tuple[BufferLifetime, ...] = ()
    in_place_safe: bool = False


@dataclass(frozen=True)
class FusionPolicy:
    """Compiler-only — kernel never reads."""

    is_boundary: bool = False
    fusable_with: tuple[str, ...] = ()
    prefer_inline_into: str | None = None


def _resolve_dispatch_mode_override(override: str | None) -> "DispatchModel":
    """M-50: turn a string override (from a SetDispatchMode candidate's
    recipe_delta) into the DispatchModel enum. Defaults to SYNC; rejects
    PERSISTENT and INLINE on NORMAL granularity (the only granularity
    M-40 materialises today). The action_space already gates these;
    this is defence-in-depth so a hand-edited candidate response
    cannot smuggle PERSISTENT past the contract.
    """
    if override is None or override == "":
        return DispatchModel.SYNC
    norm = override.lower().strip()
    if norm == "sync":
        return DispatchModel.SYNC
    if norm == "async":
        return DispatchModel.ASYNC
    if norm == "persistent":
        raise ValueError(
            "M-50: PERSISTENT dispatch requires MEGA granularity; "
            "M-40 materialises only NORMAL contracts today. The "
            "action_space marks PERSISTENT illegal-by-granularity; "
            "a candidate response that selects it is rejected here."
        )
    if norm == "inline":
        raise ValueError(
            "M-50: INLINE dispatch requires MICRO granularity; "
            "ukernel path not yet wired."
        )
    raise ValueError(f"M-50: unknown dispatch mode override {override!r}")


class DispatchModel(Enum):
    """How the dispatcher launches this kernel."""

    SYNC = "sync"  # host blocks until completion
    ASYNC = "async"  # fire-and-forget, completion via events
    PERSISTENT = "persistent"  # long-lived worker; megakernel
    INLINE = "inline"  # not dispatched at all — emitted into caller (MICRO)


@dataclass(frozen=True)
class DispatchSpec:
    """Dispatcher knobs.

    ``model``  — kernel-readable (codegen branches on it: persistent
    kernels have a top-level dispatch loop; inline ones don't have a
    function boundary).
    Other fields are compiler-only.
    """

    model: DispatchModel = DispatchModel.ASYNC
    max_concurrent_invocations: int = 0
    retry_on_recoverable_error: bool = False


@dataclass(frozen=True)
class ObservabilitySpec:
    """Compiler-only."""

    emit_dispatch_event: bool = False
    emit_completion_event: bool = False
    cost_emit_period: int = 0


@dataclass(frozen=True)
class OrchestrationSpec:
    """Container for compiler-controlled blocks."""

    execution: ExecutionEnvelope | None = None
    sync: SyncSpec = field(default_factory=SyncSpec)
    memory: MemorySpec = field(default_factory=MemorySpec)
    fusion: FusionPolicy = field(default_factory=FusionPolicy)
    dispatch: DispatchSpec = field(default_factory=DispatchSpec)
    observability: ObservabilitySpec = field(default_factory=ObservabilitySpec)


# ===========================================================================
# Megakernel composition
# ===========================================================================


@dataclass(frozen=True)
class InternalEventEdge:
    """One sync edge between two body[] sub-kernels of a MEGA kernel.

    Producer fires ``event_name`` on completion; consumer's dispatcher
    inserts a wait on it before launching.
    """

    event_name: str
    producer_idx: int
    consumer_idx: int


# ===========================================================================
# Selection
# ===========================================================================


@dataclass(frozen=True)
class ProviderHint:
    name: str
    weight: float = 1.0
    rationale: str = ""


@dataclass(frozen=True)
class SelectionHints:
    providers: tuple[ProviderHint, ...] = ()
    autocomp_budget: AutocompCostBudget | None = None


# ===========================================================================
# Read projections — what each audience may see
# ===========================================================================


@dataclass(frozen=True)
class MemoryResidencyView:
    """Kernel-readable subset of MemorySpec — drives load/store codegen."""

    input_tiers: tuple[MemoryTier, ...]
    output_tiers: tuple[MemoryTier, ...]
    aliasing: tuple[AliasPair, ...]
    in_place_safe: bool


@dataclass(frozen=True)
class KernelFacingView:
    """Read-only projection for kernel codegen.

    The kernel implementation may read every field here. Anything not in
    this view is compiler-only and the kernel must not depend on it.
    """

    op_name: str
    archetype: KernelArchetype
    granularity: Granularity
    io: IOContract
    execution: ExecutionEnvelope | None
    memory_residency: MemoryResidencyView
    event_decls: tuple[EventDecl, ...]
    dispatch_model: DispatchModel


@dataclass(frozen=True)
class CompilerOnlyView:
    """Read-only projection for the compiler / runtime planner.

    Strictly invisible to kernels. Carries the fields that drive
    scheduling, memory planning, fusion choices, and observability.
    """

    wait_on: tuple[str, ...]
    blocking: bool
    lifetimes: tuple[BufferLifetime, ...]
    fusion: FusionPolicy
    observability: ObservabilitySpec
    dispatch_max_concurrent: int
    retry_on_recoverable_error: bool


# ===========================================================================
# The contract
# ===========================================================================


@dataclass(frozen=True)
class KernelContractV3:
    """Sharp-boundary, granularity-aware kernel contract."""

    op_name: str
    archetype: KernelArchetype
    io: IOContract
    granularity: Granularity = Granularity.NORMAL
    orchestration: OrchestrationSpec = field(default_factory=OrchestrationSpec)
    selection: SelectionHints = field(default_factory=SelectionHints)
    cost: CostEstimate = field(default_factory=CostEstimate)
    body: tuple[KernelContractV3, ...] = ()
    internal_events: tuple[InternalEventEdge, ...] = ()
    legacy: KernelContractV2 | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        self._check_archetype_invariants()
        self._check_memory_arity()
        self._check_granularity_invariants()

    # --- archetype invariants (op-family) ---

    def _check_archetype_invariants(self) -> None:
        a = self.archetype
        n_in = len(self.io.inputs)
        n_out = len(self.io.outputs)
        attrs = {a.name for a in self.io.attributes}

        if a is KernelArchetype.COMPUTE_TILED and n_in < 2:
            raise ValueError(f"COMPUTE_TILED {self.op_name!r} needs ≥2 inputs (got {n_in})")
        if a is KernelArchetype.REDUCE and "axis" not in attrs:
            raise ValueError(f"REDUCE {self.op_name!r} must declare static attribute 'axis'")
        if a is KernelArchetype.MEMORY and "kind" not in attrs:
            raise ValueError(f"MEMORY {self.op_name!r} must declare static attribute 'kind'")
        if a is KernelArchetype.POINTWISE and n_out != 1:
            raise ValueError(f"POINTWISE {self.op_name!r} produces exactly 1 output (got {n_out})")
        if a is KernelArchetype.ACTIVATION and (n_in, n_out) != (1, 1):
            raise ValueError(f"ACTIVATION {self.op_name!r} is unary (got in={n_in} out={n_out})")

    # --- memory arity ---

    def _check_memory_arity(self) -> None:
        n_in = len(self.io.inputs)
        n_out = len(self.io.outputs)
        m = self.orchestration.memory
        if m.input_tiers and len(m.input_tiers) != n_in:
            raise ValueError(
                f"memory.input_tiers length ({len(m.input_tiers)}) != input arity ({n_in}) for {self.op_name!r}"
            )
        if m.output_tiers and len(m.output_tiers) != n_out:
            raise ValueError(
                f"memory.output_tiers length ({len(m.output_tiers)}) != output arity ({n_out}) for {self.op_name!r}"
            )

    # --- granularity invariants (dispatch unit) ---

    def _check_granularity_invariants(self) -> None:
        g = self.granularity
        d = self.orchestration.dispatch
        s = self.orchestration.sync
        m = self.orchestration.memory
        f = self.orchestration.fusion

        if g is Granularity.MICRO:
            if s.event_decls:
                raise ValueError(f"MICRO {self.op_name!r} must not declare event_decls; caller fires events")
            if s.wait_on:
                raise ValueError(f"MICRO {self.op_name!r} must not declare wait_on; caller orchestrates sync")
            if d.model is not DispatchModel.INLINE:
                raise ValueError(f"MICRO {self.op_name!r} must use DispatchModel.INLINE (got {d.model.value!r})")
            for tier in (*m.input_tiers, *m.output_tiers):
                if tier not in (MemoryTier.REGISTER, MemoryTier.SCRATCHPAD):
                    raise ValueError(
                        f"MICRO {self.op_name!r} memory tier must be REGISTER or SCRATCHPAD (got {tier.value!r})"
                    )
            if f.is_boundary:
                raise ValueError(f"MICRO {self.op_name!r} cannot be a fusion boundary (it is always inlined)")
            if self.body or self.internal_events:
                raise ValueError(f"MICRO {self.op_name!r} must not declare body/internal_events")

        elif g is Granularity.MEGA:
            if d.model is not DispatchModel.PERSISTENT:
                raise ValueError(f"MEGA {self.op_name!r} must use DispatchModel.PERSISTENT (got {d.model.value!r})")
            if not self.body:
                raise ValueError(f"MEGA {self.op_name!r} must declare a non-empty body of sub-kernels")
            for sub in self.body:
                if sub.granularity is Granularity.MEGA:
                    raise ValueError(f"MEGA {self.op_name!r} body must not contain nested MEGA sub-kernels")
                for tier in (*sub.orchestration.memory.input_tiers, *sub.orchestration.memory.output_tiers):
                    if tier not in (MemoryTier.REGISTER, MemoryTier.SCRATCHPAD):
                        raise ValueError(
                            f"MEGA {self.op_name!r} sub-kernel {sub.op_name!r} must keep buffers in "
                            f"REGISTER/SCRATCHPAD (got {tier.value!r}); megakernel intermediates stay resident"
                        )
            n_body = len(self.body)
            for edge in self.internal_events:
                if not (0 <= edge.producer_idx < n_body):
                    raise ValueError(
                        f"MEGA {self.op_name!r} internal_events.producer_idx={edge.producer_idx} out of range"
                    )
                if not (0 <= edge.consumer_idx < n_body):
                    raise ValueError(
                        f"MEGA {self.op_name!r} internal_events.consumer_idx={edge.consumer_idx} out of range"
                    )

        else:  # NORMAL
            if d.model is DispatchModel.INLINE:
                raise ValueError(
                    f"NORMAL {self.op_name!r} must not use DispatchModel.INLINE "
                    f"(use Granularity.MICRO if inlining is intended)"
                )
            if d.model is DispatchModel.PERSISTENT and not self.body:
                # A persistent NORMAL kernel without a body is suspicious, but
                # not strictly illegal; permit but document.
                pass
            if self.body or self.internal_events:
                raise ValueError(f"NORMAL {self.op_name!r} must not declare body/internal_events; those are MEGA-only")

    # --- audience-controlled views ---

    @classmethod
    def from_recipe(
        cls,
        *,
        candidate_selection: dict[str, Any],
        region_dossier: dict[str, Any],
        target_profile: dict[str, Any],
        declared_refinement: str = "unknown",
        dispatch_mode_override: str | None = None,
    ) -> "KernelContractV3":
        """Materialize a KernelContractV3 from a selected Recipe IR
        decision plus region facts (M-40 / Section 21).

        Inputs are dicts (already-parsed JSON / YAML) so this function
        is a pure data transformation — no I/O. The pipeline-stage
        wrapper at ``compgen.graph_compilation.kernel_contract_materialization``
        reads the on-disk artifacts and calls into here.

        Args:
            candidate_selection: ``03_recipe_planning/candidate_selection.json``
                contents. Provides candidate_kind, recipe_delta,
                cost_preview (region_dims), label, region_id.
            region_dossier: ``02_graph_analysis/region_dossiers/<...>.json``
                contents. Provides region_shape (dtype, input_shapes,
                output_shapes), kind, target_class.
            target_profile: ``configs/targets/<target>.yaml`` contents.
                Provides hardware envelope fields.
            declared_refinement: ``"bit_equality"`` |
                ``"tolerance_eps"`` | ``"unknown"``. Sourced from the
                recipe-gate verdict (M-37.13's single_k_iter rule).

        Returns:
            A populated ``KernelContractV3`` whose ``__post_init__``
            invariants pass. The contract is suitable for
            ``hash_contract()`` (M-41) and for the kernel-facing
            projection emitted as the bounded subagent task (M-42).

        Raises:
            ValueError: if the candidate kind is not yet supported.
                M-40 supports only ``set_tile_params``; non-tile
                candidates surface as ``not_applicable`` at the
                wrapper layer rather than calling here.
        """
        candidate_kind = candidate_selection.get("candidate_kind", "")
        if candidate_kind != "set_tile_params":
            raise ValueError(
                f"M-40 from_recipe supports only candidate_kind="
                f"'set_tile_params'; got {candidate_kind!r}. The "
                f"pipeline wrapper emits a typed not_applicable row "
                f"for other kinds rather than calling here."
            )

        # --- Shape: M, N, K from the region dossier's region_shape. ---
        region_shape = region_dossier.get("region_shape") or {}
        input_shapes = region_shape.get("input_shapes") or []
        if (
            len(input_shapes) < 2
            or len(input_shapes[0]) != 2
            or len(input_shapes[1]) != 2
            or input_shapes[0][1] != input_shapes[1][0]
        ):
            # Fall back to cost_preview region_dims (M-37.13 surface).
            cost_preview = candidate_selection.get("cost_preview") or {}
            region_dims = cost_preview.get("region_dims") or {}
            M_dim = int(region_dims.get("M", 0) or 0)
            N_dim = int(region_dims.get("N", 0) or 0)
            K_dim = int(region_dims.get("K", 0) or 0)
        else:
            M_dim = int(input_shapes[0][0])
            K_dim = int(input_shapes[0][1])
            N_dim = int(input_shapes[1][1])

        # --- Tile: parsed from the candidate label. ---
        import re as _re
        label = candidate_selection.get("label", "") or ""
        m = _re.search(r"tile_M(\d+)_N(\d+)_K(\d+)", label)
        if m:
            tile_M = int(m.group(1))
            tile_N = int(m.group(2))
            tile_K = int(m.group(3))
        else:
            tile_M = tile_N = tile_K = 0
            for op in candidate_selection.get("recipe_delta") or []:
                tile_M = int(op.get("M", 0) or 0)
                tile_N = int(op.get("N", 0) or 0)
                tile_K = int(op.get("K", 0) or 0)
        if tile_M <= 0 or tile_N <= 0 or tile_K <= 0:
            raise ValueError(
                f"Could not extract tile from candidate label/delta: "
                f"label={label!r}"
            )

        # --- Dtype + layout from dossier; default to f32 row-major. ---
        dtype = str(region_shape.get("dtype") or "f32")
        # NumericsSpec — refinement claim drives max_relative_error.
        # For bit_equality the structural M-12 check is exact; we still
        # surface a non-zero rtol so the `numerics.epsilon_max_rel`
        # field is monotonic across refinements (the runtime check
        # prefers the Higham bound, this is the declared headroom).
        if declared_refinement == "bit_equality":
            max_rel_err = 0.0
            deterministic = True
        elif declared_refinement == "tolerance_eps":
            max_rel_err = 1e-3
            deterministic = True
        else:
            max_rel_err = 1e-3
            deterministic = True
        numerics = NumericsSpec(
            accumulator_dtype="f32",
            fast_math=False,
            max_relative_error=max_rel_err,
            deterministic=deterministic,
        )

        # --- IO ---
        lhs = TensorIO(
            name="lhs",
            shape=ShapeClass(dims=(M_dim, K_dim)),
            dtype_class=(dtype,),
            layout=LayoutKind.ROW_MAJOR,
            alignment_bytes=64,
        )
        rhs = TensorIO(
            name="rhs",
            shape=ShapeClass(dims=(K_dim, N_dim)),
            dtype_class=(dtype,),
            layout=LayoutKind.ROW_MAJOR,
            alignment_bytes=64,
        )
        out = TensorIO(
            name="out",
            shape=ShapeClass(dims=(M_dim, N_dim)),
            dtype_class=(dtype,),
            layout=LayoutKind.ROW_MAJOR,
            alignment_bytes=64,
        )
        # Tile sizes are static attrs; the kernel reads them.
        attrs = (
            StaticAttr(name="tile_M", value=tile_M),
            StaticAttr(name="tile_N", value=tile_N),
            StaticAttr(name="tile_K", value=tile_K),
            StaticAttr(name="declared_refinement", value=declared_refinement),
        )
        io = IOContract(
            inputs=(lhs, rhs),
            outputs=(out,),
            attributes=attrs,
            numerics=numerics,
        )

        # --- Hardware envelope from target profile ---
        target_id = (
            target_profile.get("target_id")
            or candidate_selection.get("target_id")
            or region_dossier.get("target_class")
            or "host_cpu"
        )
        # target_profile carries hardware caps under various nestings;
        # we read the ones we need with conservative defaults.
        envelope = target_profile.get("hardware_envelope") or {}
        scratchpad_kib = int(envelope.get("scratchpad_kib", 64))
        register_bytes = int(envelope.get("register_bytes", 256))
        vector_lanes = int(envelope.get("vector_lanes", 8))
        native_dtypes = tuple(
            envelope.get("native_dtypes") or ("f32", "bf16")
        )
        peak_bw = float(envelope.get("peak_bandwidth_gbps", 0.0))
        hardware = HardwareEnvelope(
            target_name=str(target_id),
            vector_lanes=vector_lanes,
            scratchpad_bytes=scratchpad_kib * 1024,
            register_bytes=register_bytes,
            native_dtypes=native_dtypes,
            peak_bandwidth_gbps=peak_bw,
        )
        execution = ExecutionEnvelope(
            hardware=hardware,
            memory_budget_bytes=scratchpad_kib * 1024,
            concurrency_unit=ConcurrencyUnit.HOST_THREAD
            if "cpu" in str(target_id).lower()
            else ConcurrencyUnit.CTA,
            padding=PaddingPolicy.KERNEL_HANDLES,
            priority=PerformancePriority.BALANCED,
        )

        # --- Orchestration: SYNC for M-40 (M-50 widens to dispatch as
        # a Recipe decision). One completion event "matmul_done";
        # matmul is a fusion boundary; output goes to scratchpad so a
        # downstream pointwise consumer reads warm.
        orchestration = OrchestrationSpec(
            execution=execution,
            sync=SyncSpec(
                event_decls=(EventDecl(name="matmul_done", scope="block"),),
            ),
            memory=MemorySpec(
                input_tiers=(MemoryTier.SCRATCHPAD, MemoryTier.SCRATCHPAD),
                output_tiers=(MemoryTier.SCRATCHPAD,),
                lifetimes=(BufferLifetime(output_idx=0, live_after="next_consumer"),),
            ),
            fusion=FusionPolicy(is_boundary=True),
            # M-50: dispatch model defaults to SYNC; M-50's
            # SetDispatchMode candidate can override via
            # dispatch_mode_override. NORMAL granularity admits SYNC
            # and ASYNC; PERSISTENT requires MEGA, INLINE requires
            # MICRO (both deferred).
            dispatch=DispatchSpec(
                model=_resolve_dispatch_mode_override(dispatch_mode_override),
            ),
            observability=ObservabilitySpec(
                emit_completion_event=True, cost_emit_period=0,
            ),
        )

        # --- Selection hints — provider preferences. ---
        # cffi-C reference path is the conservative default for CPU
        # targets; Triton template for GPU; autocomp as a follow-on.
        if "cpu" in str(target_id).lower():
            providers = (
                ProviderHint(
                    name="c_reference",
                    weight=1.0,
                    rationale="cffi-C reference is the proven CPU path",
                ),
                ProviderHint(
                    name="claude_code_subagent",
                    weight=0.5,
                    rationale="LLM-driven kernel codegen (M-43+)",
                ),
            )
        else:
            providers = (
                ProviderHint(
                    name="triton_template",
                    weight=1.0,
                    rationale="parametric Triton template for GPU",
                ),
                ProviderHint(
                    name="autocomp",
                    weight=0.7,
                    rationale="autocomp kernel-search backend",
                ),
            )
        selection = SelectionHints(providers=providers)

        return cls(
            op_name="linalg.matmul",
            archetype=KernelArchetype.COMPUTE_TILED,
            io=io,
            granularity=Granularity.NORMAL,
            orchestration=orchestration,
            selection=selection,
            metadata={
                "source_candidate_id": candidate_selection.get(
                    "selected_candidate_id", ""
                ),
                "source_region_id": region_dossier.get("region_id", ""),
                "source_recipe_op_id": "recipe_0000",
                "declared_refinement": declared_refinement,
                "tile": {"M": tile_M, "N": tile_N, "K": tile_K},
                "shape": {"M": M_dim, "N": N_dim, "K": K_dim},
            },
        )

    def kernel_facing(self) -> KernelFacingView:
        """Read projection for kernel codegen.

        The kernel may read every field here. Anything outside this view
        is compiler-only and the kernel must not depend on it.
        """
        m = self.orchestration.memory
        residency = MemoryResidencyView(
            input_tiers=m.input_tiers,
            output_tiers=m.output_tiers,
            aliasing=self.orchestration.sync.aliasing,
            in_place_safe=m.in_place_safe,
        )
        return KernelFacingView(
            op_name=self.op_name,
            archetype=self.archetype,
            granularity=self.granularity,
            io=self.io,
            execution=self.orchestration.execution,
            memory_residency=residency,
            event_decls=self.orchestration.sync.event_decls,
            dispatch_model=self.orchestration.dispatch.model,
        )

    def compiler_only(self) -> CompilerOnlyView:
        """Read projection for the compiler/runtime planner.

        Strictly invisible to kernels.
        """
        return CompilerOnlyView(
            wait_on=self.orchestration.sync.wait_on,
            blocking=self.orchestration.sync.blocking,
            lifetimes=self.orchestration.memory.lifetimes,
            fusion=self.orchestration.fusion,
            observability=self.orchestration.observability,
            dispatch_max_concurrent=self.orchestration.dispatch.max_concurrent_invocations,
            retry_on_recoverable_error=self.orchestration.dispatch.retry_on_recoverable_error,
        )


__all__ = [
    "AliasPair",
    "BufferLifetime",
    "CONTRACT_VERSION",
    "CompilerOnlyView",
    "ConcurrencyUnit",
    "DispatchModel",
    "DispatchSpec",
    "EventDecl",
    "ExecutionEnvelope",
    "FusionPolicy",
    "Granularity",
    "HardwareEnvelope",
    "InternalEventEdge",
    "IOContract",
    "KernelArchetype",
    "KernelContractV3",
    "KernelFacingView",
    "LayoutKind",
    "MemoryResidencyView",
    "MemorySpec",
    "MemoryTier",
    "NumericsSpec",
    "ObservabilitySpec",
    "OrchestrationSpec",
    "PaddingPolicy",
    "PerformancePriority",
    "ProviderHint",
    "SelectionHints",
    "ShapeClass",
    "StaticAttr",
    "SyncSpec",
    "TensorIO",
]
