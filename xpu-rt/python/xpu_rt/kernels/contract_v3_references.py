"""Reference KernelContract v3 instances.

Two axes of coverage:

* archetype  — one reference per op family (6 contracts):
    COMPUTE_TILED   matmul
    REDUCE          softmax
    POINTWISE       elementwise add
    MEMORY          where
    ACTIVATION      silu
    TYPE_CONV_INDEX dtype_cast

* granularity — one reference per dispatch unit (3 contracts):
    MICRO           ukernel matmul tile (16x16x16, register-resident)
    NORMAL          (the six archetype refs above are all NORMAL)
    MEGA            attention block fusing matmul → softmax → matmul
                    with internal event-tensor sync graph
"""

from __future__ import annotations

from compgen.kernels.contract_v3 import (
    AliasPair,
    BufferLifetime,
    ConcurrencyUnit,
    DispatchModel,
    DispatchSpec,
    EventDecl,
    ExecutionEnvelope,
    FusionPolicy,
    Granularity,
    HardwareEnvelope,
    InternalEventEdge,
    IOContract,
    KernelArchetype,
    KernelContractV3,
    LayoutKind,
    MemorySpec,
    MemoryTier,
    NumericsSpec,
    ObservabilitySpec,
    OrchestrationSpec,
    PaddingPolicy,
    PerformancePriority,
    ProviderHint,
    SelectionHints,
    ShapeClass,
    StaticAttr,
    SyncSpec,
    TensorIO,
)

# ---------------------------------------------------------------------------
# COMPUTE_TILED — matmul
# ---------------------------------------------------------------------------


def reference_matmul_contract() -> KernelContractV3:
    """Reference COMPUTE_TILED contract for ``M×K @ K×N → M×N``.

    IO declares:
      * 2 inputs (LHS, RHS) of bf16/f16/f32 with dim-32 K-divisibility,
        which lets the kernel pick a tile_k of 32 without padding.
      * 1 output of bf16, accumulator forced to f32.
    Orchestration declares:
      * fires ``matmul_done`` on completion (downstream softmax waits on it).
      * matmul is a fusion boundary — nothing fuses across it.
      * outputs to scratchpad so the next consumer (softmax) reads warm.
    """
    lhs = TensorIO(
        name="lhs",
        shape=ShapeClass(
            dims=(None, None),
            divisibility=(None, 32),
        ),
        dtype_class=("bf16", "f16", "f32"),
        layout=LayoutKind.ROW_MAJOR,
        alignment_bytes=64,
    )
    rhs = TensorIO(
        name="rhs",
        shape=ShapeClass(
            dims=(None, None),
            divisibility=(32, None),
        ),
        dtype_class=("bf16", "f16", "f32"),
        layout=LayoutKind.ROW_MAJOR,
        alignment_bytes=64,
    )
    out = TensorIO(
        name="out",
        shape=ShapeClass(dims=(None, None)),
        dtype_class=("bf16",),
        layout=LayoutKind.ROW_MAJOR,
        alignment_bytes=64,
    )
    io = IOContract(
        inputs=(lhs, rhs),
        outputs=(out,),
        attributes=(),
        numerics=NumericsSpec(
            accumulator_dtype="f32",
            fast_math=False,
            max_relative_error=1e-3,
        ),
    )
    return KernelContractV3(
        op_name="linalg.matmul",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=io,
        orchestration=OrchestrationSpec(
            sync=SyncSpec(
                event_decls=(EventDecl(name="matmul_done", scope="block"),),
            ),
            memory=MemorySpec(
                input_tiers=(MemoryTier.SCRATCHPAD, MemoryTier.SCRATCHPAD),
                output_tiers=(MemoryTier.SCRATCHPAD,),
                lifetimes=(BufferLifetime(output_idx=0, live_after="next_consumer"),),
            ),
            fusion=FusionPolicy(is_boundary=True),
            dispatch=DispatchSpec(model=DispatchModel.ASYNC),
            observability=ObservabilitySpec(emit_completion_event=True, cost_emit_period=100),
        ),
        selection=SelectionHints(
            providers=(
                ProviderHint(name="autocomp", weight=1.0, rationale="primary kernel-search backend"),
                ProviderHint(name="triton_template", weight=0.5, rationale="fallback if autocomp budget blown"),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# REDUCE — softmax
# ---------------------------------------------------------------------------


def reference_softmax_contract() -> KernelContractV3:
    """REDUCE contract for numerically-stable softmax along last dim."""
    inp = TensorIO(
        name="inp",
        shape=ShapeClass(dims=(None, None, None, None)),
        dtype_class=("f32", "bf16"),
        layout=LayoutKind.ROW_MAJOR,
    )
    out = TensorIO(
        name="out",
        shape=ShapeClass(dims=(None, None, None, None)),
        dtype_class=("f32",),
        layout=LayoutKind.ROW_MAJOR,
    )
    io = IOContract(
        inputs=(inp,),
        outputs=(out,),
        attributes=(
            StaticAttr(name="axis", value=-1),
            StaticAttr(name="keepdim", value=False),
            StaticAttr(name="numerically_stable", value=True),
        ),
        numerics=NumericsSpec(
            accumulator_dtype="f32",
            fast_math=False,
            max_relative_error=1e-4,
            deterministic=True,
        ),
    )
    return KernelContractV3(
        op_name="softmax",
        archetype=KernelArchetype.REDUCE,
        io=io,
        orchestration=OrchestrationSpec(
            sync=SyncSpec(
                event_decls=(EventDecl(name="softmax_done", scope="block"),),
                wait_on=("matmul_done",),
                aliasing=(AliasPair(input_idx=0, output_idx=0),),
            ),
            memory=MemorySpec(
                input_tiers=(MemoryTier.SCRATCHPAD,),
                output_tiers=(MemoryTier.SCRATCHPAD,),
                in_place_safe=True,
            ),
            fusion=FusionPolicy(
                is_boundary=False,
                fusable_with=("activation", "pointwise"),
            ),
            dispatch=DispatchSpec(model=DispatchModel.ASYNC),
        ),
    )


# ---------------------------------------------------------------------------
# POINTWISE — elementwise add (broadcastable)
# ---------------------------------------------------------------------------


def reference_pointwise_add_contract() -> KernelContractV3:
    a = TensorIO(
        name="a",
        shape=ShapeClass(dims=(None, None)),
        dtype_class=("f32", "bf16"),
        broadcast_pattern="elementwise",
    )
    b = TensorIO(
        name="b",
        shape=ShapeClass(dims=(None, None)),
        dtype_class=("f32", "bf16"),
        broadcast_pattern="elementwise",
    )
    out = TensorIO(
        name="out",
        shape=ShapeClass(dims=(None, None)),
        dtype_class=("f32", "bf16"),
    )
    io = IOContract(
        inputs=(a, b),
        outputs=(out,),
        numerics=NumericsSpec(fast_math=True, max_relative_error=1e-5),
    )
    return KernelContractV3(
        op_name="arith.addf",
        archetype=KernelArchetype.POINTWISE,
        io=io,
        orchestration=OrchestrationSpec(
            sync=SyncSpec(),  # default bulk-sync; pointwise is cheap
            fusion=FusionPolicy(
                is_boundary=False,
                fusable_with=("pointwise", "activation", "reduce"),
            ),
            memory=MemorySpec(in_place_safe=True),
        ),
    )


# ---------------------------------------------------------------------------
# MEMORY — where (masked select)
# ---------------------------------------------------------------------------


def reference_where_contract() -> KernelContractV3:
    cond = TensorIO(
        name="cond",
        shape=ShapeClass(dims=(None, None, None, None)),
        dtype_class=("i1", "i8"),
        layout=LayoutKind.ROW_MAJOR,
    )
    x = TensorIO(
        name="x",
        shape=ShapeClass(dims=(None, None, None, None)),
        dtype_class=("f32", "bf16"),
    )
    y = TensorIO(
        name="y",
        shape=ShapeClass(dims=(None, None, None, None)),
        dtype_class=("f32", "bf16"),
    )
    out = TensorIO(
        name="out",
        shape=ShapeClass(dims=(None, None, None, None)),
        dtype_class=("f32", "bf16"),
    )
    io = IOContract(
        inputs=(cond, x, y),
        outputs=(out,),
        attributes=(StaticAttr(name="kind", value="where"),),
        numerics=NumericsSpec(deterministic=True, max_relative_error=0.0),
    )
    return KernelContractV3(
        op_name="aten_where",
        archetype=KernelArchetype.MEMORY,
        io=io,
        orchestration=OrchestrationSpec(
            sync=SyncSpec(),
            fusion=FusionPolicy(is_boundary=False, fusable_with=("pointwise",)),
        ),
    )


# ---------------------------------------------------------------------------
# ACTIVATION — silu
# ---------------------------------------------------------------------------


def reference_silu_contract() -> KernelContractV3:
    inp = TensorIO(
        name="inp",
        shape=ShapeClass(dims=(None, None)),
        dtype_class=("f32", "bf16"),
    )
    out = TensorIO(
        name="out",
        shape=ShapeClass(dims=(None, None)),
        dtype_class=("f32", "bf16"),
    )
    io = IOContract(
        inputs=(inp,),
        outputs=(out,),
        numerics=NumericsSpec(fast_math=True, max_relative_error=1e-4),
    )
    return KernelContractV3(
        op_name="silu",
        archetype=KernelArchetype.ACTIVATION,
        io=io,
        orchestration=OrchestrationSpec(
            sync=SyncSpec(),
            fusion=FusionPolicy(
                is_boundary=False,
                fusable_with=("pointwise", "reduce"),
            ),
            memory=MemorySpec(in_place_safe=True),
        ),
    )


# ---------------------------------------------------------------------------
# TYPE_CONV_INDEX — dtype_cast (bf16 → f32)
# ---------------------------------------------------------------------------


def reference_dtype_cast_contract() -> KernelContractV3:
    inp = TensorIO(
        name="inp",
        shape=ShapeClass(dims=(None,)),
        dtype_class=("bf16",),
    )
    out = TensorIO(
        name="out",
        shape=ShapeClass(dims=(None,)),
        dtype_class=("f32",),
    )
    io = IOContract(
        inputs=(inp,),
        outputs=(out,),
        attributes=(
            StaticAttr(name="src_dtype", value="bf16"),
            StaticAttr(name="dst_dtype", value="f32"),
        ),
        numerics=NumericsSpec(deterministic=True, max_relative_error=0.0),
    )
    return KernelContractV3(
        op_name="dtype_cast",
        archetype=KernelArchetype.TYPE_CONV_INDEX,
        io=io,
        orchestration=OrchestrationSpec(
            sync=SyncSpec(),
            fusion=FusionPolicy(is_boundary=False, fusable_with=("pointwise",)),
        ),
    )


# ---------------------------------------------------------------------------
# Granularity references — MICRO + MEGA
# ---------------------------------------------------------------------------


def _hexagon_envelope() -> HardwareEnvelope:
    """Tiny envelope used by the granularity references — Hexagon-shaped
    for variety against the cuda-shaped flavor implied above."""
    return HardwareEnvelope(
        target_name="openq_5165rb",
        vector_lanes=128,
        scratchpad_bytes=8 * 1024 * 1024,  # 8 MB VTCM
        register_bytes=64,
        native_dtypes=("f16", "bf16", "i8"),
        peak_bandwidth_gbps=68.0,
    )


def reference_micro_matmul_tile_contract() -> KernelContractV3:
    """MICRO ukernel: 16×16×16 fp16 matmul tile.

    Inlined into a parent kernel's body. Inputs/outputs are tile-shaped
    and register-resident; the tile fires no events and is not
    dispatched on its own.
    """
    lhs_tile = TensorIO(
        name="lhs_tile",
        shape=ShapeClass(dims=(16, 16)),
        dtype_class=("f16",),
        layout=LayoutKind.ROW_MAJOR,
        alignment_bytes=32,
    )
    rhs_tile = TensorIO(
        name="rhs_tile",
        shape=ShapeClass(dims=(16, 16)),
        dtype_class=("f16",),
        layout=LayoutKind.ROW_MAJOR,
        alignment_bytes=32,
    )
    acc_tile = TensorIO(
        name="acc_tile",
        shape=ShapeClass(dims=(16, 16)),
        dtype_class=("f32",),
        layout=LayoutKind.ROW_MAJOR,
        alignment_bytes=32,
    )
    io = IOContract(
        inputs=(lhs_tile, rhs_tile),
        outputs=(acc_tile,),
        numerics=NumericsSpec(
            accumulator_dtype="f32",
            fast_math=False,
            max_relative_error=0.0,  # bit-exact at the tile level
            deterministic=True,
        ),
    )
    return KernelContractV3(
        op_name="ukernel.matmul_tile_16x16x16_fp16",
        archetype=KernelArchetype.COMPUTE_TILED,
        granularity=Granularity.MICRO,
        io=io,
        orchestration=OrchestrationSpec(
            execution=ExecutionEnvelope(
                hardware=_hexagon_envelope(),
                memory_budget_bytes=2048,  # tile working set
                concurrency_unit=ConcurrencyUnit.WARP,
                padding=PaddingPolicy.NONE,  # tile shape is exact
                priority=PerformancePriority.LATENCY,
            ),
            sync=SyncSpec(),  # MICRO: no events, no waits
            memory=MemorySpec(
                input_tiers=(MemoryTier.REGISTER, MemoryTier.REGISTER),
                output_tiers=(MemoryTier.REGISTER,),
                in_place_safe=True,  # acc += lhs @ rhs is canonical
            ),
            fusion=FusionPolicy(is_boundary=False),
            dispatch=DispatchSpec(model=DispatchModel.INLINE),
        ),
    )


def reference_mega_attention_block_contract() -> KernelContractV3:
    """MEGA persistent kernel: attention block as one fused dispatch.

    Body composes:
      [0] Q×Kᵀ matmul       (NORMAL COMPUTE_TILED, scratchpad-resident)
      [1] softmax(scores)   (NORMAL REDUCE)
      [2] (·)×V matmul      (NORMAL COMPUTE_TILED)

    Internal events:
      qk_done       : produced by [0], consumed by [1]
      softmax_done  : produced by [1], consumed by [2]

    All sub-buffers are SCRATCHPAD/REGISTER — intermediates never spill
    to DRAM. The external IO is just (Q, K, V) → attention_out.
    """
    env = _hexagon_envelope()
    scratch_residency = MemorySpec(
        input_tiers=(MemoryTier.SCRATCHPAD, MemoryTier.SCRATCHPAD),
        output_tiers=(MemoryTier.SCRATCHPAD,),
        in_place_safe=False,
    )

    # [0] Q×Kᵀ
    qk_io = IOContract(
        inputs=(
            TensorIO(name="q", shape=ShapeClass(dims=(None, None)), dtype_class=("bf16",)),
            TensorIO(name="kt", shape=ShapeClass(dims=(None, None)), dtype_class=("bf16",)),
        ),
        outputs=(TensorIO(name="scores", shape=ShapeClass(dims=(None, None)), dtype_class=("f32",)),),
        numerics=NumericsSpec(accumulator_dtype="f32"),
    )
    qk = KernelContractV3(
        op_name="qk_matmul",
        archetype=KernelArchetype.COMPUTE_TILED,
        granularity=Granularity.NORMAL,
        io=qk_io,
        orchestration=OrchestrationSpec(
            execution=ExecutionEnvelope(hardware=env),
            sync=SyncSpec(event_decls=(EventDecl(name="qk_done"),)),
            memory=scratch_residency,
            fusion=FusionPolicy(is_boundary=True),
        ),
    )

    # [1] softmax(scores)
    sm_io = IOContract(
        inputs=(TensorIO(name="scores", shape=ShapeClass(dims=(None, None)), dtype_class=("f32",)),),
        outputs=(TensorIO(name="probs", shape=ShapeClass(dims=(None, None)), dtype_class=("f32",)),),
        attributes=(StaticAttr(name="axis", value=-1), StaticAttr(name="numerically_stable", value=True)),
        numerics=NumericsSpec(accumulator_dtype="f32"),
    )
    sm = KernelContractV3(
        op_name="softmax_inner",
        archetype=KernelArchetype.REDUCE,
        granularity=Granularity.NORMAL,
        io=sm_io,
        orchestration=OrchestrationSpec(
            execution=ExecutionEnvelope(hardware=env),
            sync=SyncSpec(event_decls=(EventDecl(name="softmax_done"),)),
            memory=MemorySpec(
                input_tiers=(MemoryTier.SCRATCHPAD,),
                output_tiers=(MemoryTier.SCRATCHPAD,),
                in_place_safe=True,
            ),
        ),
    )

    # [2] probs × V
    av_io = IOContract(
        inputs=(
            TensorIO(name="probs", shape=ShapeClass(dims=(None, None)), dtype_class=("f32",)),
            TensorIO(name="v", shape=ShapeClass(dims=(None, None)), dtype_class=("bf16",)),
        ),
        outputs=(TensorIO(name="out", shape=ShapeClass(dims=(None, None)), dtype_class=("bf16",)),),
        numerics=NumericsSpec(accumulator_dtype="f32"),
    )
    av = KernelContractV3(
        op_name="av_matmul",
        archetype=KernelArchetype.COMPUTE_TILED,
        granularity=Granularity.NORMAL,
        io=av_io,
        orchestration=OrchestrationSpec(
            execution=ExecutionEnvelope(hardware=env),
            sync=SyncSpec(event_decls=(EventDecl(name="attention_done"),)),
            memory=scratch_residency,
            fusion=FusionPolicy(is_boundary=True),
        ),
    )

    # External IO of the megakernel: (Q, K, V) → out
    mega_io = IOContract(
        inputs=(
            TensorIO(name="Q", shape=ShapeClass(dims=(None, None)), dtype_class=("bf16",)),
            TensorIO(name="K", shape=ShapeClass(dims=(None, None)), dtype_class=("bf16",)),
            TensorIO(name="V", shape=ShapeClass(dims=(None, None)), dtype_class=("bf16",)),
        ),
        outputs=(TensorIO(name="attention_out", shape=ShapeClass(dims=(None, None)), dtype_class=("bf16",)),),
        numerics=NumericsSpec(accumulator_dtype="f32", max_relative_error=1e-3),
    )

    return KernelContractV3(
        op_name="megakernel.attention_block",
        archetype=KernelArchetype.COMPUTE_TILED,
        granularity=Granularity.MEGA,
        io=mega_io,
        orchestration=OrchestrationSpec(
            execution=ExecutionEnvelope(
                hardware=env,
                memory_budget_bytes=512 * 1024,  # 512 KB scratchpad budget
                concurrency_unit=ConcurrencyUnit.BLOCK,
                padding=PaddingPolicy.KERNEL_HANDLES,
                priority=PerformancePriority.LATENCY,
            ),
            sync=SyncSpec(
                event_decls=(EventDecl(name="attention_block_done", scope="device"),),
            ),
            memory=MemorySpec(
                input_tiers=(MemoryTier.DEVICE_DRAM,) * 3,
                output_tiers=(MemoryTier.DEVICE_DRAM,),
            ),
            fusion=FusionPolicy(is_boundary=True),
            dispatch=DispatchSpec(model=DispatchModel.PERSISTENT),
        ),
        body=(qk, sm, av),
        internal_events=(
            InternalEventEdge(event_name="qk_done", producer_idx=0, consumer_idx=1),
            InternalEventEdge(event_name="softmax_done", producer_idx=1, consumer_idx=2),
        ),
    )


__all__ = [
    "reference_dtype_cast_contract",
    "reference_matmul_contract",
    "reference_mega_attention_block_contract",
    "reference_micro_matmul_tile_contract",
    "reference_pointwise_add_contract",
    "reference_silu_contract",
    "reference_softmax_contract",
    "reference_where_contract",
]
