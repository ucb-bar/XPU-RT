"""Tests for ``compgen.kernels.contract_v3`` (sharp boundary + granularity).

Locks in:

  * 6 archetype references and the MICRO + MEGA granularity references
    are all well-formed
  * archetype-mandatory IO/attribute checks reject malformed contracts
  * ``kernel_facing()`` exposes IO + execution + memory residency +
    event_decls + dispatch_model — and nothing else
  * ``compiler_only()`` exposes wait_on / blocking / lifetimes / fusion /
    observability — and the kernel-readable fields are NOT in it
  * granularity invariants: MICRO has no events, must INLINE,
    register/scratchpad-only, no body; MEGA must PERSISTENT + body, all
    sub-buffers scratchpad/register, no nested MEGA
  * memory-tier arity matches IO arity
"""

from __future__ import annotations

import pytest

from compgen.kernels.contract_v3 import (
    AliasPair,
    BufferLifetime,
    CONTRACT_VERSION,
    CompilerOnlyView,
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
    KernelFacingView,
    LayoutKind,
    MemoryResidencyView,
    MemorySpec,
    MemoryTier,
    NumericsSpec,
    ObservabilitySpec,
    OrchestrationSpec,
    PaddingPolicy,
    PerformancePriority,
    ShapeClass,
    StaticAttr,
    SyncSpec,
    TensorIO,
)
from compgen.kernels.contract_v3_references import (
    reference_dtype_cast_contract,
    reference_matmul_contract,
    reference_mega_attention_block_contract,
    reference_micro_matmul_tile_contract,
    reference_pointwise_add_contract,
    reference_silu_contract,
    reference_softmax_contract,
    reference_where_contract,
)


# ---------------------------------------------------------------------------
# Reference instances — archetype + granularity coverage, all well-formed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,expected_archetype,expected_granularity,expected_op",
    [
        (reference_matmul_contract,        KernelArchetype.COMPUTE_TILED,  Granularity.NORMAL, "linalg.matmul"),
        (reference_softmax_contract,       KernelArchetype.REDUCE,         Granularity.NORMAL, "softmax"),
        (reference_pointwise_add_contract, KernelArchetype.POINTWISE,      Granularity.NORMAL, "arith.addf"),
        (reference_where_contract,         KernelArchetype.MEMORY,         Granularity.NORMAL, "aten_where"),
        (reference_silu_contract,          KernelArchetype.ACTIVATION,     Granularity.NORMAL, "silu"),
        (reference_dtype_cast_contract,    KernelArchetype.TYPE_CONV_INDEX,Granularity.NORMAL, "dtype_cast"),
        (reference_micro_matmul_tile_contract, KernelArchetype.COMPUTE_TILED, Granularity.MICRO,
         "ukernel.matmul_tile_16x16x16_fp16"),
        (reference_mega_attention_block_contract, KernelArchetype.COMPUTE_TILED, Granularity.MEGA,
         "megakernel.attention_block"),
    ],
)
def test_reference_contract_well_formed(
    factory, expected_archetype, expected_granularity, expected_op
) -> None:
    c = factory()
    assert c.archetype is expected_archetype
    assert c.granularity is expected_granularity
    assert c.op_name == expected_op
    assert c.contract_version == CONTRACT_VERSION


def test_mega_attention_body_and_internal_events_well_formed() -> None:
    c = reference_mega_attention_block_contract()
    assert len(c.body) == 3
    assert {sub.op_name for sub in c.body} == {"qk_matmul", "softmax_inner", "av_matmul"}
    # internal events fully connect the chain
    edges = c.internal_events
    assert len(edges) == 2
    assert edges[0].producer_idx == 0 and edges[0].consumer_idx == 1
    assert edges[1].producer_idx == 1 and edges[1].consumer_idx == 2
    # All sub-buffers are scratchpad-resident — the megakernel keeps
    # intermediates in fast memory.
    for sub in c.body:
        for tier in (*sub.orchestration.memory.input_tiers,
                     *sub.orchestration.memory.output_tiers):
            assert tier in (MemoryTier.SCRATCHPAD, MemoryTier.REGISTER)


def test_micro_matmul_tile_is_inlined_and_eventless() -> None:
    c = reference_micro_matmul_tile_contract()
    assert c.orchestration.dispatch.model is DispatchModel.INLINE
    assert c.orchestration.sync.event_decls == ()
    assert c.orchestration.sync.wait_on == ()
    # All tiers register-resident; tile shape is exact (no padding).
    for tier in (*c.orchestration.memory.input_tiers,
                 *c.orchestration.memory.output_tiers):
        assert tier is MemoryTier.REGISTER
    assert c.orchestration.execution.padding is PaddingPolicy.NONE


# ---------------------------------------------------------------------------
# Audience-controlled views — kernel_facing() and compiler_only()
# ---------------------------------------------------------------------------


def test_kernel_facing_includes_io_execution_residency_events_dispatch() -> None:
    c = reference_matmul_contract()
    view = c.kernel_facing()
    assert isinstance(view, KernelFacingView)
    # Tensor data + numerics + attributes
    assert view.io.outputs and view.io.numerics is not None
    # Memory residency the kernel uses for load/store codegen
    assert isinstance(view.memory_residency, MemoryResidencyView)
    assert view.memory_residency.input_tiers == (MemoryTier.SCRATCHPAD, MemoryTier.SCRATCHPAD)
    assert view.memory_residency.output_tiers == (MemoryTier.SCRATCHPAD,)
    # Events the kernel must fire on completion
    assert view.event_decls and view.event_decls[0].name == "matmul_done"
    # Dispatch model — kernel codegen branches on PERSISTENT vs INLINE vs ASYNC
    assert view.dispatch_model is DispatchModel.ASYNC


def test_kernel_facing_excludes_compiler_only_fields() -> None:
    """The view's fields must NOT include wait_on / blocking / lifetimes /
    fusion / observability — those are compiler concerns."""
    view = reference_matmul_contract().kernel_facing()
    public = set(view.__dataclass_fields__.keys())
    assert public == {
        "op_name", "archetype", "granularity",
        "io", "execution", "memory_residency",
        "event_decls", "dispatch_model",
    }
    for forbidden in ("wait_on", "blocking", "lifetimes", "fusion", "observability"):
        assert not hasattr(view, forbidden)


def test_compiler_only_view_carries_planner_fields_only() -> None:
    c = reference_softmax_contract()
    view = c.compiler_only()
    assert isinstance(view, CompilerOnlyView)
    assert view.wait_on == ("matmul_done",)
    assert view.blocking is False
    assert isinstance(view.fusion, FusionPolicy)
    # Compiler view does NOT carry IO / dispatch_model / event_decls / residency
    public = set(view.__dataclass_fields__.keys())
    assert public == {
        "wait_on", "blocking", "lifetimes", "fusion",
        "observability", "dispatch_max_concurrent",
        "retry_on_recoverable_error",
    }
    for forbidden in ("io", "dispatch_model", "event_decls", "memory_residency"):
        assert not hasattr(view, forbidden)


# ---------------------------------------------------------------------------
# Archetype invariants (already locked in v3.0; preserved through refactor)
# ---------------------------------------------------------------------------


def _io(n_in: int = 1, n_out: int = 1, attributes=()) -> IOContract:
    return IOContract(
        inputs=tuple(
            TensorIO(name=f"in{i}", shape=ShapeClass(dims=(None,)), dtype_class=("f32",))
            for i in range(n_in)
        ),
        outputs=tuple(
            TensorIO(name=f"out{i}", shape=ShapeClass(dims=(None,)), dtype_class=("f32",))
            for i in range(n_out)
        ),
        attributes=attributes,
    )


def test_compute_tiled_with_lt_2_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="COMPUTE_TILED.*needs ≥2 inputs"):
        KernelContractV3(op_name="bad", archetype=KernelArchetype.COMPUTE_TILED, io=_io(1, 1))


def test_reduce_without_axis_rejected() -> None:
    with pytest.raises(ValueError, match="REDUCE.*declare static attribute 'axis'"):
        KernelContractV3(op_name="bad", archetype=KernelArchetype.REDUCE, io=_io(1, 1))


def test_memory_without_kind_rejected() -> None:
    with pytest.raises(ValueError, match="MEMORY.*declare static attribute 'kind'"):
        KernelContractV3(op_name="bad", archetype=KernelArchetype.MEMORY, io=_io(1, 1))


def test_pointwise_must_be_single_output() -> None:
    with pytest.raises(ValueError, match="POINTWISE.*exactly 1 output"):
        KernelContractV3(op_name="bad", archetype=KernelArchetype.POINTWISE, io=_io(2, 2))


def test_activation_must_be_unary() -> None:
    with pytest.raises(ValueError, match="ACTIVATION.*unary"):
        KernelContractV3(op_name="bad", archetype=KernelArchetype.ACTIVATION, io=_io(2, 1))


# ---------------------------------------------------------------------------
# Granularity invariants — MICRO
# ---------------------------------------------------------------------------


def _normal_micro_io() -> IOContract:
    """IO good enough for MICRO test stubs (POINTWISE-shaped)."""
    return _io(n_in=1, n_out=1)


def test_micro_with_event_decls_rejected() -> None:
    with pytest.raises(ValueError, match="MICRO.*must not declare event_decls"):
        KernelContractV3(
            op_name="bad", archetype=KernelArchetype.POINTWISE,
            io=_normal_micro_io(),
            granularity=Granularity.MICRO,
            orchestration=OrchestrationSpec(
                sync=SyncSpec(event_decls=(EventDecl(name="x"),)),
                dispatch=DispatchSpec(model=DispatchModel.INLINE),
            ),
        )


def test_micro_with_async_dispatch_rejected() -> None:
    with pytest.raises(ValueError, match="MICRO.*DispatchModel.INLINE"):
        KernelContractV3(
            op_name="bad", archetype=KernelArchetype.POINTWISE,
            io=_normal_micro_io(),
            granularity=Granularity.MICRO,
            orchestration=OrchestrationSpec(
                dispatch=DispatchSpec(model=DispatchModel.ASYNC),
            ),
        )


def test_micro_with_dram_tier_rejected() -> None:
    with pytest.raises(ValueError, match="MICRO.*REGISTER or SCRATCHPAD"):
        KernelContractV3(
            op_name="bad", archetype=KernelArchetype.POINTWISE,
            io=_normal_micro_io(),
            granularity=Granularity.MICRO,
            orchestration=OrchestrationSpec(
                dispatch=DispatchSpec(model=DispatchModel.INLINE),
                memory=MemorySpec(
                    input_tiers=(MemoryTier.DEVICE_DRAM,),
                    output_tiers=(MemoryTier.SCRATCHPAD,),
                ),
            ),
        )


def test_micro_with_body_rejected() -> None:
    with pytest.raises(ValueError, match="MICRO.*body/internal_events"):
        sub = KernelContractV3(
            op_name="sub", archetype=KernelArchetype.POINTWISE,
            io=_normal_micro_io(),
        )
        KernelContractV3(
            op_name="bad", archetype=KernelArchetype.POINTWISE,
            io=_normal_micro_io(),
            granularity=Granularity.MICRO,
            orchestration=OrchestrationSpec(dispatch=DispatchSpec(model=DispatchModel.INLINE)),
            body=(sub,),
        )


# ---------------------------------------------------------------------------
# Granularity invariants — MEGA
# ---------------------------------------------------------------------------


def _stub_normal_compute() -> KernelContractV3:
    """A NORMAL COMPUTE_TILED contract suitable as MEGA body[i]."""
    return KernelContractV3(
        op_name="stub_normal",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(
                TensorIO(name="a", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),
                TensorIO(name="b", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),
            ),
            outputs=(TensorIO(name="o", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),),
        ),
        orchestration=OrchestrationSpec(
            memory=MemorySpec(
                input_tiers=(MemoryTier.SCRATCHPAD, MemoryTier.SCRATCHPAD),
                output_tiers=(MemoryTier.SCRATCHPAD,),
            ),
        ),
    )


def test_mega_without_persistent_dispatch_rejected() -> None:
    with pytest.raises(ValueError, match="MEGA.*DispatchModel.PERSISTENT"):
        KernelContractV3(
            op_name="bad", archetype=KernelArchetype.COMPUTE_TILED,
            io=_io(2, 1),
            granularity=Granularity.MEGA,
            body=(_stub_normal_compute(),),
        )


def test_mega_with_empty_body_rejected() -> None:
    with pytest.raises(ValueError, match="MEGA.*non-empty body"):
        KernelContractV3(
            op_name="bad", archetype=KernelArchetype.COMPUTE_TILED,
            io=_io(2, 1),
            granularity=Granularity.MEGA,
            orchestration=OrchestrationSpec(dispatch=DispatchSpec(model=DispatchModel.PERSISTENT)),
        )


def test_mega_with_nested_mega_rejected() -> None:
    inner_mega = KernelContractV3(
        op_name="inner_mega",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=_io(2, 1),
        granularity=Granularity.MEGA,
        orchestration=OrchestrationSpec(dispatch=DispatchSpec(model=DispatchModel.PERSISTENT)),
        body=(_stub_normal_compute(),),
    )
    with pytest.raises(ValueError, match="must not contain nested MEGA"):
        KernelContractV3(
            op_name="bad", archetype=KernelArchetype.COMPUTE_TILED,
            io=_io(2, 1),
            granularity=Granularity.MEGA,
            orchestration=OrchestrationSpec(dispatch=DispatchSpec(model=DispatchModel.PERSISTENT)),
            body=(inner_mega,),
        )


def test_mega_sub_with_dram_tier_rejected() -> None:
    """Megakernel intermediates must stay in fast memory."""
    bad_sub = KernelContractV3(
        op_name="bad_sub",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(
                TensorIO(name="a", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),
                TensorIO(name="b", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),
            ),
            outputs=(TensorIO(name="o", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),),
        ),
        orchestration=OrchestrationSpec(
            memory=MemorySpec(
                input_tiers=(MemoryTier.DEVICE_DRAM, MemoryTier.SCRATCHPAD),
                output_tiers=(MemoryTier.SCRATCHPAD,),
            ),
        ),
    )
    with pytest.raises(ValueError, match="megakernel intermediates stay resident"):
        KernelContractV3(
            op_name="bad", archetype=KernelArchetype.COMPUTE_TILED,
            io=_io(2, 1),
            granularity=Granularity.MEGA,
            orchestration=OrchestrationSpec(dispatch=DispatchSpec(model=DispatchModel.PERSISTENT)),
            body=(bad_sub,),
        )


def test_mega_with_out_of_range_internal_event_rejected() -> None:
    sub = _stub_normal_compute()
    with pytest.raises(ValueError, match="producer_idx=5 out of range"):
        KernelContractV3(
            op_name="bad", archetype=KernelArchetype.COMPUTE_TILED,
            io=_io(2, 1),
            granularity=Granularity.MEGA,
            orchestration=OrchestrationSpec(dispatch=DispatchSpec(model=DispatchModel.PERSISTENT)),
            body=(sub,),
            internal_events=(InternalEventEdge(event_name="x", producer_idx=5, consumer_idx=0),),
        )


# ---------------------------------------------------------------------------
# Granularity invariants — NORMAL
# ---------------------------------------------------------------------------


def test_normal_with_inline_dispatch_rejected() -> None:
    with pytest.raises(ValueError, match="NORMAL.*must not use DispatchModel.INLINE"):
        KernelContractV3(
            op_name="bad", archetype=KernelArchetype.POINTWISE,
            io=_normal_micro_io(),
            granularity=Granularity.NORMAL,
            orchestration=OrchestrationSpec(dispatch=DispatchSpec(model=DispatchModel.INLINE)),
        )


def test_normal_with_body_rejected() -> None:
    with pytest.raises(ValueError, match="NORMAL.*body/internal_events.*MEGA-only"):
        KernelContractV3(
            op_name="bad", archetype=KernelArchetype.POINTWISE,
            io=_normal_micro_io(),
            granularity=Granularity.NORMAL,
            body=(_stub_normal_compute(),),
        )


# ---------------------------------------------------------------------------
# Memory-tier arity + ShapeClass invariants (preserved)
# ---------------------------------------------------------------------------


def test_memory_input_tier_arity_must_match_inputs() -> None:
    with pytest.raises(ValueError, match="memory.input_tiers length"):
        KernelContractV3(
            op_name="x", archetype=KernelArchetype.POINTWISE,
            io=_io(2, 1),
            orchestration=OrchestrationSpec(
                memory=MemorySpec(input_tiers=(MemoryTier.SCRATCHPAD,)),
            ),
        )


def test_io_with_duplicate_operand_names_rejected() -> None:
    bad = TensorIO(name="dup", shape=ShapeClass(dims=(None,)), dtype_class=("f32",))
    with pytest.raises(ValueError, match="duplicate IO operand names"):
        IOContract(inputs=(bad,), outputs=(bad,))


def test_io_without_outputs_rejected() -> None:
    with pytest.raises(ValueError, match="at least one output"):
        IOContract(inputs=(), outputs=())


def test_shape_class_max_dims_must_align() -> None:
    with pytest.raises(ValueError, match="max_dims.*align"):
        ShapeClass(dims=(None, None), max_dims=(64,))


def test_shape_class_divisibility_must_align() -> None:
    with pytest.raises(ValueError, match="divisibility.*align"):
        ShapeClass(dims=(None, None), divisibility=(32,))


# ---------------------------------------------------------------------------
# Default safety — orchestration defaults are bulk-sync friendly
# ---------------------------------------------------------------------------


def test_orchestration_defaults_are_safe() -> None:
    o = OrchestrationSpec()
    assert o.sync.event_decls == ()
    assert o.sync.wait_on == ()
    assert o.dispatch.model is DispatchModel.ASYNC
    assert o.fusion.is_boundary is False
    assert o.memory.in_place_safe is False
