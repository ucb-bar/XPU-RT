"""Tests for :mod:`compgen.promotion.contract_hash`."""

from __future__ import annotations

from compgen.kernels.contract_v3 import (
    ConcurrencyUnit,
    DispatchModel,
    DispatchSpec,
    ExecutionEnvelope,
    FusionPolicy,
    Granularity,
    HardwareEnvelope,
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
    SelectionHints,
    ShapeClass,
    SyncSpec,
    TensorIO,
)
from compgen.promotion.contract_hash import hash_contract


def _make_io(name: str, dim_a: int, dim_b: int) -> TensorIO:
    return TensorIO(
        name=name,
        shape=ShapeClass(dims=(dim_a, dim_b)),
        dtype_class=("fp32",),
        layout=LayoutKind.ROW_MAJOR,
        alignment_bytes=16,
    )


def _make_contract(
    *,
    dim_a: int = 16,
    dim_b: int = 16,
    fusion_aggressiveness: float = 0.0,
    observability_period: int = 0,
) -> KernelContractV3:
    """Build a minimal-but-valid KernelContractV3 for hashing tests.

    ``fusion_aggressiveness`` and ``observability_period`` are
    compiler-only knobs — varying them must not change the hash.
    """
    return KernelContractV3(
        op_name="matmul_tile",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(_make_io("a", dim_a, dim_b), _make_io("b", dim_a, dim_b)),
            outputs=(_make_io("c", dim_a, dim_b),),
            numerics=NumericsSpec(accumulator_dtype="fp32"),
        ),
        granularity=Granularity.NORMAL,
        orchestration=OrchestrationSpec(
            execution=ExecutionEnvelope(
                hardware=HardwareEnvelope(
                    target_name="host_cpu",
                    vector_lanes=8,
                    scratchpad_bytes=0,
                    register_bytes=256,
                    native_dtypes=("fp32",),
                ),
                memory_budget_bytes=1024,
                concurrency_unit=ConcurrencyUnit.HOST_THREAD,
                padding=PaddingPolicy.NONE,
                priority=PerformancePriority.LATENCY,
            ),
            sync=SyncSpec(),
            memory=MemorySpec(
                input_tiers=(MemoryTier.DEVICE_DRAM, MemoryTier.DEVICE_DRAM),
                output_tiers=(MemoryTier.DEVICE_DRAM,),
                in_place_safe=False,
            ),
            fusion=FusionPolicy(),
            dispatch=DispatchSpec(model=DispatchModel.ASYNC),
            observability=ObservabilitySpec(cost_emit_period=observability_period),
        ),
        selection=SelectionHints(),
    )


def test_hash_is_deterministic_across_runs() -> None:
    """Hashing the same contract twice yields the same digest."""
    h1 = hash_contract(_make_contract())
    h2 = hash_contract(_make_contract())
    assert h1 == h2


def test_hash_changes_when_io_shape_changes() -> None:
    """Different IO shapes must hash to different keys — otherwise the
    cache would silently reuse the wrong kernel."""
    h_small = hash_contract(_make_contract(dim_a=16, dim_b=16))
    h_large = hash_contract(_make_contract(dim_a=32, dim_b=32))
    assert h_small != h_large


def test_hash_invariant_to_compiler_only_fields() -> None:
    """Changing compiler-only fields (observability cadence) must not
    change the hash — the kernel codegen sees identical inputs."""
    h_quiet = hash_contract(_make_contract(observability_period=0))
    h_loud = hash_contract(_make_contract(observability_period=100))
    assert h_quiet == h_loud


def test_hash_is_truncated_hex_16_chars() -> None:
    """Hash output is short enough to embed in directory names."""
    h = hash_contract(_make_contract())
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
