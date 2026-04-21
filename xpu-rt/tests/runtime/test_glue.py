"""Tests for ``compgen.runtime.glue``.

Locks in:
  * select_adapter walks the same target taxonomy as the knowledge store
  * each adapter satisfies the RuntimeAdapter Protocol
  * supports() correctly accepts/rejects per dispatch model
  * CudaAdapter.dispatch + CpuAdapter.dispatch round-trip a callable
  * INLINE dispatch is rejected by CUDA adapter (it's a codegen-only model)
"""

from __future__ import annotations

import pytest

from compgen.kernels.contract_v3 import (
    DispatchModel, DispatchSpec, Granularity, IOContract, KernelArchetype,
    KernelContractV3, MemorySpec, MemoryTier, OrchestrationSpec, ShapeClass, TensorIO,
)
from compgen.runtime.glue import (
    BaremetalRuntimeAdapter,
    Buffer, BufferSpec,
    CpuRuntimeAdapter,
    CudaRuntimeAdapter,
    DispatchResult,
    RocmRuntimeAdapter,
    RuntimeAdapter,
    select_adapter,
)


def _io(n_in: int = 1, n_out: int = 1) -> IOContract:
    return IOContract(
        inputs=tuple(
            TensorIO(name=f"in{i}", shape=ShapeClass(dims=(None,)), dtype_class=("f32",))
            for i in range(n_in)
        ),
        outputs=tuple(
            TensorIO(name=f"out{i}", shape=ShapeClass(dims=(None,)), dtype_class=("f32",))
            for i in range(n_out)
        ),
    )


def _contract(model: DispatchModel) -> KernelContractV3:
    """Build a contract honoring v3 invariants:
       INLINE → MICRO + register/scratchpad tiers
       PERSISTENT → MEGA (which requires body[]) — for runtime tests
                    we use a minimal MEGA contract with a NORMAL stub body
       SYNC/ASYNC → NORMAL (default).
    """
    if model is DispatchModel.INLINE:
        # MICRO ukernel: register-resident, no events.
        return KernelContractV3(
            op_name="test", archetype=KernelArchetype.POINTWISE,
            io=_io(),
            granularity=Granularity.MICRO,
            orchestration=OrchestrationSpec(
                dispatch=DispatchSpec(model=DispatchModel.INLINE),
                memory=MemorySpec(
                    input_tiers=(MemoryTier.REGISTER,),
                    output_tiers=(MemoryTier.REGISTER,),
                ),
            ),
        )
    if model is DispatchModel.PERSISTENT:
        # MEGA contract with a NORMAL stub sub-kernel
        sub = KernelContractV3(
            op_name="sub", archetype=KernelArchetype.POINTWISE,
            io=_io(),
            orchestration=OrchestrationSpec(
                memory=MemorySpec(
                    input_tiers=(MemoryTier.SCRATCHPAD,),
                    output_tiers=(MemoryTier.SCRATCHPAD,),
                ),
            ),
        )
        return KernelContractV3(
            op_name="test", archetype=KernelArchetype.COMPUTE_TILED,
            io=IOContract(
                inputs=tuple(
                    TensorIO(name=f"in{i}", shape=ShapeClass(dims=(None,)),
                             dtype_class=("f32",))
                    for i in range(2)
                ),
                outputs=tuple(
                    TensorIO(name=f"out{i}", shape=ShapeClass(dims=(None,)),
                             dtype_class=("f32",))
                    for i in range(1)
                ),
            ),
            granularity=Granularity.MEGA,
            orchestration=OrchestrationSpec(
                dispatch=DispatchSpec(model=DispatchModel.PERSISTENT),
            ),
            body=(sub,),
        )
    return KernelContractV3(
        op_name="test", archetype=KernelArchetype.POINTWISE,
        io=_io(),
        orchestration=OrchestrationSpec(dispatch=DispatchSpec(model=model)),
    )


# ---------------------------------------------------------------------------
# Factory walks target taxonomy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target,expected_name", [
    ("cuda-a100",       "cuda"),
    ("cuda-titan-rtx",  "cuda"),
    ("test-gpu-simt",   "cuda"),
    ("rocm-mi250",      "rocm"),
    ("hexagon-v69",     "baremetal"),
    ("openq_5165rb",    "baremetal"),
    ("trainium1",       "baremetal"),
    ("cpu-host",        "cpu"),
    ("riscv-soc",       "cpu"),
    ("totally-unknown", "cpu"),
])
def test_select_adapter_matches_target_taxonomy(target: str, expected_name: str) -> None:
    adapter = select_adapter(target)
    assert adapter.name == expected_name
    assert isinstance(adapter, RuntimeAdapter)


# ---------------------------------------------------------------------------
# supports() — per dispatch model
# ---------------------------------------------------------------------------


def test_cpu_adapter_does_not_support_persistent_or_inline() -> None:
    cpu = CpuRuntimeAdapter()
    assert cpu.supports(_contract(DispatchModel.SYNC))
    assert cpu.supports(_contract(DispatchModel.ASYNC))
    assert not cpu.supports(_contract(DispatchModel.PERSISTENT))
    assert not cpu.supports(_contract(DispatchModel.INLINE))


def test_cuda_adapter_supports_all_dispatch_models() -> None:
    cuda = CudaRuntimeAdapter()
    for m in DispatchModel:
        assert cuda.supports(_contract(m)), f"CUDA adapter rejected {m!r}"


def test_baremetal_supports_sync_async_inline_not_persistent() -> None:
    bm = BaremetalRuntimeAdapter()
    assert bm.supports(_contract(DispatchModel.SYNC))
    assert bm.supports(_contract(DispatchModel.ASYNC))
    assert bm.supports(_contract(DispatchModel.INLINE))
    assert not bm.supports(_contract(DispatchModel.PERSISTENT))


def test_rocm_skeleton_does_not_support_anything_yet() -> None:
    rocm = RocmRuntimeAdapter()
    for m in DispatchModel:
        assert not rocm.supports(_contract(m))


# ---------------------------------------------------------------------------
# Dispatch round-trip
# ---------------------------------------------------------------------------


def test_cpu_dispatch_round_trips_a_callable() -> None:
    cpu = CpuRuntimeAdapter()
    result = cpu.dispatch(
        _contract(DispatchModel.SYNC),
        callable_kernel=lambda x: x * 2,
        args=(21,), kwargs={},
    )
    assert result.output == 42
    assert result.adapter_name == "cpu"


def test_cuda_dispatch_rejects_inline_model() -> None:
    cuda = CudaRuntimeAdapter()
    with pytest.raises(ValueError, match="INLINE"):
        cuda.dispatch(
            _contract(DispatchModel.INLINE),
            callable_kernel=lambda: None,
            args=(), kwargs={},
        )


def test_buffer_spec_carries_tier() -> None:
    spec = BufferSpec(nbytes=1024, dtype="f16", tier=MemoryTier.SCRATCHPAD)
    assert spec.tier is MemoryTier.SCRATCHPAD
    assert spec.nbytes == 1024


# ---------------------------------------------------------------------------
# CPU adapter graph capture is a no-op
# ---------------------------------------------------------------------------


def test_cpu_adapter_does_not_capture_graphs() -> None:
    cpu = CpuRuntimeAdapter()
    assert cpu.capture_graph(lambda x: x, ()) is None


def test_cpu_synchronize_is_a_noop() -> None:
    cpu = CpuRuntimeAdapter()
    cpu.synchronize()    # must not raise
