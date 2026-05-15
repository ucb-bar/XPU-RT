"""Tests for ``compgen.agent.kernel_optimizer``.

Locks in:
  * fingerprint_for is stable across instances + matches the MCP scheme
  * optimize_model with a no-op codegen + bench produces decisions
    for every contract, persists records to KernelDB, and returns
    a callable forward
  * cache hits skip codegen on the second call (same fingerprint)
  * optimize_model_multi_target produces one OptimizedModel per target
    with the per-target adapter and target name correctly set
  * the W6 loop falls back to model_fn when the adapter declines to
    capture (CPU adapter returns None for capture_graph)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.agent.hw_aware_dispatch import TargetDispatchDecision
from compgen.agent.kernel_optimizer import (
    BenchResult,
    CodegenResult,
    KernelDecision,
    OptimizedModel,
    fingerprint_for,
    optimize_model,
    optimize_model_multi_target,
)
from compgen.kernels.contract_v3 import (
    ExecutionEnvelope,
    HardwareEnvelope,
    IOContract,
    KernelArchetype,
    KernelContractV3,
    OrchestrationSpec,
    ShapeClass,
    TensorIO,
)
from compgen.memory.kernel_db import KernelDB

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _envelope(target: str = "cuda-a100") -> HardwareEnvelope:
    return HardwareEnvelope(
        target_name=target,
        vector_lanes=64,
        scratchpad_bytes=49152,
        register_bytes=256,
        native_dtypes=("f16", "f32"),
        peak_bandwidth_gbps=672.0,
    )


def _matmul(target: str = "cuda-a100") -> KernelContractV3:
    env = _envelope(target)
    return KernelContractV3(
        op_name="matmul",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(
                TensorIO(name="lhs", shape=ShapeClass(dims=(None, None)), dtype_class=("f16",)),
                TensorIO(name="rhs", shape=ShapeClass(dims=(None, None)), dtype_class=("f16",)),
            ),
            outputs=(TensorIO(name="out", shape=ShapeClass(dims=(None, None)), dtype_class=("f16",)),),
        ),
        orchestration=OrchestrationSpec(execution=ExecutionEnvelope(hardware=env)),
    )


def _pointwise(target: str = "cuda-a100", op: str = "addf") -> KernelContractV3:
    env = _envelope(target)
    return KernelContractV3(
        op_name=op,
        archetype=KernelArchetype.POINTWISE,
        io=IOContract(
            inputs=(
                TensorIO(name="a", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),
                TensorIO(name="b", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),
            ),
            outputs=(TensorIO(name="o", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),),
        ),
        orchestration=OrchestrationSpec(execution=ExecutionEnvelope(hardware=env)),
    )


@pytest.fixture
def isolated_db(tmp_path: Path) -> KernelDB:
    return KernelDB(path=tmp_path / "kernel_db.sqlite")


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_for_identical_contracts() -> None:
    a = _matmul()
    b = _matmul()
    assert fingerprint_for(a) == fingerprint_for(b)


def test_fingerprint_changes_with_target() -> None:
    a = _matmul("cuda-a100")
    b = _matmul("rocm-mi250")
    assert fingerprint_for(a) != fingerprint_for(b)


def test_fingerprint_changes_with_op_name() -> None:
    a = _pointwise(op="addf")
    b = _pointwise(op="mulf")
    assert fingerprint_for(a) != fingerprint_for(b)


def test_fingerprint_matches_mcp_scheme() -> None:
    """The MCP cache uses ``compgen.mcp.tools.kernel.contract_fingerprint``;
    so does the optimizer. They must match for the disk cache to hit
    across MCP-generated and headless-generated kernels."""
    from compgen.agent.kernel_optimizer import _v3_to_fingerprint_dict
    from compgen.mcp.tools.kernel import contract_fingerprint

    c = _matmul()
    assert fingerprint_for(c) == contract_fingerprint(_v3_to_fingerprint_dict(c))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_optimize_model_requires_at_least_one_contract(isolated_db) -> None:
    with pytest.raises(ValueError, match="at least one contract"):
        optimize_model(model_fn=None, target="cuda-a100", contracts=[], db=isolated_db)


def test_optimize_model_multi_target_requires_at_least_one_target(isolated_db) -> None:
    with pytest.raises(ValueError, match="at least one target"):
        optimize_model_multi_target(
            model_fn=None,
            targets=[],
            contracts=[_matmul()],
            db=isolated_db,
        )


# ---------------------------------------------------------------------------
# End-to-end loop with stub codegen
# ---------------------------------------------------------------------------


def _stub_codegen(contract, decision: TargetDispatchDecision) -> CodegenResult:
    return CodegenResult(
        callable_kernel=lambda *a, **kw: 42,
        provider_name="stub_provider",
        source=f"# stub for {contract.op_name}",
        language="python",
    )


def _stub_bench(contract, result) -> BenchResult:
    return BenchResult(perf_us=10.0, correct=True, notes="ok")


def test_optimize_model_runs_loop_and_returns_decisions(isolated_db) -> None:
    contracts = [_matmul(), _pointwise()]
    optim = optimize_model(
        model_fn=None,
        target="cuda-a100",
        contracts=contracts,
        codegen_fn=_stub_codegen,
        bench_fn=_stub_bench,
        db=isolated_db,
    )
    assert isinstance(optim, OptimizedModel)
    assert optim.target == "cuda-a100"
    assert optim.adapter_name == "cuda"
    assert len(optim.decisions) == 2
    for d in optim.decisions:
        assert isinstance(d, KernelDecision)
        assert d.cached is False  # first run — nothing cached
        assert d.provider_name == "stub_provider"
        assert d.perf_us == 10.0


def test_optimize_model_persists_to_kernel_db(isolated_db) -> None:
    optim = optimize_model(
        model_fn=None,
        target="cuda-a100",
        contracts=[_matmul()],
        codegen_fn=_stub_codegen,
        bench_fn=_stub_bench,
        db=isolated_db,
    )
    fp = optim.decisions[0].fingerprint
    rec = isolated_db.best_kernel_perf("cuda-a100", "compute_tiled", fp)
    assert rec is not None
    assert rec.perf_us == 10.0
    assert rec.correctness_passed


def test_second_run_hits_cache(isolated_db) -> None:
    contracts = [_matmul()]
    # First run — populates cache
    optimize_model(
        model_fn=None,
        target="cuda-a100",
        contracts=contracts,
        codegen_fn=_stub_codegen,
        bench_fn=_stub_bench,
        db=isolated_db,
    )
    # Second run — same contract, should hit the cache.
    second = optimize_model(
        model_fn=None,
        target="cuda-a100",
        contracts=contracts,
        codegen_fn=_stub_codegen,
        bench_fn=_stub_bench,
        db=isolated_db,
    )
    assert second.decisions[0].cached is True
    assert second.decisions[0].provider_name == "cache"


def test_optimize_model_summary_includes_cache_count(isolated_db) -> None:
    optimize_model(
        model_fn=None,
        target="cuda-a100",
        contracts=[_matmul()],
        codegen_fn=_stub_codegen,
        bench_fn=_stub_bench,
        db=isolated_db,
    )
    second = optimize_model(
        model_fn=None,
        target="cuda-a100",
        contracts=[_matmul()],
        codegen_fn=_stub_codegen,
        bench_fn=_stub_bench,
        db=isolated_db,
    )
    summary = second.summary()
    assert "regions optimised: 1" in summary
    assert "cache hits:        1" in summary


# ---------------------------------------------------------------------------
# Forward callable
# ---------------------------------------------------------------------------


def test_optimize_model_returns_model_fn_when_capture_unsupported(isolated_db) -> None:
    """CPU adapter returns None for capture_graph → forward is the
    user-supplied model_fn, not a graph replay wrapper."""

    def my_model(x):
        return x * 2

    optim = optimize_model(
        model_fn=my_model,
        target="cpu-host",
        contracts=[_pointwise("cpu-host")],
        codegen_fn=_stub_codegen,
        bench_fn=_stub_bench,
        db=isolated_db,
        sample_inputs=(7,),
    )
    assert optim.captured_graph is None
    assert optim.forward is my_model
    assert optim.forward(5) == 10


def test_optimize_model_with_no_model_fn_returns_noop_callable(isolated_db) -> None:
    optim = optimize_model(
        model_fn=None,
        target="cuda-a100",
        contracts=[_matmul()],
        codegen_fn=_stub_codegen,
        bench_fn=_stub_bench,
        db=isolated_db,
    )
    # forward is callable but returns None (placeholder).
    assert callable(optim.forward)
    assert optim.forward() is None


# ---------------------------------------------------------------------------
# Multi-target surface (W6.3)
# ---------------------------------------------------------------------------


def test_multi_target_returns_one_model_per_target(isolated_db) -> None:
    contracts = [_matmul()]
    out = optimize_model_multi_target(
        model_fn=None,
        targets=["cuda-a100", "cpu-host"],
        contracts=contracts,
        codegen_fn=_stub_codegen,
        bench_fn=_stub_bench,
        db=isolated_db,
    )
    assert set(out.keys()) == {"cuda-a100", "cpu-host"}
    assert out["cuda-a100"].adapter_name == "cuda"
    assert out["cpu-host"].adapter_name == "cpu"


def test_multi_target_cache_is_per_target(isolated_db) -> None:
    """Same contract optimised under two targets must hit the cache
    independently per target — fingerprint includes target."""
    contracts = [_matmul()]
    out1 = optimize_model_multi_target(
        model_fn=None,
        targets=["cuda-a100", "cpu-host"],
        contracts=contracts,
        codegen_fn=_stub_codegen,
        bench_fn=_stub_bench,
        db=isolated_db,
    )
    # First run — neither cached.
    assert out1["cuda-a100"].decisions[0].cached is False
    assert out1["cpu-host"].decisions[0].cached is False
    out2 = optimize_model_multi_target(
        model_fn=None,
        targets=["cuda-a100", "cpu-host"],
        contracts=contracts,
        codegen_fn=_stub_codegen,
        bench_fn=_stub_bench,
        db=isolated_db,
    )
    # Second run — both should hit cache, but ONLY because we used
    # different envelopes per target.
    assert out2["cuda-a100"].decisions[0].cached is True
    assert out2["cpu-host"].decisions[0].cached is True


def test_multi_target_propagates_explicit_envelopes(isolated_db) -> None:
    contracts = [_matmul()]
    cuda_env = HardwareEnvelope(
        target_name="cuda-a100",
        vector_lanes=128,
        scratchpad_bytes=99000,
        register_bytes=512,
        native_dtypes=("f16",),
        peak_bandwidth_gbps=2000.0,
    )
    cpu_env = HardwareEnvelope(
        target_name="cpu-host",
        vector_lanes=8,
        scratchpad_bytes=4096,
        register_bytes=128,
        native_dtypes=("f32",),
        peak_bandwidth_gbps=50.0,
    )
    out = optimize_model_multi_target(
        model_fn=None,
        targets=["cuda-a100", "cpu-host"],
        contracts=contracts,
        envelopes=[cuda_env, cpu_env],
        codegen_fn=_stub_codegen,
        bench_fn=_stub_bench,
        db=isolated_db,
    )
    # The metadata records the envelope target the loop saw.
    assert out["cuda-a100"].metadata["envelope_target"] == "cuda-a100"
    assert out["cpu-host"].metadata["envelope_target"] == "cpu-host"
