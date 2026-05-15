"""Tests for kernel strategy selection."""

from __future__ import annotations

from compgen.ir.payload.contracts import CostEstimate, KernelContract
from compgen.kernels.contracts import KernelSpec
from compgen.kernels.selector import (
    KernelSelector,
    KernelStrategy,
    select_strategies,
)
from compgen.targets.schema import load_profile


def test_kernel_strategy_values() -> None:
    assert KernelStrategy.NATIVE.value == "native"
    assert KernelStrategy.LIBRARY.value == "library"
    assert KernelStrategy.AUTOCOMP.value == "autocomp"
    assert KernelStrategy.FALLBACK.value == "fallback"
    assert KernelStrategy.UNSUPPORTED.value == "unsupported"


def _make_spec(op_name: str, flops: int = 0) -> KernelSpec:
    return KernelSpec(
        contract=KernelContract(op_name=op_name, cost=CostEstimate(flops=flops)),
    )


class TestKernelSelector:
    def test_native_for_arith_ops(self) -> None:
        target = load_profile("examples/target_profiles/cuda_a100.yaml")
        selector = KernelSelector(target=target)
        specs = [_make_spec("arith.addi"), _make_spec("arith.mulf")]
        decisions = selector.select(specs)
        assert len(decisions) == 2
        assert all(d.strategy == KernelStrategy.NATIVE for d in decisions)

    def test_library_for_matmul_on_gpu(self) -> None:
        target = load_profile("examples/target_profiles/cuda_a100.yaml")
        selector = KernelSelector(target=target)
        specs = [_make_spec("linalg.matmul", flops=100000)]
        decisions = selector.select(specs)
        assert len(decisions) == 1
        assert decisions[0].strategy == KernelStrategy.LIBRARY
        assert decisions[0].library_name == "cublas"

    def test_autocomp_for_generic_with_high_flops(self) -> None:
        target = load_profile("examples/target_profiles/cuda_a100.yaml")
        selector = KernelSelector(target=target)
        specs = [_make_spec("linalg.generic", flops=50000)]
        decisions = selector.select(specs)
        assert len(decisions) == 1
        assert decisions[0].strategy == KernelStrategy.AUTOCOMP

    def test_fallback_for_small_unknown_ops(self) -> None:
        target = load_profile("examples/target_profiles/cuda_a100.yaml")
        selector = KernelSelector(target=target)
        specs = [_make_spec("custom.unknown_op", flops=10)]
        decisions = selector.select(specs)
        assert len(decisions) == 1
        assert decisions[0].strategy == KernelStrategy.FALLBACK

    def test_every_spec_gets_decision(self) -> None:
        target = load_profile("examples/target_profiles/cuda_a100.yaml")
        specs = [
            _make_spec("arith.addi"),
            _make_spec("linalg.matmul", flops=100000),
            _make_spec("linalg.generic", flops=50000),
            _make_spec("custom.tiny", flops=5),
        ]
        decisions = select_strategies(specs, target)
        assert len(decisions) == len(specs)

    def test_decision_has_reason(self) -> None:
        target = load_profile("examples/target_profiles/cuda_a100.yaml")
        specs = [_make_spec("arith.addi")]
        decisions = select_strategies(specs, target)
        assert decisions[0].reason
