"""Tests for autocomp adapter integration."""

from __future__ import annotations

from compgen.agent.analyzer import PatternCluster
from compgen.kernels.autocomp_adapter import AutocompAdapter, KernelResult, _generate_reference_code
from compgen.targets.schema import DeviceSpec, TargetProfile


def test_adapter_instantiation() -> None:
    """AutocompAdapter should be instantiable with defaults."""
    adapter = AutocompAdapter()
    assert adapter.beam_size == 4
    assert adapter.max_iterations == 10


def _make_gpu_target() -> TargetProfile:
    """Create a minimal TargetProfile with a GPU device."""
    gpu = DeviceSpec(device_type="gpu", name="A100-SXM4-80GB", vendor="nvidia")
    return TargetProfile(name="test-gpu", devices=[gpu])


def test_adapter_translate_profile() -> None:
    """_generate_reference_code should produce reference code for known patterns."""
    cluster = PatternCluster(
        cluster_id="cluster_0",
        pattern_type="linear",
        node_names=("linear_0",),
        total_flops=1024,
        total_bytes=512,
        arithmetic_intensity=2.0,
        estimated_latency_per_device={"gpu": 0.5},
        best_device="gpu",
        is_bottleneck=False,
        kernel_opportunity="fused linear kernel",
        input_shapes={"x": (8, 768)},
        output_shapes={"out": (8, 768)},
    )

    ref_code = _generate_reference_code(cluster)
    assert "import torch" in ref_code
    assert "def test" in ref_code
    assert "F.linear" in ref_code


def test_adapter_search_kernel() -> None:
    """KernelResult should be constructible and _generate_reference_code handles linear_chain."""
    # Constructing a KernelResult directly to verify the dataclass
    result = KernelResult(
        cluster_id="cluster_0",
        kernel_code="# kernel code",
        language="cuda",
        latency_us=10.5,
        correct=True,
        speedup_vs_baseline=1.5,
        iterations_used=5,
        total_candidates=20,
        search_cost_tokens=1000,
        plan="fuse linear+gelu",
    )
    assert result.cluster_id == "cluster_0"
    assert result.correct is True
    assert result.speedup_vs_baseline == 1.5

    # Test reference code generation for linear_chain pattern
    cluster = PatternCluster(
        cluster_id="cluster_1",
        pattern_type="linear_chain",
        node_names=("linear_0", "gelu_0", "linear_1"),
        total_flops=2048,
        total_bytes=1024,
        arithmetic_intensity=2.0,
        estimated_latency_per_device={"gpu": 1.0},
        best_device="gpu",
        is_bottleneck=True,
        kernel_opportunity="fused MLP kernel",
        input_shapes={"x": (8, 768)},
        output_shapes={"out": (8, 768)},
    )

    ref_code = _generate_reference_code(cluster)
    assert "import torch" in ref_code
    assert "F.linear" in ref_code
    assert "F.gelu" in ref_code
