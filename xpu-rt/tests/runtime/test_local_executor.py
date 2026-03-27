"""Tests for runtime/local_executor.py — real hardware execution."""

from __future__ import annotations

import sys
from pathlib import Path

from compgen.runtime.local_executor import BenchmarkResult, LocalExecutor

EXAMPLES = Path(__file__).parent.parent.parent / "examples" / "models"


def _get_model_and_inputs():
    sys.path.insert(0, str(EXAMPLES))
    from simple_mlp import SimpleMLP, get_sample_inputs
    return SimpleMLP(), get_sample_inputs()


def test_benchmark_result_fields() -> None:
    r = BenchmarkResult(
        latency_median_us=100.0, latency_p99_us=120.0,
        throughput_samples_per_sec=10000.0, peak_memory_bytes=1024,
        device="cpu", mode="eager", num_iterations=100, warmup_iterations=10,
    )
    assert r.latency_median_us == 100.0
    assert r.device == "cpu"


def test_local_executor_benchmark_cpu() -> None:
    model, inputs = _get_model_and_inputs()
    executor = LocalExecutor()
    result = executor.benchmark(model, inputs, device="cpu", num_iterations=5)
    assert result.latency_median_us > 0
    assert result.device == "cpu"
    assert result.mode == "eager"


def test_local_executor_compare() -> None:
    model, inputs = _get_model_and_inputs()
    executor = LocalExecutor()
    comparison = executor.compare(model, inputs, num_iterations=5)
    assert comparison.eager_cpu is not None
    assert comparison.eager_cpu.latency_median_us > 0
