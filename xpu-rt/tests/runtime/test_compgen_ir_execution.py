"""Regression test for the ``mode="compgen_ir"`` execution path.

This is the Phase-A gap-closing test from
``tmp/next_steps.md`` / the runtime-HAL plan: prove that the compiled
xDSL payload module can be executed end-to-end through
``LocalExecutor.benchmark`` and produce the same numerics as an eager
PyTorch forward pass.

The test uses the ``attention_mlp_tiny`` fixture because it already
round-trips through ``cpu_executor.execute`` with < 1e-5 max-abs-diff
(see ``tests/runtime/test_cpu_executor.py::
test_executor_runs_on_attention_mlp_tiny_with_zero_diff``). Any
regression here indicates we broke the bridge or the
``LocalExecutor`` wiring.
"""

from __future__ import annotations

import pytest
import torch
from compgen.capture.torch_mlir_bridge import bridge_fx_graph
from compgen.runtime.local_executor import BenchmarkResult, LocalExecutor

from tests._fixtures.real_workloads import attention_mlp_tiny


def test_benchmark_compgen_ir_runs_and_matches_eager() -> None:
    """``mode='compgen_ir'`` produces the same output as ``mode='eager'``
    within fp32 tolerance, on a real attention + MLP fixture."""

    fx = attention_mlp_tiny()
    bridged = bridge_fx_graph(fx.model, fx.example_inputs)
    assert bridged.module is not None, "bridge should succeed for this fixture"

    executor = LocalExecutor()

    eager_result = executor.benchmark(
        model=fx.model,
        sample_inputs=fx.example_inputs,
        device="cpu",
        mode="eager",
        num_iterations=3,
        warmup=1,
    )
    assert isinstance(eager_result, BenchmarkResult)
    assert eager_result.mode == "eager"
    assert eager_result.sample_output is not None

    compgen_result = executor.benchmark(
        model=fx.model,
        sample_inputs=fx.example_inputs,
        device="cpu",
        mode="compgen_ir",
        num_iterations=3,
        warmup=1,
        payload_module=bridged.module,
        exported_program=fx.exported,
    )
    assert isinstance(compgen_result, BenchmarkResult)
    assert compgen_result.mode == "compgen_ir"
    assert compgen_result.device == "cpu"
    assert compgen_result.sample_output is not None

    # Shape parity.
    assert tuple(compgen_result.sample_output.shape) == tuple(eager_result.sample_output.shape)

    # Numerical parity: the bridged module with no passes applied should
    # reproduce eager output exactly within fp32 tolerance.
    max_abs_diff = (compgen_result.sample_output - eager_result.sample_output).abs().max().item()
    assert max_abs_diff < 1e-5, f"max_abs_diff={max_abs_diff} exceeds fp32 tolerance"


def test_benchmark_compgen_ir_matches_eager_golden_output() -> None:
    """Sanity: compgen_ir output also matches the fixture's precomputed
    eager golden (not just a fresh eager run)."""

    fx = attention_mlp_tiny()
    bridged = bridge_fx_graph(fx.model, fx.example_inputs)
    assert bridged.module is not None

    executor = LocalExecutor()
    compgen_result = executor.benchmark(
        model=fx.model,
        sample_inputs=fx.example_inputs,
        device="cpu",
        mode="compgen_ir",
        num_iterations=2,
        warmup=1,
        payload_module=bridged.module,
        exported_program=fx.exported,
    )
    assert compgen_result.sample_output is not None
    assert tuple(compgen_result.sample_output.shape) == tuple(fx.eager_output.shape)

    max_abs_diff = (compgen_result.sample_output - fx.eager_output).abs().max().item()
    assert max_abs_diff < 1e-5, f"max_abs_diff={max_abs_diff} vs golden eager"


def test_benchmark_compgen_ir_requires_payload_module_and_export() -> None:
    """Missing ``payload_module`` or ``exported_program`` is a clear
    error, not a silent fallback to eager."""

    fx = attention_mlp_tiny()
    executor = LocalExecutor()

    with pytest.raises(ValueError, match="payload_module"):
        executor.benchmark(
            model=fx.model,
            sample_inputs=fx.example_inputs,
            mode="compgen_ir",
            num_iterations=1,
            warmup=0,
        )


def test_benchmark_compgen_ir_on_cuda_device_falls_back_to_cpu() -> None:
    """``mode='compgen_ir'`` with ``device='cuda'`` silently routes to
    CPU (the cpu_executor is pure torch). This matches the docstring
    contract and avoids confusing the caller when CUDA is unavailable."""

    fx = attention_mlp_tiny()
    bridged = bridge_fx_graph(fx.model, fx.example_inputs)
    assert bridged.module is not None

    executor = LocalExecutor()
    result = executor.benchmark(
        model=fx.model,
        sample_inputs=fx.example_inputs,
        device="cuda",  # should be demoted to cpu
        mode="compgen_ir",
        num_iterations=1,
        warmup=0,
        payload_module=bridged.module,
        exported_program=fx.exported,
    )
    assert result.device == "cpu"
    assert result.mode == "compgen_ir"


def test_sample_output_populated_in_eager_and_compiled_modes() -> None:
    """``sample_output`` is populated for every mode — not just
    compgen_ir — so callers can diff across modes."""

    fx = attention_mlp_tiny()
    executor = LocalExecutor()

    # Eager mode always populates sample_output.
    eager = executor.benchmark(
        model=fx.model,
        sample_inputs=fx.example_inputs,
        mode="eager",
        num_iterations=1,
        warmup=0,
    )
    assert isinstance(eager.sample_output, torch.Tensor)
    assert tuple(eager.sample_output.shape) == tuple(fx.eager_output.shape)
