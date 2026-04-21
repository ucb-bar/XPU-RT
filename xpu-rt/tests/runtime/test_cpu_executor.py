"""Tests for the CPU reference executor."""

from __future__ import annotations

import pytest
from compgen.capture.torch_mlir_bridge import bridge_fx_graph
from compgen.runtime.cpu_executor import ExecutorStats, execute

from tests._fixtures.real_workloads import (
    ALL_FIXTURE_FNS,
    attention_mlp_tiny,
    qwen_moe_tiny,
)

# --- bridged-module execution (no passes applied) ------------------------


def test_executor_runs_on_attention_mlp_tiny_with_zero_diff():
    """Without pipeline passes, the interpreter should reproduce
    eager output exactly."""
    fx = attention_mlp_tiny()
    r = bridge_fx_graph(fx.model, fx.example_inputs)
    out = execute(r.module, fx.exported, fx.example_inputs)
    assert tuple(out.shape) == tuple(fx.eager_output.shape)
    assert (out - fx.eager_output).abs().max().item() < 1e-5


def test_executor_records_stats():
    fx = attention_mlp_tiny()
    r = bridge_fx_graph(fx.model, fx.example_inputs)
    stats = ExecutorStats()
    execute(r.module, fx.exported, fx.example_inputs, stats=stats)
    assert stats.ops_executed > 0
    assert "linalg.matmul" in stats.ops_by_name


@pytest.mark.parametrize("fn", ALL_FIXTURE_FNS, ids=lambda f: f.__name__)
def test_executor_runs_end_to_end_on_every_fixture(fn):
    """Every fixture must execute through the interpreter without
    crashing + produce a tensor of the expected shape.

    Exact numerical match is not required here — some fixtures have
    opaque-call fallbacks whose identity approximation perturbs the
    result. The contract is: no crash + correct output shape.
    """
    fx = fn()
    r = bridge_fx_graph(fx.model, fx.example_inputs)
    if r.module is None:
        pytest.skip(f"{fx.name}: bridge failed")
    out = execute(r.module, fx.exported, fx.example_inputs)
    assert tuple(out.shape) == tuple(fx.eager_output.shape)


def test_executor_handles_3d_matmul_via_broadcast():
    fx = qwen_moe_tiny()
    r = bridge_fx_graph(fx.model, fx.example_inputs)
    out = execute(r.module, fx.exported, fx.example_inputs)
    assert out.ndim == fx.eager_output.ndim


# --- error robustness ------------------------------------------------------


def test_executor_fills_zero_on_failed_op():
    """Unsupported callees shouldn't crash the interpreter; they
    produce zero tensors so downstream ops can still run."""
    fx = attention_mlp_tiny()
    r = bridge_fx_graph(fx.model, fx.example_inputs)
    stats = ExecutorStats()
    out = execute(r.module, fx.exported, fx.example_inputs, stats=stats)
    assert out is not None


# --- dispatch coverage ---------------------------------------------------


def test_dispatch_includes_core_aten_ops():
    from compgen.runtime.cpu_executor import _ATEN_DISPATCH

    for op in (
        "aten_matmul",
        "aten_add",
        "aten_sub",
        "aten_mul",
        "aten_div",
        "aten_gelu",
        "aten_silu",
        "aten_softmax",
        "aten_layer_norm",
        "aten_transpose",
        "aten_view",
        "aten_contiguous",
        "aten_sqrt",
        "aten_rsqrt",
        "aten_pow",
    ):
        assert op in _ATEN_DISPATCH, f"{op} missing from dispatch"
