"""Tests for ukernel lowering to concrete calls."""

from __future__ import annotations

import pytest
from compgen.ir.ukernel.lower import lower_ukernel_to_call


def test_lower_ukernel_to_call_exists() -> None:
    assert callable(lower_ukernel_to_call)


def test_lower_ukernel_to_call_c_backend() -> None:
    """lower_ukernel_to_call with backend='c' should produce C function calls."""
    from compgen.ir.ukernel.ops import UkernelCallOp

    calls = [
        UkernelCallOp(kernel_name="matmul_f32", operands=["a", "b"], results=["c"], workspace_bytes=256),
    ]
    result = lower_ukernel_to_call(calls, backend="c")
    assert len(result.lowered_calls) == 1
    lowered = result.lowered_calls[0]
    assert lowered.function_name == "extern_matmul_f32"
    assert lowered.backend == "c"
    assert lowered.operands == ["a", "b"]
    assert lowered.results == ["c"]
    assert lowered.workspace_bytes == 256
    assert len(result.diagnostics) == 0


def test_lower_ukernel_to_call_triton_backend() -> None:
    """lower_ukernel_to_call with backend='triton' should produce Triton kernel launches."""
    from compgen.ir.ukernel.ops import UkernelCallOp

    calls = [
        UkernelCallOp(kernel_name="fused_attention", operands=["q", "k", "v"], results=["out"]),
        UkernelCallOp(kernel_name="layernorm", operands=["x"], results=["y"]),
    ]
    result = lower_ukernel_to_call(calls, backend="triton")
    assert len(result.lowered_calls) == 2
    assert result.lowered_calls[0].function_name == "triton_kernel_fused_attention"
    assert result.lowered_calls[1].function_name == "triton_kernel_layernorm"
    assert result.lowered_calls[0].backend == "triton"
