"""Tests for ukernel dialect ops."""

from __future__ import annotations

from compgen.ir.ukernel.ops import UkernelCallOp, UkernelDeclOp


def test_ukernel_decl() -> None:
    decl = UkernelDeclOp(kernel_name="my_matmul", input_types=["tensor<128x128xf32>"], calling_convention="triton")
    assert decl.kernel_name == "my_matmul"
    assert decl.calling_convention == "triton"


def test_ukernel_call() -> None:
    call = UkernelCallOp(kernel_name="my_matmul", operands=["a", "b"], results=["c"], workspace_bytes=4096)
    assert call.workspace_bytes == 4096
