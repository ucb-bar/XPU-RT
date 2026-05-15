"""Tests for REQ-021 — ``aten.conv2d`` decomposes to im2col + matmul +
reshape so SIMT targets get conv "for free" via their matmul provider.
"""

from __future__ import annotations

import io

import torch
import torch.nn as nn
from compgen.capture.torch_export import capture_model
from compgen.ir.payload.import_fx import fx_to_xdsl
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.func import CallOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.printer import Printer


def _module_for(model: nn.Module, *args: torch.Tensor):
    ep = capture_model(model, args)
    module, _ = fx_to_xdsl(ep)
    return module


def _ir_text(module) -> str:
    buf = io.StringIO()
    Printer(stream=buf).print_op(module)
    return buf.getvalue()


def test_conv2d_decomposes_to_im2col_matmul_reshape() -> None:
    module = _module_for(nn.Conv2d(1, 8, 3).eval(), torch.randn(1, 1, 8, 8))
    text = _ir_text(module)
    # Decomposition produces all four pieces.
    assert "aten_im2col" in text
    assert "aten_flatten_weight" in text
    assert "linalg.matmul" in text
    assert "aten_reshape" in text
    # Opaque ``aten_conv2d_default`` no longer surfaces.
    assert "aten_conv2d_default" not in text
    assert "aten_convolution" not in text


def test_conv2d_matmul_carries_region_id_for_provider_dispatch() -> None:
    """The matmul region must be claimable by an existing matmul
    provider (acceptance criterion for REQ-021)."""
    module = _module_for(nn.Conv2d(1, 8, 3).eval(), torch.randn(1, 1, 8, 8))
    matmuls = [op for op in module.walk() if isinstance(op, MatmulOp)]
    assert len(matmuls) == 1
    rid = matmuls[0].attributes.get("compgen.region_id")
    did = matmuls[0].attributes.get("compgen.dispatch_id")
    assert isinstance(rid, StringAttr) and rid.data.startswith("matmul_")
    assert isinstance(did, StringAttr)


def test_conv2d_im2col_shape_matches_weight_kernel_size() -> None:
    """For ``Conv2d(1, 8, 3)`` on ``(1, 1, 8, 8)`` input:
    K = C * kH * kW = 1 * 3 * 3 = 9; output is (1, 8, 6, 6) so
    N*H'*W' = 36. The im2col output shape is (9, 36)."""
    module = _module_for(nn.Conv2d(1, 8, 3).eval(), torch.randn(1, 1, 8, 8))
    im2col = next(op for op in module.walk() if isinstance(op, CallOp) and op.callee.string_value() == "aten_im2col")
    out_shape = im2col.res[0].type.get_shape()
    assert tuple(out_shape) == (9, 36), out_shape


def test_conv2d_matmul_shape_consistent_with_im2col_and_weight() -> None:
    """``W_flat (F, K) @ im2col (K, N*H'*W') → (F, N*H'*W')``."""
    module = _module_for(nn.Conv2d(1, 8, 3).eval(), torch.randn(1, 1, 8, 8))
    matmul = next(op for op in module.walk() if isinstance(op, MatmulOp))
    a_shape = matmul.operands[0].type.get_shape()  # W_flat
    b_shape = matmul.operands[1].type.get_shape()  # im2col
    out_shape = matmul.res[0].type.get_shape()
    assert tuple(a_shape) == (8, 9)
    assert tuple(b_shape) == (9, 36)
    assert tuple(out_shape) == (8, 36)


def test_conv2d_reshape_recovers_nchw_output() -> None:
    """The final reshape op recovers the (N, F, H', W') NCHW output."""
    module = _module_for(nn.Conv2d(1, 8, 3).eval(), torch.randn(1, 1, 8, 8))
    reshape = next(op for op in module.walk() if isinstance(op, CallOp) and op.callee.string_value() == "aten_reshape")
    out_shape = reshape.res[0].type.get_shape()
    assert tuple(out_shape) == (1, 8, 6, 6)
