"""Tests for REQ-020 — ``nn.LayerNorm`` lowers without orphan
``<built-in function getitem>`` ops in payload.mlir.

PyTorch's ``aten.native_layer_norm`` returns ``(out, mean, rstd)``;
user code typically destructures the tuple via ``operator.getitem``.
The FX importer used to emit those getitem nodes as opaque
``func.call @"<built-in function getitem>"`` ops with zero operands —
no provider could match them and codegen-fallback bailed.

After REQ-020:

- ``getitem(producer, 0)`` resolves to the producer's primary tensor,
  no opaque call.
- ``getitem(producer, k)`` for ``k > 0`` is dropped entirely (the
  decomposition only models the primary output; auxiliary outputs
  are folded away).
"""

from __future__ import annotations

import io

import torch
import torch.nn as nn
from compgen.capture.torch_export import capture_model
from compgen.ir.payload.import_fx import fx_to_xdsl
from xdsl.dialects.func import CallOp
from xdsl.printer import Printer


def _module_for(model: nn.Module, *args: torch.Tensor):
    ep = capture_model(model, args)
    module, _ = fx_to_xdsl(ep)
    return module


def _ir_text(module) -> str:
    buf = io.StringIO()
    Printer(stream=buf).print_op(module)
    return buf.getvalue()


def test_plain_layernorm_yields_named_aten_call_only() -> None:
    """``nn.LayerNorm(8)`` → single ``aten_native_layer_norm`` call,
    no ``<built-in function getitem>``."""
    module = _module_for(nn.LayerNorm(8).eval(), torch.randn(1, 8))
    text = _ir_text(module)
    assert "aten_native_layer_norm" in text
    assert "<built-in function getitem>" not in text


def test_tuple_consuming_user_code_drops_getitem_entirely() -> None:
    """When user code destructures the tuple
    (``out, mean, rstd = native_layer_norm(...)``) and uses ``out``,
    payload.mlir contains the ``aten_native_layer_norm`` + the user op
    that consumes ``out`` — no orphan getitem."""

    class UseTuple(nn.Module):
        def forward(self, x, w, b):
            out, mean, rstd = torch.native_layer_norm(x, [8], w, b, 1e-5)
            return out * 2.0

    module = _module_for(
        UseTuple().eval(),
        torch.randn(1, 8),
        torch.ones(8),
        torch.zeros(8),
    )
    text = _ir_text(module)
    assert "aten_native_layer_norm" in text
    assert "<built-in function getitem>" not in text
    # The mul should consume the layer_norm output directly.
    assert "aten_mul" in text


def test_layernorm_followed_by_linear_clean() -> None:
    """LayerNorm → Linear (the typical first MLP block) lowers cleanly."""

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln = nn.LayerNorm(8)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(self.ln(x))

    module = _module_for(Net().eval(), torch.randn(1, 8))
    text = _ir_text(module)
    assert "<built-in function getitem>" not in text
    # Both regions should be tagged for dispatch.
    assert "layer_norm_0" in text
    assert "matmul_0" in text


def test_no_zero_operand_orphan_calls_anywhere() -> None:
    """Stronger invariant: every ``func.call`` in the IR has at least
    one SSA operand. (The pre-REQ-020 orphan was a zero-operand call.)"""
    module = _module_for(nn.LayerNorm(8).eval(), torch.randn(1, 8))
    for op in module.walk():
        if isinstance(op, CallOp):
            assert op.operands, f"orphan zero-operand func.call: {op.callee.string_value()}"
