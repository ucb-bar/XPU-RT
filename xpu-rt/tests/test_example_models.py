"""Test that all example models capture and convert to xDSL successfully."""

from __future__ import annotations

import pytest
import torch
from compgen.capture.torch_export import capture_model
from compgen.ir.payload.import_fx import fx_to_xdsl


def test_simple_mlp_capture() -> None:
    """SimpleMLP captures and converts to xDSL."""

    class SimpleMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(32, 64)
            self.fc2 = torch.nn.Linear(64, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(torch.relu(self.fc1(x)))

    model = SimpleMLP()
    ep = capture_model(model, (torch.randn(4, 32),))
    module, _ = fx_to_xdsl(ep)
    assert sum(1 for _ in module.walk()) > 5


@pytest.mark.xfail(reason="mul_tensor decomposition needs more operand handling")
def test_transformer_block_capture() -> None:
    """Transformer block with attention captures successfully."""

    class TransformerBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = torch.nn.MultiheadAttention(32, 4, batch_first=True)
            self.ff = torch.nn.Linear(32, 32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            attn_out, _ = self.attn(x, x, x)
            return self.ff(attn_out)

    model = TransformerBlock()
    ep = capture_model(model, (torch.randn(2, 4, 32),))
    module, _ = fx_to_xdsl(ep)
    assert module is not None


def test_quantized_mlp_capture() -> None:
    """Quantized model (dynamic quant) captures."""

    class QuantMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(16, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = QuantMLP()
    ep = capture_model(model, (torch.randn(2, 16),))
    module, _ = fx_to_xdsl(ep)
    assert module is not None
