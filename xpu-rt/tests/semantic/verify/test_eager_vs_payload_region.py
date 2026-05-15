"""Tests verifying eager execution against eager-copy for real model blocks.

These tests exercise the full harness on standard nn.Module building blocks
(MLP, Transformer encoder layer, Conv block) to ensure the verification
pipeline works end-to-end on realistic architectures.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from compgen.semantic.verify.eager_reference import build_eager_reference
from compgen.semantic.verify.harness import verify_callable_against_reference
from compgen.semantic.verify.transformed_reference import identity_transform, wrap_transformed

# -- MLP block ----------------------------------------------------------------


class _MLP(nn.Module):
    """Simple two-layer MLP for testing."""

    def __init__(self, dim: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_mlp_eager_vs_eager(tmp_path: Path) -> None:
    """Eager MLP vs identity transform should pass."""
    model = _MLP()
    inputs = (torch.randn(2, 32),)
    ref = build_eager_reference(model, inputs)
    transformed = wrap_transformed(identity_transform(model), ref.example_inputs, name="mlp-identity")

    result = verify_callable_against_reference(
        name="mlp-eager-vs-eager",
        ref_fn=ref,
        got_fn=transformed,
        out_dir=tmp_path,
    )
    assert result.passed


# -- Transformer encoder layer -----------------------------------------------


def test_transformer_layer_eager_vs_eager(tmp_path: Path) -> None:
    """Eager TransformerEncoderLayer vs identity should pass."""
    model = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
    inputs = (torch.randn(2, 8, 32),)
    ref = build_eager_reference(model, inputs)
    transformed = wrap_transformed(identity_transform(model), ref.example_inputs, name="tx-identity")

    result = verify_callable_against_reference(
        name="transformer-eager-vs-eager",
        ref_fn=ref,
        got_fn=transformed,
        out_dir=tmp_path,
    )
    assert result.passed


# -- Conv block ---------------------------------------------------------------


class _ConvBlock(nn.Module):
    """Simple Conv2d + BatchNorm + ReLU block."""

    def __init__(self, in_ch: int = 3, out_ch: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


def test_conv_block_eager_vs_eager(tmp_path: Path) -> None:
    """Eager Conv block vs identity should pass."""
    model = _ConvBlock()
    inputs = (torch.randn(1, 3, 16, 16),)
    ref = build_eager_reference(model, inputs)
    transformed = wrap_transformed(identity_transform(model), ref.example_inputs, name="conv-identity")

    result = verify_callable_against_reference(
        name="conv-eager-vs-eager",
        ref_fn=ref,
        got_fn=transformed,
        out_dir=tmp_path,
    )
    assert result.passed


# -- EagerReference round-trip ------------------------------------------------


def test_eager_reference_callable(tmp_path: Path) -> None:
    """Calling an EagerReference directly should return consistent outputs."""
    model = _MLP()
    inputs = (torch.randn(2, 32),)
    ref = build_eager_reference(model, inputs)
    out1 = ref()
    out2 = ref()
    assert torch.allclose(out1, out2)


def test_eager_reference_stores_outputs() -> None:
    """EagerReference should cache reference_outputs."""
    model = _MLP()
    inputs = (torch.randn(1, 32),)
    ref = build_eager_reference(model, inputs)
    assert ref.reference_outputs is not None
    assert isinstance(ref.reference_outputs, torch.Tensor)
