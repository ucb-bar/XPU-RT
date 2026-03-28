"""Tests for kernel-level numeric equivalence.

Verifies that simple known transformations (scaling, ReLU replacement,
matrix multiply) produce numerically correct results when checked through
the verification harness.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from compgen.verify.eager_reference import build_eager_reference
from compgen.verify.harness import verify_callable_against_reference
from compgen.verify.transformed_reference import wrap_transformed

# -- scaling kernel -----------------------------------------------------------


def test_identity_scale_kernel(tmp_path: Path) -> None:
    """Multiplying by 1.0 should be numerically equivalent to identity."""
    model = nn.Linear(16, 16, bias=False)
    inputs = (torch.randn(4, 16),)
    ref = build_eager_reference(model, inputs)

    def _scale_one(*args: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.cpu().eval()(*args) * 1.0

    transformed = wrap_transformed(_scale_one, ref.example_inputs, name="scale-1")
    result = verify_callable_against_reference(
        name="identity-scale",
        ref_fn=ref,
        got_fn=transformed,
        out_dir=tmp_path,
    )
    assert result.passed


def test_wrong_scale_kernel_detected(tmp_path: Path) -> None:
    """Multiplying by 2.0 should be detected as incorrect."""
    model = nn.Linear(16, 16, bias=False)
    inputs = (torch.randn(4, 16),)
    ref = build_eager_reference(model, inputs)

    def _scale_two(*args: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.cpu().eval()(*args) * 2.0

    transformed = wrap_transformed(_scale_two, ref.example_inputs, name="scale-2")
    result = verify_callable_against_reference(
        name="wrong-scale",
        ref_fn=ref,
        got_fn=transformed,
        out_dir=tmp_path,
    )
    assert not result.passed


# -- ReLU equivalence ---------------------------------------------------------


class _ReLUModel(nn.Module):
    """Model using ReLU for kernel equivalence test."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.fc(x))


def test_relu_clamp_equivalent(tmp_path: Path) -> None:
    """ReLU should be equivalent to clamp(min=0)."""
    model = _ReLUModel()
    inputs = (torch.randn(2, 8),)
    ref = build_eager_reference(model, inputs)

    def _clamp_relu(*args: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pre_relu = model.cpu().eval().fc(*args)
            return torch.clamp(pre_relu, min=0.0)

    transformed = wrap_transformed(_clamp_relu, ref.example_inputs, name="clamp-relu")
    result = verify_callable_against_reference(
        name="relu-vs-clamp",
        ref_fn=ref,
        got_fn=transformed,
        out_dir=tmp_path,
    )
    assert result.passed


# -- matmul equivalence -------------------------------------------------------


def test_matmul_kernel_equivalence(tmp_path: Path) -> None:
    """Manual matmul should match nn.Linear (bias=False)."""
    model = nn.Linear(16, 8, bias=False)
    inputs = (torch.randn(4, 16),)
    ref = build_eager_reference(model, inputs)

    weight = model.weight.detach().clone()

    def _manual_matmul(*args: torch.Tensor) -> torch.Tensor:
        x = args[0]
        return x @ weight.t()

    transformed = wrap_transformed(_manual_matmul, ref.example_inputs, name="manual-matmul")
    result = verify_callable_against_reference(
        name="matmul-equivalence",
        ref_fn=ref,
        got_fn=transformed,
        out_dir=tmp_path,
    )
    assert result.passed


# -- fused bias kernel --------------------------------------------------------


def test_fused_bias_add(tmp_path: Path) -> None:
    """Linear with bias should match manual matmul+bias."""
    model = nn.Linear(8, 8, bias=True)
    inputs = (torch.randn(2, 8),)
    ref = build_eager_reference(model, inputs)

    weight = model.weight.detach().clone()
    bias = model.bias.detach().clone()  # type: ignore[union-attr]

    def _manual_linear(*args: torch.Tensor) -> torch.Tensor:
        x = args[0]
        return x @ weight.t() + bias

    transformed = wrap_transformed(_manual_linear, ref.example_inputs, name="fused-bias")
    result = verify_callable_against_reference(
        name="fused-bias-add",
        ref_fn=ref,
        got_fn=transformed,
        out_dir=tmp_path,
    )
    assert result.passed
