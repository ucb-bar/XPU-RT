"""Tests for dtype/shape guard predicate matching.

Verifies that the verification harness correctly detects mismatches caused
by dtype promotion, wrong shapes, or intentional numeric perturbation --
the kinds of issues that compiler guards should catch.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from compgen.semantic.verify.compare import DTYPE_PRESETS, compare_tensors
from compgen.semantic.verify.eager_reference import build_eager_reference
from compgen.semantic.verify.harness import verify_callable_against_reference


class _Linear(nn.Module):
    """Single linear layer for guard tests."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# -- dtype guards -------------------------------------------------------------


def test_dtype_mismatch_detected() -> None:
    """Comparing float32 ref against float16 cast should detect error."""
    ref = torch.randn(4, 4)
    got = ref.half().float()  # round-trip through float16 introduces error
    preset = DTYPE_PRESETS[torch.float32]
    result = compare_tensors(ref, got, atol=preset.atol, rtol=preset.rtol)
    # The round-trip may or may not pass strict float32 tolerance depending
    # on the values, but max_abs_error should be non-zero.
    assert result.max_abs_error > 0.0


def test_float16_tolerance_accepts_roundtrip() -> None:
    """float16 preset tolerance should accept float16 round-trip error."""
    ref = torch.randn(4, 4)
    got = ref.half().float()
    preset = DTYPE_PRESETS[torch.float16]
    result = compare_tensors(ref, got, atol=preset.atol, rtol=preset.rtol)
    assert result.passed


# -- shape guards -------------------------------------------------------------


def test_shape_guard_wrong_batch(tmp_path: Path) -> None:
    """Different batch size should cause comparison failure."""
    model = _Linear()
    ref = build_eager_reference(model, (torch.randn(2, 16),))
    # Candidate produces output with different batch dimension.
    wrong_batch = torch.randn(4, 16)
    result = verify_callable_against_reference(
        name="shape-guard-batch",
        ref_fn=ref,
        got_fn=lambda: wrong_batch,
        out_dir=tmp_path,
    )
    assert not result.passed


def test_shape_guard_wrong_features(tmp_path: Path) -> None:
    """Different feature dimension should cause comparison failure."""
    model = _Linear(dim=16)
    ref = build_eager_reference(model, (torch.randn(2, 16),))
    wrong_feat = torch.randn(2, 8)
    result = verify_callable_against_reference(
        name="shape-guard-feat",
        ref_fn=ref,
        got_fn=lambda: wrong_feat,
        out_dir=tmp_path,
    )
    assert not result.passed


# -- numeric perturbation guards ----------------------------------------------


def test_small_perturbation_passes(tmp_path: Path) -> None:
    """A tiny perturbation within tolerance should pass."""
    model = _Linear()
    inputs = (torch.randn(2, 16),)
    ref = build_eager_reference(model, inputs)
    ref_out = ref()

    def _perturbed() -> torch.Tensor:
        return ref_out + torch.randn_like(ref_out) * 1e-7

    result = verify_callable_against_reference(
        name="small-perturb",
        ref_fn=ref,
        got_fn=_perturbed,
        out_dir=tmp_path,
        atol=1e-5,
        rtol=1e-5,
    )
    assert result.passed


def test_large_perturbation_fails(tmp_path: Path) -> None:
    """A large perturbation should be caught."""
    model = _Linear()
    inputs = (torch.randn(2, 16),)
    ref = build_eager_reference(model, inputs)
    ref_out = ref()

    def _perturbed() -> torch.Tensor:
        return ref_out + torch.ones_like(ref_out) * 0.5

    result = verify_callable_against_reference(
        name="large-perturb",
        ref_fn=ref,
        got_fn=_perturbed,
        out_dir=tmp_path,
        atol=1e-5,
        rtol=1e-5,
    )
    assert not result.passed
