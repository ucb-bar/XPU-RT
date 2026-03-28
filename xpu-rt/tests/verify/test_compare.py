"""Tests for numeric tensor comparison."""

from __future__ import annotations

import torch
from compgen.verify.compare import DTYPE_PRESETS, ComparisonConfig, NumericComparison, compare_tensors

# -- identical tensors --------------------------------------------------------


def test_identical_tensors_pass() -> None:
    """Two identical tensors should compare as passed with zero error."""
    t = torch.randn(4, 8)
    result = compare_tensors(t, t.clone())
    assert result.passed
    assert result.max_abs_error == 0.0
    assert result.num_mismatched == 0


def test_identical_zero_tensors() -> None:
    """All-zero tensors should pass."""
    t = torch.zeros(3, 3)
    result = compare_tensors(t, t.clone())
    assert result.passed
    assert result.max_abs_error == 0.0


def test_empty_tensors_pass() -> None:
    """Empty tensors should compare as passed."""
    t = torch.empty(0)
    result = compare_tensors(t, t.clone())
    assert result.passed
    assert result.num_mismatched == 0


# -- known differences -------------------------------------------------------


def test_known_abs_error() -> None:
    """A constant offset should be measured as max_abs_error."""
    ref = torch.ones(10)
    got = ref + 0.01
    result = compare_tensors(ref, got, atol=0.02, rtol=0.0)
    assert result.passed
    assert abs(result.max_abs_error - 0.01) < 1e-6


def test_known_diff_exceeds_tolerance() -> None:
    """A diff larger than atol should fail."""
    ref = torch.ones(10)
    got = ref + 0.1
    result = compare_tensors(ref, got, atol=0.01, rtol=0.0)
    assert not result.passed
    assert result.num_mismatched == 10


def test_relative_error_measured() -> None:
    """Relative error should be computed correctly for non-zero ref."""
    ref = torch.tensor([10.0])
    got = torch.tensor([10.5])
    result = compare_tensors(ref, got, atol=0.0, rtol=0.1)
    assert result.passed
    assert abs(result.max_rel_error - 0.05) < 1e-6


# -- shape mismatch ----------------------------------------------------------


def test_shape_mismatch_fails() -> None:
    """Tensors with different shapes should always fail."""
    ref = torch.randn(3, 4)
    got = torch.randn(4, 3)
    result = compare_tensors(ref, got)
    assert not result.passed
    assert result.max_abs_error == float("inf")
    assert result.num_mismatched == 12


def test_different_ndims_fails() -> None:
    """Tensors with different number of dimensions should fail."""
    ref = torch.randn(6)
    got = torch.randn(2, 3)
    result = compare_tensors(ref, got)
    assert not result.passed


# -- NaN handling -------------------------------------------------------------


def test_nan_in_got_fails() -> None:
    """NaN in candidate tensor should cause failure."""
    ref = torch.ones(5)
    got = torch.ones(5)
    got[2] = float("nan")
    result = compare_tensors(ref, got)
    assert not result.passed
    assert result.num_mismatched >= 1


def test_nan_in_ref_fails() -> None:
    """NaN in reference tensor should cause failure."""
    ref = torch.ones(5)
    ref[0] = float("nan")
    got = torch.ones(5)
    result = compare_tensors(ref, got)
    assert not result.passed
    assert result.num_mismatched >= 1


def test_nan_in_both_fails() -> None:
    """NaN in both tensors should still count as mismatch."""
    ref = torch.tensor([float("nan"), 1.0])
    got = torch.tensor([float("nan"), 1.0])
    result = compare_tensors(ref, got)
    assert not result.passed
    assert result.num_mismatched >= 1


# -- dtype presets ------------------------------------------------------------


def test_dtype_presets_exist() -> None:
    """Standard dtype presets should be defined."""
    assert torch.float32 in DTYPE_PRESETS
    assert torch.float16 in DTYPE_PRESETS
    assert torch.bfloat16 in DTYPE_PRESETS
    assert torch.int8 in DTYPE_PRESETS


def test_float16_preset_looser_than_float32() -> None:
    """float16 tolerances should be larger than float32 tolerances."""
    fp32 = DTYPE_PRESETS[torch.float32]
    fp16 = DTYPE_PRESETS[torch.float16]
    assert fp16.atol > fp32.atol
    assert fp16.rtol > fp32.rtol


def test_int8_preset_exact() -> None:
    """int8 preset should demand exact match."""
    preset = DTYPE_PRESETS[torch.int8]
    assert preset.atol == 0.0
    assert preset.rtol == 0.0


def test_comparison_config_frozen() -> None:
    """ComparisonConfig should be immutable."""
    cfg = ComparisonConfig(atol=1e-4, rtol=1e-4)
    try:
        cfg.atol = 0.0  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_numeric_comparison_frozen() -> None:
    """NumericComparison should be immutable."""
    cmp = NumericComparison(passed=True, max_abs_error=0.0, max_rel_error=0.0, atol=1e-5, rtol=1e-5, num_mismatched=0)
    try:
        cmp.passed = False  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass
