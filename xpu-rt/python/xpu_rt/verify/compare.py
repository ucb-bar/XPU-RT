"""Numeric tensor comparison utilities.

Provides tolerance-aware comparison of PyTorch tensors with detailed error
metrics, plus per-dtype tolerance presets used throughout the verification
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NumericComparison:
    """Result of comparing two tensors element-wise.

    Attributes:
        passed: Whether all elements are within tolerance.
        max_abs_error: Maximum absolute error across all elements.
        max_rel_error: Maximum relative error across all elements.
        atol: Absolute tolerance used for the comparison.
        rtol: Relative tolerance used for the comparison.
        num_mismatched: Number of elements that exceeded tolerance.
    """

    passed: bool
    max_abs_error: float
    max_rel_error: float
    atol: float
    rtol: float
    num_mismatched: int


@dataclass(frozen=True)
class ComparisonConfig:
    """Per-dtype tolerance presets for numeric comparison.

    Attributes:
        atol: Absolute tolerance.
        rtol: Relative tolerance.
    """

    atol: float
    rtol: float


# Standard tolerance presets by dtype.
DTYPE_PRESETS: dict[torch.dtype, ComparisonConfig] = {
    torch.float32: ComparisonConfig(atol=1e-5, rtol=1e-5),
    torch.float16: ComparisonConfig(atol=1e-3, rtol=1e-3),
    torch.bfloat16: ComparisonConfig(atol=1e-2, rtol=1e-2),
    torch.int8: ComparisonConfig(atol=0.0, rtol=0.0),
}


def compare_tensors(
    ref: torch.Tensor,
    got: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> NumericComparison:
    """Compare two tensors element-wise with tolerance.

    Handles shape mismatches, NaN values, and empty tensors gracefully.

    Args:
        ref: Reference (ground-truth) tensor.
        got: Candidate tensor to verify against *ref*.
        atol: Absolute tolerance for ``torch.isclose``.
        rtol: Relative tolerance for ``torch.isclose``.

    Returns:
        A :class:`NumericComparison` summarising the comparison.
    """
    if ref.shape != got.shape:
        return NumericComparison(
            passed=False,
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            atol=atol,
            rtol=rtol,
            num_mismatched=max(ref.numel(), got.numel()),
        )

    if ref.numel() == 0:
        return NumericComparison(
            passed=True,
            max_abs_error=0.0,
            max_rel_error=0.0,
            atol=atol,
            rtol=rtol,
            num_mismatched=0,
        )

    # Cast to float32 for uniform error computation.
    ref_f = ref.detach().float()
    got_f = got.detach().float()

    abs_err = (ref_f - got_f).abs()

    # NaN in either tensor counts as a mismatch.
    nan_mask = torch.isnan(ref_f) | torch.isnan(got_f)

    # Relative error with denominator clamped away from zero.
    rel_err = abs_err / torch.clamp(ref_f.abs(), min=1e-12)

    # For positions where ref is NaN or got is NaN, set errors to inf so
    # they always exceed tolerance and count as mismatched.
    inf_val = torch.tensor(float("inf"))
    abs_err = torch.where(nan_mask, inf_val, abs_err)
    rel_err = torch.where(nan_mask, inf_val, rel_err)

    mismatched = ~torch.isclose(ref_f, got_f, atol=atol, rtol=rtol) | nan_mask

    return NumericComparison(
        passed=int(mismatched.sum().item()) == 0,
        max_abs_error=float(abs_err.max().item()),
        max_rel_error=float(rel_err.max().item()),
        atol=atol,
        rtol=rtol,
        num_mismatched=int(mismatched.sum().item()),
    )


__all__ = [
    "ComparisonConfig",
    "DTYPE_PRESETS",
    "NumericComparison",
    "compare_tensors",
]
