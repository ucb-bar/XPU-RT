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

# Extended tolerance presets for quantization formats (wave 6 / C.4).
# Keys are canonical string tags matching ``TorchAOScheme.weight_dtype``
# optionally suffixed with granularity (e.g. ``"int4_per_group"``).
# ``torch.int4`` / ``torch.uint4`` / ``torch.nvfp4`` do not exist as
# torch dtypes today, so a sibling string-keyed dict is the pragmatic
# escape valve. ``DTYPE_PRESETS`` remains untouched for backward
# compatibility.
FORMAT_PRESETS: dict[str, ComparisonConfig] = {
    # FP8 variants (torch DOES have these dtypes, but they're not in
    # DTYPE_PRESETS today and FORMAT_PRESETS is the canonical lookup
    # for tolerance_for_format).
    "float8_e4m3fn": ComparisonConfig(atol=1e-2, rtol=1e-2),
    "float8_e5m2": ComparisonConfig(atol=1.5e-2, rtol=1.5e-2),
    "float8_e4m3fn_per_tensor": ComparisonConfig(atol=1e-2, rtol=1e-2),
    "float8_e4m3fn_per_channel": ComparisonConfig(atol=1e-2, rtol=1e-2),
    "float8_e5m2_per_tensor": ComparisonConfig(atol=1.5e-2, rtol=1.5e-2),
    # Int8 (kept consistent with DTYPE_PRESETS for exactness on
    # well-aligned quantization).
    "int8": ComparisonConfig(atol=0.0, rtol=0.0),
    "int8_per_tensor": ComparisonConfig(atol=0.0, rtol=0.0),
    "int8_per_channel": ComparisonConfig(atol=0.0, rtol=0.0),
    # Int4 — ±0.5 LSB (half-quantum) plus small relative slack for the
    # per-channel and per-group paths because groupwise scale changes
    # carry additional rounding.
    "int4": ComparisonConfig(atol=0.5, rtol=0.0),
    "int4_per_tensor": ComparisonConfig(atol=0.5, rtol=0.0),
    "int4_per_channel": ComparisonConfig(atol=0.5, rtol=2e-2),
    "int4_per_group": ComparisonConfig(atol=0.5, rtol=3e-2),
    "uint4": ComparisonConfig(atol=0.5, rtol=0.0),
    "uint4_per_tensor": ComparisonConfig(atol=0.5, rtol=0.0),
    "uint4_per_channel": ComparisonConfig(atol=0.5, rtol=2e-2),
    "uint4_per_group": ComparisonConfig(atol=0.5, rtol=3e-2),
    # Intx (1-3 bit) — much wider tolerance; these are lossy by design.
    "intx": ComparisonConfig(atol=1.0, rtol=0.1),
    "intx_per_group": ComparisonConfig(atol=1.0, rtol=0.1),
    "intx_per_channel": ComparisonConfig(atol=1.0, rtol=0.1),
    # MX family (block-wise with shared exponent).
    "mx4": ComparisonConfig(atol=2e-2, rtol=2e-2),
    "mx6": ComparisonConfig(atol=1.5e-2, rtol=1.5e-2),
    "mx9": ComparisonConfig(atol=1e-2, rtol=1e-2),
    # NVFP4 block format (NV custom 4-bit float).
    "nvfp4": ComparisonConfig(atol=1.5e-2, rtol=1.5e-2),
    "nvfp4_block": ComparisonConfig(atol=1.5e-2, rtol=1.5e-2),
    "nvfp4_per_block": ComparisonConfig(atol=1.5e-2, rtol=1.5e-2),
}


def tolerance_for_format(
    format_tag: str,
    default: ComparisonConfig | None = None,
) -> ComparisonConfig:
    """Return the ``ComparisonConfig`` for ``format_tag``.

    Accepts any of: the canonical keys in ``FORMAT_PRESETS``, the bare
    weight-dtype prefix (e.g. ``"int4"`` returns the per-tensor
    preset), or a TorchAOScheme ``name``. Falls back to ``default``
    (or a permissive FP16-like config) for unknown tags.
    """
    if format_tag in FORMAT_PRESETS:
        return FORMAT_PRESETS[format_tag]
    # Heuristic: strip common suffixes and retry.
    for suffix in ("_per_tensor", "_per_channel", "_per_group", "_per_block"):
        if format_tag.endswith(suffix):
            bare = format_tag[: -len(suffix)]
            if bare in FORMAT_PRESETS:
                return FORMAT_PRESETS[bare]
    if default is not None:
        return default
    return ComparisonConfig(atol=1e-2, rtol=1e-2)


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
