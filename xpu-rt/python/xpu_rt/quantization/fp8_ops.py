"""Core FP8 E4M3 quantization operations with power-of-two scaling.

Implements per-tensor FP8 E4M3 quantization using power-of-two (po2) scaling,
adapted from pi0-quant's ``quant_types.py``.  Po2 scales map directly to the
NPU's E8M0 scale registers (pure exponent shift, no FP multiply).

Two scaling modes are supported:

  **po2** (default) -- scale = 2^floor(log2(amax / 256)).  Hardware-optimal:
  the NPU can implement this as a shift in the exponent field.

  **scaled** -- scale = amax / 448.  Standard absmax scaling (NVIDIA
  Transformer Engine style).  Used for comparison / ablation.

All functions operate on plain ``torch.Tensor`` with no torchAO dependency.
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# FP8 E4M3 constants
# ---------------------------------------------------------------------------

FP8_E4M3_MAX: float = 448.0
"""Largest finite value representable in float8_e4m3fn (1.75 * 2^8)."""

FP8_E4M3_MAX_PO2: float = 256.0
"""Largest power-of-two representable in float8_e4m3fn (2^8)."""

FP8_E4M3_DTYPE: torch.dtype = torch.float8_e4m3fn
"""PyTorch dtype for E4M3 (4-bit exponent, 3-bit mantissa, no Inf)."""


# ---------------------------------------------------------------------------
# Po2 scaling helpers
# ---------------------------------------------------------------------------

def fp8_po2_scale(x: torch.Tensor) -> float:
    """Compute a per-tensor power-of-two scale for FP8 E4M3 quantization.

    The scale is constrained to a power of two so that the NPU can implement
    it as a pure exponent shift (via the E8M0 scale register).

    Args:
        x: Input tensor (any dtype, any shape).

    Returns:
        A positive float that is an exact power of two, or 1.0 if the tensor
        is all zeros.
    """
    amax = x.float().abs().max()
    if amax == 0:
        return 1.0
    raw_scale = amax / FP8_E4M3_MAX_PO2
    return 2.0 ** math.floor(math.log2(raw_scale.item()))


def fp8_absmax_scale(x: torch.Tensor) -> float:
    """Compute a per-tensor absmax scale for FP8 E4M3 quantization.

    Args:
        x: Input tensor (any dtype, any shape).

    Returns:
        A positive float, or 1.0 if the tensor is all zeros.
    """
    amax = x.float().abs().max()
    if amax == 0:
        return 1.0
    return (amax / FP8_E4M3_MAX).item()


# ---------------------------------------------------------------------------
# Quantize / dequantize
# ---------------------------------------------------------------------------

def quantize_fp8_e4m3_po2(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize a tensor to FP8 E4M3 using per-tensor po2 scaling.

    Args:
        x: Input tensor (any dtype).

    Returns:
        A ``(x_fp8, scale)`` tuple where ``x_fp8`` has dtype
        ``torch.float8_e4m3fn`` and ``scale`` is an exact power of two.
        To recover the original values: ``x_fp8.float() * scale``.
    """
    x_f32 = x.float()
    scale = fp8_po2_scale(x_f32)
    x_scaled = (x_f32 / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    x_fp8 = x_scaled.to(FP8_E4M3_DTYPE)
    return x_fp8, scale


def quantize_fp8_e4m3_scaled(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize a tensor to FP8 E4M3 using per-tensor absmax scaling.

    Args:
        x: Input tensor (any dtype).

    Returns:
        A ``(x_fp8, scale)`` tuple where ``x_fp8`` has dtype
        ``torch.float8_e4m3fn`` and ``scale`` is a positive float.
    """
    x_f32 = x.float()
    scale = fp8_absmax_scale(x_f32)
    x_scaled = x_f32 / scale
    x_fp8 = x_scaled.to(FP8_E4M3_DTYPE)
    return x_fp8, scale


def dequantize_fp8_e4m3(
    x_fp8: torch.Tensor,
    scale: float,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 E4M3 tensor back to a higher-precision dtype.

    Args:
        x_fp8: Quantized tensor (float8_e4m3fn).
        scale: The scale factor used during quantization.
        target_dtype: Output dtype (default: bfloat16, matching NPU accumulators).

    Returns:
        Dequantized tensor in ``target_dtype``.
    """
    return (x_fp8.float() * scale).to(target_dtype)


def quantize_dequantize_fp8_po2(
    x: torch.Tensor,
    target_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Simulate FP8 E4M3 po2 quantization noise (quantize then dequantize).

    This is the equivalent of pi0-quant's ``quant(x, QuantFormat.FLOAT8_E4M3)``
    in po2 mode.  The output has the same dtype as the input (or
    ``target_dtype`` if specified).

    Args:
        x: Input tensor.
        target_dtype: Override output dtype.  If ``None``, uses ``x.dtype``.

    Returns:
        Tensor with FP8 quantization noise baked in, in the original dtype.
    """
    out_dtype = target_dtype if target_dtype is not None else x.dtype
    x_fp8, scale = quantize_fp8_e4m3_po2(x)
    return (x_fp8.float() * scale).to(out_dtype)


def quantize_dequantize_fp8_scaled(
    x: torch.Tensor,
    target_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Simulate FP8 E4M3 absmax quantization noise (quantize then dequantize).

    Args:
        x: Input tensor.
        target_dtype: Override output dtype.  If ``None``, uses ``x.dtype``.

    Returns:
        Tensor with FP8 quantization noise baked in, in the original dtype.
    """
    out_dtype = target_dtype if target_dtype is not None else x.dtype
    x_fp8, scale = quantize_fp8_e4m3_scaled(x)
    return (x_fp8.float() * scale).to(out_dtype)


def is_power_of_two(value: float) -> bool:
    """Check whether a positive float is an exact power of two.

    Args:
        value: Must be positive and finite.

    Returns:
        ``True`` if ``value == 2^k`` for some integer ``k``.
    """
    if value <= 0 or not math.isfinite(value):
        return False
    # A power of two has exactly one bit set in its IEEE mantissa.
    # Equivalently: log2 is an exact integer.
    log2_val = math.log2(value)
    return log2_val == math.floor(log2_val)


__all__ = [
    "FP8_E4M3_DTYPE",
    "FP8_E4M3_MAX",
    "FP8_E4M3_MAX_PO2",
    "dequantize_fp8_e4m3",
    "fp8_absmax_scale",
    "fp8_po2_scale",
    "is_power_of_two",
    "quantize_dequantize_fp8_po2",
    "quantize_dequantize_fp8_scaled",
    "quantize_fp8_e4m3_po2",
    "quantize_fp8_e4m3_scaled",
]
