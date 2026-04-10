"""Tests for core FP8 E4M3 quantization operations."""

from __future__ import annotations

import math
import sys

import pytest
import torch

from compgen.quantization.fp8_ops import (
    FP8_E4M3_DTYPE,
    FP8_E4M3_MAX,
    FP8_E4M3_MAX_PO2,
    dequantize_fp8_e4m3,
    fp8_absmax_scale,
    fp8_po2_scale,
    is_power_of_two,
    quantize_dequantize_fp8_po2,
    quantize_dequantize_fp8_scaled,
    quantize_fp8_e4m3_po2,
    quantize_fp8_e4m3_scaled,
)

# Try importing pi0-quant for cross-validation
_HAS_PI0_QUANT = False
try:
    sys.path.insert(0, "/scratch2/agustin/CompGen/third_party/pi0-quant")
    from pi0_inout.quant_types import QuantFormat, quant as pi0_quant, set_fp8_mode
    _HAS_PI0_QUANT = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_fp8_e4m3_max(self) -> None:
        assert FP8_E4M3_MAX == 448.0

    def test_fp8_e4m3_max_po2(self) -> None:
        assert FP8_E4M3_MAX_PO2 == 256.0

    def test_fp8_dtype(self) -> None:
        assert FP8_E4M3_DTYPE == torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# Po2 scale computation
# ---------------------------------------------------------------------------

class TestFp8Po2Scale:
    def test_zero_tensor(self) -> None:
        x = torch.zeros(10)
        assert fp8_po2_scale(x) == 1.0

    def test_result_is_power_of_two(self) -> None:
        for mag in [0.01, 0.5, 1.0, 10.0, 100.0, 1000.0, 1e6]:
            x = torch.randn(64) * mag
            scale = fp8_po2_scale(x)
            assert is_power_of_two(scale), f"scale={scale} not po2 for mag={mag}"

    def test_known_values(self) -> None:
        # amax = 256 -> raw = 256/256 = 1.0 -> floor(log2(1)) = 0 -> scale = 1.0
        x = torch.tensor([256.0])
        assert fp8_po2_scale(x) == 1.0

        # amax = 512 -> raw = 512/256 = 2.0 -> floor(log2(2)) = 1 -> scale = 2.0
        x = torch.tensor([512.0])
        assert fp8_po2_scale(x) == 2.0

        # amax = 1.0 -> raw = 1/256 ~ 0.0039 -> floor(log2(0.0039)) = -8 -> scale = 2^-8
        x = torch.tensor([1.0])
        assert fp8_po2_scale(x) == 2.0 ** -8

    def test_bf16_input(self) -> None:
        x = torch.randn(32, dtype=torch.bfloat16) * 50
        scale = fp8_po2_scale(x)
        assert is_power_of_two(scale)

    def test_negative_values(self) -> None:
        x = torch.tensor([-100.0, -200.0, -50.0])
        scale = fp8_po2_scale(x)
        assert is_power_of_two(scale)


# ---------------------------------------------------------------------------
# Absmax scale computation
# ---------------------------------------------------------------------------

class TestFp8AbsmaxScale:
    def test_zero_tensor(self) -> None:
        assert fp8_absmax_scale(torch.zeros(10)) == 1.0

    def test_known_value(self) -> None:
        x = torch.tensor([448.0])
        assert fp8_absmax_scale(x) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Quantize (po2)
# ---------------------------------------------------------------------------

class TestQuantizeFp8Po2:
    def test_output_dtype(self) -> None:
        x = torch.randn(16)
        x_fp8, scale = quantize_fp8_e4m3_po2(x)
        assert x_fp8.dtype == torch.float8_e4m3fn

    def test_scale_is_po2(self) -> None:
        x = torch.randn(64) * 100
        _, scale = quantize_fp8_e4m3_po2(x)
        assert is_power_of_two(scale)

    def test_values_within_range(self) -> None:
        x = torch.randn(128) * 500
        x_fp8, _ = quantize_fp8_e4m3_po2(x)
        vals = x_fp8.float().abs()
        assert vals.max() <= FP8_E4M3_MAX

    def test_zero_tensor(self) -> None:
        x = torch.zeros(8)
        x_fp8, scale = quantize_fp8_e4m3_po2(x)
        assert scale == 1.0
        assert (x_fp8.float() == 0).all()

    def test_shape_preserved(self) -> None:
        x = torch.randn(4, 8, 16)
        x_fp8, _ = quantize_fp8_e4m3_po2(x)
        assert x_fp8.shape == x.shape


# ---------------------------------------------------------------------------
# Quantize (absmax)
# ---------------------------------------------------------------------------

class TestQuantizeFp8Scaled:
    def test_output_dtype(self) -> None:
        x = torch.randn(16)
        x_fp8, _ = quantize_fp8_e4m3_scaled(x)
        assert x_fp8.dtype == torch.float8_e4m3fn

    def test_shape_preserved(self) -> None:
        x = torch.randn(4, 8)
        x_fp8, _ = quantize_fp8_e4m3_scaled(x)
        assert x_fp8.shape == x.shape


# ---------------------------------------------------------------------------
# Dequantize
# ---------------------------------------------------------------------------

class TestDequantize:
    def test_default_bf16(self) -> None:
        x = torch.randn(32) * 50
        x_fp8, scale = quantize_fp8_e4m3_po2(x)
        x_deq = dequantize_fp8_e4m3(x_fp8, scale)
        assert x_deq.dtype == torch.bfloat16

    def test_custom_dtype(self) -> None:
        x = torch.randn(32) * 50
        x_fp8, scale = quantize_fp8_e4m3_po2(x)
        x_deq = dequantize_fp8_e4m3(x_fp8, scale, target_dtype=torch.float32)
        assert x_deq.dtype == torch.float32

    def test_roundtrip_accuracy(self) -> None:
        x = torch.randn(256) * 10
        x_fp8, scale = quantize_fp8_e4m3_po2(x)
        x_deq = dequantize_fp8_e4m3(x_fp8, scale, target_dtype=torch.float32)
        # FP8 E4M3 has ~3 bits of mantissa, so relative error up to ~12.5%
        # For per-tensor scaling the error should be modest on average
        rel_error = (x - x_deq).abs() / (x.abs() + 1e-8)
        assert rel_error.mean() < 0.15, f"Mean relative error {rel_error.mean():.4f} too high"


# ---------------------------------------------------------------------------
# Quantize-dequantize roundtrip (simulate noise)
# ---------------------------------------------------------------------------

class TestQuantizeDequantize:
    def test_po2_preserves_dtype(self) -> None:
        x = torch.randn(32, dtype=torch.bfloat16)
        out = quantize_dequantize_fp8_po2(x)
        assert out.dtype == torch.bfloat16

    def test_po2_custom_target_dtype(self) -> None:
        x = torch.randn(32)
        out = quantize_dequantize_fp8_po2(x, target_dtype=torch.bfloat16)
        assert out.dtype == torch.bfloat16

    def test_scaled_preserves_dtype(self) -> None:
        x = torch.randn(32, dtype=torch.bfloat16)
        out = quantize_dequantize_fp8_scaled(x)
        assert out.dtype == torch.bfloat16

    def test_zero_input(self) -> None:
        x = torch.zeros(16, dtype=torch.bfloat16)
        out = quantize_dequantize_fp8_po2(x)
        assert (out == 0).all()

    def test_bf16_baseline_zero_noise(self) -> None:
        """BF16 values that survive FP8 roundtrip should have zero error on those values."""
        # Small values that are exactly representable in E4M3
        x = torch.tensor([1.0, 2.0, 4.0, 8.0, 0.0], dtype=torch.bfloat16)
        out = quantize_dequantize_fp8_po2(x)
        # These exact powers of 2 should survive quantization
        # (they are within range and exactly representable)
        assert torch.allclose(x.float(), out.float(), atol=1e-6)


# ---------------------------------------------------------------------------
# is_power_of_two
# ---------------------------------------------------------------------------

class TestIsPowerOfTwo:
    @pytest.mark.parametrize("val", [1.0, 2.0, 4.0, 0.5, 0.25, 256.0, 2**-8, 2**15])
    def test_true_cases(self, val: float) -> None:
        assert is_power_of_two(val) is True

    @pytest.mark.parametrize("val", [3.0, 5.0, 0.3, 448.0, -1.0, 0.0, float("inf")])
    def test_false_cases(self, val: float) -> None:
        assert is_power_of_two(val) is False


# ---------------------------------------------------------------------------
# Cross-validation with pi0-quant
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_PI0_QUANT, reason="pi0-quant not available")
class TestPi0QuantEquivalence:
    """Verify numerical equivalence with pi0-quant's quant() function."""

    def test_po2_mode_matches(self) -> None:
        set_fp8_mode("po2")
        torch.manual_seed(42)
        x = torch.randn(64, dtype=torch.bfloat16) * 100

        pi0_out = pi0_quant(x, QuantFormat.FLOAT8_E4M3)
        our_out = quantize_dequantize_fp8_po2(x)

        assert torch.allclose(pi0_out.float(), our_out.float(), atol=1e-6), (
            f"Max diff: {(pi0_out.float() - our_out.float()).abs().max()}"
        )

    def test_scaled_mode_matches(self) -> None:
        set_fp8_mode("scaled")
        torch.manual_seed(42)
        x = torch.randn(64, dtype=torch.bfloat16) * 100

        pi0_out = pi0_quant(x, QuantFormat.FLOAT8_E4M3)
        our_out = quantize_dequantize_fp8_scaled(x)

        assert torch.allclose(pi0_out.float(), our_out.float(), atol=1e-6), (
            f"Max diff: {(pi0_out.float() - our_out.float()).abs().max()}"
        )
        # Reset to default
        set_fp8_mode("po2")

    def test_multiple_magnitudes(self) -> None:
        set_fp8_mode("po2")
        for mag in [0.01, 1.0, 50.0, 200.0, 1000.0]:
            torch.manual_seed(123)
            x = torch.randn(128, dtype=torch.bfloat16) * mag
            pi0_out = pi0_quant(x, QuantFormat.FLOAT8_E4M3)
            our_out = quantize_dequantize_fp8_po2(x)
            assert torch.allclose(pi0_out.float(), our_out.float(), atol=1e-6), (
                f"Mismatch at mag={mag}, max diff: "
                f"{(pi0_out.float() - our_out.float()).abs().max()}"
            )
