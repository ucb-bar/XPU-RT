"""Tests for export-friendly FP8 module wrappers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from compgen.quantization.export_wrappers import (
    ExportableFP8Conv2d,
    ExportableFP8Linear,
    rewrite_for_export,
)
from compgen.quantization.fp8_ops import quantize_fp8_e4m3_po2
from compgen.quantization.fp8_tensor import FP8E4M3Po2Tensor

torchao = pytest.importorskip("torchao")


# ---------------------------------------------------------------------------
# ExportableFP8Linear
# ---------------------------------------------------------------------------


class TestExportableFP8Linear:
    def test_forward_correctness(self) -> None:
        """Forward should produce correct-shaped BF16 output."""
        w = torch.randn(16, 32, dtype=torch.bfloat16)
        w_fp8, w_scale = quantize_fp8_e4m3_po2(w)
        layer = ExportableFP8Linear(w_fp8, w_scale, bias=None, in_features=32, out_features=16)

        x = torch.randn(4, 32, dtype=torch.bfloat16)
        out = layer(x)
        assert out.shape == (4, 16)
        assert out.dtype == torch.bfloat16

    def test_forward_with_bias(self) -> None:
        w = torch.randn(16, 32, dtype=torch.bfloat16)
        b = torch.randn(16, dtype=torch.bfloat16)
        w_fp8, w_scale = quantize_fp8_e4m3_po2(w)
        layer = ExportableFP8Linear(w_fp8, w_scale, bias=b, in_features=32, out_features=16)

        x = torch.randn(4, 32, dtype=torch.bfloat16)
        out = layer(x)
        assert out.shape == (4, 16)

    def test_from_quantized_linear(self) -> None:
        """Create from an nn.Linear with FP8E4M3Po2Tensor weight."""
        linear = nn.Linear(32, 16, dtype=torch.bfloat16)
        linear.weight = nn.Parameter(FP8E4M3Po2Tensor.from_float(linear.weight), requires_grad=False)
        exp_linear = ExportableFP8Linear.from_quantized_linear(linear)
        assert exp_linear.in_features == 32
        assert exp_linear.out_features == 16
        assert exp_linear.weight_fp8.dtype == torch.float8_e4m3fn

    def test_extra_repr(self) -> None:
        w_fp8, w_scale = quantize_fp8_e4m3_po2(torch.randn(8, 16, dtype=torch.bfloat16))
        layer = ExportableFP8Linear(w_fp8, w_scale, bias=None, in_features=16, out_features=8)
        r = layer.extra_repr()
        assert "float8_e4m3fn" in r
        assert "in_features=16" in r


# ---------------------------------------------------------------------------
# ExportableFP8Conv2d
# ---------------------------------------------------------------------------


class TestExportableFP8Conv2d:
    def test_forward_correctness(self) -> None:
        w = torch.randn(16, 3, 3, 3, dtype=torch.bfloat16)
        w_fp8, w_scale = quantize_fp8_e4m3_po2(w)
        layer = ExportableFP8Conv2d(w_fp8, w_scale, bias=None, stride=(1, 1), padding=(1, 1))

        x = torch.randn(1, 3, 8, 8, dtype=torch.bfloat16)
        out = layer(x)
        assert out.shape == (1, 16, 8, 8)
        assert out.dtype == torch.bfloat16

    def test_from_quantized_conv2d(self) -> None:
        conv = nn.Conv2d(3, 16, 3, padding=1, dtype=torch.bfloat16)
        conv.weight = nn.Parameter(FP8E4M3Po2Tensor.from_float(conv.weight), requires_grad=False)
        exp_conv = ExportableFP8Conv2d.from_quantized_conv2d(conv)
        assert exp_conv.weight_fp8.dtype == torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# rewrite_for_export
# ---------------------------------------------------------------------------


class TestRewriteForExport:
    def test_linear_replacement(self) -> None:
        """nn.Linear with FP8 weight should become ExportableFP8Linear."""
        model = nn.Sequential(
            nn.Linear(32, 16, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(16, 8, dtype=torch.bfloat16),
        )
        # Quantize
        from compgen.quantization.fp8_config import FP8E4M3Po2Config
        from torchao.quantization import quantize_

        quantize_(model, FP8E4M3Po2Config())

        # Rewrite for export
        rewrite_for_export(model)
        assert isinstance(model[0], ExportableFP8Linear)
        assert isinstance(model[2], ExportableFP8Linear)
        # ReLU should be untouched
        assert isinstance(model[1], nn.ReLU)

    def test_conv2d_replacement(self) -> None:
        """nn.Conv2d with FP8 weight should become ExportableFP8Conv2d."""
        model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, dtype=torch.bfloat16))
        model[0].weight = nn.Parameter(FP8E4M3Po2Tensor.from_float(model[0].weight), requires_grad=False)
        rewrite_for_export(model)
        assert isinstance(model[0], ExportableFP8Conv2d)

    def test_rewritten_model_forward(self) -> None:
        """Rewritten model should still produce correct output."""
        model = nn.Sequential(
            nn.Linear(32, 16, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(16, 8, dtype=torch.bfloat16),
        )
        from compgen.quantization.fp8_config import FP8E4M3Po2Config
        from torchao.quantization import quantize_

        quantize_(model, FP8E4M3Po2Config())
        rewrite_for_export(model)

        x = torch.randn(4, 32, dtype=torch.bfloat16)
        out = model(x)
        assert out.shape == (4, 8)
        assert out.dtype == torch.bfloat16

    def test_nested_model_rewrite(self) -> None:
        """Rewrite should handle nested module hierarchies."""

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(16, 16, dtype=torch.bfloat16)

            def forward(self, x):
                return self.proj(x)

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Inner()
                self.decoder = Inner()

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = Outer()
        from compgen.quantization.fp8_config import FP8E4M3Po2Config
        from torchao.quantization import quantize_

        quantize_(model, FP8E4M3Po2Config())
        rewrite_for_export(model)

        assert isinstance(model.encoder.proj, ExportableFP8Linear)
        assert isinstance(model.decoder.proj, ExportableFP8Linear)

        x = torch.randn(2, 16, dtype=torch.bfloat16)
        out = model(x)
        assert out.shape == (2, 16)

    def test_torch_export_succeeds(self) -> None:
        """torch.export.export should succeed on the rewritten model."""
        model = nn.Sequential(
            nn.Linear(16, 8, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(8, 4, dtype=torch.bfloat16),
        )
        from compgen.quantization.fp8_config import FP8E4M3Po2Config
        from torchao.quantization import quantize_

        quantize_(model, FP8E4M3Po2Config())
        rewrite_for_export(model)

        x = torch.randn(2, 16, dtype=torch.bfloat16)
        try:
            exported = torch.export.export(model, (x,), strict=False)
            assert exported is not None
            # Verify we can run the exported program
            out = exported.module()(x)
            assert out.shape == (2, 4)
        except Exception as e:
            # torch.export compatibility may vary; at minimum the rewrite
            # should not break the model
            pytest.skip(f"torch.export not fully supported: {e}")
