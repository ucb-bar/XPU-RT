"""End-to-end integration tests for the quantization pipeline."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

torchao = pytest.importorskip("torchao")


class TestCaptureIntegration:
    """Test that the capture pipeline's QuantizationConfig works with FP8 schemes."""

    def test_fp8_e4m3_po2_scheme(self) -> None:
        """QuantizationConfig(scheme='fp8_e4m3_po2') should quantize via CompGen."""
        from compgen.capture.torchao_pipeline import QuantizationConfig, apply_quantization
        from compgen.quantization.fp8_tensor import FP8E4M3Po2Tensor

        model = nn.Sequential(
            nn.Linear(32, 16, dtype=torch.bfloat16),
            nn.Linear(16, 8, dtype=torch.bfloat16),
        )
        config = QuantizationConfig(scheme="fp8_e4m3_po2")
        result = apply_quantization(model, config)

        assert isinstance(result[0].weight, FP8E4M3Po2Tensor)
        assert isinstance(result[1].weight, FP8E4M3Po2Tensor)

    def test_fp8_e4m3_po2_npu_scheme(self) -> None:
        """QuantizationConfig(scheme='fp8_e4m3_po2_npu') should use SmolVLA recipe."""
        from compgen.capture.torchao_pipeline import QuantizationConfig, apply_quantization
        from compgen.quantization.fp8_tensor import FP8E4M3Po2Tensor

        class MockSmolVLA(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = nn.Linear(32, 16, dtype=torch.bfloat16)
                self.language_model = nn.Linear(16, 8, dtype=torch.bfloat16)
                self.lm_head = nn.Linear(8, 100, dtype=torch.bfloat16)

            def forward(self, x):
                return x

        model = MockSmolVLA()
        config = QuantizationConfig(scheme="fp8_e4m3_po2_npu")
        apply_quantization(model, config)

        # Vision and language should be quantized
        assert isinstance(model.vision_tower.weight, FP8E4M3Po2Tensor)
        assert isinstance(model.language_model.weight, FP8E4M3Po2Tensor)
        # lm_head should be skipped
        assert not isinstance(model.lm_head.weight, FP8E4M3Po2Tensor)


class TestFullPipeline:
    """Test the full quantize -> verify -> export pipeline."""

    def test_quantize_verify_pipeline(self) -> None:
        """Full pipeline: quantize, verify alignment, check accuracy."""
        from compgen.quantization.fp8_config import FP8E4M3Po2Config
        from compgen.quantization.verify import compare_quantized_accuracy, npu_alignment_check
        from torchao.quantization import quantize_

        original = nn.Sequential(
            nn.Linear(32, 64, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(64, 16, dtype=torch.bfloat16),
        )
        quantized = copy.deepcopy(original)
        quantize_(quantized, FP8E4M3Po2Config())

        # Step 1: Verify NPU alignment
        alignment = npu_alignment_check(quantized)
        assert alignment.passed
        assert alignment.fp8_linear_count == 2

        # Step 2: Check accuracy
        x = torch.randn(4, 32, dtype=torch.bfloat16)
        accuracy = compare_quantized_accuracy(original, quantized, (x,))
        assert accuracy["cosine_similarity"] > 0.8

    def test_quantize_rewrite_export_pipeline(self) -> None:
        """Full pipeline: quantize -> rewrite -> forward."""
        from compgen.quantization.export_wrappers import rewrite_for_export
        from compgen.quantization.fp8_config import FP8E4M3Po2Config
        from compgen.quantization.verify import npu_alignment_check
        from torchao.quantization import quantize_

        model = nn.Sequential(
            nn.Linear(32, 16, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(16, 8, dtype=torch.bfloat16),
        )

        # Quantize
        quantize_(model, FP8E4M3Po2Config())

        # Rewrite for export
        rewrite_for_export(model)

        # Verify
        alignment = npu_alignment_check(model)
        assert alignment.passed

        # Forward should still work
        x = torch.randn(4, 32, dtype=torch.bfloat16)
        out = model(x)
        assert out.shape == (4, 8)


class TestContractsIntegration:
    """Test that payload contracts recognize FP8 dtype."""

    def test_f8e4m3_in_dtype_bytes(self) -> None:
        from compgen.ir.payload.contracts import _dtype_bytes

        assert _dtype_bytes("f8e4m3") == 1

    def test_dtype_bytes_backward_compatible(self) -> None:
        from compgen.ir.payload.contracts import _dtype_bytes

        assert _dtype_bytes("f32") == 4
        assert _dtype_bytes("bf16") == 2
        assert _dtype_bytes("f16") == 2
        assert _dtype_bytes("i8") == 1


class TestPackageImports:
    """Test that all public APIs are importable from the package."""

    def test_import_config(self) -> None:
        from compgen.quantization import FP8E4M3Po2Config
        assert FP8E4M3Po2Config is not None

    def test_import_tensor(self) -> None:
        from compgen.quantization import FP8E4M3Po2Tensor
        assert FP8E4M3Po2Tensor is not None

    def test_import_attention(self) -> None:
        from compgen.quantization import ExportableFP8Attention
        assert ExportableFP8Attention is not None

    def test_import_recipe(self) -> None:
        from compgen.quantization import SmolVLAQuantRecipe, apply_smolvla_quantization
        assert SmolVLAQuantRecipe is not None
        assert apply_smolvla_quantization is not None

    def test_import_npu_map(self) -> None:
        from compgen.quantization import NpuOpCategory, classify_op
        assert NpuOpCategory is not None
        assert classify_op is not None

    def test_import_export(self) -> None:
        from compgen.quantization import rewrite_for_export
        assert rewrite_for_export is not None

    def test_import_verify(self) -> None:
        from compgen.quantization import npu_alignment_check
        assert npu_alignment_check is not None
