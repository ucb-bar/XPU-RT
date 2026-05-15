"""Tests for NPU alignment verification and accuracy checks."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from compgen.quantization.verify import (
    compare_quantized_accuracy,
    npu_alignment_check,
)

torchao = pytest.importorskip("torchao")


def _quantized_mlp() -> nn.Module:
    """Create and quantize a simple MLP."""
    from compgen.quantization.fp8_config import FP8E4M3Po2Config
    from torchao.quantization import quantize_

    model = nn.Sequential(
        nn.Linear(32, 64, dtype=torch.bfloat16),
        nn.ReLU(),
        nn.Linear(64, 16, dtype=torch.bfloat16),
    )
    quantize_(model, FP8E4M3Po2Config())
    return model


class TestNpuAlignmentCheck:
    def test_quantized_model_passes(self) -> None:
        model = _quantized_mlp()
        result = npu_alignment_check(model)
        assert result.passed is True
        assert result.fp8_linear_count == 2
        assert len(result.non_po2_scales) == 0
        assert len(result.errors) == 0

    def test_unquantized_linear_warned(self) -> None:
        model = nn.Sequential(
            nn.Linear(16, 8, dtype=torch.bfloat16),
        )
        result = npu_alignment_check(model)
        assert len(result.unquantized_linears) == 1
        assert "0" in result.unquantized_linears[0]

    def test_lm_head_allowed_unquantized(self) -> None:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(16, 8, dtype=torch.bfloat16)
                self.lm_head = nn.Linear(8, 100, dtype=torch.bfloat16)

            def forward(self, x):
                return self.lm_head(self.proj(x))

        from compgen.quantization.fp8_config import FP8E4M3Po2Config
        from torchao.quantization import quantize_

        model = M()
        # Only quantize proj, skip lm_head
        quantize_(model, FP8E4M3Po2Config(), filter_fn=lambda m, fqn: fqn == "proj")
        result = npu_alignment_check(model, allow_unquantized={"lm_head"})
        assert result.fp8_linear_count == 1
        # lm_head should NOT appear in unquantized warnings
        assert not any("lm_head" in name for name in result.unquantized_linears)

    def test_rewritten_model_passes(self) -> None:
        from compgen.quantization.export_wrappers import rewrite_for_export

        model = _quantized_mlp()
        rewrite_for_export(model)
        result = npu_alignment_check(model)
        assert result.passed is True
        assert result.fp8_linear_count == 2


class TestCompareAccuracy:
    def test_same_model_perfect_accuracy(self) -> None:
        model = nn.Linear(16, 8, dtype=torch.bfloat16)
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        result = compare_quantized_accuracy(model, model, (x,))
        assert result["l2_error"] == 0.0
        assert result["cosine_similarity"] == pytest.approx(1.0)

    def test_quantized_vs_original(self) -> None:
        import copy

        original = nn.Sequential(
            nn.Linear(32, 16, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(16, 8, dtype=torch.bfloat16),
        )
        quantized = copy.deepcopy(original)

        from compgen.quantization.fp8_config import FP8E4M3Po2Config
        from torchao.quantization import quantize_

        quantize_(quantized, FP8E4M3Po2Config())

        x = torch.randn(4, 32, dtype=torch.bfloat16)
        result = compare_quantized_accuracy(original, quantized, (x,))

        # Should have some error but not huge
        assert result["l2_error"] > 0
        assert result["cosine_similarity"] > 0.8
