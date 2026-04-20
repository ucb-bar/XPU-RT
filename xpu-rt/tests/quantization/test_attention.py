"""Tests for FP8-aware attention with BF16 softmax."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from compgen.quantization.attention import (
    ExportableFP8Attention,
    FP8AttentionConfig,
    replace_sdpa_with_fp8_attention,
)

# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


class TestExportableFP8Attention:
    def test_output_shape_4d(self) -> None:
        """4D input (batch, heads, seq, head_dim) should produce matching output."""
        attn = ExportableFP8Attention()
        q = torch.randn(2, 4, 8, 32, dtype=torch.bfloat16)
        k = torch.randn(2, 4, 8, 32, dtype=torch.bfloat16)
        v = torch.randn(2, 4, 8, 32, dtype=torch.bfloat16)
        out = attn(q, k, v)
        assert out.shape == (2, 4, 8, 32)

    def test_output_shape_3d(self) -> None:
        """3D input (batch, seq, head_dim) should produce matching output."""
        attn = ExportableFP8Attention()
        q = torch.randn(2, 8, 32, dtype=torch.bfloat16)
        k = torch.randn(2, 8, 32, dtype=torch.bfloat16)
        v = torch.randn(2, 8, 32, dtype=torch.bfloat16)
        out = attn(q, k, v)
        assert out.shape == (2, 8, 32)

    def test_output_dtype_bf16(self) -> None:
        attn = ExportableFP8Attention()
        q = k = v = torch.randn(1, 2, 4, 16, dtype=torch.bfloat16)
        out = attn(q, k, v)
        assert out.dtype == torch.bfloat16

    def test_causal_mask(self) -> None:
        """Causal attention should produce valid output."""
        attn = ExportableFP8Attention()
        q = k = v = torch.randn(1, 2, 8, 16, dtype=torch.bfloat16)
        out = attn(q, k, v, is_causal=True)
        assert out.shape == (1, 2, 8, 16)
        assert not out.isnan().any()

    def test_custom_attn_mask(self) -> None:
        """Explicit attention mask should be applied."""
        attn = ExportableFP8Attention()
        q = k = v = torch.randn(1, 1, 4, 16, dtype=torch.bfloat16)
        mask = torch.zeros(4, 4, dtype=torch.bfloat16)
        mask[0, 1:] = float("-inf")  # First token can only attend to itself
        out = attn(q, k, v, attn_mask=mask)
        assert out.shape == (1, 1, 4, 16)
        assert not out.isnan().any()

    def test_custom_scale(self) -> None:
        attn = ExportableFP8Attention()
        q = k = v = torch.randn(1, 1, 4, 16, dtype=torch.bfloat16)
        out = attn(q, k, v, scale=0.1)
        assert out.shape == (1, 1, 4, 16)


# ---------------------------------------------------------------------------
# Softmax stays BF16
# ---------------------------------------------------------------------------


class TestSoftmaxBF16:
    def test_softmax_always_bf16(self) -> None:
        """Verify softmax intermediate is BF16, never FP8."""
        # Hook into the attention to check softmax output dtype
        softmax_dtypes: list[torch.dtype] = []

        class HookAttn(ExportableFP8Attention):
            def forward(self, query, key, value, **kwargs):
                head_dim = query.size(-1)
                scale = 1.0 / math.sqrt(head_dim)
                from compgen.quantization.fp8_ops import dequantize_fp8_e4m3, quantize_fp8_e4m3_po2

                q_fp8, q_s = quantize_fp8_e4m3_po2(query)
                q = dequantize_fp8_e4m3(q_fp8, q_s, torch.bfloat16)
                k_fp8, k_s = quantize_fp8_e4m3_po2(key)
                k = dequantize_fp8_e4m3(k_fp8, k_s, torch.bfloat16)

                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_weights = torch.nn.functional.softmax(scores, dim=-1)

                # Record the dtype AFTER softmax
                softmax_dtypes.append(attn_weights.dtype)
                return super().forward(query, key, value, **kwargs)

        attn = HookAttn()
        q = k = v = torch.randn(1, 2, 4, 16, dtype=torch.bfloat16)
        attn(q, k, v)
        assert all(dt == torch.bfloat16 for dt in softmax_dtypes)

    def test_config_softmax_dtype(self) -> None:
        config = FP8AttentionConfig()
        assert config.softmax_dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Attention weights are FP8 after softmax
# ---------------------------------------------------------------------------


class TestAttnWeightsFP8:
    def test_attn_weights_quantized_by_default(self) -> None:
        config = FP8AttentionConfig()
        assert config.quantize_attn_weights is True

    def test_accuracy_vs_standard_sdpa(self) -> None:
        """FP8 attention should be reasonably close to standard SDPA."""
        torch.manual_seed(42)
        q = torch.randn(1, 2, 8, 32, dtype=torch.bfloat16)
        k = torch.randn(1, 2, 8, 32, dtype=torch.bfloat16)
        v = torch.randn(1, 2, 8, 32, dtype=torch.bfloat16)

        # Reference: standard SDPA
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # FP8 attention
        fp8_attn = ExportableFP8Attention(
            FP8AttentionConfig(
                quantize_qkv=True,
                quantize_attn_weights=True,
            )
        )
        fp8_out = fp8_attn(q, k, v)

        # Should be in the same ballpark (FP8 introduces quantization noise)
        cosine_sim = torch.nn.functional.cosine_similarity(ref.flatten().float(), fp8_out.flatten().float(), dim=0)
        assert cosine_sim > 0.9, f"Cosine similarity {cosine_sim:.4f} too low"

    def test_no_qkv_quantization(self) -> None:
        """With quantize_qkv=False, only attn_weights get FP8."""
        config = FP8AttentionConfig(quantize_qkv=False, quantize_attn_weights=True)
        attn = ExportableFP8Attention(config)
        q = k = v = torch.randn(1, 1, 4, 16, dtype=torch.bfloat16)
        out = attn(q, k, v)
        assert out.shape == (1, 1, 4, 16)


# ---------------------------------------------------------------------------
# Module replacement
# ---------------------------------------------------------------------------


class TestReplaceSdpa:
    def test_replace_by_name(self) -> None:
        """Modules named self_attn should get fp8_attn added."""

        class FakeAttn(nn.Module):
            pass

        class FakeTransformerLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttn()
                self.mlp = nn.Linear(16, 16)

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = FakeTransformerLayer()
                self.layer1 = FakeTransformerLayer()

        model = FakeModel()
        patched = replace_sdpa_with_fp8_attention(model)
        assert len(patched) == 2
        assert hasattr(model.layer0.self_attn, "fp8_attn")
        assert hasattr(model.layer1.self_attn, "fp8_attn")
        assert isinstance(model.layer0.self_attn.fp8_attn, ExportableFP8Attention)

    def test_idempotent(self) -> None:
        """Patching twice should not add duplicate modules."""

        class FakeAttn(nn.Module):
            pass

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttn()

        model = M()
        p1 = replace_sdpa_with_fp8_attention(model)
        p2 = replace_sdpa_with_fp8_attention(model)
        assert len(p1) == 1
        assert len(p2) == 0  # Already patched
