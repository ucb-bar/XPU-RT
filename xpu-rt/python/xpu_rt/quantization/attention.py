"""FP8-aware attention with BF16 softmax for NPU deployment.

Provides ``ExportableFP8Attention``, a module that replaces PyTorch's fused
``F.scaled_dot_product_attention`` with an explicit unfused implementation
where:

  - Q, K, V are quantized to FP8 E4M3 (po2 scaling) before score matmuls
  - Softmax **always** runs in BF16 (never quantized)
  - Attention weights after softmax are **always** quantized to FP8 E4M3
    before the ``attn_weights @ V`` matmul
  - Output is optionally quantized to FP8

This matches pi0-quant's hardware-faithful attention path
(``model_patcher.py`` lines 639-829) and maps directly to the NPU's
execution model: MXU for matmuls (FP8 in, BF16 accum), VPU for softmax
(BF16), ``vpack.bf16.fp8`` for attention weight packing.

The unfused form is necessary for the NPU compiler to see individual matmuls
and apply tile scheduling to each 32x32 MXU operation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from compgen.quantization.fp8_ops import (
    dequantize_fp8_e4m3,
    quantize_fp8_e4m3_po2,
)


@dataclass(frozen=True)
class FP8AttentionConfig:
    """Configuration for FP8-aware attention.

    Attributes:
        quantize_qkv: Quantize Q, K, V to FP8 before score matmuls.
        quantize_attn_weights: Quantize attention weights after softmax to FP8.
            This is **always True** per pi0-quant spec and NPU hardware model.
        quantize_output: Quantize the final attention output to FP8.
        softmax_dtype: Softmax always runs in this dtype (BF16).
    """

    quantize_qkv: bool = True
    quantize_attn_weights: bool = True
    quantize_output: bool = False
    softmax_dtype: torch.dtype = torch.bfloat16


class ExportableFP8Attention(nn.Module):
    """Explicit unfused attention with FP8 quantization for NPU export.

    Replaces ``F.scaled_dot_product_attention`` with a sequence of standard
    ATen ops that are all individually export-friendly and mappable to NPU ISA.

    The forward pass implements:

    1. Q, K, V -> quantize to FP8 E4M3 -> dequantize to BF16 (bake in FP8 noise)
    2. scores = Q_bf16 @ K_bf16^T / sqrt(head_dim)
    3. scores += attn_mask (if provided)
    4. attn_weights = softmax(scores)  -- **BF16, never quantized**
    5. attn_weights -> quantize to FP8 E4M3 -> dequantize to BF16
    6. out = attn_weights_bf16 @ V_bf16
    7. optionally quantize output to FP8

    Args:
        config: FP8 attention configuration.
    """

    def __init__(self, config: FP8AttentionConfig | None = None) -> None:
        super().__init__()
        self.config = config or FP8AttentionConfig()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Compute FP8-aware multi-head attention.

        Args:
            query: (batch, num_heads, seq_q, head_dim) or (batch, seq_q, head_dim).
            key: Same layout as query.
            value: Same layout as query.
            attn_mask: Optional attention mask.
            dropout_p: Dropout probability (ignored during inference).
            is_causal: If True, apply causal mask.
            scale: Optional scaling factor. Defaults to 1/sqrt(head_dim).

        Returns:
            Attention output with same shape as query.
        """
        head_dim = query.size(-1)
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        # Step 1: Quantize Q, K, V to FP8 -> dequantize to BF16
        if self.config.quantize_qkv:
            q = _quantize_dequantize_bf16(query)
            k = _quantize_dequantize_bf16(key)
            v = _quantize_dequantize_bf16(value)
        else:
            q = query.to(torch.bfloat16)
            k = key.to(torch.bfloat16)
            v = value.to(torch.bfloat16)

        # Step 2: Score matmul (Q @ K^T) in BF16 -> maps to MXU vmatmul
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Step 3: Apply attention mask
        if is_causal:
            seq_q, seq_k = q.size(-2), k.size(-2)
            causal_mask = torch.triu(
                torch.full((seq_q, seq_k), float("-inf"), device=q.device, dtype=q.dtype),
                diagonal=seq_k - seq_q + 1,
            )
            scores = scores + causal_mask
        elif attn_mask is not None:
            scores = scores + attn_mask.to(scores.dtype)

        # Step 4: Softmax in BF16 -- NEVER quantized
        attn_weights = F.softmax(scores.to(self.config.softmax_dtype), dim=-1)

        # Step 5: Quantize attention weights to FP8 E4M3 -> dequantize to BF16
        # This is ALWAYS done per pi0-quant spec (hardware faithful)
        # Maps to NPU's vpack.bf16.fp8 before second MXU matmul
        if self.config.quantize_attn_weights:
            attn_weights = _quantize_dequantize_bf16(attn_weights)

        # Step 6: Output matmul (attn_weights @ V) in BF16 -> maps to MXU vmatmul
        out = torch.matmul(attn_weights, v)

        # Step 7: Optional output quantization
        if self.config.quantize_output:
            out = _quantize_dequantize_bf16(out)

        return out


def _quantize_dequantize_bf16(x: torch.Tensor) -> torch.Tensor:
    """Quantize to FP8 E4M3 (po2 scaling) and dequantize back to BF16.

    This bakes FP8 quantization noise into the tensor while keeping the
    computation in BF16, matching the NPU's MXU pipeline.
    """
    x_fp8, scale = quantize_fp8_e4m3_po2(x)
    return dequantize_fp8_e4m3(x_fp8, scale, torch.bfloat16)


def replace_sdpa_with_fp8_attention(
    model: nn.Module,
    config: FP8AttentionConfig | None = None,
) -> list[str]:
    """Walk a model and replace SDPA-using attention modules with FP8 attention.

    This function looks for modules that have a ``_fp8_attention`` attribute
    (set by the smolVLA recipe) or can be identified as attention layers by
    naming convention (containing ``"self_attn"`` or ``"attention"``).

    For each identified attention module, it adds an ``fp8_attn`` sub-module
    that should be called instead of ``F.scaled_dot_product_attention``.

    Args:
        model: The model to patch.
        config: FP8 attention config.  Defaults to standard NPU settings.

    Returns:
        List of patched module paths.
    """
    cfg = config or FP8AttentionConfig()
    patched: list[str] = []

    for name, module in model.named_modules():
        # Detect attention modules by name convention
        last_part = name.split(".")[-1] if name else ""
        if last_part in ("self_attn", "attention", "attn"):
            if not hasattr(module, "fp8_attn"):
                module.fp8_attn = ExportableFP8Attention(cfg)
                patched.append(name)

    return patched


__all__ = [
    "ExportableFP8Attention",
    "FP8AttentionConfig",
    "replace_sdpa_with_fp8_attention",
]
