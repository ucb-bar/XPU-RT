"""Transformer block model for CompGen pipeline testing.

A minimal transformer block (MultiheadAttention + FFN + LayerNorm)
that exercises more complex capture patterns than SimpleMLP:

    - Multi-head self-attention (decomposes to matmuls + softmax)
    - Layer normalization
    - GELU activation
    - Residual connections

Usage:
    python examples/models/transformer_block.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Minimal transformer block for pipeline testing."""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.linear2(F.gelu(self.linear1(x)))
        x = self.norm2(x + ffn_out)
        return x


def get_sample_inputs(
    batch_size: int = 4,
    seq_len: int = 16,
    d_model: int = 512,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    """Generate sample inputs for TransformerBlock."""
    return (torch.randn(batch_size, seq_len, d_model, dtype=dtype),)


def get_model_and_inputs(batch_size: int = 4) -> tuple[TransformerBlock, tuple[torch.Tensor, ...]]:
    """Get model and matching sample inputs."""
    model = TransformerBlock()
    inputs = get_sample_inputs(batch_size=batch_size)
    return model, inputs


if __name__ == "__main__":
    model, inputs = get_model_and_inputs()
    output = model(*inputs)
    print("Model:  TransformerBlock(512, 8 heads, 2048 FFN)")
    print(f"Input:  {inputs[0].shape} ({inputs[0].dtype})")
    print(f"Output: {output.shape} ({output.dtype})")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
