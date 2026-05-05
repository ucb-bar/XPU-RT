"""Tiny self-attention block — exercises attention/softmax/matmul patterns.

The point is to surface a *different* op histogram from ``tiny_mlp``:
QKV projection, scaled-dot-product attention, softmax, and an output
projection. After ``torch.export``'s default decompositions, this
typically yields ``aten.mm`` / ``aten.bmm`` / ``aten.softmax``-style
ops; the FX importer's decomposition table covers some of these and
falls back to opaque ``func.call`` for the rest.

Kept tiny enough (dim=32, heads=2, T=4) so capture finishes in seconds.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyAttention(nn.Module):
    def __init__(self, dim: int = 32, heads: int = 2) -> None:
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x)
        # (B, T, 3, H, D) → (3, B, H, T, D)
        qkv = qkv.reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, H, T, D)
        out = out.transpose(1, 2).reshape(B, T, self.dim)
        return self.proj(out)


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    model = TinyAttention().eval()
    x = torch.randn(2, 4, 32)
    return model, (x,)
