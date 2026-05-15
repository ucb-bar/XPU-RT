"""Proxy for Qwen-VL-style models: tiny ViT image encoder + transformer LM.

Exercises: conv2d patch embed, attention, LayerNorm, GELU, projection from
visual tokens to text-token embeddings, causal LM head over a small vocab.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 3, embed_dim: int = 32, patch: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = self.proj(pixels)  # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x


class _Block(nn.Module):
    def __init__(self, embed_dim: int = 32, n_heads: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class ProxyQwenVL(nn.Module):
    """Vision tokens + text tokens fused through 2 transformer blocks."""

    def __init__(self, vocab: int = 64, embed_dim: int = 32, image_size: int = 32) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch = _PatchEmbed(in_ch=3, embed_dim=embed_dim, patch=8)
        self.tok_emb = nn.Embedding(vocab, embed_dim)
        self.fuse = nn.Linear(embed_dim, embed_dim)
        self.block1 = _Block(embed_dim=embed_dim)
        self.block2 = _Block(embed_dim=embed_dim)
        self.head = nn.Linear(embed_dim, vocab)

    def forward(self, pixels: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        vis = self.patch(pixels)
        vis = self.fuse(vis)
        txt = self.tok_emb(input_ids)
        x = torch.cat([vis, txt], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        logits = self.head(x[:, -input_ids.shape[1] :, :])
        return F.log_softmax(logits, dim=-1)


def build_proxy(slice_id: str = "") -> tuple[nn.Module, tuple[Any, ...]]:
    torch.manual_seed(0)
    model = ProxyQwenVL()
    pixels = torch.randn(1, 3, 32, 32)
    input_ids = torch.randint(0, 64, (1, 8))
    return model, (pixels, input_ids)
