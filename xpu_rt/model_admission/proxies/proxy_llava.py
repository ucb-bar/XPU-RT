"""Proxy for LLaVA-style models: separate vision tower + language model joined
through a small MLP projector and decoded with cross-attention. Tiny but real.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class _VisionTower(nn.Module):
    def __init__(self, embed_dim: int = 24) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(2).transpose(1, 2)
        return self.ln(h)


class _Projector(nn.Module):
    def __init__(self, vis: int = 24, lm: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(vis, lm * 2)
        self.fc2 = nn.Linear(lm * 2, lm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _LMBlock(nn.Module):
    def __init__(self, embed_dim: int = 32, n_heads: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))

    def forward(self, x: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        h, _ = self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + h
        h, _ = self.cross_attn(self.ln2(x), vis, vis, need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln3(x))
        return x


class ProxyLlava(nn.Module):
    def __init__(self, vocab: int = 64, lm_dim: int = 32) -> None:
        super().__init__()
        self.vis = _VisionTower()
        self.proj = _Projector(vis=24, lm=lm_dim)
        self.tok_emb = nn.Embedding(vocab, lm_dim)
        self.block = _LMBlock(embed_dim=lm_dim)
        self.head = nn.Linear(lm_dim, vocab)

    def forward(self, pixels: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        vis = self.proj(self.vis(pixels))
        x = self.tok_emb(input_ids)
        x = self.block(x, vis)
        return self.head(x)


def build_proxy(slice_id: str = "") -> tuple[nn.Module, tuple[Any, ...]]:
    torch.manual_seed(0)
    model = ProxyLlava()
    pixels = torch.randn(1, 3, 32, 32)
    input_ids = torch.randint(0, 64, (1, 8))
    return model, (pixels, input_ids)
