"""Proxy for diffusion-based VLA policies (Octo-style).

Single denoising step: condition on (image features, proprio state, noisy
action) via a small transformer block; predict the noise. Real diffusion
training/sampling is out of scope here -- this is a forward pass over one
denoising step, which is what admission cares about.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class _ImageTokens(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, dim, kernel_size=8, stride=8)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(2).transpose(1, 2)
        return self.ln(h)


class _Block(nn.Module):
    def __init__(self, dim: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class ProxyDiffusionVLA(nn.Module):
    def __init__(self, action_dim: int = 7, dim: int = 32) -> None:
        super().__init__()
        self.img = _ImageTokens(dim=dim)
        self.prop_emb = nn.Linear(14, dim)
        self.act_emb = nn.Linear(action_dim, dim)
        self.t_emb = nn.Linear(1, dim)
        self.block = _Block(dim=dim)
        self.head = nn.Linear(dim, action_dim)

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        v = self.img(image)
        s = self.prop_emb(state).unsqueeze(1)
        a = self.act_emb(noisy_action).unsqueeze(1)
        c = self.t_emb(t).unsqueeze(1)
        seq = torch.cat([v, s, a, c], dim=1)
        seq = self.block(seq)
        return self.head(seq[:, -1, :])


def build_proxy(slice_id: str = "") -> tuple[nn.Module, tuple[Any, ...]]:
    torch.manual_seed(0)
    model = ProxyDiffusionVLA()
    image = torch.randn(1, 3, 32, 32)
    state = torch.randn(1, 14)
    noisy_action = torch.randn(1, 7)
    t = torch.randn(1, 1)
    return model, (image, state, noisy_action, t)
