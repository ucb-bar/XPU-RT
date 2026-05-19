"""Proxy for OpenVLA-style robot policies.

Image observation -> vision encoder, proprio state -> MLP, fused via concat
into an action head that emits a 7-dim continuous action (xyz + 3 rotations + gripper).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class _ImageEncoder(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, dim, kernel_size=4, stride=4),
            nn.ReLU(),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layers(x).flatten(2).mean(dim=2)  # (B, dim)
        return self.ln(h)


class _ProprioEncoder(nn.Module):
    def __init__(self, in_dim: int = 14, out_dim: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim * 2)
        self.fc2 = nn.Linear(out_dim * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))


class _ActionHead(nn.Module):
    def __init__(self, in_dim: int = 64, action_dim: int = 7) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))


class ProxyOpenVLA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img = _ImageEncoder()
        self.prop = _ProprioEncoder()
        self.head = _ActionHead()

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        v = self.img(image)
        s = self.prop(state)
        return self.head(torch.cat([v, s], dim=-1))


def build_proxy(slice_id: str = "") -> tuple[nn.Module, tuple[Any, ...]]:
    torch.manual_seed(0)
    model = ProxyOpenVLA()
    image = torch.randn(1, 3, 64, 64)
    state = torch.randn(1, 14)
    return model, (image, state)
