"""Proxy for OCR-style models (DeepSeek-OCR-3B family).

Tiny CNN encoder over a page-shaped (1x3xHxW) image and a small token
decoder that emits a logit grid over a tiny vocab.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class _CNNEncoder(nn.Module):
    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layers(x)
        return h.flatten(2).transpose(1, 2)


class _CTCHead(nn.Module):
    def __init__(self, embed_dim: int = 32, vocab: int = 64) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.ln(x))


class ProxyOCR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = _CNNEncoder()
        self.head = _CTCHead()

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(pixels))


def build_proxy(slice_id: str = "") -> tuple[nn.Module, tuple[Any, ...]]:
    torch.manual_seed(0)
    model = ProxyOCR()
    pixels = torch.randn(1, 3, 64, 64)
    return model, (pixels,)
