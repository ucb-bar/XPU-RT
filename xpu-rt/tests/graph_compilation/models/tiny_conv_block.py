"""Tiny Conv2d + BatchNorm + ReLU block — exercises conv/normalization patterns.

This stresses the lowering path with a different op family from
``tiny_mlp`` and ``tiny_attention``: ``aten.convolution``,
``aten.native_batch_norm`` (or ``aten.batch_norm``), and ``aten.relu``.
After ``torch.export`` default decompositions some of these become
``aten.convolution``/``aten.add``/``aten.mul`` patterns; the rest are
recorded honestly as opaque ``func.call``.

Kept small (3-channel 8×8 input, 8 output channels, 3×3 kernel) so
capture is cheap.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyConvBlock(nn.Module):
    def __init__(self, in_c: int = 3, out_c: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    model = TinyConvBlock().eval()
    # Calling .eval() puts BN into inference mode (running stats), which
    # is what production models do and what we want torch.export to trace.
    x = torch.randn(1, 3, 8, 8)
    return model, (x,)
