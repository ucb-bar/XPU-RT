"""Holdout: two matmuls consuming the same activation (M-31A.4).

Tests that the candidate-action space + region detector handle a
multi-consumer activation. A sloppy implementation that assumes every
intermediate has exactly one consumer would silently degrade here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HoldoutSharedInputTwoMatmuls(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_main = nn.Linear(64, 128)
        # Both branches consume the SAME activation.
        self.head_a = nn.Linear(128, 32)
        self.head_b = nn.Linear(128, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc_main(x))
        a = self.head_a(h)
        b = self.head_b(h)
        return a + b


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    model = _HoldoutSharedInputTwoMatmuls().eval()
    x = torch.randn(4, 64)
    return model, (x,)
