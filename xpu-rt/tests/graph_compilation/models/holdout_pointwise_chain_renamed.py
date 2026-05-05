"""Holdout pointwise chain with non-canonical op order (M-31A.4).

Tests that fusion / region-signature pattern matching does not depend
on a specific op-name ordering. Uses ``add → mul → relu`` (instead of
the canonical ``add → relu``) and operates on the same activation
through both branches before reducing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HoldoutPointwiseChain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((128,), 0.5))
        self.bias = nn.Parameter(torch.zeros(128))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add → mul → relu: a deliberately non-canonical pointwise
        # chain that should still be discoverable by region pattern
        # matching.
        x1 = x + self.bias
        x2 = x1 * self.scale
        x3 = F.relu(x2)
        return x3


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    model = _HoldoutPointwiseChain().eval()
    x = torch.randn(8, 128)
    return model, (x,)
