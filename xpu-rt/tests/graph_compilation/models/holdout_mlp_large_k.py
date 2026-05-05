"""Holdout MLP with extreme K dimension (M-31A.4).

K=8192, N=64, M=64 — tests that working-set / register-pressure
analyses behave on a skewed shape that doesn't fit the typical
`N≈M≈K` assumption baked into many cost models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HoldoutLargeKMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8192, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    model = _HoldoutLargeKMLP().eval()
    x = torch.randn(2, 8192)
    return model, (x,)
