"""Holdout MLP with deliberately non-clean dimensions (M-31A.4).

K=63, N=129, M=257 — none divisible by 16/32/64 — exposes hidden
clean-divide assumptions in the tiling pipeline. Reaching ``verified``
on this model is evidence that boundary-aware paths work; reaching a
typed-blocked outcome is acceptable; a silent partial pass is not.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HoldoutOddMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Deliberately picked so neither input nor hidden nor output
        # dim is a multiple of any common tile size (16, 32, 64).
        self.fc1 = nn.Linear(63, 129)
        self.fc2 = nn.Linear(129, 257)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    model = _HoldoutOddMLP().eval()
    x = torch.randn(7, 63)  # batch=7 also non-clean
    return model, (x,)
