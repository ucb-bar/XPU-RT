"""Tiny MLP test model for graph_capture stage.

The factory ``get_model_and_inputs`` returns a deterministic
``(model, sample_inputs)`` pair seeded so two graph_capture stage runs produce
the same goldens (graph_capture stage determinism test).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 64, hidden: int = 128, out_dim: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    model = TinyMLP().eval()
    x = torch.randn(4, 64)
    return model, (x,)
