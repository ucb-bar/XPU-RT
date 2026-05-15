"""Tiny model with a deliberate Python-side branch that Dynamo partitions.

Used by graph_capture stage tests to exercise the Dynamo-primary code path with
a real graph break so we can confirm:

- ``capture_dynamo_partitions`` returns >= 1 partition
- ``graph_breaks.json`` records the break honestly
- the run is still considered ``pass`` (Dynamo is primary; export may
  also pass since the branch is Python-only)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphBreakMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `print` is unconditionally a graph break under TorchDynamo.
        h = self.fc1(x)
        print("graph_break_marker")  # noqa: T201 — intentional graph break
        h = F.relu(h)
        return self.fc2(h)


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    model = GraphBreakMLP().eval()
    x = torch.randn(2, 32)
    return model, (x,)
