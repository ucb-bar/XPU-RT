"""Simple MLP model for CompGen pipeline testing.

A minimal 3-layer MLP (Linear -> GELU -> Linear) that exercises
the basic capture and IR construction pipeline:

    torch.export -> FX graph -> xDSL IR -> canonicalize

This model is intentionally simple -- the goal is to validate the
pipeline end-to-end, not to test complex model patterns.

Usage:
    python -m examples.models.simple_mlp

    # Or from the repo root:
    python examples/models/simple_mlp.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """A minimal MLP for pipeline testing.

    Architecture: Linear(input_dim, hidden_dim) -> GELU -> Linear(hidden_dim, output_dim)
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 3072,
        output_dim: int = 768,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def get_sample_inputs(
    batch_size: int = 8,
    input_dim: int = 768,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    """Generate sample inputs for the SimpleMLP."""
    return (torch.randn(batch_size, input_dim, dtype=dtype),)


def get_model_and_inputs(
    batch_size: int = 8,
) -> tuple[SimpleMLP, tuple[torch.Tensor, ...]]:
    """Get model and matching sample inputs."""
    model = SimpleMLP()
    inputs = get_sample_inputs(batch_size=batch_size)
    return model, inputs


if __name__ == "__main__":
    model, inputs = get_model_and_inputs()
    output = model(*inputs)
    print("Model:  SimpleMLP(768 -> 3072 -> 768)")
    print(f"Input:  {inputs[0].shape} ({inputs[0].dtype})")
    print(f"Output: {output.shape} ({output.dtype})")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
