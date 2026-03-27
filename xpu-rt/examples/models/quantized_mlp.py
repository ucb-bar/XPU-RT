"""Quantized MLP model for CompGen pipeline testing.

SimpleMLP with int8 weight-only quantization via TorchAO. Tests whether
torch.export handles quantized models correctly.

Note: TorchAO may not be installed in all environments. This model
gracefully falls back to the unquantized version if torchao is missing.

Usage:
    python examples/models/quantized_mlp.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Same architecture as simple_mlp.py, defined here for self-containedness."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 3072, output_dim: int = 768) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def get_sample_inputs(batch_size: int = 8, input_dim: int = 768) -> tuple[torch.Tensor, ...]:
    """Generate sample inputs."""
    return (torch.randn(batch_size, input_dim),)


def get_model_and_inputs(batch_size: int = 8) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    """Get quantized model and inputs. Falls back to unquantized if torchao unavailable."""
    model = SimpleMLP()
    inputs = get_sample_inputs(batch_size=batch_size)

    try:
        from torchao.quantization import int8_weight_only, quantize_

        quantize_(model, int8_weight_only())
        print("Quantization applied: int8_weight_only")
    except ImportError:
        print("torchao not available, using unquantized model")
    except Exception as e:
        print(f"Quantization failed ({e}), using unquantized model")

    return model, inputs


if __name__ == "__main__":
    model, inputs = get_model_and_inputs()
    output = model(*inputs)
    print(f"Input:  {inputs[0].shape} ({inputs[0].dtype})")
    print(f"Output: {output.shape} ({output.dtype})")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
