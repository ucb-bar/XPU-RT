"""Adapter for /scratch2/agustin/merlin/models/mlp/mlp.py — exposes
the merlin ``SimpleMLP`` through the standard ``get_model_and_inputs``
factory used by all CompGen graph_compilation suites.

Verifies the pipeline against a real upstream production model
definition (rather than a CompGen-internal synthetic).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_MERLIN_MLP = Path("/scratch2/agustin/merlin/models/mlp")


def _load_simple_mlp() -> type[nn.Module]:
    if str(_MERLIN_MLP) not in sys.path:
        sys.path.insert(0, str(_MERLIN_MLP))
    import importlib

    module = importlib.import_module("mlp")
    cls: type[nn.Module] = module.SimpleMLP
    return cls


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    SimpleMLP = _load_simple_mlp()
    model = SimpleMLP(input_dim=10, hidden_dim=32, output_dim=2).eval()
    x = torch.randn(1, 10)
    return model, (x,)
