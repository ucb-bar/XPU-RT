"""Adapter for /scratch2/agustin/merlin/models/mlp_wide/mlp_wide.py.

This is the variant designed for OPU 16x16 hardware tiles — wider M, K, N
dims that exercise structured matmul lowering with realistic shape
multiples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_MERLIN_MLP_WIDE = Path("/scratch2/agustin/merlin/models/mlp_wide")


def _load_wide_mlp() -> type[nn.Module]:
    if str(_MERLIN_MLP_WIDE) not in sys.path:
        sys.path.insert(0, str(_MERLIN_MLP_WIDE))
    import importlib

    module = importlib.import_module("mlp_wide")
    cls: type[nn.Module] = module.WideMLP
    return cls


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    WideMLP = _load_wide_mlp()
    model = WideMLP(input_dim=16, hidden_dim=32, output_dim=16).eval()
    x = torch.randn(16, 16)
    return model, (x,)
