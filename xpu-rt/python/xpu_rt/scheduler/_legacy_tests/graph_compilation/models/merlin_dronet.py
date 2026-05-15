"""Adapter for /scratch2/agustin/merlin/models/dronet/dronet.py.

DronetTorch is a small CNN regression / classification head for
quadcopter steering. The smaller 112x112 variant fits comfortably in
torch.compile capture and exercises real conv/batchnorm/elementwise
lowering surface in our graph_compilation pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_MERLIN_DRONET = Path("/scratch2/agustin/merlin/models/dronet")


def _load_dronet() -> type[nn.Module]:
    if str(_MERLIN_DRONET) not in sys.path:
        sys.path.insert(0, str(_MERLIN_DRONET))
    import importlib

    module = importlib.import_module("dronet")
    cls: type[nn.Module] = module.DronetTorch
    return cls


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    Dronet = _load_dronet()
    model = Dronet(img_dims=(112, 112), img_channels=3, output_dim=1).eval()
    x = torch.randn(1, 3, 112, 112)
    return model, (x,)
