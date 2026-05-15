"""Reference module source for Flash Attention 1 evaluation.

The autocomp search backend ``CompGenTorchEvalBackend`` consumes a
*string* containing a Python module that defines ``Model``,
``get_inputs``, and ``get_init_inputs``. This module exposes that
string for the FA-1 contract (B=1, H=1, S=128, D=64, fp16) used by
the Phase F validation harness.
"""

from __future__ import annotations

FA1_REF_SOURCE = '''\
import torch
import torch.nn as nn


class Model(nn.Module):
    """Eager Flash-Attention-1 reference: softmax(Q K^T / sqrt(D)) V."""

    def __init__(self, scale: float = 1.0 / (64 ** 0.5)) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, q, k, v):  # (B, H, S, D) for each
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)


def get_init_inputs():
    return (1.0 / (64 ** 0.5),)


def get_inputs():
    torch.manual_seed(0)
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, dtype=torch.float16)
    k = torch.randn(B, H, S, D, dtype=torch.float16)
    v = torch.randn(B, H, S, D, dtype=torch.float16)
    return (q, k, v)
'''
