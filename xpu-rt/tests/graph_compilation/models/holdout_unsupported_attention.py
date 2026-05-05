"""Holdout: scaled-dot-product attention CompGen does not lower (M-31A.4).

Expected outcome: typed-blocked with an ``unsupported_op`` reason.
A silent partial pass — claiming it lowered but actually skipping
the SDPA region — is the failure mode this model catches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HoldoutUnsupportedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape to [batch, heads=4, seq, head_dim=16]
        b, s = q.shape[0], q.shape[1]
        q = q.view(b, s, 4, 16).transpose(1, 2)
        k = k.view(b, s, 4, 16).transpose(1, 2)
        v = v.view(b, s, 4, 16).transpose(1, 2)
        # SDPA — CompGen does not lower this end-to-end yet.
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(b, s, 64)


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    model = _HoldoutUnsupportedAttention().eval()
    x = torch.randn(1, 16, 64)
    return model, (x,)
