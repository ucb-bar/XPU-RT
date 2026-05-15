"""Proxy vision-language-action model — exercises (vision + state) → action fusion.

Mirrors the high-level shape of SmolVLA / OpenVLA / similar robotics
policies: a vision encoder output, a low-dim proprioceptive/state
vector, and a continuous action head with bounded outputs (``tanh``).

The real models are 100M+ params; this proxy is ~1k params so it
captures cheaply but exercises the same op patterns CompGen will see
when lowering production VLAs:

- two parallel linear projections + ReLU
- concat-then-linear head
- ``tanh`` activation on the action output
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyVLA(nn.Module):
    def __init__(
        self,
        img_feature_dim: int = 32,
        state_dim: int = 8,
        hidden: int = 32,
        action_dim: int = 7,
    ) -> None:
        super().__init__()
        self.img_enc = nn.Linear(img_feature_dim, hidden)
        self.state_enc = nn.Linear(state_dim, hidden)
        self.action_head = nn.Linear(hidden * 2, action_dim)

    def forward(self, img_feat: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        i = F.relu(self.img_enc(img_feat))
        s = F.relu(self.state_enc(state))
        fused = torch.cat([i, s], dim=-1)
        return torch.tanh(self.action_head(fused))


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    model = ProxyVLA().eval()
    img_feat = torch.randn(1, 32)
    state = torch.randn(1, 8)
    return model, (img_feat, state)
