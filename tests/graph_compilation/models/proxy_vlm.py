"""Proxy vision-language model — exercises multi-input, embedding, and concat.

A real VLM (LLaVA/Qwen-VL/etc.) is too heavy to capture at every CI
run. This proxy keeps the *shape of the problem* — image features +
discrete token IDs fused into a joint representation — while staying
tiny enough to capture in <2s on CPU.

Op surface this exposes:

- ``nn.Embedding`` lookup (``aten.embedding``)
- mean reduction along a sequence dim
- concat across the last dim
- linear head

That overlap with real VLM lowering is enough to surface
``embedding``-style breakage early without running the full model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProxyVLM(nn.Module):
    def __init__(
        self,
        img_feature_dim: int = 32,
        token_vocab: int = 64,
        hidden: int = 32,
        out_dim: int = 16,
    ) -> None:
        super().__init__()
        self.img_proj = nn.Linear(img_feature_dim, hidden)
        self.tok_emb = nn.Embedding(token_vocab, hidden)
        self.fuser = nn.Linear(hidden * 2, out_dim)

    def forward(self, img_feat: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        # img_feat:  (B, img_feature_dim)
        # token_ids: (B, T) int64
        img = self.img_proj(img_feat)
        tok = self.tok_emb(token_ids).mean(dim=1)
        fused = torch.cat([img, tok], dim=-1)
        return self.fuser(fused)


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    model = ProxyVLM().eval()
    img_feat = torch.randn(2, 32)
    token_ids = torch.randint(0, 64, (2, 6), dtype=torch.long)
    return model, (img_feat, token_ids)
