"""Synthetic-but-structured frontier workload families."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from compgen.models.core import ModelSource, ModelSpec


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_sq + self.eps)
        return x * inv_rms * self.weight


class SimpleAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        attn = attn.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.hidden_dim)
        return self.out_proj(attn)


class LlamaDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int = 512, num_heads: int = 8, ff_dim: int = 1536) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = SimpleAttention(hidden_dim, num_heads)
        self.ff_norm = RMSNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, ff_dim)
        self.up_proj = nn.Linear(hidden_dim, ff_dim)
        self.down_proj = nn.Linear(ff_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.attn_norm(x))
        x = x + attn_out
        ff_in = self.ff_norm(x)
        gated = F.silu(self.gate_proj(ff_in)) * self.up_proj(ff_in)
        return x + self.down_proj(gated)


class LlamaSliceModel(nn.Module):
    def __init__(self, hidden_dim: int = 768, num_heads: int = 12, ff_dim: int = 3072, depth: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            LlamaDecoderBlock(hidden_dim=hidden_dim, num_heads=num_heads, ff_dim=ff_dim) for _ in range(depth)
        )
        self.final_norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class MoERouterExpertBlock(nn.Module):
    def __init__(self, hidden_dim: int = 512, num_experts: int = 4, ff_dim: int = 1536) -> None:
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, hidden_dim),
            )
            for _ in range(num_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing = torch.softmax(self.router(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return (routing.unsqueeze(-1) * expert_outputs).sum(dim=-2)


class DLRMv3RankingBlock(nn.Module):
    def __init__(self, dense_dim: int = 64, embed_dim: int = 64, num_tables: int = 4, vocab_size: int = 1024) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(vocab_size, embed_dim) for _ in range(num_tables))
        self.dense_proj = nn.Linear(dense_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, dense_features: torch.Tensor, sparse_features: torch.Tensor) -> torch.Tensor:
        embedded = [table(sparse_features[:, idx]) for idx, table in enumerate(self.embeddings)]
        dense_token = self.dense_proj(dense_features)
        tokens = torch.stack([dense_token, *embedded], dim=1)
        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (tokens.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.matmul(weights, v).mean(dim=1)
        return self.mlp(pooled)


class MambaStyleBlock(nn.Module):
    def __init__(self, hidden_dim: int = 256, state_dim: int = 256) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, state_dim * 2)
        self.out_proj = nn.Linear(state_dim, hidden_dim)
        self.bias = nn.Parameter(torch.randn(state_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        state = torch.zeros_like(value[:, 0, :])
        outputs: list[torch.Tensor] = []
        for step in range(value.shape[1]):
            state = torch.tanh(state + value[:, step, :] + self.bias)
            outputs.append(state * torch.sigmoid(gate[:, step, :]))
        return self.out_proj(torch.stack(outputs, dim=1))


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.pw2 = nn.Conv2d(channels * 4, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = F.gelu(self.pw1(x))
        x = self.pw2(x)
        return x + residual


class ConvNeXtStage(nn.Module):
    def __init__(self, channels: int = 96, depth: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(ConvNeXtBlock(channels) for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _register(specs: list[ModelSpec], spec: ModelSpec) -> None:
    specs.append(spec)


def _local_loader(model: nn.Module, inputs: tuple[Any, ...]) -> Any:
    def _load(_workspace: Any = None) -> tuple[nn.Module, tuple[Any, ...]]:
        return model, inputs

    return _load


def build_frontier_model_specs() -> list[ModelSpec]:
    """Return the synthetic frontier workloads used by the benchmark harness."""

    specs: list[ModelSpec] = []

    _register(
        specs,
        ModelSpec(
            model_id="llama31_decoder_block",
            family="decoder_only_transformer",
            description="Llama-style decoder block for dense-transformer benchmarking",
            loader=_local_loader(LlamaDecoderBlock(), (torch.randn(2, 32, 512),)),
            source=ModelSource(kind="synthetic_family", identifier="llama31_decoder_block"),
            source_model_id="meta-llama/Llama-3.1-8B",
            tags=("frontier", "transformer", "block"),
        ),
    )
    _register(
        specs,
        ModelSpec(
            model_id="llama31_8b_slice",
            family="decoder_only_transformer",
            description="Small multi-block slice representing a Llama 3.1 8B-style stack",
            loader=_local_loader(LlamaSliceModel(), (torch.randn(1, 128, 768),)),
            source=ModelSource(kind="synthetic_family", identifier="llama31_8b_slice"),
            source_model_id="meta-llama/Llama-3.1-8B",
            tags=("frontier", "transformer", "slice"),
        ),
    )
    _register(
        specs,
        ModelSpec(
            model_id="llama4_moe_router_expert_block",
            family="moe_transformer",
            description="Router plus expert FFN block capturing MoE routing structure",
            loader=_local_loader(MoERouterExpertBlock(), (torch.randn(2, 32, 512),)),
            source=ModelSource(kind="synthetic_family", identifier="llama4_moe_router_expert_block"),
            source_model_id="meta-llama/Llama-4",
            tags=("frontier", "moe", "block"),
        ),
    )
    _register(
        specs,
        ModelSpec(
            model_id="dlrmv3_ranking_block",
            family="recommendation",
            description="Recommendation ranking block with embeddings and attention-heavy interaction",
            loader=_local_loader(
                DLRMv3RankingBlock(),
                (torch.randn(8, 64), torch.randint(0, 1024, (8, 4))),
            ),
            source=ModelSource(kind="synthetic_family", identifier="dlrmv3_ranking_block"),
            source_model_id="dlrmv3",
            tags=("frontier", "recommendation", "block"),
        ),
    )
    _register(
        specs,
        ModelSpec(
            model_id="mamba_block",
            family="state_space",
            description="Mamba-style scan-heavy sequence block",
            loader=_local_loader(MambaStyleBlock(), (torch.randn(4, 64, 256),)),
            source=ModelSource(kind="synthetic_family", identifier="mamba_block"),
            source_model_id="state-spaces/mamba",
            tags=("frontier", "state_space", "block"),
        ),
    )
    _register(
        specs,
        ModelSpec(
            model_id="convnext_stage",
            family="vision",
            description="ConvNeXt-style stage for modern vision workload coverage",
            loader=_local_loader(ConvNeXtStage(), (torch.randn(2, 96, 56, 56),)),
            source=ModelSource(kind="synthetic_family", identifier="convnext_stage"),
            source_model_id="facebook/convnext",
            tags=("frontier", "vision", "block"),
        ),
    )

    return specs
