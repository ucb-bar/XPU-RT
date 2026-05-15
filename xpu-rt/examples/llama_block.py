"""A small Llama-shaped transformer block for MCP end-to-end testing.

Exposes ``build_model() -> (nn.Module, sample_inputs)`` as expected by
:func:`compgen.api_llm._load_model_from_python_file`.

The block carries the dominant ops we target in TinyLlama:

* Q/K/V/O linears (matmul)
* RMSNorm on inputs + post-attention
* scaled dot-product attention (softmax, matmul)
* SwiGLU MLP (gate/up/down linears + silu)

Small enough to compile quickly; structured enough that the graph digest
sees matmul / softmax / rmsnorm / silu pattern clusters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class LlamaBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.input_ln = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.post_ln = RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN + self-attention
        h = self.input_ln(x)
        B, T, _ = h.shape
        q = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (self.d_head ** 0.5)
        attn = F.softmax(scores, dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(B, T, self.d_model)
        x = x + self.o_proj(ctx)

        # Pre-LN + SwiGLU MLP
        h = self.post_ln(x)
        gate = F.silu(self.gate_proj(h))
        up = self.up_proj(h)
        x = x + self.down_proj(gate * up)
        return x


def build_model() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    model = LlamaBlock(d_model=64, n_heads=4, d_ff=128).eval()
    inputs = (torch.randn(2, 8, 64),)
    return model, inputs
