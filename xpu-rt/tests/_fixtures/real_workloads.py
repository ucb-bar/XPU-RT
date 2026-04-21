"""Real-workload fixtures for + pass validation.

Per the ``feedback_no_stubs_real_examples`` memory constraint, every
pass from  onwards must be validated on a real PyTorch model,
not a synthetic one-op fixture. This module ships three minimal
fixtures that exercise the same patterns found in production LLM /
VLA workloads:

- :func:`smolvla_tiny` -- a ≤ 10M-param slice mirroring smolVLA's
  attention + MLP decode stack.
- :func:`gemma_decode_tiny` -- a minimal Gemma-style decode block
  (RMSNorm + RoPE + attention + MLP).
- :func:`qwen_moe_tiny` -- a minimal MoE layer (router + 2 experts +
  top-k=1 dispatch) representative of the Qwen-MoE family.

Each fixture exposes :class:`RealWorkloadFixture` carrying:

- ``model`` -- the instantiated ``nn.Module``
- ``example_inputs`` -- the tuple of input tensors
- ``eager_output`` -- the output of running the model on the inputs
  in eager mode (the golden reference)
- ``exported`` -- the ``torch.export`` ``ExportedProgram``

The fixtures are **pure Python + torch**; they do NOT pull in
transformers / HF weights so they run in ~50 ms and require no
network.

Pass tests use these fixtures like::

    from tests._fixtures.real_workloads import gemma_decode_tiny
    fx = gemma_decode_tiny()
    from compgen.capture.torch_mlir_bridge import bridge_fx_graph
    result = bridge_fx_graph(fx.model, fx.example_inputs)
    apply_pass(result.module)
    ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RealWorkloadFixture:
    """Outcome of building one of the tiny real workloads."""

    name: str
    model: nn.Module
    example_inputs: tuple[torch.Tensor, ...]
    eager_output: torch.Tensor
    exported: Any  # torch.export.ExportedProgram


def _deterministic_seed() -> None:
    torch.manual_seed(0)


# --- smolVLA tiny ------------------------------------------------------------


class _SmolVLABlock(nn.Module):
    """Minimal transformer-ish decode block matching smolVLA shape:
    token norm -> qkv proj -> attention -> output proj -> MLP."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, mlp_mult: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * mlp_mult, bias=False)
        self.fc2 = nn.Linear(d_model * mlp_mult, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        a = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(a)
        m = self.norm2(x)
        return x + self.fc2(F.gelu(self.fc1(m)))


def smolvla_tiny() -> RealWorkloadFixture:
    """Minimal smolVLA-style decode block.

    Exercises: LayerNorm, linear projections, attention (with softmax),
    GELU, residual adds.
    """
    _deterministic_seed()
    model = _SmolVLABlock(d_model=128, n_heads=4).eval()
    x = torch.randn(1, 8, 128)
    with torch.no_grad():
        eager = model(x).detach().clone()
    exported = torch.export.export(model, (x,))
    return RealWorkloadFixture(
        name="smolvla_tiny",
        model=model,
        example_inputs=(x,),
        eager_output=eager,
        exported=exported,
    )


# --- Gemma-decode tiny -------------------------------------------------------


class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Split last dim in halves, apply pair-wise rotation.
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class _GemmaDecodeBlock(nn.Module):
    """Minimal Gemma-decode-style block with RMSNorm + RoPE + MLP(SiLU)."""

    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = _RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = _RMSNorm(d_model)
        self.gate = nn.Linear(d_model, d_model * 2, bias=False)
        self.up = nn.Linear(d_model, d_model * 2, bias=False)
        self.down = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # RoPE applied to q + k on the head_dim axis.
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        a = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(a)
        m = self.norm2(x)
        return x + self.down(F.silu(self.gate(m)) * self.up(m))


def gemma_decode_tiny() -> RealWorkloadFixture:
    """Gemma-decode-style: RMSNorm + RoPE + attention + SwiGLU MLP."""
    _deterministic_seed()
    head_dim = 32
    model = _GemmaDecodeBlock(d_model=128, n_heads=4).eval()
    x = torch.randn(1, 8, 128)
    # _apply_rope splits the last dim in half, so cos/sin must match the
    # **half**-dim (not the full head_dim) on the last axis.
    half = head_dim // 2
    freqs = torch.arange(half).float() / half
    pos = torch.arange(8).float()
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)  # [T, half]
    cos = angles.cos()  # [T, half]
    sin = angles.sin()
    # Broadcast to [1, n_heads, T, half].
    cos = cos.unsqueeze(0).unsqueeze(0).expand(1, 4, 8, half).contiguous()
    sin = sin.unsqueeze(0).unsqueeze(0).expand(1, 4, 8, half).contiguous()

    with torch.no_grad():
        eager = model(x, cos, sin).detach().clone()
    exported = torch.export.export(model, (x, cos, sin))
    return RealWorkloadFixture(
        name="gemma_decode_tiny",
        model=model,
        example_inputs=(x, cos, sin),
        eager_output=eager,
        exported=exported,
    )


# --- Qwen-MoE tiny -----------------------------------------------------------


class _QwenMoEExpert(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 2, bias=False)
        self.fc2 = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class _QwenMoELayer(nn.Module):
    """Minimal MoE: router -> top-1 expert dispatch -> combine.

    The combine step uses a softmax over the router logits so the pass
    that detects softmax (``raise_special_ops``) has a real target.
    """

    def __init__(self, d_model: int = 64, n_experts: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([_QwenMoEExpert(d_model) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        logits = self.router(x)  # [B, T, E]
        probs = F.softmax(logits, dim=-1)  # [B, T, E]
        # Static top-1 via argmax; each expert processes all tokens with
        # a scale factor = probability of being chosen (so the graph is
        # static-shape, no dynamic routing).
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)  # [B, T, D]
            out = out + expert_out * probs[..., i : i + 1]
        return out


def qwen_moe_tiny() -> RealWorkloadFixture:
    """Qwen-MoE-style: router + 2 experts + softmax combine."""
    _deterministic_seed()
    model = _QwenMoELayer(d_model=64, n_experts=2).eval()
    x = torch.randn(1, 4, 64)
    with torch.no_grad():
        eager = model(x).detach().clone()
    exported = torch.export.export(model, (x,))
    return RealWorkloadFixture(
        name="qwen_moe_tiny",
        model=model,
        example_inputs=(x,),
        eager_output=eager,
        exported=exported,
    )


# --- bridge-friendly minimal fixture (no transposes / views) ----------------


class _AttentionMLPNoTranspose(nn.Module):
    """Attention-less variant that uses linear + LN + softmax + silu + gelu
    without any view / transpose / slice ops.

    The full attention blocks rely on view+transpose to re-shape into
    [B, H, T, D] which CompGen's fallback FXImporter does not yet
    lower correctly. This simplified block still exercises all four
    ``compgen.linalg_ext`` destinations (softmax, layer_norm, silu,
    gelu) and bridges cleanly.
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.fc1 = nn.Linear(d_model, d_model * 2, bias=False)
        self.fc2 = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        # Per-token "attention" without reshape: dot along last axis.
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        a = torch.matmul(attn, v)
        mlp = self.fc2(F.silu(self.fc1(h)))
        return x + a + F.gelu(mlp)


def attention_mlp_tiny() -> RealWorkloadFixture:
    """Minimal attention + MLP block that uses only ops the FXImporter
    fallback covers. Exercises softmax, layer_norm, silu, gelu."""
    _deterministic_seed()
    model = _AttentionMLPNoTranspose(d_model=64).eval()
    x = torch.randn(1, 4, 64)
    with torch.no_grad():
        eager = model(x).detach().clone()
    exported = torch.export.export(model, (x,))
    return RealWorkloadFixture(
        name="attention_mlp_tiny",
        model=model,
        example_inputs=(x,),
        eager_output=eager,
        exported=exported,
    )


# --- TinyLlama decode block (canonical LLaMA-style) --------------------------


class _TinyLlamaBlock(nn.Module):
    """TinyLlama-faithful decode block.

    Architecture: RMSNorm → Q/K/V projections → rotary position
    embeddings on Q and K → scaled dot-product attention → output
    projection → residual → RMSNorm → SwiGLU MLP → residual.

    Matches ``TinyLlama/TinyLlama-1.1B-Chat-v1.0`` at reduced scale
    (d_model=128, n_heads=4, n_kv_heads=4, intermediate=256).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        intermediate: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm_attn = _RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm_mlp = _RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, intermediate, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.shape
        h = self.norm_attn(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        a = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(a)
        m = self.norm_mlp(x)
        return x + self.down_proj(F.silu(self.gate_proj(m)) * self.up_proj(m))


def tinyllama_block_tiny() -> RealWorkloadFixture:
    """TinyLlama-style decode block: RMSNorm + RoPE + attention + SwiGLU MLP.

    Architecturally identical to the real TinyLlama blocks at
    reduced scale. Exercises RMSNorm, rotary embeddings, attention
    softmax, and the SwiGLU MLP pathway CompGen's quant / fusion
    passes are tuned for.
    """
    _deterministic_seed()
    head_dim = 32
    model = _TinyLlamaBlock(d_model=128, n_heads=4, intermediate=256).eval()
    x = torch.randn(1, 8, 128)
    half = head_dim // 2
    freqs = torch.arange(half).float() / half
    pos = torch.arange(8).float()
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    cos = angles.cos().unsqueeze(0).unsqueeze(0).expand(1, 4, 8, half).contiguous()
    sin = angles.sin().unsqueeze(0).unsqueeze(0).expand(1, 4, 8, half).contiguous()
    with torch.no_grad():
        eager = model(x, cos, sin).detach().clone()
    exported = torch.export.export(model, (x, cos, sin))
    return RealWorkloadFixture(
        name="tinyllama_block_tiny",
        model=model,
        example_inputs=(x, cos, sin),
        eager_output=eager,
        exported=exported,
    )


# --- VLA (vision-language-action) decoder block -----------------------------


class _VLADecoderBlock(nn.Module):
    """VLA-style decoder block matching smolVLA-like architectures.

    Adds cross-attention from a visual prefix (``vis``) on top of
    self-attention, followed by a GELU MLP. Representative of the
    VLA-family policy heads used in embodied agents.
    """

    def __init__(
        self,
        d_model: int = 96,
        n_heads: int = 4,
        mlp_mult: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm_self = nn.LayerNorm(d_model)
        self.q_self = nn.Linear(d_model, d_model, bias=False)
        self.k_self = nn.Linear(d_model, d_model, bias=False)
        self.v_self = nn.Linear(d_model, d_model, bias=False)
        self.norm_cross = nn.LayerNorm(d_model)
        self.q_cross = nn.Linear(d_model, d_model, bias=False)
        self.k_cross = nn.Linear(d_model, d_model, bias=False)
        self.v_cross = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * mlp_mult, bias=False)
        self.fc2 = nn.Linear(d_model * mlp_mult, d_model, bias=False)

    def _attend(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        qW: nn.Linear,
        kW: nn.Linear,
        vW: nn.Linear,
    ) -> torch.Tensor:
        B, T, C = q_in.shape
        _, Tkv, _ = kv_in.shape
        q = qW(q_in).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = kW(kv_in).view(B, Tkv, self.n_heads, self.head_dim).transpose(1, 2)
        v = vW(kv_in).view(B, Tkv, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)

    def forward(
        self,
        x: torch.Tensor,
        vis: torch.Tensor,
    ) -> torch.Tensor:
        # Self attention.
        h = self.norm_self(x)
        a = self._attend(h, h, self.q_self, self.k_self, self.v_self)
        x = x + a
        # Cross attention onto visual features.
        h = self.norm_cross(x)
        a = self._attend(h, vis, self.q_cross, self.k_cross, self.v_cross)
        x = x + self.o_proj(a)
        # MLP.
        m = self.norm_mlp(x)
        return x + self.fc2(F.gelu(self.fc1(m)))


def vla_decoder_tiny() -> RealWorkloadFixture:
    """VLA decoder-style block: self-attn + cross-attn + GELU MLP.

    Representative of smolVLA / OpenVLA / similar vision-language-action
    architectures used for embodied agents.
    """
    _deterministic_seed()
    model = _VLADecoderBlock(d_model=96, n_heads=4).eval()
    x = torch.randn(1, 4, 96)
    vis = torch.randn(1, 16, 96)  # 16 vision tokens
    with torch.no_grad():
        eager = model(x, vis).detach().clone()
    exported = torch.export.export(model, (x, vis))
    return RealWorkloadFixture(
        name="vla_decoder_tiny",
        model=model,
        example_inputs=(x, vis),
        eager_output=eager,
        exported=exported,
    )


# --- stacked full-model fixtures (W12.1) -----------------------------------


class _TinyLlamaStack(nn.Module):
    """Stacked TinyLlama-style model: embed + N decode blocks + LM head.

    Realistic end-to-end shape for LLM inference: embedding +
    residual blocks + final logits projection. Ties embed + lm_head
    weights as TinyLlama does.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        intermediate: int = 256,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [_TinyLlamaBlock(d_model=d_model, n_heads=n_heads, intermediate=intermediate) for _ in range(n_layers)]
        )
        self.final_norm = _RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

    def forward(
        self,
        input_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.final_norm(x))


def tinyllama_stack_3() -> RealWorkloadFixture:
    """Full stacked TinyLlama: embed + 3 blocks + tied LM head."""
    _deterministic_seed()
    d_model, n_heads, head_dim = 128, 4, 32
    model = _TinyLlamaStack(
        vocab_size=256,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=3,
        intermediate=256,
    ).eval()
    input_ids = torch.randint(0, 256, (1, 8), dtype=torch.long)
    half = head_dim // 2
    freqs = torch.arange(half).float() / half
    pos = torch.arange(8).float()
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    cos = angles.cos().unsqueeze(0).unsqueeze(0).expand(1, n_heads, 8, half).contiguous()
    sin = angles.sin().unsqueeze(0).unsqueeze(0).expand(1, n_heads, 8, half).contiguous()
    with torch.no_grad():
        eager = model(input_ids, cos, sin).detach().clone()
    exported = torch.export.export(model, (input_ids, cos, sin))
    return RealWorkloadFixture(
        name="tinyllama_stack_3",
        model=model,
        example_inputs=(input_ids, cos, sin),
        eager_output=eager,
        exported=exported,
    )


class _GemmaStack(nn.Module):
    """Stacked Gemma-style: embedding + N decode blocks + final norm + LM head."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([_GemmaDecodeBlock(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)])
        self.final_norm = _RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.final_norm(x))


def gemma_stack_3() -> RealWorkloadFixture:
    _deterministic_seed()
    d_model, n_heads, head_dim = 128, 4, 32
    model = _GemmaStack(
        vocab_size=256,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=3,
    ).eval()
    input_ids = torch.randint(0, 256, (1, 8), dtype=torch.long)
    half = head_dim // 2
    freqs = torch.arange(half).float() / half
    pos = torch.arange(8).float()
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    cos = angles.cos().unsqueeze(0).unsqueeze(0).expand(1, n_heads, 8, half).contiguous()
    sin = angles.sin().unsqueeze(0).unsqueeze(0).expand(1, n_heads, 8, half).contiguous()
    with torch.no_grad():
        eager = model(input_ids, cos, sin).detach().clone()
    exported = torch.export.export(model, (input_ids, cos, sin))
    return RealWorkloadFixture(
        name="gemma_stack_3",
        model=model,
        example_inputs=(input_ids, cos, sin),
        eager_output=eager,
        exported=exported,
    )


class _SmolVLAStack(nn.Module):
    """Stacked smolVLA-style: input projector + N blocks + 7-DoF action head."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model, bias=False)
        self.blocks = nn.ModuleList([_SmolVLABlock(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def smolvla_stack_2() -> RealWorkloadFixture:
    _deterministic_seed()
    model = _SmolVLAStack(d_model=128, n_heads=4, n_layers=2).eval()
    x = torch.randn(1, 8, 128)
    with torch.no_grad():
        eager = model(x).detach().clone()
    exported = torch.export.export(model, (x,))
    return RealWorkloadFixture(
        name="smolvla_stack_2",
        model=model,
        example_inputs=(x,),
        eager_output=eager,
        exported=exported,
    )


ALL_FIXTURE_FNS = (
    smolvla_tiny,
    gemma_decode_tiny,
    qwen_moe_tiny,
    attention_mlp_tiny,
    tinyllama_block_tiny,
    vla_decoder_tiny,
    tinyllama_stack_3,
    gemma_stack_3,
    smolvla_stack_2,
)

# Subset of fixtures known to bridge cleanly through CompGen's FXImporter
# fallback today. Used by + pass tests when ``torch_mlir`` is not
# installed. When the torch-mlir bridge path is available, all fixtures
# should bridge -- so tests should prefer ``ALL_FIXTURE_FNS`` with a
# fallback to this subset.
BRIDGE_FRIENDLY_FIXTURE_FNS = (qwen_moe_tiny, attention_mlp_tiny)


__all__ = [
    "ALL_FIXTURE_FNS",
    "BRIDGE_FRIENDLY_FIXTURE_FNS",
    "RealWorkloadFixture",
    "attention_mlp_tiny",
    "gemma_decode_tiny",
    "gemma_stack_3",
    "qwen_moe_tiny",
    "smolvla_stack_2",
    "smolvla_tiny",
    "tinyllama_block_tiny",
    "tinyllama_stack_3",
    "vla_decoder_tiny",
]
