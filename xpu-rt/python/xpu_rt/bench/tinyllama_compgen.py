"""TinyLlama-1.1B forward pass on our Triton kernels.

Runs the real HF TinyLlama model end-to-end with our kernels replacing the
dominant ops; uses torch native for the glue (reshape/transpose/embedding/
simple elementwise). The goal is:

  1. Prove our kernel set covers the whole model — every dominant op
     routes through a Triton kernel we wrote.
  2. Measure correctness vs HF eager across the full forward.
  3. Produce a per-op-family timing breakdown ranked by total wall-time
     → the autocomp-candidate list (ops ≥ 5% of total).

Kernel coverage for TinyLlama-1.1B-Chat:
  * linears (Q/K/V/O/gate/up/down/lm_head)  → matmul_fp16_v3   [ours]
  * attention QKᵀ and ·V                    → bmm via our matmul in loop  [ours]
  * RMSNorm (input_ln + post_attn_ln)       → rmsnorm_fp16     [ours]
  * softmax along last dim                  → softmax_fp32_last_dim  [ours]
  * silu on gate proj                       → silu_fp16        [ours]
  * embedding lookup                        → torch.nn.functional.embedding
  * residual adds                           → torch (tiny; not perf-critical)
  * RoPE apply                              → torch (trig + rotate)
  * attention mask add                      → torch (broadcast add)
  * reshape / transpose                     → torch (views, no-op)

This is "our kernels on the dominant ops" not "every single aten op".
That's the cost-correct coverage — dominant ops are where perf and
autocomp-escalation decisions live; glue ops are 1% of wall-time.
"""

from __future__ import annotations

import dataclasses
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from compgen.bench.flash_attention_kernel import flash_attention_fp16
from compgen.bench.turing_kernels import (
    bmm_fp16,
    matmul_fp16_v3,
    rmsnorm_fp16,
    silu_fp16,
    softmax_fp32_last_dim,
)


# ---------------------------------------------------------------------------
# Per-kernel timing accumulator
# ---------------------------------------------------------------------------


@dataclass
class KernelTime:
    name: str
    calls: int = 0
    total_us: float = 0.0

    def add(self, us: float) -> None:
        self.calls += 1
        self.total_us += us

    @property
    def avg_us(self) -> float:
        return self.total_us / self.calls if self.calls else 0.0


class KernelProfiler:
    """Accumulates per-op-family CUDA-event timing across the whole forward."""

    def __init__(self) -> None:
        self._bins: dict[str, KernelTime] = {}

    def _bin(self, name: str) -> KernelTime:
        b = self._bins.get(name)
        if b is None:
            b = KernelTime(name=name)
            self._bins[name] = b
        return b

    def time(self, name: str, fn: Callable[[], Any]) -> Any:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        self._bin(name).add(start.elapsed_time(end) * 1000.0)  # ms → μs
        return out


class NoOpProfiler:
    """Inert profiler — no events, no sync. Required inside CUDA-graph
    capture (capture forbids ``torch.cuda.synchronize()`` mid-stream)."""

    def time(self, _name: str, fn: Callable[[], Any]) -> Any:
        return fn()

    def snapshot(self) -> list:
        return []

    def render(self) -> str:
        return "(no-op profiler — no measurements)"

    def snapshot(self) -> list[KernelTime]:
        return [b for b in self._bins.values() if b.calls > 0]

    def render(self) -> str:
        rows = sorted(self.snapshot(), key=lambda r: -r.total_us)
        total = sum(r.total_us for r in rows)
        lines = [f"{'op':22s} {'calls':>6s} {'total_us':>12s} {'avg_us':>10s} {'% of total':>10s}"]
        for r in rows:
            pct = 100.0 * r.total_us / total if total else 0.0
            lines.append(
                f"{r.name:22s} {r.calls:>6d} {r.total_us:>12.1f} "
                f"{r.avg_us:>10.2f} {pct:>9.1f}%"
            )
        lines.append(f"\nTOTAL: {total:.1f} μs ({total/1000:.2f} ms)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RoPE — precomputed cos/sin tables + apply helper
# ---------------------------------------------------------------------------


def make_rope_tables(seq_len: int, head_dim: int, base: float, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard HF Llama RoPE cos/sin, half-rotate form."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, S, H, D). cos/sin: (S, D)."""
    # HF's rotate-half: split last dim in two halves, negate the second.
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos[None, :, None, :] + rot * sin[None, :, None, :]


# ---------------------------------------------------------------------------
# One decoder layer
# ---------------------------------------------------------------------------


@dataclass
class LayerWeights:
    """Flat struct of one TinyLlama decoder layer's weights (fp16 tensors)."""

    input_layernorm: torch.Tensor   # (D,)
    q_proj: torch.Tensor            # (D, D)   — linear weight as (out, in)
    k_proj: torch.Tensor            # (D_kv, D)
    v_proj: torch.Tensor            # (D_kv, D)
    o_proj: torch.Tensor            # (D, D)
    post_attention_layernorm: torch.Tensor
    gate_proj: torch.Tensor         # (I, D)
    up_proj: torch.Tensor           # (I, D)
    down_proj: torch.Tensor         # (D, I)


@dataclass
class ModelConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int       # TinyLlama uses GQA (32/4)
    head_dim: int
    intermediate_size: int
    rms_eps: float
    rope_theta: float


def decoder_layer_forward(
    x: torch.Tensor,           # (B, S, D) fp16
    w: LayerWeights,
    cfg: ModelConfig,
    cos: torch.Tensor, sin: torch.Tensor,   # (S, D_h)
    attention_mask: torch.Tensor,            # (B, 1, S, S) or None
    prof: KernelProfiler,
) -> torch.Tensor:
    B, S, D = x.shape
    H, H_kv, D_h = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    D_kv = H_kv * D_h

    # ---- Self-attention block ----
    residual = x
    x_norm = prof.time("rmsnorm", lambda: rmsnorm_fp16(x.reshape(-1, D), w.input_layernorm, cfg.rms_eps).reshape(B, S, D))

    # Q/K/V linears: torch.Linear applies weight.T, i.e. (in) @ (out,in).T.
    # Our matmul takes (M,K) @ (K,N), so we pass weight.T as rhs.
    q = prof.time("matmul_linear", lambda: matmul_fp16_v3(x_norm.reshape(-1, D), w.q_proj.T.contiguous())).reshape(B, S, H,   D_h)
    k = prof.time("matmul_linear", lambda: matmul_fp16_v3(x_norm.reshape(-1, D), w.k_proj.T.contiguous())).reshape(B, S, H_kv, D_h)
    v = prof.time("matmul_linear", lambda: matmul_fp16_v3(x_norm.reshape(-1, D), w.v_proj.T.contiguous())).reshape(B, S, H_kv, D_h)

    q = prof.time("rope_apply", lambda: apply_rope(q, cos, sin))
    k = prof.time("rope_apply", lambda: apply_rope(k, cos, sin))

    # GQA: broadcast K/V across the Q head groups via repeat_interleave.
    if H_kv != H:
        n_rep = H // H_kv
        k = prof.time("transpose_reshape", lambda: k.repeat_interleave(n_rep, dim=2))
        v = prof.time("transpose_reshape", lambda: v.repeat_interleave(n_rep, dim=2))

    # (B, S, H, D_h) → (B, H, S, D_h)
    q = prof.time("transpose_reshape", lambda: q.transpose(1, 2).contiguous())
    k = prof.time("transpose_reshape", lambda: k.transpose(1, 2).contiguous())
    v = prof.time("transpose_reshape", lambda: v.transpose(1, 2).contiguous())

    # FlashAttention MEGA — fused QKᵀ→softmax→·V in one kernel, online softmax.
    # Replaces the bmm + softmax + bmm trio. The (S,S) scores never hit DRAM.
    scale = 1.0 / (D_h ** 0.5)
    BH = B * H
    q2 = q.reshape(BH, S, D_h).contiguous()
    k2 = k.reshape(BH, S, D_h).contiguous()
    v2 = v.reshape(BH, S, D_h).contiguous()
    attn_out = prof.time(
        "flash_attention",
        lambda: flash_attention_fp16(q2, k2, v2, scale, causal=True),
    ).reshape(B, H, S, D_h)
    attn_out = prof.time("transpose_reshape", lambda: attn_out.transpose(1, 2).contiguous().reshape(B, S, D))

    attn_proj = prof.time("matmul_linear", lambda: matmul_fp16_v3(attn_out.reshape(-1, D), w.o_proj.T.contiguous())).reshape(B, S, D)
    x = prof.time("residual_add", lambda: residual + attn_proj)

    # ---- MLP block ----
    residual = x
    x_norm = prof.time("rmsnorm", lambda: rmsnorm_fp16(x.reshape(-1, D), w.post_attention_layernorm, cfg.rms_eps).reshape(B, S, D))

    gate = prof.time("matmul_linear", lambda: matmul_fp16_v3(x_norm.reshape(-1, D), w.gate_proj.T.contiguous()))
    up = prof.time("matmul_linear", lambda: matmul_fp16_v3(x_norm.reshape(-1, D), w.up_proj.T.contiguous()))
    gate_act = prof.time("silu", lambda: silu_fp16(gate))
    mlp_mid = prof.time("residual_add", lambda: gate_act * up)  # gated element-wise mul
    mlp_out = prof.time("matmul_linear", lambda: matmul_fp16_v3(mlp_mid, w.down_proj.T.contiguous())).reshape(B, S, D)
    x = prof.time("residual_add", lambda: residual + mlp_out)

    return x


# ---------------------------------------------------------------------------
# Whole-model forward
# ---------------------------------------------------------------------------


@dataclass
class TinyLlamaWeights:
    embed: torch.Tensor                     # (V, D)
    layers: list[LayerWeights]
    final_norm: torch.Tensor                # (D,)
    lm_head: torch.Tensor                   # (V, D)   (Llama ties weights to embed by default)
    cfg: ModelConfig


def tinyllama_forward(
    input_ids: torch.Tensor,           # (B, S) long
    weights: TinyLlamaWeights,
    prof: KernelProfiler | NoOpProfiler | None = None,
) -> torch.Tensor:
    """Full forward: embed → 22 decoder layers → final RMS → LM head.

    Returns logits (B, S, V).  No KV cache — this is prefill only.

    ``prof=None`` defaults to ``NoOpProfiler()`` which adds zero overhead;
    pass an explicit ``KernelProfiler()`` for per-op timing breakdown
    (incompatible with CUDA-graph capture due to per-call sync).
    """
    if prof is None:
        prof = NoOpProfiler()

    device = input_ids.device
    B, S = input_ids.shape
    cfg = weights.cfg
    D = cfg.hidden_size

    # Embedding
    x = prof.time("embedding", lambda: F.embedding(input_ids, weights.embed))  # (B, S, D)

    # RoPE tables, causal mask
    cos, sin = make_rope_tables(S, cfg.head_dim, cfg.rope_theta, str(device), torch.float16)
    causal = torch.triu(
        torch.full((S, S), float("-inf"), device=device, dtype=torch.float16),
        diagonal=1,
    )
    causal = causal[None, None, :, :]  # (1, 1, S, S)

    # 22 decoder layers
    for i, lw in enumerate(weights.layers):
        x = decoder_layer_forward(x, lw, cfg, cos, sin, causal, prof)

    # Final RMS + LM head
    x = prof.time("rmsnorm", lambda: rmsnorm_fp16(x.reshape(-1, D), weights.final_norm, cfg.rms_eps).reshape(B, S, D))
    logits = prof.time("matmul_linear", lambda: matmul_fp16_v3(x.reshape(-1, D), weights.lm_head.T.contiguous())).reshape(B, S, -1)

    return logits


# ---------------------------------------------------------------------------
# Load TinyLlama weights from the HF-style state_dict shape
# ---------------------------------------------------------------------------


def load_tinyllama_weights(hf_state_dict: dict, cfg: ModelConfig, device: str = "cuda") -> TinyLlamaWeights:
    """Adapter: takes an HF-Llama state_dict, returns our flat weights struct."""
    d = {k: v.to(device=device, dtype=torch.float16) for k, v in hf_state_dict.items()}

    def pick(key: str) -> torch.Tensor:
        if key in d:
            return d[key]
        # Some versions prefix with "model."
        if f"model.{key}" in d:
            return d[f"model.{key}"]
        raise KeyError(f"missing: {key}")

    embed = pick("embed_tokens.weight")
    final_norm = pick("norm.weight")
    # TinyLlama ties lm_head to embed. Try the tied weight first.
    try:
        lm_head = d["lm_head.weight"]
    except KeyError:
        lm_head = embed

    num_layers = 0
    while True:
        if f"layers.{num_layers}.input_layernorm.weight" in d or f"model.layers.{num_layers}.input_layernorm.weight" in d:
            num_layers += 1
        else:
            break

    layers = []
    for i in range(num_layers):
        layers.append(LayerWeights(
            input_layernorm=pick(f"layers.{i}.input_layernorm.weight"),
            q_proj=pick(f"layers.{i}.self_attn.q_proj.weight"),
            k_proj=pick(f"layers.{i}.self_attn.k_proj.weight"),
            v_proj=pick(f"layers.{i}.self_attn.v_proj.weight"),
            o_proj=pick(f"layers.{i}.self_attn.o_proj.weight"),
            post_attention_layernorm=pick(f"layers.{i}.post_attention_layernorm.weight"),
            gate_proj=pick(f"layers.{i}.mlp.gate_proj.weight"),
            up_proj=pick(f"layers.{i}.mlp.up_proj.weight"),
            down_proj=pick(f"layers.{i}.mlp.down_proj.weight"),
        ))

    return TinyLlamaWeights(
        embed=embed, layers=layers, final_norm=final_norm,
        lm_head=lm_head, cfg=cfg,
    )


__all__ = [
    "KernelProfiler", "KernelTime",
    "LayerWeights", "ModelConfig", "TinyLlamaWeights",
    "apply_rope", "decoder_layer_forward",
    "load_tinyllama_weights", "make_rope_tables", "tinyllama_forward",
]
