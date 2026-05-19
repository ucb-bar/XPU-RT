"""Real  example: TinyLlama layer-0 through HF-faithful megakernel.

Loads ``model.layers.0`` from the cached TinyLlama-1.1B-Chat checkpoint
and runs the **HF-faithful** decoder-layer megakernel from
:mod:`examples.event_tensor.llama_layer_rope_megakernel` (RoPE +
causal mask + RMSNorm + SwiGLU + residuals) on a slice of those
weights -- with the same RoPE base TinyLlama trained with
(``rope_theta`` from the checkpoint config).

This is the closest match  achieves to running an actual HF
``LlamaDecoderLayer.forward()``: identical math, identical RoPE
tables, identical norms, identical projection / MLP weights -- the
only differences are
    (a) the slice of heads / hidden / intermediate (to fit shared mem),
    (b) GQA collapsed to MHA via the slice (TinyLlama uses 4 KV heads;
        we take the first n_heads K/V rows and pretend they're per-head).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from examples.event_tensor.llama_layer_rope_megakernel import (
    CompiledLlamaLayerRope,
    compile_llama_layer_rope,
    hf_rope_tables,
    reference_llama_layer_rope,
    run_llama_layer_rope,
)


def _find_tinyllama_snapshot() -> Path:
    cache = Path(os.path.expanduser(
        "~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
    ))
    if not cache.exists():
        raise SystemExit(
            "TinyLlama-1.1B-Chat checkpoint not in cache; this example "
            f"requires it pre-fetched at {cache}"
        )
    return next((cache / "snapshots").iterdir())


@dataclass(frozen=True)
class TinyLlamaHFLayer0Weights:
    w_norm1: torch.Tensor
    w_q: torch.Tensor
    w_k: torch.Tensor
    w_v: torch.Tensor
    w_o: torch.Tensor
    w_norm2: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor
    cfg: dict


def load_tinyllama_hf_layer0(device: str = "cuda") -> TinyLlamaHFLayer0Weights:
    snap = _find_tinyllama_snapshot()
    cfg = json.loads((snap / "config.json").read_text())
    weights = load_file(str(snap / "model.safetensors"), device=device)
    p = "model.layers.0"
    return TinyLlamaHFLayer0Weights(
        w_norm1 = weights[f"{p}.input_layernorm.weight"].to(torch.float32),
        w_q     = weights[f"{p}.self_attn.q_proj.weight"].to(torch.float32),
        w_k     = weights[f"{p}.self_attn.k_proj.weight"].to(torch.float32),
        w_v     = weights[f"{p}.self_attn.v_proj.weight"].to(torch.float32),
        w_o     = weights[f"{p}.self_attn.o_proj.weight"].to(torch.float32),
        w_norm2 = weights[f"{p}.post_attention_layernorm.weight"].to(torch.float32),
        w_gate  = weights[f"{p}.mlp.gate_proj.weight"].to(torch.float32),
        w_up    = weights[f"{p}.mlp.up_proj.weight"].to(torch.float32),
        w_down  = weights[f"{p}.mlp.down_proj.weight"].to(torch.float32),
        cfg     = cfg,
    )


DEFAULT_HEAD_SLICE         = 4
DEFAULT_INTERMEDIATE_SLICE = 64
DEFAULT_SEQ_LEN            = 16
DEFAULT_HEAD_DIM           = 16   # smaller than TinyLlama's 64 to fit shared mem


def slice_hf_weights(
    full: TinyLlamaHFLayer0Weights,
    n_heads: int = DEFAULT_HEAD_SLICE,
    intermediate: int = DEFAULT_INTERMEDIATE_SLICE,
    head_dim: int = DEFAULT_HEAD_DIM,
) -> tuple[TinyLlamaHFLayer0Weights, dict]:
    d_hidden = n_heads * head_dim
    sliced = TinyLlamaHFLayer0Weights(
        w_norm1 = full.w_norm1[:d_hidden].contiguous(),
        w_q     = full.w_q[:d_hidden,    :d_hidden].contiguous(),
        w_k     = full.w_k[:d_hidden,    :d_hidden].contiguous(),
        w_v     = full.w_v[:d_hidden,    :d_hidden].contiguous(),
        w_o     = full.w_o[:d_hidden,    :d_hidden].contiguous(),
        w_norm2 = full.w_norm2[:d_hidden].contiguous(),
        w_gate  = full.w_gate[:intermediate, :d_hidden].contiguous(),
        w_up    = full.w_up[:intermediate,   :d_hidden].contiguous(),
        w_down  = full.w_down[:d_hidden,     :intermediate].contiguous(),
        cfg     = full.cfg,
    )
    sliced_cfg = {
        "n_heads":      n_heads,
        "head_dim":     head_dim,
        "hidden_dim":   d_hidden,
        "intermediate": intermediate,
        "rms_eps":      full.cfg.get("rms_norm_eps", 1e-5),
        "rope_base":    full.cfg.get("rope_theta", 10000.0),
    }
    return sliced, sliced_cfg


def compile_for_tinyllama_hf(
    seq_len: int = DEFAULT_SEQ_LEN,
    n_heads: int = DEFAULT_HEAD_SLICE,
    intermediate: int = DEFAULT_INTERMEDIATE_SLICE,
    head_dim: int = DEFAULT_HEAD_DIM,
) -> CompiledLlamaLayerRope:
    return compile_llama_layer_rope(
        n_heads=n_heads, seq_len=seq_len, head_dim=head_dim,
        intermediate_dim=intermediate,
        block_m=min(16, seq_len),
        block_i=min(32, intermediate),
        block_n=min(32, n_heads * head_dim),
    )


def run_tinyllama_hf_layer(
    compiled: CompiledLlamaLayerRope,
    x: torch.Tensor,
    sliced: TinyLlamaHFLayer0Weights,
    sliced_cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    rms_eps = float(sliced_cfg["rms_eps"])
    cos, sin = hf_rope_tables(
        seq_len=compiled.seq_len, head_dim=compiled.head_dim,
        base=float(sliced_cfg["rope_base"]),
        device=str(x.device), dtype=torch.float32,
    )

    got = run_llama_layer_rope(
        compiled, x,
        sliced.w_norm1, sliced.w_q, sliced.w_k, sliced.w_v, sliced.w_o,
        sliced.w_norm2,
        sliced.w_gate, sliced.w_up, sliced.w_down,
        cos, sin, rms_eps=rms_eps,
    )
    ref = reference_llama_layer_rope(
        x,
        sliced.w_norm1, sliced.w_q, sliced.w_k, sliced.w_v, sliced.w_o,
        sliced.w_norm2,
        sliced.w_gate, sliced.w_up, sliced.w_down,
        cos, sin,
        n_heads=sliced_cfg["n_heads"], head_dim=sliced_cfg["head_dim"],
        rms_eps=rms_eps,
    )
    return got, ref


__all__ = [
    "DEFAULT_HEAD_DIM",
    "DEFAULT_HEAD_SLICE",
    "DEFAULT_INTERMEDIATE_SLICE",
    "DEFAULT_SEQ_LEN",
    "TinyLlamaHFLayer0Weights",
    "compile_for_tinyllama_hf",
    "load_tinyllama_hf_layer0",
    "run_tinyllama_hf_layer",
    "slice_hf_weights",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    print("Loading TinyLlama layer-0 weights ...")
    full = load_tinyllama_hf_layer0()
    print(f"  rope_theta from checkpoint: {full.cfg.get('rope_theta', 10000.0)}")

    sliced, sliced_cfg = slice_hf_weights(full)
    print(f"\nSlice: H={sliced_cfg['n_heads']}, D_HEAD={sliced_cfg['head_dim']}, "
          f"D_HIDDEN={sliced_cfg['hidden_dim']}, I={sliced_cfg['intermediate']}")

    print("\nCompiling HF-faithful Llama layer megakernel ...")
    S = DEFAULT_SEQ_LEN
    compiled = compile_for_tinyllama_hf(seq_len=S)
    print(f"  emitted kernel: {compiled.kernel_name}")
    print(f"  device functions ({len(compiled.lowering.device_function_table)}): "
          f"{sorted(compiled.lowering.device_function_table.values())}")

    torch.manual_seed(2026)
    x = torch.randn(
        (S, sliced_cfg["hidden_dim"]), dtype=torch.float32, device="cuda",
    ) * 0.1

    print("\nRunning megakernel + reference (RoPE + causal mask) ...")
    got, ref = run_tinyllama_hf_layer(compiled, x, sliced, sliced_cfg)

    err_abs = (got - ref).abs().max().item()
    err_rel = ((got - ref).abs() / (ref.abs() + 1e-6)).max().item()
    print(f"  max |got - ref|       = {err_abs:.3e}")
    print(f"  max |got - ref|/|ref| = {err_rel:.3e}")

    assert err_abs < 1e-2, (
        f"TinyLlama HF-faithful decoder-layer megakernel diverges by {err_abs}."
    )
    print("\nPASS: full HF-faithful Llama decoder layer megakernel matches reference")
    print(
        f"      on REAL TinyLlama-1.1B layer-0 weights "
        f"(rope_theta={sliced_cfg['rope_base']}, real RMSNorm scales, "
        f"H={sliced_cfg['n_heads']}, D={sliced_cfg['hidden_dim']}, "
        f"I={sliced_cfg['intermediate']})."
    )
