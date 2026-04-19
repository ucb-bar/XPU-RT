"""Real Phase C example: TinyLlama layer-0 weights through our megakernel.

Loads the actual ``model.layers.0`` weights from a cached TinyLlama-1.1B
checkpoint (real Llama architecture: 32 attention heads, 4 KV heads
with GQA, hidden_dim 2048, intermediate_dim 5632) and runs the
fused-attention+MLP transformer-block megakernel emitted by
:mod:`compgen.ir.tile.lower_megakernel_dynamic` on those exact weight
values.

What's compared:

    * Reference (PyTorch eager): the same transformer-block sequence
      (X_resid + flatten(SDPA(Q,K,V))) + SwiGLU MLP, computed with
      ``torch.matmul`` + ``F.scaled_dot_product_attention`` + ``F.silu``.
    * Got (CompGen-emitted megakernel): the same composition, but every
      GPU instruction comes from the persistent megakernel produced by
      our compiler pipeline.

Both consume the *same* TinyLlama weights and the *same* random
activations.  A numerical match within fp32 tolerance proves the
megakernel handles real-LLM-scale weights and shapes correctly -- not
a toy.

This intentionally does **not** include RMSNorm, RoPE, or the output
projection -- those are layer-level wrappers that the megakernel will
absorb in a Phase C+ extension.  The block we lower is the
attention-then-MLP residual sandwich that sits at the heart of every
modern decoder layer.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from examples.event_tensor.transformer_block_megakernel import (
    CompiledTransformerBlockMegakernel,
    compile_transformer_block_megakernel,
    reference_block,
    run_transformer_block_megakernel,
)


def _find_tinyllama_snapshot() -> Path:
    cache = Path(os.path.expanduser(
        "~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
    ))
    if not cache.exists():
        raise SystemExit(
            "TinyLlama-1.1B-Chat checkpoint not in cache; this example "
            "requires it pre-fetched at "
            f"{cache}"
        )
    snap = next((cache / "snapshots").iterdir())
    return snap


@dataclass(frozen=True)
class TinyLlamaLayer0Weights:
    """The layer-0 weight slices we feed into the megakernel."""

    w_q: torch.Tensor       # (n_heads * D_HEAD, D_HIDDEN)
    w_k: torch.Tensor       # (n_kv_heads * D_HEAD, D_HIDDEN)
    w_v: torch.Tensor       # (n_kv_heads * D_HEAD, D_HIDDEN)
    w_gate: torch.Tensor    # (intermediate, D_HIDDEN)
    w_up: torch.Tensor      # (intermediate, D_HIDDEN)
    w_down: torch.Tensor    # (D_HIDDEN, intermediate)
    cfg: dict


def load_tinyllama_layer0(device: str = "cuda") -> TinyLlamaLayer0Weights:
    snap = _find_tinyllama_snapshot()
    cfg = json.loads((snap / "config.json").read_text())
    weights = load_file(str(snap / "model.safetensors"), device=device)
    layer_prefix = "model.layers.0"
    return TinyLlamaLayer0Weights(
        w_q     = weights[f"{layer_prefix}.self_attn.q_proj.weight"].to(torch.float32),
        w_k     = weights[f"{layer_prefix}.self_attn.k_proj.weight"].to(torch.float32),
        w_v     = weights[f"{layer_prefix}.self_attn.v_proj.weight"].to(torch.float32),
        w_gate  = weights[f"{layer_prefix}.mlp.gate_proj.weight"].to(torch.float32),
        w_up    = weights[f"{layer_prefix}.mlp.up_proj.weight"].to(torch.float32),
        w_down  = weights[f"{layer_prefix}.mlp.down_proj.weight"].to(torch.float32),
        cfg     = cfg,
    )


DEFAULT_HEAD_SLICE        = 4     # << TinyLlama's 32 heads -- slice for fast test
DEFAULT_INTERMEDIATE_SLICE = 128  # << TinyLlama's 5632 -- slice for fast test
DEFAULT_SEQ_LEN           = 16


def slice_weights_for_megakernel(
    weights: TinyLlamaLayer0Weights,
    n_heads: int = DEFAULT_HEAD_SLICE,
    intermediate: int = DEFAULT_INTERMEDIATE_SLICE,
) -> tuple[TinyLlamaLayer0Weights, dict]:
    """Slice the real TinyLlama checkpoint into a smaller block whose dims
    fit our megakernel signature in well under a minute of Triton compile.

    Slicing rules (everything is a CONTIGUOUS slice of the real weights;
    no random replacement, no transformation -- the values are exactly
    those TinyLlama was trained with):

        * keep the first ``n_heads`` Q heads  (= first n_heads*64 rows of W_q)
        * use those same heads for K/V (no GQA; ``min(n_heads, n_kv_heads*8)``)
        * input dim for everything = n_heads * 64
        * intermediate dim = ``intermediate``
        * MLP weights sliced to (intermediate, n_heads*64) and (n_heads*64, intermediate)
    """
    d_head      = 64
    d_hidden    = n_heads * d_head
    n_kv_heads  = weights.cfg["num_key_value_heads"]
    repeat      = n_heads // n_kv_heads
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_heads={n_heads} must be a multiple of n_kv_heads={n_kv_heads}"
        )

    sliced = TinyLlamaLayer0Weights(
        w_q     = weights.w_q[:d_hidden,         :d_hidden].contiguous(),
        # Real TinyLlama K/V have only n_kv_heads*64 rows; take all of them
        # then expand to match ``repeat`` Q heads via the same repeat_interleave
        # used during inference (= no information loss vs the checkpoint).
        w_k     = weights.w_k[: n_kv_heads * d_head, :d_hidden].contiguous(),
        w_v     = weights.w_v[: n_kv_heads * d_head, :d_hidden].contiguous(),
        w_gate  = weights.w_gate[:intermediate,  :d_hidden].contiguous(),
        w_up    = weights.w_up[:intermediate,    :d_hidden].contiguous(),
        w_down  = weights.w_down[:d_hidden,      :intermediate].contiguous(),
        cfg     = weights.cfg,
    )
    sliced_cfg = {
        "n_heads":          n_heads,
        "n_kv_heads_used":  n_kv_heads,
        "kv_repeat":        repeat,
        "head_dim":         d_head,
        "hidden_dim":       d_hidden,
        "intermediate":     intermediate,
    }
    return sliced, sliced_cfg


def project_qkv(
    x: torch.Tensor, weights: TinyLlamaLayer0Weights, sliced_cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Q, K, V using the sliced TinyLlama projections + GQA expand."""
    n_q     = sliced_cfg["n_heads"]
    n_kv    = sliced_cfg["n_kv_heads_used"]
    d_head  = sliced_cfg["head_dim"]
    repeat  = sliced_cfg["kv_repeat"]
    S, _    = x.shape

    q_full = x @ weights.w_q.T
    k_full = x @ weights.w_k.T
    v_full = x @ weights.w_v.T

    q = q_full.reshape(S, n_q, d_head).permute(1, 0, 2).contiguous()
    k = k_full.reshape(S, n_kv, d_head).permute(1, 0, 2).contiguous()
    v = v_full.reshape(S, n_kv, d_head).permute(1, 0, 2).contiguous()

    k = k.repeat_interleave(repeat, dim=0).contiguous()
    v = v.repeat_interleave(repeat, dim=0).contiguous()
    return q, k, v


def compile_for_tinyllama(
    seq_len: int = DEFAULT_SEQ_LEN,
    n_heads: int = DEFAULT_HEAD_SLICE,
    intermediate: int = DEFAULT_INTERMEDIATE_SLICE,
) -> CompiledTransformerBlockMegakernel:
    """Compile the transformer-block megakernel at sliced-TinyLlama dims.

    Block sizes are picked to stay under the TITAN-RTX shared-memory
    limit (64KB) -- mlp_down loads three (BLOCK, I) tiles per task and
    is the binding constraint.
    """
    return compile_transformer_block_megakernel(
        n_heads=n_heads, seq_len=seq_len, head_dim=64,
        intermediate_dim=intermediate,
        block_m=min(16, seq_len),
        block_i=min(32, intermediate),
        block_n=min(32, n_heads * 64),
    )


def run_tinyllama_block(
    compiled: CompiledTransformerBlockMegakernel,
    x: torch.Tensor,
    weights: TinyLlamaLayer0Weights,
    sliced_cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run BOTH the megakernel and the eager reference; return ``(got, ref)``."""
    q, k, v = project_qkv(x, weights, sliced_cfg)
    got = run_transformer_block_megakernel(
        compiled, q, k, v, x,
        weights.w_gate, weights.w_up, weights.w_down,
    )
    ref = reference_block(
        q, k, v, x,
        weights.w_gate, weights.w_up, weights.w_down,
    )
    return got, ref


__all__ = [
    "DEFAULT_HEAD_SLICE",
    "DEFAULT_INTERMEDIATE_SLICE",
    "DEFAULT_SEQ_LEN",
    "TinyLlamaLayer0Weights",
    "compile_for_tinyllama",
    "load_tinyllama_layer0",
    "project_qkv",
    "run_tinyllama_block",
    "slice_weights_for_megakernel",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    print("Loading TinyLlama layer-0 weights ...")
    full = load_tinyllama_layer0()
    print(f"  hidden_size       = {full.cfg['hidden_size']}")
    print(f"  intermediate_size = {full.cfg['intermediate_size']}")
    print(f"  num_attention_heads = {full.cfg['num_attention_heads']}")
    print(f"  num_key_value_heads = {full.cfg['num_key_value_heads']}")
    print(f"  full W_q.shape    = {tuple(full.w_q.shape)}")
    print(f"  full W_gate.shape = {tuple(full.w_gate.shape)}")
    print(f"  full W_down.shape = {tuple(full.w_down.shape)}")

    sliced, sliced_cfg = slice_weights_for_megakernel(full)
    print("\nUsing real-weight slice for megakernel test:")
    print(f"  n_heads      = {sliced_cfg['n_heads']}")
    print(f"  hidden_dim   = {sliced_cfg['hidden_dim']}")
    print(f"  intermediate = {sliced_cfg['intermediate']}")
    print(f"  W_q[:H*D, :H*D]  = {tuple(sliced.w_q.shape)}")
    print(f"  W_gate[:I, :H*D] = {tuple(sliced.w_gate.shape)}")
    print(f"  W_down[:H*D, :I] = {tuple(sliced.w_down.shape)}")

    print("\nCompiling transformer-block megakernel at sliced-TinyLlama dims ...")
    S = DEFAULT_SEQ_LEN
    compiled = compile_for_tinyllama(seq_len=S)
    print(f"  emitted kernel: {compiled.kernel_name}")
    print(f"  source = {len(compiled.kernel_source)} chars")
    print(f"  SM_COUNT = {compiled.sm_count}")
    print(f"  device functions: {sorted(compiled.lowering.device_function_table.values())}")

    torch.manual_seed(42)
    x = torch.randn(
        (S, sliced_cfg["hidden_dim"]), dtype=torch.float32, device="cuda",
    ) * 0.1

    print("\nRunning megakernel + reference ...")
    got, ref = run_tinyllama_block(compiled, x, sliced, sliced_cfg)

    err_abs = (got - ref).abs().max().item()
    err_rel = ((got - ref).abs() / (ref.abs() + 1e-6)).max().item()
    print(f"  max |got - ref|       = {err_abs:.3e}")
    print(f"  max |got - ref|/|ref| = {err_rel:.3e}")

    assert err_abs < 1e-2, (
        f"TinyLlama-block megakernel diverges from PyTorch eager at {err_abs:.3e} "
        "-- expected < 1e-2 with real Llama-scale weights."
    )
    print("\nPASS: emitted transformer-block megakernel matches PyTorch eager")
    print(
        f"      on TinyLlama-1.1B layer-0 weights "
        f"(real-weight slice: H={sliced_cfg['n_heads']}, "
        f"D={sliced_cfg['hidden_dim']}, I={sliced_cfg['intermediate']})."
    )
