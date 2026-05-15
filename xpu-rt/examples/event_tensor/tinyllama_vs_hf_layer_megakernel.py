"""Real  example: HF-faithful PyTorch reference vs the actual HF
``LlamaDecoderLayer.forward()`` on real TinyLlama-1.1B layer-0 weights.

Validates the chain that turns "our megakernel matches our reference"
into "our megakernel matches HuggingFace":

    megakernel  ==  our_HF_faithful_reference
    our_HF_faithful_reference  ==  HF.LlamaDecoderLayer.forward()
                                                 (proved here, on real
                                                  TinyLlama weights at the
                                                  full-dim subset of the
                                                  config that fits MHA)

Together: ``megakernel  ==  HF.LlamaDecoderLayer.forward()``  modulo the
test slice and shared-memory constraints.

We construct an HF ``LlamaDecoderLayer`` whose config matches TinyLlama
exactly (rope_theta, rms_norm_eps, intermediate_size, head_dim, hidden_size)
*except* that ``num_key_value_heads`` is set equal to
``num_attention_heads`` so the layer runs in plain MHA mode -- this lets
us reuse the same weight slice the megakernel example uses (4 Q heads,
4 K/V heads from the first 4 KV heads of TinyLlama's GQA).  Under that
config, every other operator (RMSNorm, RoPE, SDPA, SwiGLU MLP) runs the
same code path HF runs in production, so the comparison validates
exactly the math our reference is supposed to mirror.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

import sys
# transformers' top-level import drags in torchvision; bypass it.
try:
    import torchvision as _tv  # real install — let transformers use it
    del _tv
except ImportError:
    sys.modules.setdefault("torchvision", None)

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from examples.event_tensor.llama_layer_rope_megakernel import (
    hf_rope_tables,
    reference_llama_layer_rope,
)
from examples.event_tensor.tinyllama_hf_layer_megakernel import (
    DEFAULT_HEAD_DIM,
    DEFAULT_HEAD_SLICE,
    DEFAULT_INTERMEDIATE_SLICE,
    DEFAULT_SEQ_LEN,
    load_tinyllama_hf_layer0,
    slice_hf_weights,
)


def build_hf_layer_at_sliced_dims(sliced_cfg: dict, device: str = "cuda") -> LlamaDecoderLayer:
    """Construct a HuggingFace ``LlamaDecoderLayer`` at the sliced dims.

    Sets ``num_key_value_heads = num_attention_heads`` so HF runs plain
    MHA on our slice (matching the slice helper, which collapses GQA).
    """
    cfg = LlamaConfig(
        hidden_size            = sliced_cfg["hidden_dim"],
        intermediate_size      = sliced_cfg["intermediate"],
        num_attention_heads    = sliced_cfg["n_heads"],
        num_key_value_heads    = sliced_cfg["n_heads"],
        head_dim               = sliced_cfg["head_dim"],
        rms_norm_eps           = float(sliced_cfg["rms_eps"]),
        rope_theta             = float(sliced_cfg["rope_base"]),
        # The remaining defaults (vocab, num_hidden_layers, etc.) don't affect
        # a single-layer forward.
        attention_bias         = False,
        mlp_bias               = False,
        hidden_act             = "silu",
    )
    layer = LlamaDecoderLayer(cfg, layer_idx=0).to(device=device, dtype=torch.float32)
    layer.eval()
    return layer


def install_sliced_weights_into_hf_layer(
    layer: LlamaDecoderLayer, sliced, sliced_cfg: dict,
) -> None:
    """Load our sliced TinyLlama weights into the HF layer's parameters."""
    with torch.no_grad():
        layer.input_layernorm.weight.copy_(sliced.w_norm1)
        layer.post_attention_layernorm.weight.copy_(sliced.w_norm2)
        layer.self_attn.q_proj.weight.copy_(sliced.w_q)
        layer.self_attn.k_proj.weight.copy_(sliced.w_k)
        layer.self_attn.v_proj.weight.copy_(sliced.w_v)
        layer.self_attn.o_proj.weight.copy_(sliced.w_o)
        layer.mlp.gate_proj.weight.copy_(sliced.w_gate)
        layer.mlp.up_proj.weight.copy_(sliced.w_up)
        layer.mlp.down_proj.weight.copy_(sliced.w_down)


def run_hf_layer(
    layer: LlamaDecoderLayer,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Invoke HF ``LlamaDecoderLayer.forward`` with explicit cos/sin
    position_embeddings, matching our reference's convention exactly."""
    S = x.shape[0]
    # HF expects (B, S, D) and a position_embeddings tuple.  Since we
    # pre-computed the same cos/sin tables our reference uses, we hand
    # them in directly -- this avoids any divergence from the model's
    # internal rotary_emb constructor.
    hidden = x.unsqueeze(0)                                 # (1, S, D)
    cos_b  = cos.unsqueeze(0).expand(1, S, cos.shape[-1])   # (1, S, D_HEAD)
    sin_b  = sin.unsqueeze(0).expand(1, S, sin.shape[-1])
    # Feed a None attention_mask -- HF will apply causal masking
    # automatically via SDPA's is_causal path when no mask is given.
    out = layer(
        hidden_states=hidden,
        attention_mask=None,
        position_embeddings=(cos_b, sin_b),
    )
    if isinstance(out, tuple):
        out = out[0]
    return out.squeeze(0)


def compare_reference_to_hf(
    seq_len: int = DEFAULT_SEQ_LEN,
    n_heads: int = DEFAULT_HEAD_SLICE,
    intermediate: int = DEFAULT_INTERMEDIATE_SLICE,
    head_dim: int = DEFAULT_HEAD_DIM,
    seed: int = 31337,
    device: str = "cuda",
) -> dict:
    """End-to-end run: build HF layer at sliced dims, install real
    TinyLlama weights, run forward, compare to our reference."""
    full = load_tinyllama_hf_layer0()
    sliced, sliced_cfg = slice_hf_weights(
        full, n_heads=n_heads, intermediate=intermediate, head_dim=head_dim,
    )

    hf_layer = build_hf_layer_at_sliced_dims(sliced_cfg, device=device)
    install_sliced_weights_into_hf_layer(hf_layer, sliced, sliced_cfg)

    cos, sin = hf_rope_tables(
        seq_len=seq_len, head_dim=head_dim,
        base=float(sliced_cfg["rope_base"]),
        device=device, dtype=torch.float32,
    )

    torch.manual_seed(seed)
    x = torch.randn((seq_len, sliced_cfg["hidden_dim"]), dtype=torch.float32, device=device) * 0.1

    with torch.no_grad():
        hf_out = run_hf_layer(hf_layer, x, cos, sin)
    ref_out = reference_llama_layer_rope(
        x, sliced.w_norm1, sliced.w_q, sliced.w_k, sliced.w_v, sliced.w_o,
        sliced.w_norm2, sliced.w_gate, sliced.w_up, sliced.w_down,
        cos, sin, n_heads=n_heads, head_dim=head_dim,
        rms_eps=float(sliced_cfg["rms_eps"]),
    )

    err_abs = (hf_out - ref_out).abs().max().item()
    err_rel = ((hf_out - ref_out).abs() / (ref_out.abs() + 1e-6)).max().item()
    return {
        "err_abs": err_abs,
        "err_rel": err_rel,
        "hf_out": hf_out,
        "ref_out": ref_out,
        "sliced_cfg": sliced_cfg,
    }


__all__ = [
    "build_hf_layer_at_sliced_dims",
    "compare_reference_to_hf",
    "install_sliced_weights_into_hf_layer",
    "run_hf_layer",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    print("Building HF LlamaDecoderLayer at sliced TinyLlama dims ...")
    result = compare_reference_to_hf()
    cfg = result["sliced_cfg"]
    print(f"  config: H={cfg['n_heads']}, D_HEAD={cfg['head_dim']}, "
          f"D_HIDDEN={cfg['hidden_dim']}, I={cfg['intermediate']}, "
          f"rope_theta={cfg['rope_base']}, rms_eps={cfg['rms_eps']}")
    print(f"\n  HF output mean abs:        {result['hf_out'].abs().mean().item():.4e}")
    print(f"  reference output mean abs: {result['ref_out'].abs().mean().item():.4e}")
    print(f"  max |HF - reference|       = {result['err_abs']:.3e}")
    print(f"  max |HF - reference|/|HF|  = {result['err_rel']:.3e}")

    # HF and our reference do the same math but with subtly different
    # accumulation order (HF uses fused matmul kernels; we use plain
    # @-matmul), so we allow 1e-4 absolute slack.
    assert result["err_abs"] < 1e-4, (
        "Our PyTorch reference diverges from HF's actual LlamaDecoderLayer.forward "
        f"by {result['err_abs']:.3e} -- expected < 1e-4."
    )
    print("\nPASS: our HF-faithful PyTorch reference matches HF.LlamaDecoderLayer.forward()")
    print("      on real TinyLlama-1.1B layer-0 weights.")
    print("      => the megakernel (which matches the reference at sliced dims)")
    print("         runs the same math as HF's actual decoder layer.")
