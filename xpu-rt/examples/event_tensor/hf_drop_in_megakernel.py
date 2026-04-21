"""Real  example: megakernel as a drop-in HF Llama decoder layer.

The proof:

    1. Build a real ``transformers.LlamaForCausalLM`` at reduced
       (megakernel-fittable) dims.
    2. Run a real ``model.forward(input_ids)`` end-to-end, including
       ``embed_tokens`` -> N decoder layers -> final RMSNorm ->
       ``lm_head``.  Save the logits.
    3. Re-run the forward, but for one chosen layer use **our emitted
       megakernel** instead of HF's ``LlamaDecoderLayer.forward()``.
       All other layers + embedding + final norm + lm_head still come
       from HF.
    4. Verify the resulting logits match the pure-HF logits to within
       float-32 tolerance.

This proves the megakernel is a true **drop-in replacement** for HF's
production decoder-layer code path inside an actual
``LlamaForCausalLM``.

 already proved
    our_PyTorch_reference == HF.LlamaDecoderLayer.forward()
on real TinyLlama weights at the layer level.   scales that to
the model level: the substitution survives all the wiring HF does
around a layer (residual norms, position-embedding plumbing, output
unpacking, lm_head).

Validated on the GQA megakernel from
:mod:`examples.event_tensor.llama_layer_gqa_megakernel` so we exercise
the actual TinyLlama-style attention structure (n_kv_heads <
n_attention_heads).
"""

from __future__ import annotations

import sys
sys.modules.setdefault("torchvision", None)

from dataclasses import dataclass

import torch

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from examples.event_tensor.llama_layer_gqa_megakernel import (
    CompiledLlamaLayerGQA,
    compile_llama_layer_gqa,
    run_llama_layer_gqa,
)
from examples.event_tensor.llama_layer_rope_megakernel import hf_rope_tables


@dataclass
class HFDropInBundle:
    """Everything needed to substitute layer ``layer_idx`` with our megakernel."""

    model: LlamaForCausalLM
    config: LlamaConfig
    compiled: CompiledLlamaLayerGQA
    layer_idx: int
    cos: torch.Tensor          # (S, head_dim)
    sin: torch.Tensor


def build_small_llama_model(
    vocab_size: int = 64,
    hidden_size: int = 64,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    head_dim: int = 16,
    intermediate_size: int = 64,
    num_layers: int = 2,
    rope_theta: float = 10000.0,
    rms_eps: float = 1e-5,
    seed: int = 9999,
    device: str = "cuda",
) -> LlamaForCausalLM:
    """Construct a real HF ``LlamaForCausalLM`` at megakernel-fittable dims.

    Real architecture (real HF code path); just smaller so the megakernel
    fits inside the TITAN-RTX 64KB shared-memory budget.
    """
    config = LlamaConfig(
        vocab_size            = vocab_size,
        hidden_size           = hidden_size,
        intermediate_size     = intermediate_size,
        num_attention_heads   = n_heads,
        num_key_value_heads   = n_kv_heads,
        head_dim              = head_dim,
        num_hidden_layers     = num_layers,
        rms_norm_eps          = rms_eps,
        rope_parameters       = {"rope_type": "default", "rope_theta": rope_theta},
        attention_bias        = False,
        mlp_bias              = False,
        hidden_act            = "silu",
        tie_word_embeddings   = False,
    )
    torch.manual_seed(seed)
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.float32)
    model.eval()
    return model


def compile_megakernel_for(model: LlamaForCausalLM, seq_len: int) -> CompiledLlamaLayerGQA:
    cfg = model.config
    return compile_llama_layer_gqa(
        n_heads          = cfg.num_attention_heads,
        n_kv_heads       = cfg.num_key_value_heads,
        seq_len          = seq_len,
        head_dim         = cfg.head_dim,
        intermediate_dim = cfg.intermediate_size,
        block_m          = min(16, seq_len),
        block_i          = min(32, cfg.intermediate_size),
        block_n          = min(32, cfg.hidden_size),
    )


def _layer_megakernel_output(
    bundle: HFDropInBundle, hidden: torch.Tensor,
) -> torch.Tensor:
    """Run our megakernel on the chosen layer's weights."""
    cfg = bundle.config
    layer = bundle.model.model.layers[bundle.layer_idx]
    return run_llama_layer_gqa(
        bundle.compiled,
        hidden,
        layer.input_layernorm.weight,
        layer.self_attn.q_proj.weight,
        layer.self_attn.k_proj.weight,
        layer.self_attn.v_proj.weight,
        layer.self_attn.o_proj.weight,
        layer.post_attention_layernorm.weight,
        layer.mlp.gate_proj.weight,
        layer.mlp.up_proj.weight,
        layer.mlp.down_proj.weight,
        bundle.cos, bundle.sin,
        rms_eps=cfg.rms_norm_eps,
    )


def hf_only_forward(model: LlamaForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    """Pure-HF logits."""
    with torch.no_grad():
        return model(input_ids).logits.squeeze(0)        # (S, vocab)


def substituted_forward(
    bundle: HFDropInBundle, input_ids: torch.Tensor,
) -> torch.Tensor:
    """Run the model end-to-end but substitute ``layer_idx`` with our megakernel.

    We replicate HF's ``LlamaModel.forward`` step-by-step, because our
    megakernel returns a (S, D) tensor and we want to plug it into HF's
    own pipeline cleanly.
    """
    model  = bundle.model
    cfg    = model.config
    cos    = bundle.cos
    sin    = bundle.sin
    layer_idx = bundle.layer_idx

    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids).squeeze(0)        # (S, D)

        S = hidden.shape[0]
        cos_b = cos.unsqueeze(0).expand(1, S, cos.shape[-1])
        sin_b = sin.unsqueeze(0).expand(1, S, sin.shape[-1])

        for i, layer in enumerate(model.model.layers):
            if i == layer_idx:
                hidden = _layer_megakernel_output(bundle, hidden)
            else:
                out = layer(
                    hidden_states         = hidden.unsqueeze(0),
                    attention_mask        = None,
                    position_embeddings   = (cos_b, sin_b),
                )
                if isinstance(out, tuple):
                    out = out[0]
                hidden = out.squeeze(0)

        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)
    return logits                                         # (S, vocab)


def fully_substituted_forward(
    bundle: HFDropInBundle, input_ids: torch.Tensor,
) -> torch.Tensor:
    """Same as :func:`substituted_forward` but our megakernel handles
    *every* decoder layer."""
    model  = bundle.model
    cos    = bundle.cos
    sin    = bundle.sin

    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids).squeeze(0)
        for i in range(len(model.model.layers)):
            mb = HFDropInBundle(
                model=model, config=model.config,
                compiled=bundle.compiled, layer_idx=i,
                cos=cos, sin=sin,
            )
            hidden = _layer_megakernel_output(mb, hidden)
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)
    return logits


def make_bundle(
    seq_len: int = 16, layer_idx: int = 0, **build_kwargs,
) -> HFDropInBundle:
    model = build_small_llama_model(**build_kwargs)
    compiled = compile_megakernel_for(model, seq_len=seq_len)
    rope_theta = float(
        model.config.rope_parameters.get("rope_theta", 10000.0)
        if hasattr(model.config, "rope_parameters")
        else getattr(model.config, "rope_theta", 10000.0)
    )
    cos, sin = hf_rope_tables(
        seq_len=seq_len, head_dim=model.config.head_dim,
        base=rope_theta,
        device=str(next(model.parameters()).device), dtype=torch.float32,
    )
    return HFDropInBundle(
        model=model, config=model.config, compiled=compiled,
        layer_idx=layer_idx, cos=cos, sin=sin,
    )


__all__ = [
    "HFDropInBundle",
    "build_small_llama_model",
    "compile_megakernel_for",
    "fully_substituted_forward",
    "hf_only_forward",
    "make_bundle",
    "substituted_forward",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    SEQ_LEN = 16
    print("Building real LlamaForCausalLM at reduced dims ...")
    bundle = make_bundle(seq_len=SEQ_LEN, layer_idx=0)
    cfg = bundle.config
    rope_theta = (
        cfg.rope_parameters.get("rope_theta", 10000.0)
        if hasattr(cfg, "rope_parameters")
        else getattr(cfg, "rope_theta", 10000.0)
    )
    print(f"  vocab={cfg.vocab_size}, hidden={cfg.hidden_size}, "
          f"H={cfg.num_attention_heads}, N_KV={cfg.num_key_value_heads}, "
          f"head_dim={cfg.head_dim}, intermediate={cfg.intermediate_size}, "
          f"num_layers={cfg.num_hidden_layers}, rope_theta={rope_theta}")
    print(f"  emitted GQA megakernel: {bundle.compiled.kernel_name}")

    torch.manual_seed(54321)
    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), device="cuda")
    print(f"  input_ids: {input_ids.tolist()}")

    print("\nRunning pure-HF forward ...")
    hf_logits = hf_only_forward(bundle.model, input_ids)
    print(f"  logits shape = {tuple(hf_logits.shape)}")

    print("\nRunning forward with megakernel substituted for layer 0 ...")
    sub_logits = substituted_forward(bundle, input_ids)
    err1 = (hf_logits - sub_logits).abs().max().item()
    print(f"  max |HF - substituted_layer0|         = {err1:.3e}")

    print("\nRunning forward with megakernel substituted for ALL layers ...")
    full_logits = fully_substituted_forward(bundle, input_ids)
    err2 = (hf_logits - full_logits).abs().max().item()
    print(f"  max |HF - all_layers_substituted|     = {err2:.3e}")

    # HF's matmul accumulation order differs subtly from ours.  At fp32
    # the per-layer accumulation drift is bounded; we require both
    # paths to stay within an order of magnitude of float-32 noise.
    assert err1 < 5e-4, f"single-layer substitution diverges by {err1}"
    assert err2 < 5e-3, f"all-layers substitution diverges by {err2}"

    # Sanity: greedy next-token must match HF's choice.
    hf_next  = int(hf_logits[-1].argmax().item())
    sub_next = int(sub_logits[-1].argmax().item())
    full_next = int(full_logits[-1].argmax().item())
    print(f"\nGreedy next-token from each path:")
    print(f"  HF only                = {hf_next}")
    print(f"  layer-0 substituted    = {sub_next}")
    print(f"  all layers substituted = {full_next}")
    assert hf_next == sub_next  == full_next, "greedy next-token disagrees"

    print("\nPASS: emitted megakernel is a drop-in replacement for HF's")
    print("      LlamaDecoderLayer.forward() inside an actual LlamaForCausalLM.")
    print("      Greedy next-token matches between pure-HF and megakernel paths.")
