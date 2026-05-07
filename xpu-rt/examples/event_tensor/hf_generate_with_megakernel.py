"""Real  example: greedy generation with our megakernel.

End-to-end demonstration that the emitted persistent megakernel can
**generate tokens** the same way HuggingFace's actual ``LlamaForCausalLM``
does.

Pipeline:

    1. Build a real ``LlamaForCausalLM`` at reduced (megakernel-fittable)
       dims (real HF code path; just smaller config so the megakernel
       fits inside the TITAN-RTX 64KB shared-memory budget).
    2. Take a real prompt of input_ids, run greedy autoregressive
       generation for N_NEW tokens via:
          (a) HF's own ``model.generate(..., do_sample=False)`` --
              the production path.
          (b) Our manual greedy loop where every decoder layer is
              executed by **our emitted megakernel** instead of HF's
              ``LlamaDecoderLayer.forward()``.
    3. Verify the two paths produce the **same generated token IDs**.

This is the strongest "real LLM" demonstration the test surface can
make: we generate the same sequence of tokens that HF generates, with
every transformer-layer call being our compiler's emitted code.

Note on KV cache: this  example re-encodes the entire growing
sequence on each step (no KV cache reuse).  That is wasteful but keeps
the demo focused on layer-level correctness; KV-cache append is the
next emitter optimisation.
"""

from __future__ import annotations

import sys
try:
    import torchvision as _tv  # real install — let transformers use it
    del _tv
except ImportError:
    sys.modules.setdefault("torchvision", None)

import torch

from transformers.generation.configuration_utils import GenerationConfig

from examples.event_tensor.hf_drop_in_megakernel import (
    HFDropInBundle,
    _layer_megakernel_output,
    build_small_llama_model,
    compile_megakernel_for,
)
from examples.event_tensor.llama_layer_rope_megakernel import hf_rope_tables


def hf_greedy_generate(
    model, input_ids: torch.Tensor, max_new_tokens: int,
) -> torch.Tensor:
    """HF's own greedy path (calls ``model.generate``)."""
    cfg = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=False,            # match our manual loop's no-cache path
        pad_token_id=0,
        eos_token_id=None,
    )
    with torch.no_grad():
        out = model.generate(input_ids, generation_config=cfg)
    return out                                          # (1, S+max_new)


def megakernel_greedy_generate(
    model, compiled, input_ids: torch.Tensor, max_new_tokens: int,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Greedy generation where every decoder layer is our megakernel."""
    cfg = model.config
    device = input_ids.device
    cur = input_ids.clone()                             # (1, S)

    with torch.no_grad():
        for _step in range(max_new_tokens):
            S = cur.shape[1]
            cos, sin = hf_rope_tables(
                seq_len=S, head_dim=cfg.head_dim,
                base=rope_theta, device=str(device), dtype=torch.float32,
            )

            hidden = model.model.embed_tokens(cur).squeeze(0)           # (S, D)

            for i in range(cfg.num_hidden_layers):
                bundle = HFDropInBundle(
                    model=model, config=cfg, compiled=compiled,
                    layer_idx=i, cos=cos, sin=sin,
                )
                hidden = _layer_megakernel_output(bundle, hidden)

            hidden = model.model.norm(hidden)
            logits = model.lm_head(hidden)                              # (S, vocab)
            next_id = int(logits[-1].argmax().item())
            cur = torch.cat(
                [cur, torch.tensor([[next_id]], device=device, dtype=cur.dtype)],
                dim=1,
            )
    return cur


__all__ = [
    "hf_greedy_generate",
    "megakernel_greedy_generate",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    SEQ_LEN     = 16
    MAX_NEW     = 8

    print("Building real LlamaForCausalLM ...")
    model = build_small_llama_model()
    cfg = model.config
    print(f"  num_layers={cfg.num_hidden_layers}, vocab={cfg.vocab_size}, "
          f"hidden={cfg.hidden_size}, H={cfg.num_attention_heads}, "
          f"N_KV={cfg.num_key_value_heads}")

    print("\nCompiling megakernel for the prompt-length pass ...")
    # We must recompile per growing sequence length; for this demo we
    # accept that and JIT the kernel inside the loop on first use.
    print(f"  initial compile for SEQ_LEN={SEQ_LEN}")
    initial_compiled = compile_megakernel_for(model, seq_len=SEQ_LEN)

    torch.manual_seed(20260418)
    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), device="cuda")
    print(f"  prompt input_ids: {input_ids.tolist()}")

    print("\n[1/2] Generating with HF's own model.generate(do_sample=False) ...")
    hf_seq = hf_greedy_generate(model, input_ids, MAX_NEW)
    hf_new = hf_seq[0, SEQ_LEN:].tolist()
    print(f"  HF generated tokens: {hf_new}")

    print("\n[2/2] Generating with our megakernel substituted for ALL layers ...")
    # The megakernel bakes M_TILES = S / BLOCK_M into the compiled queue,
    # so we precompile a single kernel at the max sequence length we'll
    # ever encounter (SEQ_LEN + MAX_NEW rounded up to BLOCK_M) and pad
    # each step's input to that fixed length.  Causal masking makes the
    # padded positions inert -- they only attend to themselves and don't
    # affect the real positions' outputs.  We index logits at the last
    # *real* position to pick the next token.
    BLOCK_M = 16
    final_max_S = ((SEQ_LEN + MAX_NEW + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    print(f"  compiling megakernel for fixed padded-S={final_max_S}")
    compiled_padded = compile_megakernel_for(model, seq_len=final_max_S)
    cos_full, sin_full = hf_rope_tables(
        seq_len=final_max_S, head_dim=cfg.head_dim,
        base=10000.0, device="cuda", dtype=torch.float32,
    )

    cur = input_ids.clone()
    for step in range(MAX_NEW):
        real_S = cur.shape[1]
        # Pad with 0s up to final_max_S; causal mask keeps real positions correct.
        pad_len = final_max_S - real_S
        padded = torch.cat(
            [cur, torch.zeros((1, pad_len), dtype=cur.dtype, device="cuda")], dim=1,
        )
        with torch.no_grad():
            hidden = model.model.embed_tokens(padded).squeeze(0)
            for i in range(cfg.num_hidden_layers):
                bundle = HFDropInBundle(
                    model=model, config=cfg, compiled=compiled_padded,
                    layer_idx=i, cos=cos_full, sin=sin_full,
                )
                hidden = _layer_megakernel_output(bundle, hidden)
            hidden = model.model.norm(hidden)
            logits = model.lm_head(hidden)
            # Index logits at the LAST REAL position, not the last padded one.
            next_id = int(logits[real_S - 1].argmax().item())
        cur = torch.cat(
            [cur, torch.tensor([[next_id]], device="cuda", dtype=cur.dtype)],
            dim=1,
        )

    mk_new = cur[0, SEQ_LEN:].tolist()
    print(f"  megakernel generated tokens: {mk_new}")

    print(f"\nHF                tokens: {hf_new}")
    print(f"megakernel        tokens: {mk_new}")
    matches = sum(int(a == b) for a, b in zip(hf_new, mk_new))
    print(f"matching positions: {matches} / {len(hf_new)}")

    assert hf_new == mk_new, (
        "Greedy generation diverges between HF and our megakernel "
        f"at step {next(i for i,(a,b) in enumerate(zip(hf_new, mk_new)) if a != b)}: "
        f"HF picked {hf_new}, megakernel picked {mk_new}"
    )
    print("\nPASS: megakernel-driven greedy generation produces the SAME tokens as HF.")
