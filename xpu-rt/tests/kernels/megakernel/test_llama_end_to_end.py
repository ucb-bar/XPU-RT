"""End-to-end Llama megakernel regression tests (drop-in HF layer + greedy generation).

End-to-end demonstration that the emitted megakernel is a drop-in
replacement for HuggingFace's ``LlamaDecoderLayer.forward()`` inside
an actual ``LlamaForCausalLM`` -- both for single-step
``model.forward()`` and for autoregressive ``model.generate()``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
# transformers is in the optional `demo` extra (heavy HF dep). When the
# CompGen install doesn't include it, skip rather than fail at import-time
# of the example module — matches the torch/triton importorskip pattern.
pytest.importorskip(
    "transformers",
    reason="HF parity tests require the `demo` extra (uv sync --extra demo)",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torchvision as _tv  # real install — let transformers use it
    del _tv
except ImportError:
    sys.modules.setdefault("torchvision", None)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="real-example tests require CUDA")


# ---------------------------------------------------------------------------
# G.1 -- megakernel as drop-in for HF Llama layer in real model.forward()
# ---------------------------------------------------------------------------


def test_megakernel_substitutes_one_layer_in_real_llama_forward() -> None:
    """Substitute layer-0 with our megakernel inside a real LlamaForCausalLM
    and confirm the model's logits stay within float-32 noise of pure HF."""
    from examples.event_tensor.hf_drop_in_megakernel import (
        hf_only_forward,
        make_bundle,
        substituted_forward,
    )

    bundle = make_bundle(seq_len=16, layer_idx=0)
    torch.manual_seed(54321)
    input_ids = torch.randint(0, bundle.config.vocab_size, (1, 16), device="cuda")

    hf_logits = hf_only_forward(bundle.model, input_ids)
    sub_logits = substituted_forward(bundle, input_ids)
    err = (hf_logits - sub_logits).abs().max().item()
    assert err < 5e-4, f"single-layer substitution diverges by {err}"
    assert int(hf_logits[-1].argmax()) == int(sub_logits[-1].argmax())


def test_megakernel_substitutes_all_layers_in_real_llama_forward() -> None:
    """Same as above but our megakernel handles EVERY decoder layer."""
    from examples.event_tensor.hf_drop_in_megakernel import (
        fully_substituted_forward,
        hf_only_forward,
        make_bundle,
    )

    bundle = make_bundle(seq_len=16, layer_idx=0)
    torch.manual_seed(54321)
    input_ids = torch.randint(0, bundle.config.vocab_size, (1, 16), device="cuda")

    hf_logits = hf_only_forward(bundle.model, input_ids)
    full_logits = fully_substituted_forward(bundle, input_ids)
    err = (hf_logits - full_logits).abs().max().item()
    assert err < 5e-3, f"all-layers substitution diverges by {err}"
    assert int(hf_logits[-1].argmax()) == int(full_logits[-1].argmax())


# ---------------------------------------------------------------------------
# G.2 -- megakernel-driven greedy generation matches HF.model.generate()
# ---------------------------------------------------------------------------


def test_megakernel_greedy_generation_matches_hf_generate() -> None:
    """Run greedy autoregressive generation with our megakernel handling
    EVERY decoder layer; verify token-by-token match against
    ``HF.model.generate(do_sample=False)``."""
    from examples.event_tensor.hf_drop_in_megakernel import (
        HFDropInBundle,
        _layer_megakernel_output,
        build_small_llama_model,
        compile_megakernel_for,
    )
    from examples.event_tensor.hf_generate_with_megakernel import hf_greedy_generate
    from examples.event_tensor.llama_layer_rope_megakernel import hf_rope_tables

    SEQ_LEN, MAX_NEW, BLOCK_M = 16, 8, 16
    final_max_S = ((SEQ_LEN + MAX_NEW + BLOCK_M - 1) // BLOCK_M) * BLOCK_M

    model = build_small_llama_model()
    cfg = model.config
    compiled = compile_megakernel_for(model, seq_len=final_max_S)
    cos_full, sin_full = hf_rope_tables(
        seq_len=final_max_S,
        head_dim=cfg.head_dim,
        base=10000.0,
        device="cuda",
        dtype=torch.float32,
    )

    torch.manual_seed(20260418)
    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), device="cuda")

    # HF reference path
    hf_seq = hf_greedy_generate(model, input_ids, MAX_NEW)
    hf_new = hf_seq[0, SEQ_LEN:].tolist()

    # Megakernel path: pad to final_max_S each step; index logits at last real position.
    cur = input_ids.clone()
    for _step in range(MAX_NEW):
        real_S = cur.shape[1]
        padded = torch.cat(
            [cur, torch.zeros((1, final_max_S - real_S), dtype=cur.dtype, device="cuda")],
            dim=1,
        )
        with torch.no_grad():
            hidden = model.model.embed_tokens(padded).squeeze(0)
            for i in range(cfg.num_hidden_layers):
                bundle = HFDropInBundle(
                    model=model,
                    config=cfg,
                    compiled=compiled,
                    layer_idx=i,
                    cos=cos_full,
                    sin=sin_full,
                )
                hidden = _layer_megakernel_output(bundle, hidden)
            hidden = model.model.norm(hidden)
            logits = model.lm_head(hidden)
            next_id = int(logits[real_S - 1].argmax().item())
        cur = torch.cat(
            [cur, torch.tensor([[next_id]], device="cuda", dtype=cur.dtype)],
            dim=1,
        )
    mk_new = cur[0, SEQ_LEN:].tolist()

    assert hf_new == mk_new, f"Greedy generation diverges:\n  HF: {hf_new}\n  MK: {mk_new}"
