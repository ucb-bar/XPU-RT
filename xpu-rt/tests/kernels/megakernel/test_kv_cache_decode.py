"""KV-cache decode-step megakernel regression tests (prefill + decode pattern).

Adds KV-cache support: a new decode-step megakernel that processes one
new token at a time using cached K/V from previous steps.  Composed
with the Phase G prefill megakernel and validated against
``HF.model.generate(do_sample=False)`` -- the production LLM-serving
pattern of "encode prompt once, decode incrementally with cache".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.modules.setdefault("torchvision", None)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="real-example tests require CUDA"
)


# ---------------------------------------------------------------------------
# H.1 -- decode-step megakernel correctness across multiple cache states
# ---------------------------------------------------------------------------


def test_decode_step_megakernel_matches_pytorch_reference_across_steps() -> None:
    from examples.event_tensor.llama_decode_step_megakernel import (
        compile_decode_step,
        reference_decode_step,
        run_decode_step,
    )
    from examples.event_tensor.llama_layer_rope_megakernel import hf_rope_tables

    H, N_KV, D_HEAD, I, S_MAX = 4, 2, 16, 64, 32
    D_HIDDEN = H * D_HEAD
    compiled = compile_decode_step(
        n_heads=H, n_kv_heads=N_KV, head_dim=D_HEAD,
        intermediate_dim=I, s_max=S_MAX,
    )
    cos, sin = hf_rope_tables(S_MAX, D_HEAD)

    torch.manual_seed(2026)
    w_norm1 = torch.randn((D_HIDDEN,),                  dtype=torch.float32, device="cuda") * 0.1 + 1.0
    w_q     = torch.randn((D_HIDDEN, D_HIDDEN),         dtype=torch.float32, device="cuda") * 0.05
    w_k     = torch.randn((N_KV * D_HEAD, D_HIDDEN),    dtype=torch.float32, device="cuda") * 0.05
    w_v     = torch.randn((N_KV * D_HEAD, D_HIDDEN),    dtype=torch.float32, device="cuda") * 0.05
    w_o     = torch.randn((D_HIDDEN, D_HIDDEN),         dtype=torch.float32, device="cuda") * 0.05
    w_norm2 = torch.randn((D_HIDDEN,),                  dtype=torch.float32, device="cuda") * 0.1 + 1.0
    w_gate  = torch.randn((I, D_HIDDEN),                dtype=torch.float32, device="cuda") * 0.05
    w_up    = torch.randn((I, D_HIDDEN),                dtype=torch.float32, device="cuda") * 0.05
    w_down  = torch.randn((D_HIDDEN, I),                dtype=torch.float32, device="cuda") * 0.05

    k_cache_mk  = torch.zeros((N_KV, S_MAX, D_HEAD), dtype=torch.float32, device="cuda")
    v_cache_mk  = torch.zeros((N_KV, S_MAX, D_HEAD), dtype=torch.float32, device="cuda")
    k_cache_ref = torch.zeros((N_KV, S_MAX, D_HEAD), dtype=torch.float32, device="cuda")
    v_cache_ref = torch.zeros((N_KV, S_MAX, D_HEAD), dtype=torch.float32, device="cuda")

    max_err = 0.0
    for step in range(5):
        x = torch.randn((D_HIDDEN,), dtype=torch.float32, device="cuda") * 0.1
        y_mk  = run_decode_step(
            compiled, x, k_cache_mk, v_cache_mk, context_len=step,
            w_norm1=w_norm1, w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o,
            w_norm2=w_norm2, w_gate=w_gate, w_up=w_up, w_down=w_down,
            cos=cos, sin=sin,
        )
        y_ref = reference_decode_step(
            x, k_cache_ref, v_cache_ref, context_len=step,
            w_norm1=w_norm1, w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o,
            w_norm2=w_norm2, w_gate=w_gate, w_up=w_up, w_down=w_down,
            cos=cos, sin=sin, n_heads=H, n_kv_heads=N_KV, head_dim=D_HEAD,
        )
        max_err = max(max_err, (y_mk - y_ref).abs().max().item())

    assert max_err < 5e-4, f"decode step diverges by {max_err}"


def test_decode_step_emits_kv_cache_machinery() -> None:
    from examples.event_tensor.llama_decode_step_megakernel import compile_decode_step

    compiled = compile_decode_step(n_heads=2, n_kv_heads=1, head_dim=16, intermediate_dim=32, s_max=16)
    src = compiled.kernel_source
    # KV cache pointers + per-token write at slot CONTEXT_LEN.
    assert "KCACHE_ptr" in src
    assert "VCACHE_ptr" in src
    assert "CONTEXT_LEN" in src
    # Causal-with-cache mask.
    assert "key_pos <= CONTEXT_LEN" in src
    # Sum-based attention (no tl.dot needed for 1-row Q).
    assert "tl.sum(k * q[None, :], axis=1)" in src or "tl.sum(v * probs" in src


# ---------------------------------------------------------------------------
# H.2 -- KV-cache greedy generation matches HF.model.generate()
# ---------------------------------------------------------------------------


def test_kv_cache_generation_matches_hf_generate() -> None:
    """Prefill megakernel + decode-step megakernel composed over a full
    greedy generation; verify token-by-token match with HF."""
    from examples.event_tensor.hf_drop_in_megakernel import (
        HFDropInBundle,
        build_small_llama_model,
        compile_megakernel_for,
    )
    from examples.event_tensor.hf_generate_with_megakernel import hf_greedy_generate
    from examples.event_tensor.hf_generate_with_kv_cache import kv_cache_generate
    from examples.event_tensor.llama_decode_step_megakernel import compile_decode_step
    from examples.event_tensor.llama_layer_rope_megakernel import hf_rope_tables

    SEQ_LEN, MAX_NEW = 16, 8
    S_MAX = ((SEQ_LEN + MAX_NEW + 15) // 16) * 16

    model = build_small_llama_model()
    cfg = model.config
    prefill_compiled = compile_megakernel_for(model, seq_len=S_MAX)
    decode_compiled = compile_decode_step(
        n_heads=cfg.num_attention_heads, n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, intermediate_dim=cfg.intermediate_size,
        s_max=S_MAX,
    )
    cos_full, sin_full = hf_rope_tables(
        seq_len=S_MAX, head_dim=cfg.head_dim, base=10000.0,
        device="cuda", dtype=torch.float32,
    )
    bundle = HFDropInBundle(
        model=model, config=cfg, compiled=prefill_compiled,
        layer_idx=0, cos=cos_full, sin=sin_full,
    )

    torch.manual_seed(20260418)
    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), device="cuda")

    hf_seq = hf_greedy_generate(model, input_ids, MAX_NEW)
    hf_new = hf_seq[0, SEQ_LEN:].tolist()

    mk_seq = kv_cache_generate(bundle, decode_compiled, input_ids, MAX_NEW, S_MAX)
    mk_new = mk_seq[0, SEQ_LEN:].tolist()

    assert hf_new == mk_new, (
        f"KV-cache generation diverges:\n  HF: {hf_new}\n  MK: {mk_new}"
    )
