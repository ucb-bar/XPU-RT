"""HF-faithful Llama decoder layer (RoPE + causal) megakernel regression tests.

Adds the operators that turn the  Llama decoder layer into an
HF-faithful one: RoPE (half-rotation, matching HF's
``apply_rotary_pos_emb``) and causal attention mask.  Every test
executes the actually-emitted persistent megakernel on a real GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="real-example tests require CUDA")


# ---------------------------------------------------------------------------
# E.1 -- Llama layer megakernel with RoPE + causal (synthetic weights)
# ---------------------------------------------------------------------------


def test_llama_layer_with_rope_and_causal_matches_reference() -> None:
    from examples.event_tensor.llama_layer_rope_megakernel import (
        compile_llama_layer_rope,
        hf_rope_tables,
        reference_llama_layer_rope,
        run_llama_layer_rope,
    )

    H, S, D_HEAD, I = 4, 16, 16, 64
    D_HIDDEN = H * D_HEAD
    compiled = compile_llama_layer_rope(
        n_heads=H,
        seq_len=S,
        head_dim=D_HEAD,
        intermediate_dim=I,
    )
    cos, sin = hf_rope_tables(S, D_HEAD)
    torch.manual_seed(101)
    x = torch.randn((S, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.1
    w_norm1 = torch.randn((D_HIDDEN,), dtype=torch.float32, device="cuda") * 0.1 + 1.0
    w_q = torch.randn((D_HIDDEN, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_k = torch.randn((D_HIDDEN, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_v = torch.randn((D_HIDDEN, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_o = torch.randn((D_HIDDEN, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_norm2 = torch.randn((D_HIDDEN,), dtype=torch.float32, device="cuda") * 0.1 + 1.0
    w_gate = torch.randn((I, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_up = torch.randn((I, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_down = torch.randn((D_HIDDEN, I), dtype=torch.float32, device="cuda") * 0.05

    got = run_llama_layer_rope(
        compiled,
        x,
        w_norm1,
        w_q,
        w_k,
        w_v,
        w_o,
        w_norm2,
        w_gate,
        w_up,
        w_down,
        cos,
        sin,
    )
    ref = reference_llama_layer_rope(
        x,
        w_norm1,
        w_q,
        w_k,
        w_v,
        w_o,
        w_norm2,
        w_gate,
        w_up,
        w_down,
        cos,
        sin,
        n_heads=H,
        head_dim=D_HEAD,
    )
    err = (got - ref).abs().max().item()
    assert err < 5e-3, f"layer (RoPE+causal) diverges by {err}"


def test_emitted_kernel_contains_rope_and_causal_logic() -> None:
    from examples.event_tensor.llama_layer_rope_megakernel import (
        compile_llama_layer_rope,
    )

    compiled = compile_llama_layer_rope(
        n_heads=2,
        seq_len=16,
        head_dim=16,
        intermediate_dim=32,
    )
    src = compiled.kernel_source
    # RoPE-specific markers
    assert "_run_rope_apply" in src
    assert "EROPE_ptr" in src
    assert "COS_ptr" in src and "SIN_ptr" in src
    assert "D_HEAD_HALF" in src
    # Causal-mask marker (the literal -1e30 sentinel from compute_scores)
    assert "-1e30" in src or "-1e+30" in src
    # All ten device functions present
    for fn in (
        "_run_input_norm",
        "_run_qkv_proj",
        "_run_rope_apply",
        "_run_compute_scores",
        "_run_apply_values",
        "_run_o_proj_residual",
        "_run_post_attn_norm",
        "_run_mlp_gate_proj",
        "_run_mlp_up_proj",
        "_run_mlp_down_proj",
    ):
        assert fn in src, f"{fn} missing from emitted RoPE megakernel"


# ---------------------------------------------------------------------------
# E.2 -- real TinyLlama layer through HF-faithful megakernel
# ---------------------------------------------------------------------------


_TINYLLAMA_CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"))


@pytest.mark.skipif(
    not _TINYLLAMA_CACHE.exists(),
    reason="TinyLlama-1.1B-Chat checkpoint not in HF cache",
)
def test_tinyllama_hf_layer_megakernel_matches_reference() -> None:
    from examples.event_tensor.tinyllama_hf_layer_megakernel import (
        DEFAULT_SEQ_LEN,
        compile_for_tinyllama_hf,
        load_tinyllama_hf_layer0,
        run_tinyllama_hf_layer,
        slice_hf_weights,
    )

    full = load_tinyllama_hf_layer0()
    sliced, sliced_cfg = slice_hf_weights(full)
    compiled = compile_for_tinyllama_hf(seq_len=DEFAULT_SEQ_LEN)

    torch.manual_seed(2026)
    x = (
        torch.randn(
            (DEFAULT_SEQ_LEN, sliced_cfg["hidden_dim"]),
            dtype=torch.float32,
            device="cuda",
        )
        * 0.1
    )
    got, ref = run_tinyllama_hf_layer(compiled, x, sliced, sliced_cfg)
    err = (got - ref).abs().max().item()
    assert err < 1e-2, f"TinyLlama HF-faithful decoder-layer megakernel diverges by {err}."


@pytest.mark.skipif(
    not _TINYLLAMA_CACHE.exists(),
    reason="TinyLlama-1.1B-Chat checkpoint not in HF cache",
)
def test_tinyllama_hf_layer_uses_real_rope_theta() -> None:
    """The slice must capture rope_theta from the actual TinyLlama config."""
    from examples.event_tensor.tinyllama_hf_layer_megakernel import (
        load_tinyllama_hf_layer0,
        slice_hf_weights,
    )

    full = load_tinyllama_hf_layer0()
    _, sliced_cfg = slice_hf_weights(full)
    # TinyLlama uses the standard 10000.0 base; this verifies we read the
    # real config and didn't accidentally hard-code the default elsewhere.
    assert sliced_cfg["rope_base"] == full.cfg.get("rope_theta", 10000.0)
    assert sliced_cfg["rms_eps"] == full.cfg.get("rms_norm_eps", 1e-5)


def test_hf_rope_tables_match_hf_formula() -> None:
    """Independent reproduction of HF's RoPE formula to validate our table builder."""
    from examples.event_tensor.llama_layer_rope_megakernel import hf_rope_tables

    S, D_HEAD = 8, 16
    cos, sin = hf_rope_tables(S, D_HEAD, base=10000.0)
    # HF formula: inv_freq = 1 / base^(2i/D); freqs = pos * inv_freq;
    #             emb = cat([freqs, freqs], -1); cos/sin of emb.
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D_HEAD, 2, device=cos.device, dtype=torch.float32) / D_HEAD))
    pos = torch.arange(S, device=cos.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    expected_cos = emb.cos()
    expected_sin = emb.sin()
    assert torch.allclose(cos, expected_cos, atol=1e-6)
    assert torch.allclose(sin, expected_sin, atol=1e-6)
