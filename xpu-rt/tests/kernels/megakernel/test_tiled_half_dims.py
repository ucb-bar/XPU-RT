"""Tiled megakernel regression tests at half-TinyLlama dims.

Validates the tiled-matmul layer megakernel at dims that overflow the
Phase F shared-memory budget, including running on REAL TinyLlama-1.1B
weights at HALF-TinyLlama dims with the checkpoint's actual head_dim
and rope_theta.
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

sys.modules.setdefault("torchvision", None)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="real-example tests require CUDA")


_TINYLLAMA_CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"))


# ---------------------------------------------------------------------------
# I.1 -- tiled megakernel at H=8 (the dim that overflowed Phase F)
# ---------------------------------------------------------------------------


def test_tiled_megakernel_runs_at_h8_dim_phase_f_couldnt() -> None:
    from examples.event_tensor.llama_layer_gqa_megakernel import (
        reference_llama_layer_gqa,
    )
    from examples.event_tensor.llama_layer_rope_megakernel import hf_rope_tables
    from examples.event_tensor.llama_layer_tiled_megakernel import (
        compile_llama_layer_tiled,
        run_llama_layer_tiled,
    )

    H, N_KV, S, D_HEAD, I = 8, 2, 16, 16, 256
    D_HIDDEN = H * D_HEAD  # = 128
    compiled = compile_llama_layer_tiled(
        n_heads=H,
        n_kv_heads=N_KV,
        seq_len=S,
        head_dim=D_HEAD,
        intermediate_dim=I,
    )
    cos, sin = hf_rope_tables(S, D_HEAD)
    torch.manual_seed(2028)
    x = torch.randn((S, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.1
    w_norm1 = torch.randn((D_HIDDEN,), dtype=torch.float32, device="cuda") * 0.1 + 1.0
    w_q = torch.randn((D_HIDDEN, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_k = torch.randn((N_KV * D_HEAD, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_v = torch.randn((N_KV * D_HEAD, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_o = torch.randn((D_HIDDEN, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_norm2 = torch.randn((D_HIDDEN,), dtype=torch.float32, device="cuda") * 0.1 + 1.0
    w_gate = torch.randn((I, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_up = torch.randn((I, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_down = torch.randn((D_HIDDEN, I), dtype=torch.float32, device="cuda") * 0.05

    got = run_llama_layer_tiled(
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
    ref = reference_llama_layer_gqa(
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
        n_kv_heads=N_KV,
        head_dim=D_HEAD,
    )
    err = (got - ref).abs().max().item()
    assert err < 5e-3, f"tiled layer at H=8 diverges by {err}"


def test_tiled_megakernel_emits_inner_k_loop() -> None:
    from examples.event_tensor.llama_layer_tiled_megakernel import compile_llama_layer_tiled

    compiled = compile_llama_layer_tiled(
        n_heads=4,
        n_kv_heads=2,
        seq_len=16,
        head_dim=16,
        intermediate_dim=64,
    )
    src = compiled.kernel_source
    # The tiled emitter must thread BLOCK_K + K_TILES through the kernel signature
    # and use them in the static_range inner loop within the heavy bodies.
    assert "BLOCK_K" in src
    assert "K_TILES" in src
    assert "for k_block in tl.static_range(0, K_TILES)" in src
    # o_proj is now per (m_tile, n_tile)
    assert "n_tile = task_id %  N_TILES" in src or "n_tile = task_id % N_TILES" in src


# ---------------------------------------------------------------------------
# I.2 -- real TinyLlama at HALF-TinyLlama dims (Phase F couldn't reach this)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _TINYLLAMA_CACHE.exists(),
    reason="TinyLlama-1.1B-Chat checkpoint not in HF cache",
)
def test_tiled_megakernel_on_real_tinyllama_half_dims() -> None:
    from examples.event_tensor.llama_layer_gqa_megakernel import (
        reference_llama_layer_gqa,
    )
    from examples.event_tensor.llama_layer_rope_megakernel import hf_rope_tables
    from examples.event_tensor.llama_layer_tiled_megakernel import (
        compile_llama_layer_tiled,
        run_llama_layer_tiled,
    )
    from examples.event_tensor.tinyllama_hf_layer_megakernel import (
        load_tinyllama_hf_layer0,
    )

    full = load_tinyllama_hf_layer0()
    # HALF-TinyLlama: H=16 (TinyLlama: 32), N_KV=4 (TinyLlama: 4), D_HEAD=64
    # (TinyLlama actual), hidden=1024, intermediate=2048.
    H, N_KV, D_HEAD = 16, 4, 64
    INTER = 2048
    D_HIDDEN = H * D_HEAD  # = 1024
    S = 16

    w_norm1 = full.w_norm1[:D_HIDDEN].contiguous()
    w_q = full.w_q[:D_HIDDEN, :D_HIDDEN].contiguous()
    w_k = full.w_k[: N_KV * D_HEAD, :D_HIDDEN].contiguous()
    w_v = full.w_v[: N_KV * D_HEAD, :D_HIDDEN].contiguous()
    w_o = full.w_o[:D_HIDDEN, :D_HIDDEN].contiguous()
    w_norm2 = full.w_norm2[:D_HIDDEN].contiguous()
    w_gate = full.w_gate[:INTER, :D_HIDDEN].contiguous()
    w_up = full.w_up[:INTER, :D_HIDDEN].contiguous()
    w_down = full.w_down[:D_HIDDEN, :INTER].contiguous()

    compiled = compile_llama_layer_tiled(
        n_heads=H,
        n_kv_heads=N_KV,
        seq_len=S,
        head_dim=D_HEAD,
        intermediate_dim=INTER,
    )
    base = float(full.cfg.get("rope_theta", 10000.0))
    rms_eps = float(full.cfg.get("rms_norm_eps", 1e-5))
    cos, sin = hf_rope_tables(S, D_HEAD, base=base)

    torch.manual_seed(2031)
    x = torch.randn((S, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.1

    got = run_llama_layer_tiled(
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
        rms_eps=rms_eps,
    )
    ref = reference_llama_layer_gqa(
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
        n_kv_heads=N_KV,
        head_dim=D_HEAD,
        rms_eps=rms_eps,
    )
    err = (got - ref).abs().max().item()
    assert err < 5e-2, f"tiled megakernel on real TinyLlama half-dims diverges by {err}"
