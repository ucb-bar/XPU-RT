"""Llama decoder-layer megakernel regression tests (full decoder layer in one megakernel).

Every test executes the **actually-emitted** persistent megakernel on a
real GPU.  Phase D adds the operators that turn the Phase C
transformer-block megakernel into a full Llama decoder layer
(RMSNorm + QKV proj + O proj + RMSNorm), validated on real
TinyLlama-1.1B layer-0 weights.
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
# D.1 -- full Llama decoder-layer megakernel (synthetic weights)
# ---------------------------------------------------------------------------


def test_llama_decoder_layer_matches_pytorch_reference() -> None:
    from examples.event_tensor.llama_decoder_layer_megakernel import (
        compile_llama_decoder_layer,
        reference_decoder_layer,
        run_llama_decoder_layer,
    )

    H, S, D_HEAD, I = 4, 16, 16, 64
    D_HIDDEN = H * D_HEAD
    compiled = compile_llama_decoder_layer(
        n_heads=H,
        seq_len=S,
        head_dim=D_HEAD,
        intermediate_dim=I,
    )
    torch.manual_seed(7)
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

    got = run_llama_decoder_layer(
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
    )
    ref = reference_decoder_layer(
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
        n_heads=H,
        head_dim=D_HEAD,
    )
    err = (got - ref).abs().max().item()
    assert err < 5e-3, f"decoder layer diverges by {err}"


def test_llama_decoder_layer_emits_nine_device_functions() -> None:
    """Sanity: the emitted kernel must contain bodies for every paper-faithful
    decoder-layer device function."""
    from examples.event_tensor.llama_decoder_layer_megakernel import (
        compile_llama_decoder_layer,
    )

    compiled = compile_llama_decoder_layer(
        n_heads=2,
        seq_len=16,
        head_dim=16,
        intermediate_dim=32,
    )
    src = compiled.kernel_source
    for fn in (
        "_run_input_norm",
        "_run_qkv_proj",
        "_run_compute_scores",
        "_run_apply_values",
        "_run_o_proj_residual",
        "_run_post_attn_norm",
        "_run_mlp_gate_proj",
        "_run_mlp_up_proj",
        "_run_mlp_down_proj",
    ):
        assert fn in src, f"{fn} missing from emitted decoder-layer megakernel"

    # Eight event tensors threaded through the persistent kernel signature.
    for ev in (
        "ENORM1_ptr",
        "EQKV_ptr",
        "ESCORES_ptr",
        "EATTN_ptr",
        "EOPROJ_ptr",
        "ENORM2_ptr",
        "EGATE_ptr",
        "EUP_ptr",
    ):
        assert ev in src, f"{ev} not threaded through the megakernel"


# ---------------------------------------------------------------------------
# D.2 -- real TinyLlama-1.1B layer-0 weights through the full megakernel
# ---------------------------------------------------------------------------


_TINYLLAMA_CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"))


@pytest.mark.skipif(
    not _TINYLLAMA_CACHE.exists(),
    reason="TinyLlama-1.1B-Chat checkpoint not in HF cache",
)
def test_tinyllama_full_decoder_layer_matches_pytorch_reference() -> None:
    from examples.event_tensor.tinyllama_full_layer_megakernel import (
        DEFAULT_SEQ_LEN,
        compile_for_tinyllama_full,
        load_tinyllama_full_layer0,
        run_tinyllama_full_layer,
        slice_full_weights_for_megakernel,
    )

    full = load_tinyllama_full_layer0()
    sliced, sliced_cfg = slice_full_weights_for_megakernel(full)
    compiled = compile_for_tinyllama_full(seq_len=DEFAULT_SEQ_LEN)

    torch.manual_seed(2026)
    x = (
        torch.randn(
            (DEFAULT_SEQ_LEN, sliced_cfg["hidden_dim"]),
            dtype=torch.float32,
            device="cuda",
        )
        * 0.1
    )

    got, ref = run_tinyllama_full_layer(compiled, x, sliced, sliced_cfg)
    err = (got - ref).abs().max().item()
    assert err < 1e-2, (
        f"TinyLlama full decoder-layer megakernel diverges by {err} from "
        "PyTorch eager on real Llama weights -- expected < 1e-2."
    )


@pytest.mark.skipif(
    not _TINYLLAMA_CACHE.exists(),
    reason="TinyLlama-1.1B-Chat checkpoint not in HF cache",
)
def test_tinyllama_layernorm_scales_are_loaded_from_real_checkpoint() -> None:
    """The norm scales used by the megakernel must be the trained
    TinyLlama values, not zero-initialised buffers.  We don't constrain
    the magnitude (TinyLlama's input_layernorm is unusually small at
    layer 0 -- around 4e-3 -- which is exactly the kind of trained
    value a sanity test should not be fooled by)."""
    from examples.event_tensor.tinyllama_full_layer_megakernel import (
        load_tinyllama_full_layer0,
        slice_full_weights_for_megakernel,
    )

    full = load_tinyllama_full_layer0()
    sliced, _ = slice_full_weights_for_megakernel(full)
    for name, w in (
        ("w_norm1", sliced.w_norm1),
        ("w_norm2", sliced.w_norm2),
    ):
        std = float(w.std().item())
        nonzero_frac = float((w != 0).float().mean().item())
        assert std > 1e-5, f"{name} std={std} -- looks zero-initialised"
        assert nonzero_frac > 0.95, f"{name} {nonzero_frac:.2%} non-zero -- looks like a sparse buffer"
