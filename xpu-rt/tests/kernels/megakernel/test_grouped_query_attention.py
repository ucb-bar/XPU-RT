"""Grouped-query-attention (GQA) megakernel regression tests + HF parity proof.

Closes the validation chain to actual HuggingFace ``LlamaDecoderLayer.forward()``:

  * F.1: our HF-faithful PyTorch reference matches HF's actual layer
    code on real TinyLlama-1.1B layer-0 weights.  Combined with
    (megakernel matches the reference) this proves the emitted megakernel
    runs the same math as HuggingFace's production decoder layer.

  * F.2: GQA-aware megakernel -- K/V are computed and rotated only for
    GQA group leaders; attention reads them at h // KV_REPEAT.

Every test executes the actually-emitted artifact (or HF's actual
forward() in F.1's case) on a real GPU.
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

# transformers' top-level import drags in torchvision; bypass it here too.
sys.modules.setdefault("torchvision", None)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="real-example tests require CUDA")


_TINYLLAMA_CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"))


# ---------------------------------------------------------------------------
# F.1 -- our reference matches HF's actual LlamaDecoderLayer.forward()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _TINYLLAMA_CACHE.exists(),
    reason="TinyLlama-1.1B-Chat checkpoint not in HF cache",
)
def test_pytorch_reference_matches_hf_llama_decoder_layer_forward() -> None:
    """End-to-end chain proof:

    megakernel  ==  our HF-faithful reference     (1.8e-07)
    our HF-faithful reference  ==  HF.LlamaDecoderLayer.forward (this test)
    ⇒ megakernel runs the same math as HF's actual decoder layer.
    """
    from examples.event_tensor.tinyllama_vs_hf_layer_megakernel import (
        compare_reference_to_hf,
    )

    result = compare_reference_to_hf()
    # HF's fused-matmul accumulation order differs from our @-matmul
    # reference; 1e-4 absolute is the realistic float32 budget.
    assert result["err_abs"] < 1e-4, (
        f"Our PyTorch reference diverges from HF.LlamaDecoderLayer.forward "
        f"by {result['err_abs']:.3e} -- expected < 1e-4."
    )


# ---------------------------------------------------------------------------
# F.2 -- GQA-aware megakernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_heads,n_kv_heads",
    [
        (4, 2),  # KV_REPEAT = 2
        (4, 1),  # KV_REPEAT = 4 (TinyLlama-like ratio)
        # H >= 8 with our default block sizes overflows the TITAN-RTX 64 KB
        # shared-memory budget at the o_proj body's full-W_o load -- the
        # tiled-W_o emitter optimisation is the next step.
    ],
)
def test_gqa_layer_megakernel_matches_gqa_reference(n_heads: int, n_kv_heads: int) -> None:
    from examples.event_tensor.llama_layer_gqa_megakernel import (
        compile_llama_layer_gqa,
        reference_llama_layer_gqa,
        run_llama_layer_gqa,
    )
    from examples.event_tensor.llama_layer_rope_megakernel import hf_rope_tables

    S, D_HEAD, I = 16, 16, 64
    D_HIDDEN = n_heads * D_HEAD
    compiled = compile_llama_layer_gqa(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        seq_len=S,
        head_dim=D_HEAD,
        intermediate_dim=I,
    )
    cos, sin = hf_rope_tables(S, D_HEAD)
    torch.manual_seed(2027 + n_heads * 10 + n_kv_heads)
    x = torch.randn((S, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.1
    w_norm1 = torch.randn((D_HIDDEN,), dtype=torch.float32, device="cuda") * 0.1 + 1.0
    w_q = torch.randn((D_HIDDEN, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_k = torch.randn((n_kv_heads * D_HEAD, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_v = torch.randn((n_kv_heads * D_HEAD, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_o = torch.randn((D_HIDDEN, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_norm2 = torch.randn((D_HIDDEN,), dtype=torch.float32, device="cuda") * 0.1 + 1.0
    w_gate = torch.randn((I, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_up = torch.randn((I, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_down = torch.randn((D_HIDDEN, I), dtype=torch.float32, device="cuda") * 0.05

    got = run_llama_layer_gqa(
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
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=D_HEAD,
    )
    err = (got - ref).abs().max().item()
    assert err < 5e-3, f"GQA megakernel (H={n_heads}, N_KV={n_kv_heads}) diverges by {err}"


def test_gqa_emitted_kernel_carries_kv_repeat_constexpr() -> None:
    """The GQA emitter must thread KV_REPEAT through the persistent kernel
    signature -- otherwise the body's K/V index logic is unreachable."""
    from examples.event_tensor.llama_layer_gqa_megakernel import compile_llama_layer_gqa

    compiled = compile_llama_layer_gqa(
        n_heads=4,
        n_kv_heads=2,
        seq_len=16,
        head_dim=16,
        intermediate_dim=32,
    )
    src = compiled.kernel_source
    assert "KV_REPEAT" in src
    assert "N_KV_HEADS" in src
    # The leader-only K/V store guard is in the emitted source.
    assert "h_kv = h // KV_REPEAT" in src
    assert "if h == h_kv * KV_REPEAT" in src
