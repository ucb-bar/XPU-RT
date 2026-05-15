"""Transformer-block megakernel regression tests (real TinyLlama weights + AOT warmup).

Every test here executes the **actually-emitted** persistent megakernel
on a real GPU.  No stubs, no toys, no hand-written parallel kernels.
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
# C.1 -- multi-stage transformer-block megakernel (synthetic weights)
# ---------------------------------------------------------------------------


def test_transformer_block_megakernel_matches_pytorch_reference() -> None:
    from examples.event_tensor.transformer_block_megakernel import (
        compile_transformer_block_megakernel,
        reference_block,
        run_transformer_block_megakernel,
    )

    H, S, D_HEAD, I = 4, 32, 32, 128
    D_HIDDEN = H * D_HEAD
    compiled = compile_transformer_block_megakernel(
        n_heads=H,
        seq_len=S,
        head_dim=D_HEAD,
        intermediate_dim=I,
    )
    torch.manual_seed(31)
    q = torch.randn((H, S, D_HEAD), dtype=torch.float32, device="cuda")
    k = torch.randn((H, S, D_HEAD), dtype=torch.float32, device="cuda")
    v = torch.randn((H, S, D_HEAD), dtype=torch.float32, device="cuda")
    x = torch.randn((S, D_HIDDEN), dtype=torch.float32, device="cuda")
    wg = torch.randn((I, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    wu = torch.randn((I, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    wd = torch.randn((D_HIDDEN, I), dtype=torch.float32, device="cuda") * 0.05

    got = run_transformer_block_megakernel(compiled, q, k, v, x, wg, wu, wd)
    ref = reference_block(q, k, v, x, wg, wu, wd)
    err = (got - ref).abs().max().item()
    assert err < 5e-3, f"transformer block diverges by {err}"


def test_transformer_block_emits_five_device_functions_and_four_event_tensors() -> None:
    from examples.event_tensor.transformer_block_megakernel import (
        compile_transformer_block_megakernel,
    )

    compiled = compile_transformer_block_megakernel(
        n_heads=2,
        seq_len=16,
        head_dim=32,
        intermediate_dim=64,
    )
    src = compiled.kernel_source
    # All five paper-faithful device-function bodies are present in the
    # emitted source.
    for fn in (
        "_run_compute_scores",
        "_run_apply_values",
        "_run_mlp_gate_proj",
        "_run_mlp_up_proj",
        "_run_mlp_down_proj",
    ):
        assert fn in src, f"{fn} missing from emitted megakernel"
    # All four event tensors thread through the persistent kernel signature.
    for ev in ("ESCORES_ptr", "EATTN_ptr", "EGATE_ptr", "EUP_ptr"):
        assert ev in src, f"{ev} not threaded through the megakernel"


# ---------------------------------------------------------------------------
# C.2 -- real TinyLlama-1.1B layer-0 weights through the same megakernel
# ---------------------------------------------------------------------------


_TINYLLAMA_CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"))


@pytest.mark.skipif(
    not _TINYLLAMA_CACHE.exists(),
    reason="TinyLlama-1.1B-Chat checkpoint not in HF cache",
)
def test_tinyllama_layer0_weights_match_pytorch_reference() -> None:
    from examples.event_tensor.tinyllama_layer_megakernel import (
        DEFAULT_SEQ_LEN,
        compile_for_tinyllama,
        load_tinyllama_layer0,
        run_tinyllama_block,
        slice_weights_for_megakernel,
    )

    full = load_tinyllama_layer0()
    sliced, sliced_cfg = slice_weights_for_megakernel(full)
    compiled = compile_for_tinyllama(seq_len=DEFAULT_SEQ_LEN)

    torch.manual_seed(42)
    x = (
        torch.randn(
            (DEFAULT_SEQ_LEN, sliced_cfg["hidden_dim"]),
            dtype=torch.float32,
            device="cuda",
        )
        * 0.1
    )

    got, ref = run_tinyllama_block(compiled, x, sliced, sliced_cfg)
    err = (got - ref).abs().max().item()
    assert err < 1e-2, (
        f"TinyLlama-block megakernel diverges by {err} from PyTorch eager on real Llama weights -- expected < 1e-2."
    )


@pytest.mark.skipif(
    not _TINYLLAMA_CACHE.exists(),
    reason="TinyLlama-1.1B-Chat checkpoint not in HF cache",
)
def test_tinyllama_real_weights_have_realistic_dynamic_range() -> None:
    """Sanity: the loaded slice contains real Llama-trained values, not
    a cleared / zeroed buffer."""
    from examples.event_tensor.tinyllama_layer_megakernel import (
        load_tinyllama_layer0,
        slice_weights_for_megakernel,
    )

    full = load_tinyllama_layer0()
    sliced, _ = slice_weights_for_megakernel(full)
    for name, w in (
        ("w_q", sliced.w_q),
        ("w_k", sliced.w_k),
        ("w_v", sliced.w_v),
        ("w_gate", sliced.w_gate),
        ("w_up", sliced.w_up),
        ("w_down", sliced.w_down),
    ):
        std = float(w.std().item())
        nonzero_frac = float((w != 0).float().mean().item())
        assert std > 1e-4, f"{name} std={std} -- looks zero-initialized"
        assert nonzero_frac > 0.9, f"{name} {nonzero_frac:.2%} non-zero -- looks sparse"


# ---------------------------------------------------------------------------
# C.3 -- AOT warmup benchmark sanity (just ensure it runs, do not assert
# magnitudes since they vary by GPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _TINYLLAMA_CACHE.exists(),
    reason="warmup benchmark needs TinyLlama-1.1B-Chat in HF cache",
)
@pytest.mark.slow
def test_aot_warmup_benchmark_runs_to_completion() -> None:
    """Runs the AOT warmup benchmark.  We only assert that both paths
    produce non-zero timings -- the AOT/JIT magnitude depends on GPU,
    Triton cache state, and is the subject of separate measurement."""
    from benchmarks.megakernel_warmup import (
        DEFAULT_SEQ_LEN,
        load_tinyllama_layer0,
        measure_megakernel_aot,
        measure_torch_compile_jit,
        slice_weights_for_megakernel,
    )

    full = load_tinyllama_layer0()
    sliced, sliced_cfg = slice_weights_for_megakernel(full)
    torch.manual_seed(123)
    x = (
        torch.randn(
            (DEFAULT_SEQ_LEN, sliced_cfg["hidden_dim"]),
            dtype=torch.float32,
            device="cuda",
        )
        * 0.1
    )

    aot = measure_megakernel_aot(sliced, sliced_cfg, x)
    assert aot.cold_seconds > 0
    assert aot.warm_seconds > 0
    # Warm path must be substantially faster than cold (Triton cache hit).
    assert aot.warm_seconds < aot.cold_seconds, (
        f"AOT warm ({aot.warm_seconds:.3f} s) should be < cold ({aot.cold_seconds:.3f} s)"
    )

    jit = measure_torch_compile_jit(sliced, sliced_cfg, x)
    assert jit.cold_seconds > 0
    assert jit.warm_seconds > 0
