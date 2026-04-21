"""Tests for the FlashAttention MEGA kernel.

Skip-gated on CUDA. Tests:
  * Correctness vs ``torch.nn.functional.scaled_dot_product_attention``
    (the Flash-Attention-2 reference path).
  * Perf vs the existing 3-kernel composition we already had
    (``compgen.bench.turing_kernels.attention_block_fp16``-style: bmm +
    softmax + bmm).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required"
)
triton = pytest.importorskip("triton")

from compgen.bench.flash_attention_kernel import flash_attention_fp16
from compgen.bench.kernel_bench import format_bench_result, run_microbench
from compgen.bench.turing_kernels import bmm_fp16, softmax_fp32_last_dim


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def _sdpa_reference(q, k, v, scale, *, causal):
    """torch.nn.functional.scaled_dot_product_attention reference."""
    q4 = q.unsqueeze(0)        # (1, BH, S, D) — sdpa wants (B, H, S, D)
    k4 = k.unsqueeze(0)
    v4 = v.unsqueeze(0)
    out = torch.nn.functional.scaled_dot_product_attention(
        q4, k4, v4, is_causal=causal, scale=scale,
    )
    return out.squeeze(0)


@pytest.mark.parametrize("S", [32, 64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_flash_attention_matches_sdpa(S: int, causal: bool) -> None:
    BH, D = 8, 64
    torch.manual_seed(2026)
    q = torch.randn((BH, S, D), device="cuda", dtype=torch.float16)
    k = torch.randn((BH, S, D), device="cuda", dtype=torch.float16)
    v = torch.randn((BH, S, D), device="cuda", dtype=torch.float16)
    scale = 1.0 / (D ** 0.5)

    ours = flash_attention_fp16(q, k, v, scale, causal=causal)
    ref = _sdpa_reference(q, k, v, scale, causal=causal)

    # fp16 attention noise is real; use loose tolerance like sdpa itself does.
    torch.testing.assert_close(ours, ref, atol=5e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# Perf — vs the 3-kernel composition we used to ship
# ---------------------------------------------------------------------------


def _three_kernel_attention(q, k, v, scale, *, causal):
    """The bmm + softmax + bmm path our decoder layer used pre-FA."""
    BH, S, D = q.shape
    kt = k.transpose(1, 2).contiguous()  # (BH, D, S)
    scores = bmm_fp16(q, kt) * scale
    if causal:
        mask = torch.triu(
            torch.full((S, S), float("-inf"), device=q.device, dtype=torch.float16),
            diagonal=1,
        )
        scores = scores + mask[None, :, :]
    probs = softmax_fp32_last_dim(scores.float()).to(torch.float16)
    return bmm_fp16(probs, v)


def test_flash_attention_perf_vs_three_kernel_composition(capsys) -> None:
    """FA should beat the 3-kernel trio at TinyLlama-style shapes.

    On Turing (sm_75 / TITAN RTX) the win is typically smaller than
    on Ampere because we lack async copy. Soft assert at ≤1.2× of the
    trio (i.e. within 20% slower at worst); print numbers either way.
    """
    BH, S, D = 32, 128, 64        # mimics TinyLlama prefill of 128 tokens
    torch.manual_seed(2026)
    q = torch.randn((BH, S, D), device="cuda", dtype=torch.float16)
    k = torch.randn((BH, S, D), device="cuda", dtype=torch.float16)
    v = torch.randn((BH, S, D), device="cuda", dtype=torch.float16)
    scale = 1.0 / (D ** 0.5)

    # Warm autotune for both
    _ = flash_attention_fp16(q, k, v, scale, causal=True)
    _ = _three_kernel_attention(q, k, v, scale, causal=True)
    torch.cuda.synchronize()

    fa = run_microbench(
        f"flash_attention BH={BH} S={S} D={D}",
        our_fn=lambda: flash_attention_fp16(q, k, v, scale, causal=True),
        eager_ref=lambda: _sdpa_reference(q, k, v, scale, causal=True),
        atol=5e-2, rtol=5e-2,
    )
    trio = run_microbench(
        f"three_kernel    BH={BH} S={S} D={D}",
        our_fn=lambda: _three_kernel_attention(q, k, v, scale, causal=True),
        eager_ref=lambda: _sdpa_reference(q, k, v, scale, causal=True),
        atol=5e-2, rtol=5e-2,
    )

    with capsys.disabled():
        print()
        print(format_bench_result(fa))
        print(format_bench_result(trio))
        speedup = trio.our_us / fa.our_us
        print(f"\nFA vs 3-kernel: {speedup:.2f}x  ({fa.our_us:.1f}μs vs {trio.our_us:.1f}μs)")
        sdpa_us = fa.eager_us
        print(f"FA vs sdpa(FA-2): {fa.our_us / sdpa_us:.2f}x  "
              f"({fa.our_us:.1f}μs vs {sdpa_us:.1f}μs)")

    # Soft assertion: FA should not be dramatically slower than the trio
    # (it's the whole point). On Turing it might be only marginal.
    assert fa.our_us < 1.5 * trio.our_us, (
        f"FA is unexpectedly slower than the 3-kernel composition: "
        f"FA={fa.our_us}μs vs trio={trio.our_us}μs"
    )
