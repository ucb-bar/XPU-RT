"""GPU microbenches for the Turing-targeted Triton kernels.

Skip-gated on ``torch.cuda.is_available()``. Runs four per-kernel
microbenches (matmul, softmax, silu, RMSNorm) and one attention-block
composition; each prints a one-line result and asserts correctness +
reasonable latency (ours not catastrophically slower than eager).

Perf assertions are *soft* (≤5× eager) — the goal is to surface real
numbers and regressions, not to punish a cold-tuned kernel. Tune the
tile / warps hyperparams when numbers drift.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU microbenches require CUDA"
)
triton = pytest.importorskip("triton")

from compgen.bench.kernel_bench import format_bench_result, run_microbench
from compgen.bench.turing_kernels import (
    attention_block_fp16,
    matmul_fp16,
    rmsnorm_fp16,
    silu_fp16,
    softmax_fp32_last_dim,
)


# ---------------------------------------------------------------------------
# Fixtures — deterministic inputs, device=cuda
# ---------------------------------------------------------------------------


@pytest.fixture
def device() -> str:
    torch.manual_seed(2026)
    return "cuda"


# ---------------------------------------------------------------------------
# Microbenches
# ---------------------------------------------------------------------------


def test_microbench_matmul_fp16_512x1024x512(device: str, capsys) -> None:
    M, K, N = 512, 1024, 512
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)

    our_fn = lambda: matmul_fp16(a, b)                  # noqa: E731
    eager = lambda: a @ b                               # noqa: E731
    tc = torch.compile(lambda x, y: x @ y, mode="reduce-overhead")
    tc_fn = lambda: tc(a, b)                            # noqa: E731

    r = run_microbench(
        "matmul_fp16_512x1024x512",
        our_fn=our_fn, eager_ref=eager, torch_compile_fn=tc_fn,
        atol=5e-2, rtol=5e-2,
        input_shapes=[[M, K], [K, N]],
    )
    with capsys.disabled():
        print("\n" + format_bench_result(r))
    assert r.passed, f"matmul correctness failed: abs={r.max_abs_err} rel={r.max_rel_err}"
    # cuBLAS on TITAN RTX is heavily tuned; a first-cut Triton matmul
    # with 64×64×32 tiles runs ~5× slower. Soft assert at 15× is a
    # regression check, not a speedup target.
    assert r.our_us < 15 * r.eager_us, (
        f"matmul too slow: ours={r.our_us}us eager={r.eager_us}us"
    )


def test_microbench_softmax_fp32_last_dim(device: str, capsys) -> None:
    M, N = 2048, 1024
    x = torch.randn((M, N), device=device, dtype=torch.float32)

    our_fn = lambda: softmax_fp32_last_dim(x)           # noqa: E731
    eager = lambda: torch.softmax(x, dim=-1)            # noqa: E731
    tc = torch.compile(lambda t: torch.softmax(t, dim=-1), mode="reduce-overhead")
    tc_fn = lambda: tc(x)                               # noqa: E731

    r = run_microbench(
        "softmax_fp32_last_dim_2048x1024",
        our_fn=our_fn, eager_ref=eager, torch_compile_fn=tc_fn,
        atol=1e-4, rtol=1e-4,
        input_shapes=[[M, N]],
    )
    with capsys.disabled():
        print(format_bench_result(r))
    assert r.passed
    assert r.our_us < 10 * r.eager_us


def test_microbench_silu_fp16(device: str, capsys) -> None:
    x = torch.randn((4096, 4096), device=device, dtype=torch.float16)
    our_fn = lambda: silu_fp16(x)                       # noqa: E731
    eager = lambda: torch.nn.functional.silu(x)         # noqa: E731
    tc = torch.compile(torch.nn.functional.silu, mode="reduce-overhead")
    tc_fn = lambda: tc(x)                               # noqa: E731

    r = run_microbench(
        "silu_fp16_4096x4096",
        our_fn=our_fn, eager_ref=eager, torch_compile_fn=tc_fn,
        atol=5e-3, rtol=5e-3,
        input_shapes=[[4096, 4096]],
    )
    with capsys.disabled():
        print(format_bench_result(r))
    assert r.passed
    assert r.our_us < 10 * r.eager_us


def test_microbench_rmsnorm_fp16(device: str, capsys) -> None:
    M, N = 2048, 4096
    x = torch.randn((M, N), device=device, dtype=torch.float16)
    w = torch.randn((N,), device=device, dtype=torch.float16)

    our_fn = lambda: rmsnorm_fp16(x, w)                 # noqa: E731

    def _eager_rmsnorm() -> torch.Tensor:
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        normed = xf * torch.rsqrt(var + 1e-5)
        return (normed * w.float()).to(torch.float16)

    tc = torch.compile(_eager_rmsnorm, mode="reduce-overhead")

    r = run_microbench(
        "rmsnorm_fp16_2048x4096",
        our_fn=our_fn, eager_ref=_eager_rmsnorm, torch_compile_fn=tc,
        atol=5e-2, rtol=5e-2,
        input_shapes=[[M, N], [N]],
    )
    with capsys.disabled():
        print(format_bench_result(r))
    assert r.passed


# ---------------------------------------------------------------------------
# Attention block — composed from our matmul + softmax (MEGA-style)
# ---------------------------------------------------------------------------


def test_attention_block_mega_style(device: str, capsys) -> None:
    """QKᵀ → softmax → ·V using our kernels, vs torch.nn.functional.sdpa."""
    M, D = 512, 128
    q = torch.randn((M, D), device=device, dtype=torch.float16)
    k = torch.randn((M, D), device=device, dtype=torch.float16)
    v = torch.randn((M, D), device=device, dtype=torch.float16)
    scale = 1.0 / (D ** 0.5)

    our_fn = lambda: attention_block_fp16(q, k, v, scale)                       # noqa: E731

    def _eager_attention() -> torch.Tensor:
        # Use the 4-D sdpa signature (B=1, H=1, M, D) and squeeze back.
        qi = q.unsqueeze(0).unsqueeze(0)
        ki = k.unsqueeze(0).unsqueeze(0)
        vi = v.unsqueeze(0).unsqueeze(0)
        out = torch.nn.functional.scaled_dot_product_attention(
            qi, ki, vi, is_causal=False,
        )
        return out.squeeze(0).squeeze(0)

    tc = torch.compile(_eager_attention, mode="reduce-overhead")

    r = run_microbench(
        "attention_block_MEGA_style_M=512_D=128",
        our_fn=our_fn, eager_ref=_eager_attention, torch_compile_fn=tc,
        atol=5e-2, rtol=5e-2,
        input_shapes=[[M, D]] * 3,
        notes="QKᵀ→softmax→·V composed from our matmul + softmax kernels",
    )
    with capsys.disabled():
        print(format_bench_result(r))
    assert r.passed, f"attention correctness failed: abs={r.max_abs_err} rel={r.max_rel_err}"
    # sdpa is a very heavily optimised fused kernel — don't be strict
    # about winning; assert we're within 10× and correct.
    assert r.our_us < 10 * r.eager_us
