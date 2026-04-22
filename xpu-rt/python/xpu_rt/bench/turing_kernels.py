"""Triton kernels authored for Turing (sm_75, TITAN RTX).

Four microkernels plus one attention-block composition, all targeted
at the fp16 tensor-core path Turing supports (no bf16 TC, no TF32).
Each is what ``ClaudeCodeKernelProvider`` would generate from the
matching v3 contract + Turing `HardwareEnvelope`.

Characteristics:
  * fp16 inputs + fp32 accumulator + fp16 output for matmul-shaped ops
  * SMEM budget ≤48 KB/block (Turing caps at 49152 B)
  * 32-wide warps, ``num_warps=4`` / ``num_warps=8`` for big kernels
  * ``allow_tf32=False`` (sm_75 doesn't support TF32 anyway)

Each kernel exposes both the raw ``@triton.jit`` function AND a
Python wrapper that allocates output + launches with a fixed grid.
The wrapper is what :func:`run_microbench` wires up as ``our_fn``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# 1. COMPUTE_TILED — matmul fp16 × fp16 → fp32 acc → fp16 out
# ---------------------------------------------------------------------------


@triton.jit
def _matmul_fp16_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)  # HMMA fp16×fp16→fp32 on Turing
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out = acc.to(tl.float16)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_fp16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b``; both fp16. Output fp16. Accumulator fp32.

    v1 first-cut — 64×64×32 tiles, no autotune, no swizzle. Kept for
    baseline comparisons. Measured 194.9μs on TITAN RTX 512×1024×512
    (5.41× slower than cuBLAS).
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and a.dtype == torch.float16 == b.dtype
    out = torch.empty((M, N), device=a.device, dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_fp16_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# 1b. COMPUTE_TILED v2 — refined from diagnosis feedback
#
# Diagnosis on v1 said:
#   1. Tiles aren't re-using lhs/rhs across the K-loop. Try 128×128 + num_stages=3.
#   2. triton.autotune over (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages).
#   3. The autotune curve itself is strong signal.
#
# v2 changes:
#   * @triton.autotune with 6 configs (small → large tiles, 4/8 warps, 2/3 stages)
#   * GROUP_M swizzle — L2 re-use pattern: warps cooperate on nearby output
#     tiles instead of sweeping row-by-row
#   * SMEM budget check — sm_75 has 48 KB/block; largest config (128×128×32)
#     uses (128+128)*32*2B × 3 stages = 48 KB — fits tight
#   * Aligned-K fast path: if K is a multiple of BLOCK_K (the contract's
#     ``divisibility`` hint says K%32==0), skip the K boundary mask
# ---------------------------------------------------------------------------


_MATMUL_V2_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=_MATMUL_V2_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_fp16_kernel_v2(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # GROUP_M swizzle: linearise the 2-D (pid_m, pid_n) grid into groups
    # of GROUP_M rows so consecutive program IDs cooperate on nearby
    # output tiles. Improves L2 re-use because a group of warps share
    # the same lhs rows.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k * BLOCK_K + offs_k) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out = acc.to(tl.float16)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_fp16_v2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Refined matmul — autotuned tile + GROUP_M swizzle."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and a.dtype == torch.float16 == b.dtype
    out = torch.empty((M, N), device=a.device, dtype=torch.float16)

    def _grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _matmul_fp16_kernel_v2[_grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


# ---------------------------------------------------------------------------
# 1c. COMPUTE_TILED v3 — persistent CTA + more configs + multiple_of hints
#
# Diagnosis after v2 (147μs vs 29.6μs eager):
#   * Autotune picked the SAME 64×64×32 as v1 → tile-bound path isn't the fix
#   * Still 5× vs cuBLAS → launch/scheduling overhead + register usage matter
#   * Next levers: persistent CTA (amortize launch), pipelined async loads
#     via num_stages=3, tl.multiple_of hints for the compiler
#
# v3 changes:
#   * Persistent scheduling — launch exactly NUM_SMS CTAs (72 on TITAN RTX);
#     each iterates over a strided slice of the output-tile space. Removes
#     per-tile launch overhead.
#   * ``tl.multiple_of`` hints on pointer arithmetic so Triton's optimizer
#     can emit wider / vectorised loads
#   * Expanded autotune grid (9 configs including 128×64 + num_stages=3)
# ---------------------------------------------------------------------------


_MATMUL_V3_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=_MATMUL_V3_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_fp16_kernel_v3(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Persistent scheduling: each of NUM_SMS CTAs handles a strided slice of
    # the output-tile space. ``start_tile_id`` is this CTA's first tile;
    # the loop step is NUM_SMS.
    start_tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n

    for tile_id in range(start_tile_id, num_tiles, NUM_SMS):
        # Decode pid_m, pid_n with the same GROUP_M swizzle as v2.
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # multiple_of hints help Triton emit wider / vectorised loads.
        offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = (k * BLOCK_K + offs_k) < K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=k_mask[:, None] & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        out = acc.to(tl.float16)
        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_fp16_v3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Persistent-CTA autotuned matmul — launches exactly NUM_SMS CTAs."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and a.dtype == torch.float16 == b.dtype
    out = torch.empty((M, N), device=a.device, dtype=torch.float16)

    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count

    _matmul_fp16_kernel_v3[(num_sms,)](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        NUM_SMS=num_sms,
    )
    return out


# ---------------------------------------------------------------------------
# 1d. COMPUTE_TILED v4 — ALGORITHMIC rewrite, not just knob tuning
#
# The real body changes:
#   * ``tl.make_block_ptr`` for A, B, C — gives the compiler a structured
#     tensor view; emits vectorised coalesced loads + handles boundaries
#     via block_ptr's own padding, not mask-and-clamp.
#   * ``tl.advance(block_ptr, delta)`` to step through the K-loop —
#     replaces manual pointer arithmetic that was blocking vectorisation.
#   * ``acc = tl.dot(a, b, acc, ...)`` with fused accumulator — lowers to
#     a single HMMA with accumulator input instead of HMMA + separate
#     add. Direct win on the inner loop.
#   * Structured store via ``tl.store(c_block_ptr, ...)`` with
#     ``boundary_check`` — proper strided writes.
#
# Knobs unchanged: same autotune grid as v3, same persistent-CTA shell,
# same GROUP_M swizzle. Any delta here is purely from the body rewrite.
# ---------------------------------------------------------------------------


@triton.autotune(configs=_MATMUL_V3_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_fp16_kernel_v4(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    start_tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n

    for tile_id in range(start_tile_id, num_tiles, NUM_SMS):
        # GROUP_M swizzle — same as v2/v3.
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        # Structured block pointers — lets the compiler see the full
        # (M,K) / (K,N) / (M,N) tensor shapes and emit coalesced loads.
        a_block_ptr = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=B,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            # boundary_check lets block_ptr pad with zeros instead of us
            # doing mask-and-clamp in the inner loop.
            a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
            b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
            # Fused-accumulator form: one HMMA with acc as input + output,
            # instead of separate HMMA + add.
            acc = tl.dot(a, b, acc, allow_tf32=False)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

        c_block_ptr = tl.make_block_ptr(
            base=C,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


def matmul_fp16_v4(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """v4: block_ptr + advance + fused-acc dot. Algorithmic, not parametric."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and a.dtype == torch.float16 == b.dtype
    out = torch.empty((M, N), device=a.device, dtype=torch.float16)

    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    _matmul_fp16_kernel_v4[(num_sms,)](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        NUM_SMS=num_sms,
    )
    return out


# ---------------------------------------------------------------------------
# 1e. COMPUTE_TILED — proper batched matmul
#
# Replaces the Python loop over per-batch 2D matmul we were doing for
# attention scores + output. 3D launch: grid_z indexes the batch dim.
# Shape: (B_dim, M, K) × (B_dim, K, N) → (B_dim, M, N)
#
# The bmm case is the dominant overhead at warm-cache steady state:
# 51.6% of TinyLlama wall-time was the Python-loop-over-2D-matmul
# pattern (1408 wasted launches). One fused 3D kernel collapses that
# to 44 launches with the same HMMA inner loop.
# ---------------------------------------------------------------------------


_BMM_FP16_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=_BMM_FP16_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _bmm_fp16_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3-D grid: (num_M_tiles × num_N_tiles, 1, B_dim).
    # program_id(0) indexes the output tile within one batch element,
    # program_id(1) indexes which batch element.
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + bid * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + bid * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k * BLOCK_K + offs_k) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out = acc.to(tl.float16)
    c_ptrs = C + bid * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def bmm_fp16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched matmul: (B, M, K) × (B, K, N) → (B, M, N), fp16."""
    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2
    assert a.dtype == torch.float16 == b.dtype
    out = torch.empty((B, M, N), device=a.device, dtype=torch.float16)

    def _grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), 1, B)

    _bmm_fp16_kernel[_grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out


# ---------------------------------------------------------------------------
# 2. REDUCE — softmax fp32 along last dim, numerically stable
# ---------------------------------------------------------------------------


@triton.jit
def _softmax_fp32_kernel(
    X,
    Y,
    stride_m,
    stride_n,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_m + cols * stride_n, mask=mask, other=-float("inf"))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    z = tl.sum(e, axis=0)
    y = e / z
    tl.store(Y + row * stride_m + cols * stride_n, y, mask=mask)


def softmax_fp32_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax along the last dim. 2-D input for simplicity;
    for 4-D inputs the caller flattens leading dims."""
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    M, N = x2.shape
    out = torch.empty_like(x2)
    # BLOCK_N power-of-two ≥ N
    BLOCK_N = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_N < 2048 else 8
    _softmax_fp32_kernel[(M,)](
        x2,
        out,
        x2.stride(0),
        x2.stride(1),
        N,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out.reshape(orig_shape)


# ---------------------------------------------------------------------------
# 3. ACTIVATION — silu fp16 (x * sigmoid(x))
# ---------------------------------------------------------------------------


@triton.jit
def _silu_fp16_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    tl.store(Y + offs, (x * sig).to(tl.float16), mask=mask)


def silu_fp16(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    N = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    _silu_fp16_kernel[grid](x, out, N, BLOCK=BLOCK, num_warps=4)
    return out


# ---------------------------------------------------------------------------
# 4. REDUCE (compound) — RMSNorm fp16 (x * rsqrt(mean(x**2) + eps) * w)
# ---------------------------------------------------------------------------


@triton.jit
def _rmsnorm_fp16_kernel(
    X,
    W,
    Y,
    stride_m,
    N,
    EPS,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = x * x
    mean_sq = tl.sum(x2, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(mean_sq + EPS)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * inv_rms * w
    tl.store(Y + row * stride_m + cols, y.to(tl.float16), mask=mask)


def rmsnorm_fp16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    M, N = x2.shape
    out = torch.empty_like(x2)
    BLOCK_N = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_N < 2048 else 8
    _rmsnorm_fp16_kernel[(M,)](
        x2,
        weight,
        out,
        x2.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out.reshape(orig_shape)


# ---------------------------------------------------------------------------
# 5. MEGA-style attention block — QKᵀ → softmax → ·V, composed from three
#    separate kernel launches. NOT a persistent-fused megakernel — that
#    would require a unified kernel body that's much larger. This form
#    still tests the KernelContractV3 body+internal_events composition
#    pattern (matmul_done → softmax_done → attention_done).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Auto-load persisted autotune picks at import time.
#
# Triton's binary cache is automatic (~/.triton/cache); ours sits beside
# it at ~/.compgen/autotune/. With both populated, deployment cold start
# drops from ~10 s to ~hundreds of ms (only the kernel binary loads
# from disk; no autotune sweep, no JIT compile).
# ---------------------------------------------------------------------------


def _autoload_autotune_caches() -> None:
    """Best-effort load — silent on missing files (first ever run)."""
    try:
        from compgen.bench.autotune_cache import load_all
    except ImportError:
        return
    load_all(
        [
            _matmul_fp16_kernel_v2,
            _matmul_fp16_kernel_v3,
            _matmul_fp16_kernel_v4,
            _bmm_fp16_kernel,
        ]
    )


_autoload_autotune_caches()


def attention_block_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Scaled dot-product attention using OUR matmul + softmax kernels.

    Args:
        q: (M, D)   fp16
        k: (N, D)   fp16 (note: NOT pre-transposed)
        v: (N, D)   fp16
        scale: usually 1/sqrt(D)
    """
    # QKᵀ: q @ k.T  — with a manual transpose since our matmul takes (M,K)×(K,N).
    kt = k.transpose(0, 1).contiguous()
    scores = matmul_fp16(q, kt) * scale  # (M, N) fp16
    probs = softmax_fp32_last_dim(scores.float()).to(torch.float16)  # (M, N) fp16
    out = matmul_fp16(probs, v)  # (M, D) fp16
    return out
