"""FlashAttention MEGA kernel — fused QKᵀ → softmax → ·V in one persistent pass.

The KernelContractV3 archetype is COMPUTE_TILED, granularity MEGA: one
kernel does what our previous decoder layer did in three (bmm + softmax +
bmm) — and it does so without ever materialising the (S, S) attention
scores matrix to DRAM.

Algorithm (FlashAttention-2, online softmax):

    for each Q tile (BLOCK_M rows):
        acc, l, m = 0, 0, -inf      # running output, running denom, running max
        for each K, V tile (BLOCK_N rows):
            S = Q @ K.T * scale     # (BLOCK_M, BLOCK_N) in registers
            if causal: apply causal mask
            m_new = max(m, S.max(-1))
            P = exp(S - m_new)
            alpha = exp(m - m_new)
            l = l * alpha + P.sum(-1)
            acc = acc * alpha + P @ V
            m = m_new
        write acc / l

For Turing sm_75 (TITAN RTX, 48 KB SMEM/CTA, no async copy):
    * BLOCK_M = 64 rows of Q, BLOCK_N = 64 rows of K/V
    * head_dim D up to 128 (TinyLlama uses 64)
    * SMEM budget per CTA ≈ Q(64×D×2) + K(64×D×2) + V(64×D×2) + acc(64×D×4)
      = 4(64D) + 16(D) bytes... at D=64: 24KB; fits with double-buffer

Public surface:

    flash_attention_fp16(q, k, v, scale, causal=True) -> Tensor

    q, k, v: (B*H, S, D) fp16, contiguous
    output : (B*H, S, D) fp16
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune config grid — Turing-friendly (no async copy, smaller smem)
# ---------------------------------------------------------------------------


_FLASH_ATTN_CONFIGS = [
    # BLOCK_M, BLOCK_N, num_warps, num_stages
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=2),
]


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@triton.autotune(configs=_FLASH_ATTN_CONFIGS, key=["S", "D"])
@triton.jit
def _flash_attention_fwd_kernel(
    Q, K, V, Out,
    L,                           # logsumexp output, optional (set to dummy if not needed)
    stride_qb, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_ob, stride_os, stride_od,
    SCALE,
    S, D,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,         # padded D to next power of 2
):
    # Each program processes BLOCK_M rows of one head.
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)       # batch-head index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    # Load Q tile once — stays resident across the K/V loop.
    q_ptrs = (Q + pid_b * stride_qb
              + offs_m[:, None] * stride_qs
              + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs,
                mask=(offs_m[:, None] < S) & (offs_d[None, :] < D),
                other=0.0)

    # Online softmax accumulators.
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # K/V loop. For causal we can stop at the diagonal Q-block end.
    n_max = S
    if CAUSAL:
        # Last K-row this Q-block can attend to is offs_m.max()
        n_max = (pid_m + 1) * BLOCK_M
        n_max = tl.minimum(n_max, S)

    for n_start in range(0, n_max, BLOCK_N):
        n_offs = n_start + offs_n

        # Load K tile (BLOCK_N, BLOCK_D)
        k_ptrs = (K + pid_b * stride_kb
                  + n_offs[:, None] * stride_ks
                  + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs,
                    mask=(n_offs[:, None] < S) & (offs_d[None, :] < D),
                    other=0.0)

        # Load V tile (BLOCK_N, BLOCK_D)
        v_ptrs = (V + pid_b * stride_vb
                  + n_offs[:, None] * stride_vs
                  + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs,
                    mask=(n_offs[:, None] < S) & (offs_d[None, :] < D),
                    other=0.0)

        # S = Q @ K.T  (BLOCK_M, BLOCK_N)  scaled
        s = tl.dot(q, tl.trans(k), allow_tf32=False) * SCALE

        # Causal mask + sequence-length mask
        if CAUSAL:
            causal_mask = offs_m[:, None] >= n_offs[None, :]
            s = tl.where(causal_mask, s, -float("inf"))
        s = tl.where(n_offs[None, :] < S, s, -float("inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        # Renormalise running quantities
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, allow_tf32=False)
        m_i = m_new

    # Final normalise + write
    acc = acc / l_i[:, None]
    out = acc.to(tl.float16)

    o_ptrs = (Out + pid_b * stride_ob
              + offs_m[:, None] * stride_os
              + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, out,
             mask=(offs_m[:, None] < S) & (offs_d[None, :] < D))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def flash_attention_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Fused attention. q/k/v: (BH, S, D) fp16 contiguous. Returns (BH, S, D) fp16.

    No DRAM materialisation of attention scores. Single kernel launch
    per ``BH × cdiv(S, BLOCK_M)`` programs.
    """
    assert q.shape == k.shape == v.shape, "q, k, v must have identical shape"
    assert q.dtype == torch.float16 == k.dtype == v.dtype
    BH, S, D = q.shape
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    BLOCK_D = _next_power_of_2(D)
    out = torch.empty_like(q)
    # L is the per-row logsumexp; we don't use it externally for now but
    # it's needed by the autograd of FA. Leave a tiny placeholder for the
    # signature.
    L = torch.empty((BH, S), device=q.device, dtype=torch.float32)

    def _grid(meta):
        return (triton.cdiv(S, meta["BLOCK_M"]), BH)

    _flash_attention_fwd_kernel[_grid](
        q, k, v, out, L,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        scale, S, D,
        CAUSAL=causal,
        BLOCK_D=BLOCK_D,
    )
    return out


# ---------------------------------------------------------------------------
# Auto-load persisted autotune picks at import (matches turing_kernels.py pattern)
# ---------------------------------------------------------------------------


def _autoload() -> None:
    try:
        from compgen.bench.autotune_cache import load
    except ImportError:
        return
    load(_flash_attention_fwd_kernel)


_autoload()


__all__ = ["flash_attention_fp16"]
