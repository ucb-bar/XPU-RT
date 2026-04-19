"""Real Phase H example: decode-step megakernel with KV cache.

This is the second of the two megakernels a real LLM serving stack
needs:

    * Prefill kernel  (Phases C-G):  encodes a full prompt of S tokens,
      attention is S x S, every layer task processes BLOCK_M rows.
    * Decode kernel   (this file):    encodes a single new token using
      cached K/V from prior steps; Q is 1 row, attention is 1 x S_kv.

The decode-step megakernel:

    Inputs:
        x          : (D_HIDDEN,)              -- the new token's hidden state
        K_cache    : (N_KV, S_MAX, D_HEAD)    -- preallocated; first
                                                 ``context_len`` positions
                                                 are valid coming in
        V_cache    : (N_KV, S_MAX, D_HEAD)    -- same
        weights    : RMSNorm scales + Q/K/V/O proj + RMSNorm + SwiGLU
        cos, sin   : (S_MAX, D_HEAD) precomputed RoPE tables
        context_len: int (passed via constexpr per launch) -- number
                     of valid positions in the cache *before* this step

    Output:
        y          : (D_HIDDEN,) -- the new token's output hidden state
        K_cache[:, context_len, :]  <- new K, written by the megakernel
        V_cache[:, context_len, :]  <- new V, written by the megakernel

Differences from the prefill megakernel:

    * Q has 1 row total (per head), so we don't ``tl.dot`` on the
      attention path -- ``tl.sum(q * k, axis=-1)`` is enough.
    * MLP is one row of activations, so each MLP body is a tiny matmul.
    * o_proj, RMSNorm, SwiGLU all operate on a single row.

Validated by composing this kernel with the Phase G prefill kernel:
the prefill processes the prompt and populates the cache; the decode
kernel produces each subsequent token.  Token-by-token match against
``HF.model.generate(do_sample=False)`` is the acceptance criterion.
"""

from __future__ import annotations

import importlib.util
import linecache
import os
import tempfile
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Block, Region

from compgen.ir.event.attrs import EventCoordAttr, EventTensorTypeAttr
from compgen.ir.event.ops import CallDeviceOp, EventTensorOp, GraphOp
from compgen.ir.tile.lower_megakernel_dynamic import (
    DynamicDeviceFunctionSpec,
    DynamicMegakernelLoweringResult,
    DynamicMegakernelLoweringSpec,
    lower_megakernel_dynamic,
)


# ---------------------------------------------------------------------------
# Triton bodies (8 device functions for the decode step)
# ---------------------------------------------------------------------------


_NORM1_BODY = r"""
# Single-row RMSNorm of x.
d_idx  = tl.arange(0, D_HIDDEN)
x      = tl.load(X_ptr + d_idx)
ms     = tl.sum(x * x) / D_HIDDEN
inv    = 1.0 / tl.sqrt(ms + RMS_EPS)
w      = tl.load(WNORM1_ptr + d_idx)
xn     = x * inv * w
tl.store(XN1_ptr + d_idx, xn)
tl.atomic_add(ENORM1_ptr + 0, -1)
"""


# qkv_proj_decode: per head h.  Always projects Q[h]; if h is a GQA
# leader, projects K[h_kv] and V[h_kv] and writes them at index
# ``CONTEXT_LEN`` of the cache (the slot for this new token).
_QKV_DECODE_BODY = r"""
h = task_id

counter = tl.atomic_or(ENORM1_ptr + 0, 0)
while counter > 0:
    counter = tl.atomic_or(ENORM1_ptr + 0, 0)

d_in  = tl.arange(0, D_HIDDEN)
d_out = tl.arange(0, D_HEAD)

xn = tl.load(XN1_ptr + d_in)              # (D_HIDDEN,)

# Q[h] = xn @ W_q[h_block, :].T   -- shape (D_HEAD,)
wq_rows = h * D_HEAD + tl.arange(0, D_HEAD)
wq      = tl.load(WQ_ptr + wq_rows[:, None] * D_HIDDEN + d_in[None, :])
q       = tl.sum(wq * xn[None, :], axis=1)
tl.store(Q_ptr + h * D_HEAD + d_out, q)

h_kv = h // KV_REPEAT
if h == h_kv * KV_REPEAT:
    wk_rows = h_kv * D_HEAD + tl.arange(0, D_HEAD)

    wk = tl.load(WK_ptr + wk_rows[:, None] * D_HIDDEN + d_in[None, :])
    k  = tl.sum(wk * xn[None, :], axis=1)
    # Append new K at slot CONTEXT_LEN of K_cache[h_kv].
    tl.store(KCACHE_ptr + h_kv * S_MAX * D_HEAD + CONTEXT_LEN * D_HEAD + d_out, k)

    wv = tl.load(WV_ptr + wk_rows[:, None] * D_HIDDEN + d_in[None, :])
    v  = tl.sum(wv * xn[None, :], axis=1)
    tl.store(VCACHE_ptr + h_kv * S_MAX * D_HEAD + CONTEXT_LEN * D_HEAD + d_out, v)

tl.atomic_add(EQKV_ptr + 0, -1)
"""


# rope_decode: rotate the new Q[h] and (if leader) the new K[h_kv at slot CONTEXT_LEN].
_ROPE_DECODE_BODY = r"""
h = task_id

counter = tl.atomic_or(EQKV_ptr + 0, 0)
while counter > 0:
    counter = tl.atomic_or(EQKV_ptr + 0, 0)

d_first  = tl.arange(0, D_HEAD_HALF)
d_second = D_HEAD_HALF + tl.arange(0, D_HEAD_HALF)

# Position of the new token == CONTEXT_LEN.  Read the matching cos/sin row.
cos1 = tl.load(COS_ptr + CONTEXT_LEN * D_HEAD + d_first)
cos2 = tl.load(COS_ptr + CONTEXT_LEN * D_HEAD + d_second)
sin1 = tl.load(SIN_ptr + CONTEXT_LEN * D_HEAD + d_first)
sin2 = tl.load(SIN_ptr + CONTEXT_LEN * D_HEAD + d_second)

q1_ptrs = Q_ptr + h * D_HEAD + d_first
q2_ptrs = Q_ptr + h * D_HEAD + d_second
q1 = tl.load(q1_ptrs); q2 = tl.load(q2_ptrs)
tl.store(q1_ptrs, q1 * cos1 - q2 * sin1)
tl.store(q2_ptrs, q2 * cos2 + q1 * sin2)

h_kv = h // KV_REPEAT
if h == h_kv * KV_REPEAT:
    k_off = h_kv * S_MAX * D_HEAD + CONTEXT_LEN * D_HEAD
    k1_ptrs = KCACHE_ptr + k_off + d_first
    k2_ptrs = KCACHE_ptr + k_off + d_second
    k1 = tl.load(k1_ptrs); k2 = tl.load(k2_ptrs)
    tl.store(k1_ptrs, k1 * cos1 - k2 * sin1)
    tl.store(k2_ptrs, k2 * cos2 + k1 * sin2)

tl.atomic_add(EROPE_ptr + 0, -1)
"""


# attention_decode: per Q head h. Compute attention output for the single
# new query against all S_KV = CONTEXT_LEN + 1 cached positions.
_ATTENTION_DECODE_BODY = r"""
h = task_id

counter = tl.atomic_or(EROPE_ptr + 0, 0)
while counter > 0:
    counter = tl.atomic_or(EROPE_ptr + 0, 0)

d_idx    = tl.arange(0, D_HEAD)
key_pos  = tl.arange(0, S_MAX)

# Load the new query (D_HEAD,)
q = tl.load(Q_ptr + h * D_HEAD + d_idx)

h_kv = h // KV_REPEAT

# Load all S_MAX keys for this KV head.
k_ptrs = KCACHE_ptr + h_kv * S_MAX * D_HEAD + key_pos[:, None] * D_HEAD + d_idx[None, :]
k      = tl.load(k_ptrs)              # (S_MAX, D_HEAD)

# Score for each position: dot product q . k[i] / sqrt(D)
scores = tl.sum(k * q[None, :], axis=1) * INV_SQRT_D    # (S_MAX,)

# Causal-with-cache mask: only positions [0..CONTEXT_LEN] are valid.
valid_mask = key_pos <= CONTEXT_LEN
scores = tl.where(valid_mask, scores, -1e30)

# Softmax (numerically stable).
sm = tl.max(scores)
scores = scores - sm
exps   = tl.exp(scores)
denom  = tl.sum(exps)
probs  = exps / denom

# Apply to V.
v_ptrs = VCACHE_ptr + h_kv * S_MAX * D_HEAD + key_pos[:, None] * D_HEAD + d_idx[None, :]
v      = tl.load(v_ptrs)              # (S_MAX, D_HEAD)
out    = tl.sum(v * probs[:, None], axis=0)             # (D_HEAD,)

tl.store(A_ptr + h * D_HEAD + d_idx, out)

tl.atomic_add(EATTN_ptr + 0, -1)
"""


_O_PROJ_DECODE_BODY = r"""
counter = tl.atomic_or(EATTN_ptr + 0, 0)
while counter > 0:
    counter = tl.atomic_or(EATTN_ptr + 0, 0)

d_idx = tl.arange(0, D_HIDDEN)

# Flatten A: A[h, d] -> A_flat[h*D_HEAD + d]
a_flat = tl.load(A_ptr + d_idx)        # (D_HIDDEN,)

# o_block = a_flat @ W_o.T   -- shape (D_HIDDEN,)
wo = tl.load(WO_ptr + d_idx[:, None] * D_HIDDEN + d_idx[None, :])
o  = tl.sum(wo * a_flat[None, :], axis=1)

x  = tl.load(X_ptr + d_idx)
tl.store(HI_ptr + d_idx, x + o)

tl.atomic_add(EOPROJ_ptr + 0, -1)
"""


_NORM2_DECODE_BODY = r"""
counter = tl.atomic_or(EOPROJ_ptr + 0, 0)
while counter > 0:
    counter = tl.atomic_or(EOPROJ_ptr + 0, 0)

d_idx = tl.arange(0, D_HIDDEN)
hi    = tl.load(HI_ptr + d_idx)
ms    = tl.sum(hi * hi) / D_HIDDEN
inv   = 1.0 / tl.sqrt(ms + RMS_EPS)
w     = tl.load(WNORM2_ptr + d_idx)
tl.store(XN2_ptr + d_idx, hi * inv * w)
tl.atomic_add(ENORM2_ptr + 0, -1)
"""


_MLP_GATE_UP_DOWN_DECODE_BODY = r"""
# Single-row MLP -- compute gate, up, down all at once.  Cheap because
# everything is one row of the activation matrix.
counter = tl.atomic_or(ENORM2_ptr + 0, 0)
while counter > 0:
    counter = tl.atomic_or(ENORM2_ptr + 0, 0)

d_idx = tl.arange(0, D_HIDDEN)
i_idx = tl.arange(0, I)

xn = tl.load(XN2_ptr + d_idx)

# gate, up: each is xn @ W[i_idx, :].T -> (I,)
wg = tl.load(WG_ptr + i_idx[:, None] * D_HIDDEN + d_idx[None, :])
wu = tl.load(WU_ptr + i_idx[:, None] * D_HIDDEN + d_idx[None, :])
gate = tl.sum(wg * xn[None, :], axis=1)
up   = tl.sum(wu * xn[None, :], axis=1)
gated = gate * tl.sigmoid(gate) * up                # SwiGLU activated mid-state

# down: gated @ W_down.T -> (D_HIDDEN,)
wd = tl.load(WD_ptr + d_idx[:, None] * I + i_idx[None, :])
mlp_out = tl.sum(wd * gated[None, :], axis=1)

# Final residual.
hi = tl.load(HI_ptr + d_idx)
tl.store(Y_ptr + d_idx, hi + mlp_out)
"""


# ---------------------------------------------------------------------------
# IR construction
# ---------------------------------------------------------------------------


def build_decode_step_event_graph(n_heads: int) -> tuple[ModuleOp, GraphOp]:
    block = Block()
    for ev_name, wait in [
        ("ENORM1",  1),
        ("EQKV",    n_heads),
        ("EROPE",   n_heads),
        ("EATTN",   n_heads),
        ("EOPROJ",  1),
        ("ENORM2",  1),
    ]:
        block.add_op(EventTensorOp.create(properties={
            "sym_name": StringAttr(ev_name),
            "event_type": EventTensorTypeAttr([1]),
            "wait_count": IntegerAttr(wait, IntegerType(64)),
        }))

    for fn, count in [
        ("input_norm",       1),
        ("qkv_proj",         n_heads),
        ("rope_apply",       n_heads),
        ("attention_decode", n_heads),
        ("o_proj_residual",  1),
        ("post_attn_norm",   1),
        ("mlp_step",         1),
    ]:
        block.add_op(CallDeviceOp.create(properties={
            "device_func": SymbolRefAttr(fn),
            "task_shape": ArrayAttr([IntegerAttr(count, IntegerType(64))]),
        }))

    total = 1 + n_heads + n_heads + n_heads + 1 + 1 + 1
    sm_count = max(1, min(total, 16))
    graph = GraphOp(
        sym_name="llama_decode_step",
        policy="dynamic",
        sm_count=sm_count,
        body=Region([block]),
    )
    mod = ModuleOp([])
    mod.body.block.add_op(graph)
    return mod, graph


# ---------------------------------------------------------------------------
# Compile + run
# ---------------------------------------------------------------------------


@dataclass
class CompiledDecodeStep:
    kernel_name: str
    kernel_source: str
    kernel_callable: object
    lowering: DynamicMegakernelLoweringResult
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    intermediate_dim: int
    s_max: int
    sm_count: int


def compile_decode_step(
    n_heads: int = 4, n_kv_heads: int = 2, head_dim: int = 16,
    intermediate_dim: int = 64, s_max: int = 32,
) -> CompiledDecodeStep:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires even head_dim")
    if n_heads % n_kv_heads:
        raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
    hidden_dim = n_heads * head_dim

    mod, graph = build_decode_step_event_graph(n_heads)
    spec = DynamicMegakernelLoweringSpec(
        data_pointers=(
            "X_ptr", "XN1_ptr", "WNORM1_ptr",
            "WQ_ptr", "WK_ptr", "WV_ptr",
            "Q_ptr", "KCACHE_ptr", "VCACHE_ptr", "A_ptr",
            "COS_ptr", "SIN_ptr",
            "WO_ptr", "HI_ptr", "WNORM2_ptr", "XN2_ptr",
            "WG_ptr", "WU_ptr", "WD_ptr",
            "Y_ptr",
        ),
        constexpr_args=(
            "S_MAX", "D_HEAD", "D_HEAD_HALF", "D_HIDDEN", "I",
            "H_HEADS", "N_KV_HEADS", "KV_REPEAT",
            "CONTEXT_LEN",
            "INV_SQRT_D", "RMS_EPS",
        ),
        device_functions=(
            DynamicDeviceFunctionSpec(name="input_norm",       body_source=_NORM1_BODY),
            DynamicDeviceFunctionSpec(name="qkv_proj",         body_source=_QKV_DECODE_BODY),
            DynamicDeviceFunctionSpec(name="rope_apply",       body_source=_ROPE_DECODE_BODY),
            DynamicDeviceFunctionSpec(name="attention_decode", body_source=_ATTENTION_DECODE_BODY),
            DynamicDeviceFunctionSpec(name="o_proj_residual",  body_source=_O_PROJ_DECODE_BODY),
            DynamicDeviceFunctionSpec(name="post_attn_norm",   body_source=_NORM2_DECODE_BODY),
            DynamicDeviceFunctionSpec(name="mlp_step",         body_source=_MLP_GATE_UP_DOWN_DECODE_BODY),
        ),
    )
    lowering = lower_megakernel_dynamic(graph, spec=spec)

    fd, path = tempfile.mkstemp(prefix=f"{lowering.kernel_name}_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(lowering.kernel_source)
    linecache.checkcache(path)
    module_spec = importlib.util.spec_from_file_location(lowering.kernel_name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"failed to build importlib spec for {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kernel_callable = getattr(module, lowering.kernel_name)

    return CompiledDecodeStep(
        kernel_name=lowering.kernel_name,
        kernel_source=lowering.kernel_source,
        kernel_callable=kernel_callable,
        lowering=lowering,
        n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
        hidden_dim=hidden_dim, intermediate_dim=intermediate_dim,
        s_max=s_max, sm_count=int(lowering.launch_config["grid"]),
    )


def run_decode_step(
    compiled: CompiledDecodeStep,
    x: torch.Tensor,                    # (D_HIDDEN,)
    k_cache: torch.Tensor,              # (N_KV, S_MAX, D_HEAD), modified in place
    v_cache: torch.Tensor,              # (N_KV, S_MAX, D_HEAD), modified in place
    context_len: int,
    w_norm1: torch.Tensor, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor,
    w_o: torch.Tensor, w_norm2: torch.Tensor,
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    rms_eps: float = 1e-5,
) -> torch.Tensor:
    """Run one decode step.  Updates k_cache, v_cache in place."""
    H, N_KV = compiled.n_heads, compiled.n_kv_heads
    D_HEAD, D_HIDDEN, I = compiled.head_dim, compiled.hidden_dim, compiled.intermediate_dim
    S_MAX = compiled.s_max
    KV_REPEAT = H // N_KV
    device = x.device

    if x.shape != (D_HIDDEN,): raise ValueError("x shape mismatch")
    if k_cache.shape != (N_KV, S_MAX, D_HEAD): raise ValueError("k_cache shape mismatch")
    if v_cache.shape != (N_KV, S_MAX, D_HEAD): raise ValueError("v_cache shape mismatch")

    xn1 = torch.zeros_like(x)
    xn2 = torch.zeros_like(x)
    q   = torch.zeros((H * D_HEAD,), dtype=torch.float32, device=device)
    a   = torch.zeros((H * D_HEAD,), dtype=torch.float32, device=device)
    hi  = torch.zeros_like(x)
    y   = torch.zeros_like(x)

    e_norm1  = torch.full((1,), 1, dtype=torch.int32, device=device)
    e_qkv    = torch.full((1,), H, dtype=torch.int32, device=device)
    e_rope   = torch.full((1,), H, dtype=torch.int32, device=device)
    e_attn   = torch.full((1,), H, dtype=torch.int32, device=device)
    e_oproj  = torch.full((1,), 1, dtype=torch.int32, device=device)
    e_norm2  = torch.full((1,), 1, dtype=torch.int32, device=device)

    total_tasks = 1 + H + H + H + 1 + 1 + 1
    max_queue = total_tasks * 2

    kind_of = {fn: k for k, fn in compiled.lowering.device_function_table.items()}
    queue_pool  = torch.zeros((max_queue, 2), dtype=torch.int32, device=device)
    queue_valid = torch.zeros((max_queue,),   dtype=torch.int32, device=device)
    slot = 0
    for fn_name, count in [
        ("input_norm",       1),
        ("qkv_proj",         H),
        ("rope_apply",       H),
        ("attention_decode", H),
        ("o_proj_residual",  1),
        ("post_attn_norm",   1),
        ("mlp_step",         1),
    ]:
        kind = kind_of[fn_name]
        for tid in range(count):
            queue_pool[slot, 0] = tid
            queue_pool[slot, 1] = kind
            queue_valid[slot]   = 1
            slot += 1
    queue_head = torch.zeros((1,), dtype=torch.int32, device=device)
    queue_tail = torch.tensor([slot], dtype=torch.int32, device=device)
    inv_sqrt_d = 1.0 / (D_HEAD ** 0.5)

    compiled.kernel_callable[(compiled.sm_count,)](
        # data ptrs
        x, xn1, w_norm1,
        w_q, w_k, w_v,
        q, k_cache, v_cache, a,
        cos, sin,
        w_o, hi, w_norm2, xn2,
        w_gate, w_up, w_down,
        y,
        # event ptrs (declaration order)
        e_norm1, e_qkv, e_rope, e_attn, e_oproj, e_norm2,
        # queue ptrs
        queue_pool, queue_head, queue_tail, queue_valid,
        # constexprs
        S_MAX, D_HEAD, D_HEAD // 2, D_HIDDEN, I,
        H, N_KV, KV_REPEAT,
        context_len,
        inv_sqrt_d, rms_eps,
        # implicit constexprs
        compiled.sm_count, total_tasks, max_queue,
        num_warps=compiled.lowering.launch_config["num_warps"],
        num_stages=compiled.lowering.launch_config["num_stages"],
    )
    torch.cuda.synchronize()
    return y


# ---------------------------------------------------------------------------
# PyTorch reference -- HF-faithful single-token decode with KV cache
# ---------------------------------------------------------------------------


def reference_decode_step(
    x: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor, context_len: int,
    w_norm1: torch.Tensor, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor,
    w_o: torch.Tensor, w_norm2: torch.Tensor,
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int,
    rms_eps: float = 1e-5,
) -> torch.Tensor:
    """Updates k_cache, v_cache in place; returns y (D_HIDDEN,)."""
    D_HIDDEN = x.shape[0]
    repeat = n_heads // n_kv_heads

    def rmsn(z, w): return z * torch.rsqrt(z.pow(2).mean() + rms_eps) * w
    def rotate_half(t):
        half = t.shape[-1] // 2
        return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

    xn1 = rmsn(x, w_norm1)
    q = (xn1 @ w_q.T).reshape(n_heads,    head_dim)
    k = (xn1 @ w_k.T).reshape(n_kv_heads, head_dim)
    v = (xn1 @ w_v.T).reshape(n_kv_heads, head_dim)

    cos_pos = cos[context_len]                   # (head_dim,)
    sin_pos = sin[context_len]
    q = q * cos_pos[None, :] + rotate_half(q) * sin_pos[None, :]
    k = k * cos_pos[None, :] + rotate_half(k) * sin_pos[None, :]

    k_cache[:, context_len, :] = k
    v_cache[:, context_len, :] = v

    # Attention: q is (H, D_HEAD); use cached K[:context_len+1], V[:context_len+1].
    k_full = k_cache[:, :context_len + 1, :].repeat_interleave(repeat, dim=0)  # (H, S, D)
    v_full = v_cache[:, :context_len + 1, :].repeat_interleave(repeat, dim=0)

    scores = (q[:, None, :] * k_full).sum(dim=-1) / (head_dim ** 0.5)         # (H, S)
    probs  = torch.softmax(scores, dim=-1)
    a      = (probs[:, :, None] * v_full).sum(dim=1)                          # (H, D_HEAD)
    a_flat = a.reshape(D_HIDDEN)

    o  = a_flat @ w_o.T
    hi = x + o
    xn2 = rmsn(hi, w_norm2)
    gated = F.silu(xn2 @ w_gate.T) * (xn2 @ w_up.T)
    return hi + gated @ w_down.T


__all__ = [
    "CompiledDecodeStep",
    "build_decode_step_event_graph",
    "compile_decode_step",
    "reference_decode_step",
    "run_decode_step",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    H, N_KV, D_HEAD, I, S_MAX = 4, 2, 16, 64, 32
    D_HIDDEN = H * D_HEAD
    compiled = compile_decode_step(
        n_heads=H, n_kv_heads=N_KV, head_dim=D_HEAD,
        intermediate_dim=I, s_max=S_MAX,
    )
    print(f"Emitted decode-step megakernel: {compiled.kernel_name}")
    print(f"  H={H}, N_KV={N_KV}, D_HEAD={D_HEAD}, D_HIDDEN={D_HIDDEN}, I={I}, S_MAX={S_MAX}")
    print(f"  source = {len(compiled.kernel_source)} chars; SM_COUNT={compiled.sm_count}")
    print(f"  device functions ({len(compiled.lowering.device_function_table)}): "
          f"{sorted(compiled.lowering.device_function_table.values())}")

    # Synthetic test: run a few decode steps; verify against PyTorch reference.
    from examples.event_tensor.llama_layer_rope_megakernel import hf_rope_tables
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

    # Two parallel KV caches -- one for megakernel, one for reference.
    k_cache_mk  = torch.zeros((N_KV, S_MAX, D_HEAD), dtype=torch.float32, device="cuda")
    v_cache_mk  = torch.zeros((N_KV, S_MAX, D_HEAD), dtype=torch.float32, device="cuda")
    k_cache_ref = torch.zeros((N_KV, S_MAX, D_HEAD), dtype=torch.float32, device="cuda")
    v_cache_ref = torch.zeros((N_KV, S_MAX, D_HEAD), dtype=torch.float32, device="cuda")

    print("\nRunning 5 decode steps with growing KV cache ...")
    max_err = 0.0
    for step in range(5):
        x = torch.randn((D_HIDDEN,), dtype=torch.float32, device="cuda") * 0.1
        y_mk = run_decode_step(
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
        err = (y_mk - y_ref).abs().max().item()
        max_err = max(max_err, err)
        print(f"  step {step}: context_len={step}, max |y_mk - y_ref| = {err:.3e}")

    assert max_err < 5e-4, f"decode step diverges by {max_err}"
    print(f"\nPASS: emitted decode-step megakernel matches PyTorch reference across 5 steps")
    print(f"      (max abs error = {max_err:.3e}).")
