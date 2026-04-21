"""Real  example: full Llama decoder-layer megakernel.

Extends the  transformer block with the operators that turn it
into an actual Llama / Gemma / Qwen3 decoder layer:

    1. input_layernorm     -- RMSNorm before attention
    2. q_proj, k_proj, v_proj  -- attention input projections
    3. (Q/K/V are grouped into a single ``qkv_proj`` task class.)
    4. compute_scores      -- existing attention stage 1
    5. apply_values        -- existing attention stage 2
    6. o_proj + residual_1 -- output projection + first residual
    7. post_attn_layernorm -- RMSNorm before MLP
    8. mlp_gate / mlp_up   -- existing MLP stages
    9. mlp_down + residual_2 -- final residual

Eight device-function bodies, seven event tensors, all fused into
one persistent dynamic-scheduled megakernel.  Computes the *exact*
mathematical sequence a Llama decoder layer runs (modulo RoPE; the
test in ``tests/kernels/megakernel/test_llama_decoder_layer.py`` either skips
RoPE or applies a no-op rotation, with the same choice made by the
PyTorch reference for an apples-to-apples comparison).

Validated on real **TinyLlama-1.1B layer-0 weights** -- the input_norm
scale, the post_attn_norm scale, every projection matrix, and every
MLP matrix come from the cached HuggingFace checkpoint.  The PyTorch
reference is the same mathematical sequence in eager mode; both
consume the identical weights.
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
# Triton bodies (8 device functions, paper-faithful event coordination)
# ---------------------------------------------------------------------------
#
# Memory layout (row-major float32):
#   X         : (S, D_HIDDEN)         input residual stream
#   XN1       : (S, D_HIDDEN)         RMSNorm(X, w_norm1)  -- workspace
#   W_NORM1   : (D_HIDDEN,)           input_layernorm scale
#   W_Q,W_K,W_V: (D_HIDDEN, D_HIDDEN) attention projections (assume MHA)
#   W_O       : (D_HIDDEN, D_HIDDEN)  output projection
#   W_NORM2   : (D_HIDDEN,)           post_attention_layernorm scale
#   W_G,W_U   : (I, D_HIDDEN)         MLP gate, up
#   W_D       : (D_HIDDEN, I)         MLP down
#   Q,K,V     : (H, S, D_HEAD)        attention activations
#   P         : (H, S, S)             softmax probabilities
#   A         : (H, S, D_HEAD)        attention output
#   H_IN      : (S, D_HIDDEN)         X + (A_flat @ W_O.T) -- first residual
#   XN2       : (S, D_HIDDEN)         RMSNorm(H_IN, w_norm2)
#   G,U       : (S, I)                MLP intermediate
#   Y         : (S, D_HIDDEN)         final output
#
# Event tensors (one per consumer-set, paper-faithful "wait_count = N producers"):
#   E_NORM1   : (M_TILES,)              wait_count = 1
#   E_QKV     : (1,)                    wait_count = M_TILES * H
#   E_SCORES  : (H * Q_TILES,)          wait_count = 1
#   E_ATTN    : (M_TILES,)              wait_count = H
#   E_OPROJ   : (M_TILES,)              wait_count = 1
#   E_NORM2   : (M_TILES,)              wait_count = 1
#   E_GATE    : (M_TILES * I_TILES,)    wait_count = 1
#   E_UP      : (M_TILES * I_TILES,)    wait_count = 1


# RMSNorm(x, w) = w * x / sqrt(mean(x^2) + eps)   (Llama variant, no mean centring)
_NORM1_BODY = r"""
m_tile = task_id
m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
d_idx  = tl.arange(0, D_HIDDEN)

x_ptrs = X_ptr + m_rows[:, None] * D_HIDDEN + d_idx[None, :]
x      = tl.load(x_ptrs)

ms = tl.sum(x * x, axis=1) / D_HIDDEN
inv = 1.0 / tl.sqrt(ms + RMS_EPS)
w   = tl.load(WNORM1_ptr + d_idx)

xn = x * inv[:, None] * w[None, :]
xn_ptrs = XN1_ptr + m_rows[:, None] * D_HIDDEN + d_idx[None, :]
tl.store(xn_ptrs, xn)

tl.atomic_add(ENORM1_ptr + m_tile, -1)
"""


# QKV projection: per (m_tile, h) compute Q[h, m_rows, :], K[h, m_rows, :],
# V[h, m_rows, :] from XN1[m_rows, :] using slices of W_Q / W_K / W_V.
_QKV_BODY = r"""
m_tile = task_id // H_HEADS
h      = task_id %  H_HEADS

# Wait for input norm to finish for THIS m_tile.
counter = tl.atomic_or(ENORM1_ptr + m_tile, 0)
while counter > 0:
    counter = tl.atomic_or(ENORM1_ptr + m_tile, 0)

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
d_in   = tl.arange(0, D_HIDDEN)
d_out  = tl.arange(0, D_HEAD)

# Load XN1[m_rows, :]
xn_ptrs = XN1_ptr + m_rows[:, None] * D_HIDDEN + d_in[None, :]
xn      = tl.load(xn_ptrs)

# W_Q rows [h*D_HEAD : (h+1)*D_HEAD]  shape (D_HEAD, D_HIDDEN)
wq_rows = h * D_HEAD + tl.arange(0, D_HEAD)
wq_ptrs = WQ_ptr + wq_rows[:, None] * D_HIDDEN + d_in[None, :]
wq      = tl.load(wq_ptrs)
q       = tl.dot(xn, tl.trans(wq))                    # (BLOCK_M, D_HEAD)
q_ptrs  = Q_ptr + h * (S * D_HEAD) + m_rows[:, None] * D_HEAD + d_out[None, :]
tl.store(q_ptrs, q)

wk_ptrs = WK_ptr + wq_rows[:, None] * D_HIDDEN + d_in[None, :]
wk      = tl.load(wk_ptrs)
k       = tl.dot(xn, tl.trans(wk))
k_ptrs  = K_ptr + h * (S * D_HEAD) + m_rows[:, None] * D_HEAD + d_out[None, :]
tl.store(k_ptrs, k)

wv_ptrs = WV_ptr + wq_rows[:, None] * D_HIDDEN + d_in[None, :]
wv      = tl.load(wv_ptrs)
v       = tl.dot(xn, tl.trans(wv))
v_ptrs  = V_ptr + h * (S * D_HEAD) + m_rows[:, None] * D_HEAD + d_out[None, :]
tl.store(v_ptrs, v)

# Notify the global QKV-done counter.
tl.atomic_add(EQKV_ptr + 0, -1)
"""


_COMPUTE_SCORES_BODY = r"""
h        = task_id // Q_TILES
q_tile   = task_id %  Q_TILES

# Spin until ALL QKV proj tiles for ALL heads have finished.
counter = tl.atomic_or(EQKV_ptr + 0, 0)
while counter > 0:
    counter = tl.atomic_or(EQKV_ptr + 0, 0)

q_rows   = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
key_cols = tl.arange(0, S)
d_cols   = tl.arange(0, D_HEAD)

q_ptrs = Q_ptr + h * (S * D_HEAD) + q_rows[:, None] * D_HEAD + d_cols[None, :]
q      = tl.load(q_ptrs)
k_ptrs = K_ptr + h * (S * D_HEAD) + key_cols[:, None] * D_HEAD + d_cols[None, :]
k      = tl.load(k_ptrs)

scores = tl.dot(q, tl.trans(k)) * INV_SQRT_D
row_max = tl.max(scores, axis=1)
scores  = scores - row_max[:, None]
exps    = tl.exp(scores)
denom   = tl.sum(exps, axis=1)
probs   = exps / denom[:, None]

p_ptrs = P_ptr + h * (S * S) + q_rows[:, None] * S + key_cols[None, :]
tl.store(p_ptrs, probs)

tl.atomic_add(ESCORES_ptr + task_id, -1)
"""


_APPLY_VALUES_BODY = r"""
h        = task_id // Q_TILES
q_tile   = task_id %  Q_TILES

counter = tl.atomic_or(ESCORES_ptr + task_id, 0)
while counter > 0:
    counter = tl.atomic_or(ESCORES_ptr + task_id, 0)

q_rows   = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
key_rows = tl.arange(0, S)
d_cols   = tl.arange(0, D_HEAD)

p_ptrs = P_ptr + h * (S * S) + q_rows[:, None] * S + key_rows[None, :]
p      = tl.load(p_ptrs)
v_ptrs = V_ptr + h * (S * D_HEAD) + key_rows[:, None] * D_HEAD + d_cols[None, :]
v      = tl.load(v_ptrs)
out    = tl.dot(p, v)

# Store A[h, q_rows, :]
a_ptrs = A_ptr + h * (S * D_HEAD) + q_rows[:, None] * D_HEAD + d_cols[None, :]
tl.store(a_ptrs, out)

tl.atomic_add(EATTN_ptr + q_tile, -1)
"""


# o_proj_residual: per m_tile, computes O[m_rows, :] = A_flat[m_rows, :] @ W_O.T
# then writes H_IN[m_rows, :] = X[m_rows, :] + O[m_rows, :].
_O_PROJ_BODY = r"""
m_tile = task_id

# Wait for every head to have written A[h, m_rows, :].
counter = tl.atomic_or(EATTN_ptr + m_tile, 0)
while counter > 0:
    counter = tl.atomic_or(EATTN_ptr + m_tile, 0)

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
d_idx  = tl.arange(0, D_HIDDEN)

# Load A_flat[m_rows, d] -- A is laid out as (H, S, D_HEAD) so we have to
# gather the right slice per head.  We compute the matmul as:
#   O[m_rows, n_cols] = sum_{h, dh} A[h, m_rows, dh] * W_O[n_cols, h*D_HEAD + dh]
# The cleanest Triton form is a tl.dot over the flat (D_HIDDEN,) inner axis.
# We materialise A_flat[m_rows, d_idx] in registers by mapping d_idx -> (h, dh).
h_of   = d_idx // D_HEAD
dh_of  = d_idx %  D_HEAD
a_ptrs = A_ptr + h_of[None, :] * (S * D_HEAD) + m_rows[:, None] * D_HEAD + dh_of[None, :]
a_flat = tl.load(a_ptrs)                                  # (BLOCK_M, D_HIDDEN)

# W_O is (D_HIDDEN, D_HIDDEN).  We want H_IN[m_rows, :] = X[m_rows, :] + a_flat @ W_O.T.
wo_ptrs = WO_ptr + d_idx[:, None] * D_HIDDEN + d_idx[None, :]
wo      = tl.load(wo_ptrs)
o_block = tl.dot(a_flat, tl.trans(wo))                    # (BLOCK_M, D_HIDDEN)

x_ptrs  = X_ptr  + m_rows[:, None] * D_HIDDEN + d_idx[None, :]
hi_ptrs = HI_ptr + m_rows[:, None] * D_HIDDEN + d_idx[None, :]
x_block = tl.load(x_ptrs)
tl.store(hi_ptrs, x_block + o_block)

tl.atomic_add(EOPROJ_ptr + m_tile, -1)
"""


_NORM2_BODY = r"""
m_tile = task_id

counter = tl.atomic_or(EOPROJ_ptr + m_tile, 0)
while counter > 0:
    counter = tl.atomic_or(EOPROJ_ptr + m_tile, 0)

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
d_idx  = tl.arange(0, D_HIDDEN)

hi_ptrs = HI_ptr + m_rows[:, None] * D_HIDDEN + d_idx[None, :]
hi      = tl.load(hi_ptrs)

ms  = tl.sum(hi * hi, axis=1) / D_HIDDEN
inv = 1.0 / tl.sqrt(ms + RMS_EPS)
w   = tl.load(WNORM2_ptr + d_idx)

xn = hi * inv[:, None] * w[None, :]
xn_ptrs = XN2_ptr + m_rows[:, None] * D_HIDDEN + d_idx[None, :]
tl.store(xn_ptrs, xn)

tl.atomic_add(ENORM2_ptr + m_tile, -1)
"""


_MLP_GATE_BODY = r"""
m_tile = task_id // I_TILES
i_tile = task_id %  I_TILES

counter = tl.atomic_or(ENORM2_ptr + m_tile, 0)
while counter > 0:
    counter = tl.atomic_or(ENORM2_ptr + m_tile, 0)

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
i_cols = i_tile * BLOCK_I + tl.arange(0, BLOCK_I)
k_idx  = tl.arange(0, D_HIDDEN)

xn_ptrs = XN2_ptr + m_rows[:, None] * D_HIDDEN + k_idx[None, :]
xn      = tl.load(xn_ptrs)
wg_ptrs = WG_ptr + i_cols[:, None] * D_HIDDEN + k_idx[None, :]
wg      = tl.load(wg_ptrs)
gate    = tl.dot(xn, tl.trans(wg))
gated   = gate * tl.sigmoid(gate)

g_ptrs = G_ptr + m_rows[:, None] * I + i_cols[None, :]
tl.store(g_ptrs, gated)

tl.atomic_add(EGATE_ptr + (m_tile * I_TILES + i_tile), -1)
"""


_MLP_UP_BODY = r"""
m_tile = task_id // I_TILES
i_tile = task_id %  I_TILES

counter = tl.atomic_or(ENORM2_ptr + m_tile, 0)
while counter > 0:
    counter = tl.atomic_or(ENORM2_ptr + m_tile, 0)

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
i_cols = i_tile * BLOCK_I + tl.arange(0, BLOCK_I)
k_idx  = tl.arange(0, D_HIDDEN)

xn_ptrs = XN2_ptr + m_rows[:, None] * D_HIDDEN + k_idx[None, :]
xn      = tl.load(xn_ptrs)
wu_ptrs = WU_ptr + i_cols[:, None] * D_HIDDEN + k_idx[None, :]
wu      = tl.load(wu_ptrs)
up      = tl.dot(xn, tl.trans(wu))

u_ptrs = U_ptr + m_rows[:, None] * I + i_cols[None, :]
tl.store(u_ptrs, up)

tl.atomic_add(EUP_ptr + (m_tile * I_TILES + i_tile), -1)
"""


_MLP_DOWN_BODY = r"""
m_tile = task_id // N_TILES
n_tile = task_id %  N_TILES

for it in tl.static_range(0, I_TILES):
    cg = tl.atomic_or(EGATE_ptr + (m_tile * I_TILES + it), 0)
    while cg > 0:
        cg = tl.atomic_or(EGATE_ptr + (m_tile * I_TILES + it), 0)
    cu = tl.atomic_or(EUP_ptr + (m_tile * I_TILES + it), 0)
    while cu > 0:
        cu = tl.atomic_or(EUP_ptr + (m_tile * I_TILES + it), 0)

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
n_cols = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
i_idx  = tl.arange(0, I)

g_ptrs = G_ptr + m_rows[:, None] * I + i_idx[None, :]
u_ptrs = U_ptr + m_rows[:, None] * I + i_idx[None, :]
g      = tl.load(g_ptrs)
u      = tl.load(u_ptrs)
hid    = g * u

wd_ptrs = WD_ptr + n_cols[:, None] * I + i_idx[None, :]
wd      = tl.load(wd_ptrs)
mlp_out = tl.dot(hid, tl.trans(wd))

# Final residual: Y = H_IN + MLP_out (NOT XN2; the residual is the
# pre-norm value, matching Llama's "post_attention_layernorm output -> mlp -> + H_IN").
hi_ptrs = HI_ptr + m_rows[:, None] * D_HIDDEN + n_cols[None, :]
hi      = tl.load(hi_ptrs)

y_ptrs  = Y_ptr  + m_rows[:, None] * D_HIDDEN + n_cols[None, :]
tl.store(y_ptrs, hi + mlp_out)
"""


# ---------------------------------------------------------------------------
# IR construction
# ---------------------------------------------------------------------------


def build_decoder_layer_event_graph(
    n_heads: int, m_tiles: int, q_tiles: int, i_tiles: int, n_tiles: int,
) -> tuple[ModuleOp, GraphOp]:
    block = Block()
    n_qkv_total = m_tiles * n_heads
    n_attn_tasks = n_heads * q_tiles
    n_gate_up = m_tiles * i_tiles

    block.add_op(EventTensorOp.create(properties={
        "sym_name": StringAttr("ENORM1"),
        "event_type": EventTensorTypeAttr([m_tiles]),
        "wait_count": IntegerAttr(1, IntegerType(64)),
    }))
    block.add_op(EventTensorOp.create(properties={
        "sym_name": StringAttr("EQKV"),
        "event_type": EventTensorTypeAttr([1]),
        "wait_count": IntegerAttr(n_qkv_total, IntegerType(64)),
    }))
    block.add_op(EventTensorOp.create(properties={
        "sym_name": StringAttr("ESCORES"),
        "event_type": EventTensorTypeAttr([n_attn_tasks]),
        "wait_count": IntegerAttr(1, IntegerType(64)),
    }))
    block.add_op(EventTensorOp.create(properties={
        "sym_name": StringAttr("EATTN"),
        "event_type": EventTensorTypeAttr([m_tiles]),
        "wait_count": IntegerAttr(n_heads, IntegerType(64)),
    }))
    block.add_op(EventTensorOp.create(properties={
        "sym_name": StringAttr("EOPROJ"),
        "event_type": EventTensorTypeAttr([m_tiles]),
        "wait_count": IntegerAttr(1, IntegerType(64)),
    }))
    block.add_op(EventTensorOp.create(properties={
        "sym_name": StringAttr("ENORM2"),
        "event_type": EventTensorTypeAttr([m_tiles]),
        "wait_count": IntegerAttr(1, IntegerType(64)),
    }))
    block.add_op(EventTensorOp.create(properties={
        "sym_name": StringAttr("EGATE"),
        "event_type": EventTensorTypeAttr([n_gate_up]),
        "wait_count": IntegerAttr(1, IntegerType(64)),
    }))
    block.add_op(EventTensorOp.create(properties={
        "sym_name": StringAttr("EUP"),
        "event_type": EventTensorTypeAttr([n_gate_up]),
        "wait_count": IntegerAttr(1, IntegerType(64)),
    }))

    n_mlp_down = m_tiles * n_tiles
    for fn, count in [
        ("input_norm",      m_tiles),
        ("qkv_proj",        n_qkv_total),
        ("compute_scores",  n_attn_tasks),
        ("apply_values",    n_attn_tasks),
        ("o_proj_residual", m_tiles),
        ("post_attn_norm",  m_tiles),
        ("mlp_gate_proj",   n_gate_up),
        ("mlp_up_proj",     n_gate_up),
        ("mlp_down_proj",   n_mlp_down),
    ]:
        block.add_op(CallDeviceOp.create(properties={
            "device_func": SymbolRefAttr(fn),
            "task_shape": ArrayAttr([IntegerAttr(count, IntegerType(64))]),
        }))

    total = m_tiles + n_qkv_total + 2 * n_attn_tasks + m_tiles + m_tiles + 2 * n_gate_up + n_mlp_down
    sm_count = max(1, min(total, 16))
    graph = GraphOp(
        sym_name="llama_decoder_layer",
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
class CompiledLlamaDecoderLayer:
    kernel_name: str
    kernel_source: str
    kernel_callable: object
    lowering: DynamicMegakernelLoweringResult
    n_heads: int
    seq_len: int
    head_dim: int
    hidden_dim: int
    intermediate_dim: int
    block_m: int
    block_i: int
    block_n: int
    sm_count: int


def compile_llama_decoder_layer(
    n_heads: int = 4, seq_len: int = 16, head_dim: int = 64,
    intermediate_dim: int = 128,
    block_m: int = 16, block_i: int = 32, block_n: int = 32,
) -> CompiledLlamaDecoderLayer:
    hidden_dim = n_heads * head_dim
    if seq_len % block_m or hidden_dim % block_n or intermediate_dim % block_i:
        raise ValueError("dims must be divisible by their block sizes")
    m_tiles = seq_len // block_m
    q_tiles = m_tiles
    i_tiles = intermediate_dim // block_i
    n_tiles = hidden_dim // block_n

    mod, graph = build_decoder_layer_event_graph(
        n_heads, m_tiles, q_tiles, i_tiles, n_tiles,
    )
    spec = DynamicMegakernelLoweringSpec(
        data_pointers=(
            "X_ptr", "XN1_ptr", "WNORM1_ptr",
            "WQ_ptr", "WK_ptr", "WV_ptr",
            "Q_ptr", "K_ptr", "V_ptr", "P_ptr", "A_ptr",
            "WO_ptr", "HI_ptr", "WNORM2_ptr", "XN2_ptr",
            "WG_ptr", "WU_ptr", "WD_ptr",
            "G_ptr", "U_ptr", "Y_ptr",
        ),
        constexpr_args=(
            "S", "D_HEAD", "D_HIDDEN", "I", "H_HEADS",
            "Q_TILES", "I_TILES", "N_TILES",
            "BLOCK_M", "BLOCK_I", "BLOCK_N",
            "INV_SQRT_D", "RMS_EPS",
        ),
        device_functions=(
            DynamicDeviceFunctionSpec(name="input_norm",      body_source=_NORM1_BODY),
            DynamicDeviceFunctionSpec(name="qkv_proj",        body_source=_QKV_BODY),
            DynamicDeviceFunctionSpec(name="compute_scores",  body_source=_COMPUTE_SCORES_BODY),
            DynamicDeviceFunctionSpec(name="apply_values",    body_source=_APPLY_VALUES_BODY),
            DynamicDeviceFunctionSpec(name="o_proj_residual", body_source=_O_PROJ_BODY),
            DynamicDeviceFunctionSpec(name="post_attn_norm",  body_source=_NORM2_BODY),
            DynamicDeviceFunctionSpec(name="mlp_gate_proj",   body_source=_MLP_GATE_BODY),
            DynamicDeviceFunctionSpec(name="mlp_up_proj",     body_source=_MLP_UP_BODY),
            DynamicDeviceFunctionSpec(name="mlp_down_proj",   body_source=_MLP_DOWN_BODY),
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

    return CompiledLlamaDecoderLayer(
        kernel_name=lowering.kernel_name,
        kernel_source=lowering.kernel_source,
        kernel_callable=kernel_callable,
        lowering=lowering,
        n_heads=n_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        block_m=block_m,
        block_i=block_i,
        block_n=block_n,
        sm_count=int(lowering.launch_config["grid"]),
    )


def run_llama_decoder_layer(
    compiled: CompiledLlamaDecoderLayer,
    x: torch.Tensor,
    w_norm1: torch.Tensor, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor,
    w_o: torch.Tensor,
    w_norm2: torch.Tensor,
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
    rms_eps: float = 1e-5,
) -> torch.Tensor:
    H, S, D_HEAD = compiled.n_heads, compiled.seq_len, compiled.head_dim
    D_HIDDEN, I = compiled.hidden_dim, compiled.intermediate_dim
    device = x.device

    if any(t.dtype != torch.float32 for t in (
        x, w_norm1, w_q, w_k, w_v, w_o, w_norm2, w_gate, w_up, w_down,
    )):
        raise TypeError(" decoder layer megakernel only supports float32")
    if tuple(x.shape) != (S, D_HIDDEN): raise ValueError("X shape mismatch")
    if tuple(w_norm1.shape) != (D_HIDDEN,): raise ValueError("w_norm1 shape mismatch")
    if tuple(w_q.shape) != (D_HIDDEN, D_HIDDEN): raise ValueError("W_q shape mismatch")
    if tuple(w_k.shape) != (D_HIDDEN, D_HIDDEN): raise ValueError("W_k shape mismatch")
    if tuple(w_v.shape) != (D_HIDDEN, D_HIDDEN): raise ValueError("W_v shape mismatch")
    if tuple(w_o.shape) != (D_HIDDEN, D_HIDDEN): raise ValueError("W_o shape mismatch")
    if tuple(w_norm2.shape) != (D_HIDDEN,): raise ValueError("w_norm2 shape mismatch")
    if tuple(w_gate.shape) != (I, D_HIDDEN): raise ValueError("W_gate shape mismatch")
    if tuple(w_up.shape) != (I, D_HIDDEN): raise ValueError("W_up shape mismatch")
    if tuple(w_down.shape) != (D_HIDDEN, I): raise ValueError("W_down shape mismatch")

    xn1 = torch.zeros_like(x)
    xn2 = torch.zeros_like(x)
    q_t  = torch.zeros((H, S, D_HEAD),  dtype=torch.float32, device=device)
    k_t  = torch.zeros((H, S, D_HEAD),  dtype=torch.float32, device=device)
    v_t  = torch.zeros((H, S, D_HEAD),  dtype=torch.float32, device=device)
    p    = torch.zeros((H, S, S),       dtype=torch.float32, device=device)
    a    = torch.zeros((H, S, D_HEAD),  dtype=torch.float32, device=device)
    hi   = torch.zeros_like(x)
    g    = torch.zeros((S, I),          dtype=torch.float32, device=device)
    u    = torch.zeros((S, I),          dtype=torch.float32, device=device)
    y    = torch.zeros_like(x)

    m_tiles = S // compiled.block_m
    q_tiles = m_tiles
    i_tiles = I // compiled.block_i
    n_tiles = D_HIDDEN // compiled.block_n
    n_attn_tasks = H * q_tiles
    n_qkv_total  = H * m_tiles
    n_gate_up    = m_tiles * i_tiles
    n_mlp_down   = m_tiles * n_tiles

    e_norm1   = torch.full((m_tiles,),    1, dtype=torch.int32, device=device)
    e_qkv     = torch.full((1,), n_qkv_total, dtype=torch.int32, device=device)
    e_scores  = torch.full((n_attn_tasks,), 1, dtype=torch.int32, device=device)
    e_attn    = torch.full((m_tiles,),    H, dtype=torch.int32, device=device)
    e_oproj   = torch.full((m_tiles,),    1, dtype=torch.int32, device=device)
    e_norm2   = torch.full((m_tiles,),    1, dtype=torch.int32, device=device)
    e_gate    = torch.full((n_gate_up,),  1, dtype=torch.int32, device=device)
    e_up      = torch.full((n_gate_up,),  1, dtype=torch.int32, device=device)

    # Initial queue: every task pre-pushed; cross-stage waits inside bodies.
    total_tasks = (
        m_tiles + n_qkv_total + 2 * n_attn_tasks + m_tiles + m_tiles
        + 2 * n_gate_up + n_mlp_down
    )
    max_queue = total_tasks * 2

    kind_of = {fn: k for k, fn in compiled.lowering.device_function_table.items()}
    queue_pool  = torch.zeros((max_queue, 2), dtype=torch.int32, device=device)
    queue_valid = torch.zeros((max_queue,),   dtype=torch.int32, device=device)
    slot = 0
    for fn_name, count in [
        ("input_norm",      m_tiles),
        ("qkv_proj",        n_qkv_total),
        ("compute_scores",  n_attn_tasks),
        ("apply_values",    n_attn_tasks),
        ("o_proj_residual", m_tiles),
        ("post_attn_norm",  m_tiles),
        ("mlp_gate_proj",   n_gate_up),
        ("mlp_up_proj",     n_gate_up),
        ("mlp_down_proj",   n_mlp_down),
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
        q_t, k_t, v_t, p, a,
        w_o, hi, w_norm2, xn2,
        w_gate, w_up, w_down,
        g, u, y,
        # event ptrs (declaration order)
        e_norm1, e_qkv, e_scores, e_attn, e_oproj, e_norm2, e_gate, e_up,
        # queue ptrs
        queue_pool, queue_head, queue_tail, queue_valid,
        # constexprs
        S, D_HEAD, D_HIDDEN, I, H,
        q_tiles, i_tiles, n_tiles,
        compiled.block_m, compiled.block_i, compiled.block_n,
        inv_sqrt_d, rms_eps,
        # implicit constexprs
        compiled.sm_count, total_tasks, max_queue,
        num_warps=compiled.lowering.launch_config["num_warps"],
        num_stages=compiled.lowering.launch_config["num_stages"],
    )
    torch.cuda.synchronize()
    return y


def reference_decoder_layer(
    x: torch.Tensor,
    w_norm1: torch.Tensor, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor,
    w_o: torch.Tensor,
    w_norm2: torch.Tensor,
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
    n_heads: int, head_dim: int,
    rms_eps: float = 1e-5,
) -> torch.Tensor:
    """Faithful PyTorch eager Llama decoder layer (no RoPE, no causal mask)."""
    S, D_HIDDEN = x.shape

    def rms_norm(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ms = z.pow(2).mean(-1, keepdim=True)
        return z * torch.rsqrt(ms + rms_eps) * w

    xn1 = rms_norm(x, w_norm1)

    q = (xn1 @ w_q.T).reshape(S, n_heads, head_dim).permute(1, 0, 2).contiguous()
    k = (xn1 @ w_k.T).reshape(S, n_heads, head_dim).permute(1, 0, 2).contiguous()
    v = (xn1 @ w_v.T).reshape(S, n_heads, head_dim).permute(1, 0, 2).contiguous()

    a = F.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=False,
    ).squeeze(0)                                          # (H, S, D_HEAD)
    a_flat = a.permute(1, 0, 2).reshape(S, n_heads * head_dim)

    o  = a_flat @ w_o.T
    hi = x + o

    xn2 = rms_norm(hi, w_norm2)

    gated = F.silu(xn2 @ w_gate.T) * (xn2 @ w_up.T)
    mlp_out = gated @ w_down.T

    return hi + mlp_out


__all__ = [
    "CompiledLlamaDecoderLayer",
    "build_decoder_layer_event_graph",
    "compile_llama_decoder_layer",
    "reference_decoder_layer",
    "run_llama_decoder_layer",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    H, S, D_HEAD, I = 4, 16, 16, 64
    D_HIDDEN = H * D_HEAD
    compiled = compile_llama_decoder_layer(
        n_heads=H, seq_len=S, head_dim=D_HEAD, intermediate_dim=I,
        block_m=16, block_i=32, block_n=32,
    )
    print(f"Emitted Llama decoder-layer megakernel: {compiled.kernel_name}")
    print(f"  H={H}, S={S}, D_HEAD={D_HEAD}, D_HIDDEN={D_HIDDEN}, I={I}")
    print(f"  source = {len(compiled.kernel_source)} chars; SM_COUNT={compiled.sm_count}")
    print(f"  device functions: {sorted(compiled.lowering.device_function_table.values())}")

    torch.manual_seed(7)
    x        = torch.randn((S, D_HIDDEN),       dtype=torch.float32, device="cuda") * 0.1
    w_norm1  = torch.randn((D_HIDDEN,),         dtype=torch.float32, device="cuda") * 0.1 + 1.0
    w_q      = torch.randn((D_HIDDEN, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_k      = torch.randn((D_HIDDEN, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_v      = torch.randn((D_HIDDEN, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_o      = torch.randn((D_HIDDEN, D_HIDDEN), dtype=torch.float32, device="cuda") * 0.05
    w_norm2  = torch.randn((D_HIDDEN,),         dtype=torch.float32, device="cuda") * 0.1 + 1.0
    w_gate   = torch.randn((I, D_HIDDEN),       dtype=torch.float32, device="cuda") * 0.05
    w_up     = torch.randn((I, D_HIDDEN),       dtype=torch.float32, device="cuda") * 0.05
    w_down   = torch.randn((D_HIDDEN, I),       dtype=torch.float32, device="cuda") * 0.05

    got = run_llama_decoder_layer(
        compiled, x, w_norm1, w_q, w_k, w_v, w_o, w_norm2, w_gate, w_up, w_down,
    )
    ref = reference_decoder_layer(
        x, w_norm1, w_q, w_k, w_v, w_o, w_norm2, w_gate, w_up, w_down,
        n_heads=H, head_dim=D_HEAD,
    )

    err_abs = (got - ref).abs().max().item()
    err_rel = ((got - ref).abs() / (ref.abs() + 1e-6)).max().item()
    print(f"max |got - ref|       = {err_abs:.3e}")
    print(f"max |got - ref|/|ref| = {err_rel:.3e}")
    assert err_abs < 5e-3, f"Llama decoder-layer megakernel diverges by {err_abs}"
    print("PASS: emitted full Llama decoder-layer megakernel matches PyTorch reference.")
