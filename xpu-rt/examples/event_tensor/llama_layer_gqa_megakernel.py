"""Real Phase F example: Llama decoder layer megakernel with GQA + RoPE + causal.

Extends Phase E's RoPE+causal megakernel with **Grouped-Query Attention**
(GQA): K and V are computed and rotated only for ``N_KV_HEADS`` heads
(< ``N_HEADS``), and attention indexes them at ``h // KV_REPEAT``.  This
matches the actual structure of TinyLlama, Llama-2/3, Gemma, Qwen, and
every modern decoder-only LLM.

Per-task structure (only the bits that change from Phase E):

    qkv_proj    : per (m_tile, h_q).  Always computes Q.  K/V only if
                  ``h_q == (h_q // KV_REPEAT) * KV_REPEAT`` -- i.e. h_q
                  is the leader of its GQA group.  K/V are stored at
                  index ``h_kv = h_q // KV_REPEAT``.

    rope_apply  : per (m_tile, h_q).  Always rotates Q.  Rotates K only
                  for leader heads.  Same trick.

    compute_scores / apply_values : read K, V at index ``h // KV_REPEAT``.

K/V buffers are sized ``(N_KV_HEADS, S, D_HEAD)`` instead of
``(H, S, D_HEAD)``.

Validated against an HF-faithful PyTorch reference that does the same
thing (compute K/V at ``N_KV_HEADS`` heads, then ``repeat_interleave``
to ``H`` heads before SDPA -- mathematically equivalent).
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
from examples.event_tensor.llama_layer_rope_megakernel import hf_rope_tables


# ---------------------------------------------------------------------------
# Triton bodies (10 device functions; GQA-aware where it matters)
# ---------------------------------------------------------------------------


_NORM1_BODY = r"""
m_tile = task_id
m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
d_idx  = tl.arange(0, D_HIDDEN)

x_ptrs = X_ptr + m_rows[:, None] * D_HIDDEN + d_idx[None, :]
x      = tl.load(x_ptrs)

ms  = tl.sum(x * x, axis=1) / D_HIDDEN
inv = 1.0 / tl.sqrt(ms + RMS_EPS)
w   = tl.load(WNORM1_ptr + d_idx)

xn = x * inv[:, None] * w[None, :]
xn_ptrs = XN1_ptr + m_rows[:, None] * D_HIDDEN + d_idx[None, :]
tl.store(xn_ptrs, xn)

tl.atomic_add(ENORM1_ptr + m_tile, -1)
"""


# qkv_proj_gqa: always projects Q; K/V only for GQA group leaders.
_QKV_GQA_BODY = r"""
m_tile = task_id // H_HEADS
h      = task_id %  H_HEADS

counter = tl.atomic_or(ENORM1_ptr + m_tile, 0)
while counter > 0:
    counter = tl.atomic_or(ENORM1_ptr + m_tile, 0)

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
d_in   = tl.arange(0, D_HIDDEN)
d_out  = tl.arange(0, D_HEAD)

xn_ptrs = XN1_ptr + m_rows[:, None] * D_HIDDEN + d_in[None, :]
xn      = tl.load(xn_ptrs)

# Always compute Q[h]
wq_rows = h * D_HEAD + tl.arange(0, D_HEAD)
wq_ptrs = WQ_ptr + wq_rows[:, None] * D_HIDDEN + d_in[None, :]
wq      = tl.load(wq_ptrs)
q       = tl.dot(xn, tl.trans(wq))
q_ptrs  = Q_ptr + h * (S * D_HEAD) + m_rows[:, None] * D_HEAD + d_out[None, :]
tl.store(q_ptrs, q)

# K/V only if this Q head is the leader of its GQA group.
h_kv = h // KV_REPEAT
if h == h_kv * KV_REPEAT:
    wk_rows = h_kv * D_HEAD + tl.arange(0, D_HEAD)
    wk_ptrs = WK_ptr + wk_rows[:, None] * D_HIDDEN + d_in[None, :]
    wk      = tl.load(wk_ptrs)
    k       = tl.dot(xn, tl.trans(wk))
    k_ptrs  = K_ptr + h_kv * (S * D_HEAD) + m_rows[:, None] * D_HEAD + d_out[None, :]
    tl.store(k_ptrs, k)

    wv_ptrs = WV_ptr + wk_rows[:, None] * D_HIDDEN + d_in[None, :]
    wv      = tl.load(wv_ptrs)
    v       = tl.dot(xn, tl.trans(wv))
    v_ptrs  = V_ptr + h_kv * (S * D_HEAD) + m_rows[:, None] * D_HEAD + d_out[None, :]
    tl.store(v_ptrs, v)

tl.atomic_add(EQKV_ptr + 0, -1)
"""


# rope_apply_gqa: rotate Q always, K only for leader heads.
_ROPE_GQA_BODY = r"""
m_tile = task_id // H_HEADS
h      = task_id %  H_HEADS

counter = tl.atomic_or(EQKV_ptr + 0, 0)
while counter > 0:
    counter = tl.atomic_or(EQKV_ptr + 0, 0)

m_rows   = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
d_first  = tl.arange(0, D_HEAD_HALF)
d_second = D_HEAD_HALF + tl.arange(0, D_HEAD_HALF)

cos1 = tl.load(COS_ptr + m_rows[:, None] * D_HEAD + d_first[None, :])
cos2 = tl.load(COS_ptr + m_rows[:, None] * D_HEAD + d_second[None, :])
sin1 = tl.load(SIN_ptr + m_rows[:, None] * D_HEAD + d_first[None, :])
sin2 = tl.load(SIN_ptr + m_rows[:, None] * D_HEAD + d_second[None, :])

# Rotate Q[h]
q1_ptrs = Q_ptr + h * (S * D_HEAD) + m_rows[:, None] * D_HEAD + d_first[None, :]
q2_ptrs = Q_ptr + h * (S * D_HEAD) + m_rows[:, None] * D_HEAD + d_second[None, :]
q1 = tl.load(q1_ptrs); q2 = tl.load(q2_ptrs)
new_q1 = q1 * cos1 - q2 * sin1
new_q2 = q2 * cos2 + q1 * sin2
tl.store(q1_ptrs, new_q1)
tl.store(q2_ptrs, new_q2)

# Rotate K[h_kv] only for leader heads.
h_kv = h // KV_REPEAT
if h == h_kv * KV_REPEAT:
    k1_ptrs = K_ptr + h_kv * (S * D_HEAD) + m_rows[:, None] * D_HEAD + d_first[None, :]
    k2_ptrs = K_ptr + h_kv * (S * D_HEAD) + m_rows[:, None] * D_HEAD + d_second[None, :]
    k1 = tl.load(k1_ptrs); k2 = tl.load(k2_ptrs)
    new_k1 = k1 * cos1 - k2 * sin1
    new_k2 = k2 * cos2 + k1 * sin2
    tl.store(k1_ptrs, new_k1)
    tl.store(k2_ptrs, new_k2)

tl.atomic_add(EROPE_ptr + 0, -1)
"""


# compute_scores_gqa: K is read at h // KV_REPEAT.
_COMPUTE_SCORES_GQA_BODY = r"""
h        = task_id // Q_TILES
q_tile   = task_id %  Q_TILES

counter = tl.atomic_or(EROPE_ptr + 0, 0)
while counter > 0:
    counter = tl.atomic_or(EROPE_ptr + 0, 0)

q_rows   = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
key_cols = tl.arange(0, S)
d_cols   = tl.arange(0, D_HEAD)

q_ptrs = Q_ptr + h * (S * D_HEAD) + q_rows[:, None] * D_HEAD + d_cols[None, :]
q      = tl.load(q_ptrs)
h_kv   = h // KV_REPEAT
k_ptrs = K_ptr + h_kv * (S * D_HEAD) + key_cols[:, None] * D_HEAD + d_cols[None, :]
k      = tl.load(k_ptrs)

scores = tl.dot(q, tl.trans(k)) * INV_SQRT_D
mask   = q_rows[:, None] >= key_cols[None, :]
scores = tl.where(mask, scores, -1e30)

row_max = tl.max(scores, axis=1)
scores  = scores - row_max[:, None]
exps    = tl.exp(scores)
denom   = tl.sum(exps, axis=1)
probs   = exps / denom[:, None]

p_ptrs = P_ptr + h * (S * S) + q_rows[:, None] * S + key_cols[None, :]
tl.store(p_ptrs, probs)

tl.atomic_add(ESCORES_ptr + task_id, -1)
"""


# apply_values_gqa: V is read at h // KV_REPEAT.
_APPLY_VALUES_GQA_BODY = r"""
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
h_kv   = h // KV_REPEAT
v_ptrs = V_ptr + h_kv * (S * D_HEAD) + key_rows[:, None] * D_HEAD + d_cols[None, :]
v      = tl.load(v_ptrs)
out    = tl.dot(p, v)

a_ptrs = A_ptr + h * (S * D_HEAD) + q_rows[:, None] * D_HEAD + d_cols[None, :]
tl.store(a_ptrs, out)

tl.atomic_add(EATTN_ptr + q_tile, -1)
"""


# Stages 5-10 (o_proj, norm2, MLP) reuse Phase E bodies unchanged.
_O_PROJ_BODY = r"""
m_tile = task_id

counter = tl.atomic_or(EATTN_ptr + m_tile, 0)
while counter > 0:
    counter = tl.atomic_or(EATTN_ptr + m_tile, 0)

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
d_idx  = tl.arange(0, D_HIDDEN)

h_of   = d_idx // D_HEAD
dh_of  = d_idx %  D_HEAD
a_ptrs = A_ptr + h_of[None, :] * (S * D_HEAD) + m_rows[:, None] * D_HEAD + dh_of[None, :]
a_flat = tl.load(a_ptrs)

wo_ptrs = WO_ptr + d_idx[:, None] * D_HIDDEN + d_idx[None, :]
wo      = tl.load(wo_ptrs)
o_block = tl.dot(a_flat, tl.trans(wo))

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

hi_ptrs = HI_ptr + m_rows[:, None] * D_HIDDEN + n_cols[None, :]
hi      = tl.load(hi_ptrs)

y_ptrs  = Y_ptr  + m_rows[:, None] * D_HIDDEN + n_cols[None, :]
tl.store(y_ptrs, hi + mlp_out)
"""


# ---------------------------------------------------------------------------
# IR construction
# ---------------------------------------------------------------------------


def build_gqa_event_graph(
    n_heads: int, n_kv_heads: int,
    m_tiles: int, q_tiles: int, i_tiles: int, n_tiles: int,
) -> tuple[ModuleOp, GraphOp]:
    block = Block()
    n_qkv_total  = m_tiles * n_heads
    n_attn_tasks = n_heads * q_tiles
    n_gate_up    = m_tiles * i_tiles
    n_mlp_down   = m_tiles * n_tiles

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
        "sym_name": StringAttr("EROPE"),
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

    for fn, count in [
        ("input_norm",      m_tiles),
        ("qkv_proj",        n_qkv_total),
        ("rope_apply",      n_qkv_total),
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

    total = (m_tiles + 2 * n_qkv_total + 2 * n_attn_tasks
             + m_tiles + m_tiles + 2 * n_gate_up + n_mlp_down)
    sm_count = max(1, min(total, 16))
    graph = GraphOp(
        sym_name="llama_layer_gqa",
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
class CompiledLlamaLayerGQA:
    kernel_name: str
    kernel_source: str
    kernel_callable: object
    lowering: DynamicMegakernelLoweringResult
    n_heads: int
    n_kv_heads: int
    seq_len: int
    head_dim: int
    hidden_dim: int
    intermediate_dim: int
    block_m: int
    block_i: int
    block_n: int
    sm_count: int


def compile_llama_layer_gqa(
    n_heads: int = 4, n_kv_heads: int = 2, seq_len: int = 16, head_dim: int = 16,
    intermediate_dim: int = 64,
    block_m: int = 16, block_i: int = 32, block_n: int = 32,
) -> CompiledLlamaLayerGQA:
    if n_heads % n_kv_heads:
        raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
    if head_dim % 2:
        raise ValueError("RoPE requires even head_dim")
    hidden_dim = n_heads * head_dim
    if seq_len % block_m or hidden_dim % block_n or intermediate_dim % block_i:
        raise ValueError("dims must be divisible by their block sizes")
    m_tiles = seq_len // block_m
    q_tiles = m_tiles
    i_tiles = intermediate_dim // block_i
    n_tiles = hidden_dim // block_n

    mod, graph = build_gqa_event_graph(
        n_heads, n_kv_heads, m_tiles, q_tiles, i_tiles, n_tiles,
    )
    spec = DynamicMegakernelLoweringSpec(
        data_pointers=(
            "X_ptr", "XN1_ptr", "WNORM1_ptr",
            "WQ_ptr", "WK_ptr", "WV_ptr",
            "Q_ptr", "K_ptr", "V_ptr", "P_ptr", "A_ptr",
            "COS_ptr", "SIN_ptr",
            "WO_ptr", "HI_ptr", "WNORM2_ptr", "XN2_ptr",
            "WG_ptr", "WU_ptr", "WD_ptr",
            "G_ptr", "U_ptr", "Y_ptr",
        ),
        constexpr_args=(
            "S", "D_HEAD", "D_HEAD_HALF", "D_HIDDEN", "I",
            "H_HEADS", "N_KV_HEADS", "KV_REPEAT",
            "Q_TILES", "I_TILES", "N_TILES",
            "BLOCK_M", "BLOCK_I", "BLOCK_N",
            "INV_SQRT_D", "RMS_EPS",
        ),
        device_functions=(
            DynamicDeviceFunctionSpec(name="input_norm",      body_source=_NORM1_BODY),
            DynamicDeviceFunctionSpec(name="qkv_proj",        body_source=_QKV_GQA_BODY),
            DynamicDeviceFunctionSpec(name="rope_apply",      body_source=_ROPE_GQA_BODY),
            DynamicDeviceFunctionSpec(name="compute_scores",  body_source=_COMPUTE_SCORES_GQA_BODY),
            DynamicDeviceFunctionSpec(name="apply_values",    body_source=_APPLY_VALUES_GQA_BODY),
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

    return CompiledLlamaLayerGQA(
        kernel_name=lowering.kernel_name,
        kernel_source=lowering.kernel_source,
        kernel_callable=kernel_callable,
        lowering=lowering,
        n_heads=n_heads, n_kv_heads=n_kv_heads,
        seq_len=seq_len, head_dim=head_dim,
        hidden_dim=hidden_dim, intermediate_dim=intermediate_dim,
        block_m=block_m, block_i=block_i, block_n=block_n,
        sm_count=int(lowering.launch_config["grid"]),
    )


def run_llama_layer_gqa(
    compiled: CompiledLlamaLayerGQA,
    x: torch.Tensor,
    w_norm1: torch.Tensor, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor,
    w_o: torch.Tensor, w_norm2: torch.Tensor,
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    rms_eps: float = 1e-5,
) -> torch.Tensor:
    H, N_KV = compiled.n_heads, compiled.n_kv_heads
    S, D_HEAD = compiled.seq_len, compiled.head_dim
    D_HIDDEN, I = compiled.hidden_dim, compiled.intermediate_dim
    KV_REPEAT = H // N_KV
    device = x.device

    if tuple(w_q.shape) != (D_HIDDEN, D_HIDDEN): raise ValueError("W_q shape mismatch")
    if tuple(w_k.shape) != (N_KV * D_HEAD, D_HIDDEN): raise ValueError(f"W_k shape mismatch: got {tuple(w_k.shape)}, want ({N_KV * D_HEAD}, {D_HIDDEN})")
    if tuple(w_v.shape) != (N_KV * D_HEAD, D_HIDDEN): raise ValueError("W_v shape mismatch")

    xn1 = torch.zeros_like(x)
    xn2 = torch.zeros_like(x)
    q_t  = torch.zeros((H,    S, D_HEAD), dtype=torch.float32, device=device)
    k_t  = torch.zeros((N_KV, S, D_HEAD), dtype=torch.float32, device=device)
    v_t  = torch.zeros((N_KV, S, D_HEAD), dtype=torch.float32, device=device)
    p    = torch.zeros((H,    S, S),      dtype=torch.float32, device=device)
    a    = torch.zeros((H,    S, D_HEAD), dtype=torch.float32, device=device)
    hi   = torch.zeros_like(x)
    g    = torch.zeros((S, I),            dtype=torch.float32, device=device)
    u    = torch.zeros((S, I),            dtype=torch.float32, device=device)
    y    = torch.zeros_like(x)

    m_tiles = S // compiled.block_m
    q_tiles = m_tiles
    i_tiles = I // compiled.block_i
    n_tiles = D_HIDDEN // compiled.block_n
    n_attn_tasks = H * q_tiles
    n_qkv_total  = H * m_tiles
    n_gate_up    = m_tiles * i_tiles
    n_mlp_down   = m_tiles * n_tiles

    e_norm1   = torch.full((m_tiles,),       1, dtype=torch.int32, device=device)
    e_qkv     = torch.full((1,),   n_qkv_total, dtype=torch.int32, device=device)
    e_rope    = torch.full((1,),   n_qkv_total, dtype=torch.int32, device=device)
    e_scores  = torch.full((n_attn_tasks,),  1, dtype=torch.int32, device=device)
    e_attn    = torch.full((m_tiles,),       H, dtype=torch.int32, device=device)
    e_oproj   = torch.full((m_tiles,),       1, dtype=torch.int32, device=device)
    e_norm2   = torch.full((m_tiles,),       1, dtype=torch.int32, device=device)
    e_gate    = torch.full((n_gate_up,),     1, dtype=torch.int32, device=device)
    e_up      = torch.full((n_gate_up,),     1, dtype=torch.int32, device=device)

    total_tasks = (
        m_tiles + 2 * n_qkv_total + 2 * n_attn_tasks
        + m_tiles + m_tiles + 2 * n_gate_up + n_mlp_down
    )
    max_queue = total_tasks * 2

    kind_of = {fn: k for k, fn in compiled.lowering.device_function_table.items()}
    queue_pool  = torch.zeros((max_queue, 2), dtype=torch.int32, device=device)
    queue_valid = torch.zeros((max_queue,),   dtype=torch.int32, device=device)
    slot = 0
    for fn_name, count in [
        ("input_norm",      m_tiles),
        ("qkv_proj",        n_qkv_total),
        ("rope_apply",      n_qkv_total),
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
        x, xn1, w_norm1, w_q, w_k, w_v,
        q_t, k_t, v_t, p, a, cos, sin,
        w_o, hi, w_norm2, xn2,
        w_gate, w_up, w_down, g, u, y,
        e_norm1, e_qkv, e_rope, e_scores, e_attn, e_oproj, e_norm2, e_gate, e_up,
        queue_pool, queue_head, queue_tail, queue_valid,
        S, D_HEAD, D_HEAD // 2, D_HIDDEN, I,
        H, N_KV, KV_REPEAT,
        q_tiles, i_tiles, n_tiles,
        compiled.block_m, compiled.block_i, compiled.block_n,
        inv_sqrt_d, rms_eps,
        compiled.sm_count, total_tasks, max_queue,
        num_warps=compiled.lowering.launch_config["num_warps"],
        num_stages=compiled.lowering.launch_config["num_stages"],
    )
    torch.cuda.synchronize()
    return y


def reference_llama_layer_gqa(
    x: torch.Tensor,
    w_norm1: torch.Tensor, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor,
    w_o: torch.Tensor, w_norm2: torch.Tensor,
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int,
    rms_eps: float = 1e-5,
) -> torch.Tensor:
    """HF-faithful Llama decoder-layer reference WITH true GQA."""
    S, D_HIDDEN = x.shape
    repeat = n_heads // n_kv_heads

    def rms_norm(z, w): return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + rms_eps) * w

    def rotate_half(t):
        half = t.shape[-1] // 2
        return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

    xn1 = rms_norm(x, w_norm1)

    q = (xn1 @ w_q.T).reshape(S, n_heads,    head_dim).permute(1, 0, 2).contiguous()
    k = (xn1 @ w_k.T).reshape(S, n_kv_heads, head_dim).permute(1, 0, 2).contiguous()
    v = (xn1 @ w_v.T).reshape(S, n_kv_heads, head_dim).permute(1, 0, 2).contiguous()

    q = q * cos[None, :, :] + rotate_half(q) * sin[None, :, :]
    k = k * cos[None, :, :] + rotate_half(k) * sin[None, :, :]

    # GQA expand: each KV head services ``repeat`` Q heads.
    k_full = k.repeat_interleave(repeat, dim=0)
    v_full = v.repeat_interleave(repeat, dim=0)

    a = F.scaled_dot_product_attention(
        q.unsqueeze(0), k_full.unsqueeze(0), v_full.unsqueeze(0), is_causal=True,
    ).squeeze(0)
    a_flat = a.permute(1, 0, 2).reshape(S, n_heads * head_dim)

    o  = a_flat @ w_o.T
    hi = x + o
    xn2 = rms_norm(hi, w_norm2)
    gated = F.silu(xn2 @ w_gate.T) * (xn2 @ w_up.T)
    return hi + gated @ w_down.T


__all__ = [
    "CompiledLlamaLayerGQA",
    "build_gqa_event_graph",
    "compile_llama_layer_gqa",
    "reference_llama_layer_gqa",
    "run_llama_layer_gqa",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    H, N_KV = 4, 2          # KV_REPEAT = 2
    S, D_HEAD, I = 16, 16, 64
    D_HIDDEN = H * D_HEAD
    compiled = compile_llama_layer_gqa(
        n_heads=H, n_kv_heads=N_KV, seq_len=S, head_dim=D_HEAD, intermediate_dim=I,
    )
    print(f"Emitted GQA Llama layer megakernel: {compiled.kernel_name}")
    print(f"  H={H}, N_KV={N_KV}, KV_REPEAT={H//N_KV}, S={S}, D_HEAD={D_HEAD}, D_HIDDEN={D_HIDDEN}, I={I}")
    print(f"  source = {len(compiled.kernel_source)} chars; SM_COUNT={compiled.sm_count}")
    print(f"  device functions ({len(compiled.lowering.device_function_table)}): "
          f"{sorted(compiled.lowering.device_function_table.values())}")

    cos, sin = hf_rope_tables(S, D_HEAD)
    torch.manual_seed(2027)
    x       = torch.randn((S, D_HIDDEN),                 dtype=torch.float32, device="cuda") * 0.1
    w_norm1 = torch.randn((D_HIDDEN,),                   dtype=torch.float32, device="cuda") * 0.1 + 1.0
    w_q     = torch.randn((D_HIDDEN, D_HIDDEN),          dtype=torch.float32, device="cuda") * 0.05
    w_k     = torch.randn((N_KV * D_HEAD, D_HIDDEN),     dtype=torch.float32, device="cuda") * 0.05
    w_v     = torch.randn((N_KV * D_HEAD, D_HIDDEN),     dtype=torch.float32, device="cuda") * 0.05
    w_o     = torch.randn((D_HIDDEN, D_HIDDEN),          dtype=torch.float32, device="cuda") * 0.05
    w_norm2 = torch.randn((D_HIDDEN,),                   dtype=torch.float32, device="cuda") * 0.1 + 1.0
    w_gate  = torch.randn((I, D_HIDDEN),                 dtype=torch.float32, device="cuda") * 0.05
    w_up    = torch.randn((I, D_HIDDEN),                 dtype=torch.float32, device="cuda") * 0.05
    w_down  = torch.randn((D_HIDDEN, I),                 dtype=torch.float32, device="cuda") * 0.05

    got = run_llama_layer_gqa(
        compiled, x, w_norm1, w_q, w_k, w_v, w_o, w_norm2, w_gate, w_up, w_down,
        cos, sin,
    )
    ref = reference_llama_layer_gqa(
        x, w_norm1, w_q, w_k, w_v, w_o, w_norm2, w_gate, w_up, w_down,
        cos, sin, n_heads=H, n_kv_heads=N_KV, head_dim=D_HEAD,
    )
    err = (got - ref).abs().max().item()
    print(f"max |got - ref| = {err:.3e}")
    assert err < 5e-3, f"GQA layer diverges by {err}"
    print("PASS: emitted GQA Llama layer megakernel matches GQA reference.")
