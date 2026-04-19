"""Real Phase C example: fused attention + MLP transformer block megakernel.

Composes two heavy LLM stages -- multi-head attention and a SwiGLU MLP --
into a single persistent megakernel coordinated by event tensors.  This
is the Phase C demonstration that the abstraction *composes*: more than
one operator stage, more than one event tensor, multiple device function
families, all fused into one kernel.

Workload (per layer of a Llama / Gemma / Qwen3 / smolVLA decoder block,
modulo the elementwise norms which we run in PyTorch wrappers):

    Q, K, V  : (H, S, D_HEAD)      -- pre-projected
    X_resid  : (S, D_HIDDEN)        -- residual stream from upstream
    W_gate   : (I, D_HIDDEN)
    W_up     : (I, D_HIDDEN)
    W_down   : (D_HIDDEN, I)

    A         = SDPA(Q, K, V)            (H, S, D_HEAD)
    A_flat    = reshape(A, (S, H*D_HEAD)) (S, D_HIDDEN)   -- view; D_HIDDEN = H*D_HEAD
    H_in      = X_resid + A_flat
    M         = down( silu(H_in @ W_gate.T) * (H_in @ W_up.T) ).T
    Y         = H_in + M

The megakernel emits five device functions, all dispatched from one
persistent kernel:

    Stage 1  compute_scores   notify E_attn[m_tile]   (per head, per q_tile)
    Stage 2  apply_values     wait  E_attn[m_tile]    (per head, per q_tile)
                              notify E_attn_done[m_tile]
    Stage 3  mlp_gate_proj    wait  E_attn_done[m_tile]
                              notify E_gate[m_tile, i_tile]
    Stage 4  mlp_up_proj      wait  E_attn_done[m_tile]
                              notify E_up[m_tile, i_tile]
    Stage 5  mlp_down_proj    wait  E_gate[m_tile, *] + E_up[m_tile, *]
                              writes Y

For the dynamic emitter's auto-emitted notify-and-push logic to work
with this many distinct events, every body still does its own
``tl.atomic_add`` / spin-wait calls -- this matches the path the MoE
example took, and lets us focus the Phase C demonstration on stage
*composition* rather than on extending the dynamic emitter further.

Validated against a PyTorch eager reference that runs the same
mathematical sequence with ``F.scaled_dot_product_attention`` and a
SwiGLU MLP.  Real HF model weights are tested in
``tests/kernels/megakernel/test_transformer_block.py``.
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
# Triton bodies (5 device functions, multi-stage event coordination)
# ---------------------------------------------------------------------------
#
# Memory layout (row-major float32):
#   Q, K, V    : (H, S, D_HEAD)
#   P          : (H, S, S)              workspace for softmax probs
#   A          : (H, S, D_HEAD)         attention output
#   X          : (S, D_HIDDEN)          residual stream
#   H_IN       : (S, D_HIDDEN)          X + A (computed by apply_values)
#   G_WS, U_WS : (S, I)                 MLP intermediate workspaces
#   Y          : (S, D_HIDDEN)          final output
#
# Event tensors (paper-faithful "one event per consumer task" topology):
#   E_ATTN     : (M_TILES,) int32, wait_count=H -- attention done for tile
#   E_ATTND    : (M_TILES,) int32, wait_count=1 -- H_IN written for tile
#   E_GATE     : (M_TILES * I_TILES,) int32, wait_count=1
#   E_UP       : (M_TILES * I_TILES,) int32, wait_count=1


_COMPUTE_SCORES_BODY = r"""
h        = task_id // Q_TILES
q_tile   = task_id %  Q_TILES
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

# Notify the scores->values handoff event so apply_values may proceed.
tl.atomic_add(ESCORES_ptr + task_id, -1)
"""


_APPLY_VALUES_BODY = r"""
h        = task_id // Q_TILES
q_tile   = task_id %  Q_TILES

# Wait for the matching compute_scores task to finish writing P[h, q_tile, :].
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

out = tl.dot(p, v)

# Store A[h, q_rows, :]
a_ptrs = A_ptr + h * (S * D_HEAD) + q_rows[:, None] * D_HEAD + d_cols[None, :]
tl.store(a_ptrs, out)

# Now write A_flat into H_IN[q_rows, h*D_HEAD : (h+1)*D_HEAD] += residual contribution.
# We compute H_IN[q_rows, hd_cols] = X[q_rows, hd_cols] + A[h, q_rows, :] using atomic-add
# so multiple heads can write to disjoint columns concurrently without locking.
hd_cols = h * D_HEAD + tl.arange(0, D_HEAD)
x_ptrs  = X_ptr  + q_rows[:, None] * D_HIDDEN + hd_cols[None, :]
hi_ptrs = HI_ptr + q_rows[:, None] * D_HIDDEN + hd_cols[None, :]
x_block = tl.load(x_ptrs)
tl.atomic_add(hi_ptrs, x_block + out)

# Notify E_ATTN[q_tile] (decrement counter; wait_count==H so all heads must fire).
tl.atomic_add(EATTN_ptr + q_tile, -1)
"""


_MLP_GATE_BODY = r"""
m_tile = task_id // I_TILES
i_tile = task_id %  I_TILES

# Wait until every head has finished writing H_IN for this m_tile.
counter = tl.atomic_or(EATTN_ptr + m_tile, 0)
while counter > 0:
    counter = tl.atomic_or(EATTN_ptr + m_tile, 0)

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
i_cols = i_tile * BLOCK_I + tl.arange(0, BLOCK_I)
k_idx  = tl.arange(0, D_HIDDEN)

hi_ptrs = HI_ptr + m_rows[:, None] * D_HIDDEN + k_idx[None, :]
hi      = tl.load(hi_ptrs)
wg_ptrs = WG_ptr + i_cols[:, None] * D_HIDDEN + k_idx[None, :]
wg      = tl.load(wg_ptrs)
gate    = tl.dot(hi, tl.trans(wg))
gated   = gate * tl.sigmoid(gate)

g_ptrs = G_ptr + m_rows[:, None] * I + i_cols[None, :]
tl.store(g_ptrs, gated)

# Notify E_GATE[m_tile, i_tile]
tl.atomic_add(EGATE_ptr + (m_tile * I_TILES + i_tile), -1)
"""


_MLP_UP_BODY = r"""
m_tile = task_id // I_TILES
i_tile = task_id %  I_TILES

counter = tl.atomic_or(EATTN_ptr + m_tile, 0)
while counter > 0:
    counter = tl.atomic_or(EATTN_ptr + m_tile, 0)

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
i_cols = i_tile * BLOCK_I + tl.arange(0, BLOCK_I)
k_idx  = tl.arange(0, D_HIDDEN)

hi_ptrs = HI_ptr + m_rows[:, None] * D_HIDDEN + k_idx[None, :]
hi      = tl.load(hi_ptrs)
wu_ptrs = WU_ptr + i_cols[:, None] * D_HIDDEN + k_idx[None, :]
wu      = tl.load(wu_ptrs)
up      = tl.dot(hi, tl.trans(wu))

u_ptrs = U_ptr + m_rows[:, None] * I + i_cols[None, :]
tl.store(u_ptrs, up)

tl.atomic_add(EUP_ptr + (m_tile * I_TILES + i_tile), -1)
"""


_MLP_DOWN_BODY = r"""
m_tile = task_id // N_TILES
n_tile = task_id %  N_TILES

# Wait every (m_tile, *) gate AND up tile.
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

# Read H_IN residual back and add to MLP output -> Y.
hi_ptrs = HI_ptr + m_rows[:, None] * D_HIDDEN + n_cols[None, :]
hi      = tl.load(hi_ptrs)

y_ptrs  = Y_ptr  + m_rows[:, None] * D_HIDDEN + n_cols[None, :]
tl.store(y_ptrs, hi + mlp_out)
"""


# ---------------------------------------------------------------------------
# IR construction
# ---------------------------------------------------------------------------


def build_block_event_graph(
    n_heads: int, m_tiles: int, q_tiles: int, i_tiles: int, n_tiles: int,
) -> tuple[ModuleOp, GraphOp]:
    """Build the transformer-block event.graph.

    Per-stage task counts:
        compute_scores: n_heads * q_tiles
        apply_values:   n_heads * q_tiles
        mlp_gate_proj:  m_tiles * i_tiles
        mlp_up_proj:    m_tiles * i_tiles
        mlp_down_proj:  m_tiles * n_tiles
    """
    block = Block()
    n_attn = m_tiles                   # E_ATTN size = M_TILES, wait_count = n_heads
    n_attnd = m_tiles                  # E_ATTND placeholder (unused by current bodies)
    n_gate_up = m_tiles * i_tiles
    n_scores = n_heads * q_tiles
    block.add_op(EventTensorOp.create(properties={
        "sym_name": StringAttr("ESCORES"),
        "event_type": EventTensorTypeAttr([n_scores]),
        "wait_count": IntegerAttr(1, IntegerType(64)),
    }))
    block.add_op(EventTensorOp.create(properties={
        "sym_name": StringAttr("EATTN"),
        "event_type": EventTensorTypeAttr([n_attn]),
        "wait_count": IntegerAttr(n_heads, IntegerType(64)),
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

    n_attn_tasks = n_heads * q_tiles
    n_mlp_pre   = m_tiles * i_tiles
    n_mlp_down  = m_tiles * n_tiles

    block.add_op(CallDeviceOp.create(properties={
        "device_func": SymbolRefAttr("compute_scores"),
        "task_shape": ArrayAttr([IntegerAttr(n_attn_tasks, IntegerType(64))]),
    }))
    block.add_op(CallDeviceOp.create(properties={
        "device_func": SymbolRefAttr("apply_values"),
        "task_shape": ArrayAttr([IntegerAttr(n_attn_tasks, IntegerType(64))]),
    }))
    block.add_op(CallDeviceOp.create(properties={
        "device_func": SymbolRefAttr("mlp_gate_proj"),
        "task_shape": ArrayAttr([IntegerAttr(n_mlp_pre, IntegerType(64))]),
    }))
    block.add_op(CallDeviceOp.create(properties={
        "device_func": SymbolRefAttr("mlp_up_proj"),
        "task_shape": ArrayAttr([IntegerAttr(n_mlp_pre, IntegerType(64))]),
    }))
    block.add_op(CallDeviceOp.create(properties={
        "device_func": SymbolRefAttr("mlp_down_proj"),
        "task_shape": ArrayAttr([IntegerAttr(n_mlp_down, IntegerType(64))]),
    }))

    total = 2 * n_attn_tasks + 2 * n_mlp_pre + n_mlp_down
    sm_count = max(1, min(total, 16))
    graph = GraphOp(
        sym_name="transformer_block",
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
class CompiledTransformerBlockMegakernel:
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


def compile_transformer_block_megakernel(
    n_heads: int = 4,
    seq_len: int = 32,
    head_dim: int = 32,
    intermediate_dim: int = 128,
    block_m: int = 16,
    block_i: int = 32,
    block_n: int = 32,
) -> CompiledTransformerBlockMegakernel:
    hidden_dim = n_heads * head_dim
    if seq_len % block_m or hidden_dim % block_n or intermediate_dim % block_i:
        raise ValueError("dims must be divisible by their block sizes")
    m_tiles = seq_len // block_m
    q_tiles = m_tiles
    i_tiles = intermediate_dim // block_i
    n_tiles = hidden_dim // block_n

    mod, graph = build_block_event_graph(n_heads, m_tiles, q_tiles, i_tiles, n_tiles)

    spec = DynamicMegakernelLoweringSpec(
        data_pointers=(
            "Q_ptr", "K_ptr", "V_ptr", "P_ptr", "A_ptr",
            "X_ptr", "HI_ptr",
            "WG_ptr", "WU_ptr", "WD_ptr",
            "G_ptr", "U_ptr", "Y_ptr",
        ),
        constexpr_args=(
            "S", "D_HEAD", "D_HIDDEN", "I",
            "Q_TILES", "I_TILES", "N_TILES",
            "BLOCK_M", "BLOCK_I", "BLOCK_N",
            "INV_SQRT_D",
        ),
        device_functions=(
            DynamicDeviceFunctionSpec(name="compute_scores",  body_source=_COMPUTE_SCORES_BODY),
            DynamicDeviceFunctionSpec(name="apply_values",    body_source=_APPLY_VALUES_BODY),
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

    return CompiledTransformerBlockMegakernel(
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


def run_transformer_block_megakernel(
    compiled: CompiledTransformerBlockMegakernel,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    x_resid: torch.Tensor,
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
) -> torch.Tensor:
    """Launch the emitted kernel and return Y of shape (S, D_HIDDEN)."""
    H, S, D_HEAD = compiled.n_heads, compiled.seq_len, compiled.head_dim
    D_HIDDEN, I = compiled.hidden_dim, compiled.intermediate_dim
    device = q.device
    if any(t.dtype != torch.float32 for t in (q, k, v, x_resid, w_gate, w_up, w_down)):
        raise TypeError("Phase C transformer block megakernel only supports float32")
    if tuple(q.shape) != (H, S, D_HEAD): raise ValueError("Q shape mismatch")
    if tuple(k.shape) != (H, S, D_HEAD): raise ValueError("K shape mismatch")
    if tuple(v.shape) != (H, S, D_HEAD): raise ValueError("V shape mismatch")
    if tuple(x_resid.shape) != (S, D_HIDDEN): raise ValueError("X shape mismatch")
    if tuple(w_gate.shape) != (I, D_HIDDEN): raise ValueError("W_gate shape mismatch")
    if tuple(w_up.shape)   != (I, D_HIDDEN): raise ValueError("W_up shape mismatch")
    if tuple(w_down.shape) != (D_HIDDEN, I): raise ValueError("W_down shape mismatch")

    # Workspace allocations.
    p  = torch.zeros((H, S, S),       dtype=torch.float32, device=device)
    a  = torch.zeros((H, S, D_HEAD),  dtype=torch.float32, device=device)
    hi = torch.zeros((S, D_HIDDEN),   dtype=torch.float32, device=device)
    g  = torch.zeros((S, I),          dtype=torch.float32, device=device)
    u  = torch.zeros((S, I),          dtype=torch.float32, device=device)
    y  = torch.zeros((S, D_HIDDEN),   dtype=torch.float32, device=device)

    m_tiles = S // compiled.block_m
    i_tiles = I // compiled.block_i
    n_tiles = D_HIDDEN // compiled.block_n

    n_attn_tasks = H * m_tiles
    e_scores = torch.full((n_attn_tasks,),        1, dtype=torch.int32, device=device)
    e_attn   = torch.full((m_tiles,),             H, dtype=torch.int32, device=device)
    e_gate   = torch.full((m_tiles * i_tiles,),   1, dtype=torch.int32, device=device)
    e_up     = torch.full((m_tiles * i_tiles,),   1, dtype=torch.int32, device=device)

    # Initial queue: every task pre-pushed (the bodies handle the cross-stage
    # waits via spin-loops on the right event tensors).
    n_mlp_pre    = m_tiles * i_tiles
    n_mlp_down   = m_tiles * n_tiles
    total_tasks  = 2 * n_attn_tasks + 2 * n_mlp_pre + n_mlp_down
    max_queue    = total_tasks * 2

    kind_of = {fn: k for k, fn in compiled.lowering.device_function_table.items()}
    queue_pool  = torch.zeros((max_queue, 2), dtype=torch.int32, device=device)
    queue_valid = torch.zeros((max_queue,),  dtype=torch.int32, device=device)
    slot = 0
    for tid in range(n_attn_tasks):
        queue_pool[slot, 0] = tid
        queue_pool[slot, 1] = kind_of["compute_scores"]
        queue_valid[slot]   = 1
        slot += 1
    for tid in range(n_attn_tasks):
        queue_pool[slot, 0] = tid
        queue_pool[slot, 1] = kind_of["apply_values"]
        queue_valid[slot]   = 1
        slot += 1
    for tid in range(n_mlp_pre):
        queue_pool[slot, 0] = tid
        queue_pool[slot, 1] = kind_of["mlp_gate_proj"]
        queue_valid[slot]   = 1
        slot += 1
    for tid in range(n_mlp_pre):
        queue_pool[slot, 0] = tid
        queue_pool[slot, 1] = kind_of["mlp_up_proj"]
        queue_valid[slot]   = 1
        slot += 1
    for tid in range(n_mlp_down):
        queue_pool[slot, 0] = tid
        queue_pool[slot, 1] = kind_of["mlp_down_proj"]
        queue_valid[slot]   = 1
        slot += 1

    queue_head = torch.zeros((1,), dtype=torch.int32, device=device)
    queue_tail = torch.tensor([slot], dtype=torch.int32, device=device)

    inv_sqrt_d = 1.0 / (D_HEAD ** 0.5)

    compiled.kernel_callable[(compiled.sm_count,)](
        # data ptrs
        q, k, v, p, a,
        x_resid, hi,
        w_gate, w_up, w_down,
        g, u, y,
        # event ptrs (order: matches the IR's EventTensorOp declaration order)
        e_scores, e_attn, e_gate, e_up,
        # queue ptrs
        queue_pool, queue_head, queue_tail, queue_valid,
        # constexprs (order matches spec.constexpr_args)
        S, D_HEAD, D_HIDDEN, I,
        m_tiles, i_tiles, n_tiles,
        compiled.block_m, compiled.block_i, compiled.block_n,
        inv_sqrt_d,
        # implicit constexprs
        compiled.sm_count, total_tasks, max_queue,
        num_warps=compiled.lowering.launch_config["num_warps"],
        num_stages=compiled.lowering.launch_config["num_stages"],
    )
    torch.cuda.synchronize()
    return y


def reference_block(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    x_resid: torch.Tensor,
    w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
) -> torch.Tensor:
    """Faithful PyTorch reference for the fused block.

    Y = (X + flatten(SDPA(Q,K,V))) + MLP(X + flatten(SDPA(Q,K,V)))
      where MLP(z) = silu(z @ Wg.T) * (z @ Wu.T) @ Wd.T
    """
    H, S, D_HEAD = q.shape
    a = F.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=False,
    ).squeeze(0)                                  # (H, S, D_HEAD)
    a_flat = a.transpose(0, 1).reshape(S, H * D_HEAD)
    h_in = x_resid + a_flat
    gated = F.silu(h_in @ w_gate.T) * (h_in @ w_up.T)
    mlp_out = gated @ w_down.T
    return h_in + mlp_out


__all__ = [
    "CompiledTransformerBlockMegakernel",
    "build_block_event_graph",
    "compile_transformer_block_megakernel",
    "reference_block",
    "run_transformer_block_megakernel",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    H, S, D_HEAD, I = 4, 32, 32, 128
    D_HIDDEN = H * D_HEAD
    compiled = compile_transformer_block_megakernel(
        n_heads=H, seq_len=S, head_dim=D_HEAD, intermediate_dim=I,
    )
    print(f"Emitted transformer-block megakernel: {compiled.kernel_name}")
    print(f"  H={H}, S={S}, D_HEAD={D_HEAD}, D_HIDDEN={D_HIDDEN}, I={I}")
    print(f"  source = {len(compiled.kernel_source)} chars; SM_COUNT={compiled.sm_count}")
    print(f"  device functions: {sorted(compiled.lowering.device_function_table.values())}")

    torch.manual_seed(31)
    q       = torch.randn((H, S, D_HEAD),  dtype=torch.float32, device="cuda")
    k       = torch.randn((H, S, D_HEAD),  dtype=torch.float32, device="cuda")
    v       = torch.randn((H, S, D_HEAD),  dtype=torch.float32, device="cuda")
    x       = torch.randn((S, D_HIDDEN),   dtype=torch.float32, device="cuda")
    w_gate  = torch.randn((I, D_HIDDEN),   dtype=torch.float32, device="cuda") * 0.05
    w_up    = torch.randn((I, D_HIDDEN),   dtype=torch.float32, device="cuda") * 0.05
    w_down  = torch.randn((D_HIDDEN, I),   dtype=torch.float32, device="cuda") * 0.05

    got = run_transformer_block_megakernel(compiled, q, k, v, x, w_gate, w_up, w_down)
    ref = reference_block(q, k, v, x, w_gate, w_up, w_down)
    err = (got - ref).abs().max().item()
    rel = ((got - ref).abs() / (ref.abs() + 1e-6)).max().item()
    print(f"max |got - ref|       = {err:.3e}")
    print(f"max |got - ref|/|ref| = {rel:.3e}")
    assert err < 5e-3, f"transformer block diverges by {err}"
    print("PASS: emitted fused transformer-block megakernel matches PyTorch reference.")
