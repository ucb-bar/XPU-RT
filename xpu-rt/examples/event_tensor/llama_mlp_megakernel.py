"""Real  example: Llama / Gemma SwiGLU MLP via Event Tensor megakernel.

Llama, Gemma, smolVLA, and Qwen3 all use the same gated-MLP block:

    out = down_proj( silu(gate_proj(x)) * up_proj(x) )

The megakernel splits this into three classes of tile tasks coordinated
by Event Tensors, exactly as the ETC paper (Jin et al., MLSys '26) does
in its Qwen3 end-to-end pipeline (Section 4.3):

    Stage 1a  (gate_proj_tile)  -- one task per (m_tile, i_tile):
        G[m_rows, i_cols] = silu( X[m_rows] @ W_gate[:, i_cols] )
        notify E_gate[m_tile, i_tile]

    Stage 1b  (up_proj_tile)    -- one task per (m_tile, i_tile):
        U[m_rows, i_cols] = X[m_rows] @ W_up[:, i_cols]
        notify E_up[m_tile, i_tile]

    Stage 2   (down_proj_tile)  -- one task per (m_tile, n_tile):
        wait E_gate[m_tile, *] and E_up[m_tile, *]
        H[m_rows, :] = G[m_rows, :] * U[m_rows, :]
        Y[m_rows, n_cols] = H[m_rows, :] @ W_down[:, n_cols]

The two intermediate tensors G and U live in workspace memory; the
event tensors are the *only* synchronisation primitive.

Validation: build the IR, run StaticMegakernelSchedule, lower to Triton,
import + launch on the GPU, compare to a PyTorch reference computed with
``F.silu(x @ W_gate.T) * (x @ W_up.T) @ W_down.T``.

Shapes default to a Llama-2-7B-style block scaled down so the test runs
on any GPU; the megakernel structure is unchanged at full Llama shapes.
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
from compgen.ir.payload.passes.megakernel_static_schedule import (
    StaticMegakernelSchedule,
)
from compgen.ir.tile.lower_megakernel import (
    DeviceFunctionSpec,
    MegakernelLoweringResult,
    MegakernelLoweringSpec,
    lower_megakernel,
)


# ---------------------------------------------------------------------------
# Triton bodies for the three device functions
# ---------------------------------------------------------------------------
#
# Tensor layout (row-major):
#   X       : (M, K)            input activations
#   W_gate  : (I, K)            gate-projection weights (out_features, in_features)
#   W_up    : (I, K)            up-projection   weights
#   W_down  : (N, I)            down-projection weights
#   G_ws    : (M, I)            workspace: silu(X @ W_gate.T)
#   U_ws    : (M, I)            workspace: X @ W_up.T
#   Y       : (M, N)            output
#
#   E_gate  : (M_TILES * I_TILES,) int32 -- one counter per (m_tile, i_tile)
#   E_up    : (M_TILES * I_TILES,) int32

_GATE_PROJ_BODY = r"""
m_tile = task_id // I_TILES
i_tile = task_id %  I_TILES

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)        # (BLOCK_M,)
i_cols = i_tile * BLOCK_I + tl.arange(0, BLOCK_I)        # (BLOCK_I,)
k_idx  = tl.arange(0, K)                                  # (K,) -- assumes K fits

# X[m_rows, k_idx]  shape (BLOCK_M, K)
x_ptrs = X_ptr + m_rows[:, None] * K + k_idx[None, :]
x      = tl.load(x_ptrs)

# W_gate[i_cols, k_idx]  shape (BLOCK_I, K)
wg_ptrs = WG_ptr + i_cols[:, None] * K + k_idx[None, :]
wg      = tl.load(wg_ptrs)

# (BLOCK_M, K) @ (K, BLOCK_I) -> (BLOCK_M, BLOCK_I)
proj    = tl.dot(x, tl.trans(wg))

# silu(z) = z * sigmoid(z)
gated   = proj * tl.sigmoid(proj)

# Store into G_ws[m_rows, i_cols]
g_ptrs  = G_ptr + m_rows[:, None] * I + i_cols[None, :]
tl.store(g_ptrs, gated)

tl.atomic_add(EG_ptr + (m_tile * I_TILES + i_tile), -1)
"""


_UP_PROJ_BODY = r"""
m_tile = task_id // I_TILES
i_tile = task_id %  I_TILES

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
i_cols = i_tile * BLOCK_I + tl.arange(0, BLOCK_I)
k_idx  = tl.arange(0, K)

x_ptrs = X_ptr + m_rows[:, None] * K + k_idx[None, :]
x      = tl.load(x_ptrs)

wu_ptrs = WU_ptr + i_cols[:, None] * K + k_idx[None, :]
wu      = tl.load(wu_ptrs)

proj    = tl.dot(x, tl.trans(wu))

u_ptrs  = U_ptr + m_rows[:, None] * I + i_cols[None, :]
tl.store(u_ptrs, proj)

tl.atomic_add(EU_ptr + (m_tile * I_TILES + i_tile), -1)
"""


_DOWN_PROJ_BODY = r"""
m_tile = task_id // N_TILES
n_tile = task_id %  N_TILES

# Wait until every (m_tile, *) gate and up tile is ready.
for it in tl.static_range(0, I_TILES):
    cg = tl.atomic_or(EG_ptr + (m_tile * I_TILES + it), 0)
    while cg > 0:
        cg = tl.atomic_or(EG_ptr + (m_tile * I_TILES + it), 0)
    cu = tl.atomic_or(EU_ptr + (m_tile * I_TILES + it), 0)
    while cu > 0:
        cu = tl.atomic_or(EU_ptr + (m_tile * I_TILES + it), 0)

m_rows = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
n_cols = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
i_idx  = tl.arange(0, I)

# G_ws[m_rows, i_idx] * U_ws[m_rows, i_idx] -> (BLOCK_M, I)
g_ptrs = G_ptr + m_rows[:, None] * I + i_idx[None, :]
u_ptrs = U_ptr + m_rows[:, None] * I + i_idx[None, :]
g      = tl.load(g_ptrs)
u      = tl.load(u_ptrs)
hid    = g * u

# W_down[n_cols, i_idx]  shape (BLOCK_N, I)
wd_ptrs = WD_ptr + n_cols[:, None] * I + i_idx[None, :]
wd      = tl.load(wd_ptrs)

# (BLOCK_M, I) @ (I, BLOCK_N) -> (BLOCK_M, BLOCK_N)
y       = tl.dot(hid, tl.trans(wd))

y_ptrs  = Y_ptr + m_rows[:, None] * N + n_cols[None, :]
tl.store(y_ptrs, y)
"""


# ---------------------------------------------------------------------------
# IR construction
# ---------------------------------------------------------------------------


def build_mlp_event_graph(m_tiles: int, i_tiles: int, n_tiles: int) -> tuple[ModuleOp, GraphOp]:
    n_intermediate_events = m_tiles * i_tiles
    n_down_tasks = m_tiles * n_tiles

    block = Block()
    block.add_op(
        EventTensorOp.create(
            properties={
                "sym_name": StringAttr("EG"),
                "event_type": EventTensorTypeAttr([n_intermediate_events]),
                "wait_count": IntegerAttr(1, IntegerType(64)),
            },
        ),
    )
    block.add_op(
        EventTensorOp.create(
            properties={
                "sym_name": StringAttr("EU"),
                "event_type": EventTensorTypeAttr([n_intermediate_events]),
                "wait_count": IntegerAttr(1, IntegerType(64)),
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("gate_proj_tile"),
                "task_shape": ArrayAttr([IntegerAttr(n_intermediate_events, IntegerType(64))]),
                "out_edges": ArrayAttr(
                    [EventCoordAttr("EG", [str(k)], 1) for k in range(n_intermediate_events)],
                ),
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("up_proj_tile"),
                "task_shape": ArrayAttr([IntegerAttr(n_intermediate_events, IntegerType(64))]),
                "out_edges": ArrayAttr(
                    [EventCoordAttr("EU", [str(k)], 1) for k in range(n_intermediate_events)],
                ),
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("down_proj_tile"),
                "task_shape": ArrayAttr([IntegerAttr(n_down_tasks, IntegerType(64))]),
                "in_edges": ArrayAttr(
                    [EventCoordAttr("EG", [str(k)], 1) for k in range(n_intermediate_events)]
                    + [EventCoordAttr("EU", [str(k)], 1) for k in range(n_intermediate_events)],
                ),
            },
        ),
    )
    sm_count = max(1, min(n_intermediate_events * 2 + n_down_tasks, 16))
    graph = GraphOp(
        sym_name="llama_mlp",
        policy="static",
        sm_count=sm_count,
        body=Region([block]),
    )
    mod = ModuleOp([])
    mod.body.block.add_op(graph)
    return mod, graph


# ---------------------------------------------------------------------------
# End-to-end compile + run
# ---------------------------------------------------------------------------


@dataclass
class CompiledMLPMegakernel:
    kernel_name: str
    kernel_source: str
    kernel_callable: object
    lowering: MegakernelLoweringResult
    M: int
    K: int
    I: int
    N: int
    BLOCK_M: int
    BLOCK_I: int
    BLOCK_N: int
    sm_count: int
    max_qlen: int


def compile_mlp_megakernel(
    M: int = 32,           # batch * seq tokens
    K: int = 64,           # hidden dim (Llama-2-7B: 4096; we use 64 for fast test)
    I: int = 128,          # intermediate dim (Llama-2-7B: 11008; we use 128)
    N: int = 64,           # output dim (== K in real Llama; we keep separate for shape clarity)
    BLOCK_M: int = 16,
    BLOCK_I: int = 32,
    BLOCK_N: int = 16,
) -> CompiledMLPMegakernel:
    if M % BLOCK_M or I % BLOCK_I or N % BLOCK_N:
        raise ValueError("dims must be divisible by their block sizes")

    m_tiles = M // BLOCK_M
    i_tiles = I // BLOCK_I
    n_tiles = N // BLOCK_N

    mod, graph = build_mlp_event_graph(m_tiles, i_tiles, n_tiles)
    StaticMegakernelSchedule().run(mod)

    spec = MegakernelLoweringSpec(
        data_pointers=("X_ptr", "WG_ptr", "WU_ptr", "WD_ptr", "G_ptr", "U_ptr", "Y_ptr"),
        constexpr_args=(
            "M", "K", "I", "N",
            "BLOCK_M", "BLOCK_I", "BLOCK_N",
            "I_TILES", "N_TILES",
        ),
        device_functions=(
            DeviceFunctionSpec(name="gate_proj_tile", body_source=_GATE_PROJ_BODY),
            DeviceFunctionSpec(name="up_proj_tile",   body_source=_UP_PROJ_BODY),
            DeviceFunctionSpec(name="down_proj_tile", body_source=_DOWN_PROJ_BODY),
        ),
    )
    lowering = lower_megakernel(graph, spec=spec)
    if not lowering.kernel_source:
        raise RuntimeError(f"emitter rejected the graph: {lowering.diagnostics}")

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

    return CompiledMLPMegakernel(
        kernel_name=lowering.kernel_name,
        kernel_source=lowering.kernel_source,
        kernel_callable=kernel_callable,
        lowering=lowering,
        M=M, K=K, I=I, N=N,
        BLOCK_M=BLOCK_M, BLOCK_I=BLOCK_I, BLOCK_N=BLOCK_N,
        sm_count=int(lowering.launch_config["grid"]),
        max_qlen=max((len(q) for q in lowering.task_queue.values()), default=1),
    )


def _flatten_queue(
    compiled: CompiledMLPMegakernel, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    queue = torch.zeros(
        (compiled.sm_count, compiled.max_qlen, 2),
        dtype=torch.int32, device=device,
    )
    lens = torch.zeros((compiled.sm_count,), dtype=torch.int32, device=device)
    for sm, entries in compiled.lowering.task_queue.items():
        for slot, (tid, kind) in enumerate(entries):
            task_id_int = int(tid.split(":")[1])
            queue[sm, slot, 0] = task_id_int
            queue[sm, slot, 1] = kind
        lens[sm] = len(entries)
    return queue, lens


def run_mlp_megakernel(
    compiled: CompiledMLPMegakernel,
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Launch the emitted MLP megakernel and return Y of shape (M, N)."""
    if any(t.dtype != torch.float32 for t in (x, w_gate, w_up, w_down)):
        raise TypeError(" MLP megakernel only supports float32")
    if tuple(x.shape) != (compiled.M, compiled.K):
        raise ValueError(f"X shape {tuple(x.shape)} != ({compiled.M}, {compiled.K})")
    if tuple(w_gate.shape) != (compiled.I, compiled.K):
        raise ValueError(f"W_gate shape {tuple(w_gate.shape)} != ({compiled.I}, {compiled.K})")
    if tuple(w_up.shape) != (compiled.I, compiled.K):
        raise ValueError(f"W_up shape {tuple(w_up.shape)} != ({compiled.I}, {compiled.K})")
    if tuple(w_down.shape) != (compiled.N, compiled.I):
        raise ValueError(f"W_down shape {tuple(w_down.shape)} != ({compiled.N}, {compiled.I})")

    device = x.device
    g_ws = torch.zeros((compiled.M, compiled.I), dtype=torch.float32, device=device)
    u_ws = torch.zeros((compiled.M, compiled.I), dtype=torch.float32, device=device)
    y    = torch.zeros((compiled.M, compiled.N), dtype=torch.float32, device=device)

    m_tiles = compiled.M // compiled.BLOCK_M
    i_tiles = compiled.I // compiled.BLOCK_I
    n_tiles = compiled.N // compiled.BLOCK_N
    n_intermediate_events = m_tiles * i_tiles
    eg = torch.full((n_intermediate_events,), 1, dtype=torch.int32, device=device)
    eu = torch.full((n_intermediate_events,), 1, dtype=torch.int32, device=device)

    queue, lens = _flatten_queue(compiled, device)

    compiled.kernel_callable[(compiled.sm_count,)](
        # data ptrs
        x, w_gate, w_up, w_down, g_ws, u_ws, y,
        # event ptrs
        eg, eu,
        # queue + lens
        queue, lens,
        # constexprs
        compiled.M, compiled.K, compiled.I, compiled.N,
        compiled.BLOCK_M, compiled.BLOCK_I, compiled.BLOCK_N,
        i_tiles, n_tiles,
        # implicit constexprs
        compiled.sm_count, compiled.max_qlen,
        num_warps=compiled.lowering.launch_config["num_warps"],
        num_stages=compiled.lowering.launch_config["num_stages"],
    )
    torch.cuda.synchronize()

    if not bool(torch.all(eg == 0)) or not bool(torch.all(eu == 0)):
        raise RuntimeError(
            f"event counters did not drain (eg={eg.tolist()[:8]}..., eu={eu.tolist()[:8]}...)"
        )
    return y


def reference_mlp(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
) -> torch.Tensor:
    """PyTorch eager reference: down( silu(x @ W_gate.T) * (x @ W_up.T) ).T."""
    gated = F.silu(x @ w_gate.T)
    upped = x @ w_up.T
    hidden = gated * upped
    return hidden @ w_down.T


__all__ = [
    "CompiledMLPMegakernel",
    "build_mlp_event_graph",
    "compile_mlp_megakernel",
    "reference_mlp",
    "run_mlp_megakernel",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    M, K, I, N = 32, 64, 128, 64
    compiled = compile_mlp_megakernel(M=M, K=K, I=I, N=N)
    print(f"Emitted Llama MLP megakernel: {compiled.kernel_name}")
    print(f"  M={M}, K={K}, I={I}, N={N}, SM_COUNT={compiled.sm_count}, MAX_QLEN={compiled.max_qlen}")
    print(f"  source = {len(compiled.kernel_source)} chars")

    torch.manual_seed(42)
    x = torch.randn((M, K), dtype=torch.float32, device="cuda")
    w_gate = torch.randn((I, K), dtype=torch.float32, device="cuda") * 0.05
    w_up   = torch.randn((I, K), dtype=torch.float32, device="cuda") * 0.05
    w_down = torch.randn((N, I), dtype=torch.float32, device="cuda") * 0.05

    got = run_mlp_megakernel(compiled, x, w_gate, w_up, w_down)
    ref = reference_mlp(x, w_gate, w_up, w_down)

    err_abs = (got - ref).abs().max().item()
    err_rel = ((got - ref).abs() / (ref.abs() + 1e-6)).max().item()
    print(f"max |got - ref|       = {err_abs:.3e}")
    print(f"max |got - ref|/|ref| = {err_rel:.3e}")
    assert err_abs < 5e-3, "Llama MLP megakernel deviates from PyTorch reference"
    print("PASS: emitted Llama-style SwiGLU MLP megakernel matches PyTorch reference.")
