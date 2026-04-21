"""Real  example: Llama/Gemma-style attention via Event Tensor megakernel.

A two-stage tiled attention block coordinated by an Event Tensor:

    Stage 1 (compute_scores) -- one task per (head, q_tile):
        S[h, q_rows, :] = Q[h, q_rows] @ K[h].T  /  sqrt(D)
        P[h, q_rows, :] = softmax(S[h, q_rows, :])      (along key axis)
        notify E[h, q_tile]

    Stage 2 (apply_values) -- one task per (head, q_tile):
        wait E[h, q_tile]
        O[h, q_rows] = P[h, q_rows] @ V[h]

This is the actual structure of the per-decode-step attention path in
Llama, Gemma, smolVLA, Qwen3, and every other transformer the ETC paper
targets at the model level.   scope keeps it as a single attention
block (static schedule, no dynamic dispatch); the same megakernel
synthesis lifts unchanged into a Phase-C full-model template.

Validation: emit the Triton source via the CompGen pipeline, import it,
launch on the GPU, compare to ``torch.nn.functional.scaled_dot_product_attention``.
There is no hand-written GPU code here.
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
# Triton bodies for the two device functions (passed verbatim to the emitter)
# ---------------------------------------------------------------------------
#
# Memory layout:
#   Q, K, V : (H, S, D) row-major float32
#   P       : (H, S, S) row-major workspace (softmax probabilities)
#   O       : (H, S, D) row-major output
#   E       : (H * Q_TILES,) int32 event counters

_COMPUTE_SCORES_BODY = r"""
h        = task_id // Q_TILES
q_tile   = task_id %  Q_TILES

q_rows   = q_tile * Q_TILE + tl.arange(0, Q_TILE)        # (Q_TILE,)
key_cols = tl.arange(0, S)                                # (S,)
d_cols   = tl.arange(0, D)                                # (D,)

# Q[h, q_rows, :]  shape (Q_TILE, D)
q_ptrs = Q_ptr + h * (S * D) + q_rows[:, None] * D + d_cols[None, :]
q      = tl.load(q_ptrs)

# K[h, key_cols, :]  shape (S, D)
k_ptrs = K_ptr + h * (S * D) + key_cols[:, None] * D + d_cols[None, :]
k      = tl.load(k_ptrs)

# Q @ K^T -> (Q_TILE, S)
scores = tl.dot(q, tl.trans(k)) * INV_SQRT_D

# Softmax along the key (last) axis.
row_max  = tl.max(scores, axis=1)
scores   = scores - row_max[:, None]
exp_scs  = tl.exp(scores)
denom    = tl.sum(exp_scs, axis=1)
probs    = exp_scs / denom[:, None]

# Store P[h, q_rows, :]  shape (Q_TILE, S)
p_ptrs = P_ptr + h * (S * S) + q_rows[:, None] * S + key_cols[None, :]
tl.store(p_ptrs, probs)

# event.notify -- decrement the per-(h, q_tile) counter
tl.atomic_add(E_ptr + (h * Q_TILES + q_tile), -1)
"""


_APPLY_VALUES_BODY = r"""
h        = task_id // Q_TILES
q_tile   = task_id %  Q_TILES

# event.wait on E[h, q_tile]
counter = tl.atomic_or(E_ptr + (h * Q_TILES + q_tile), 0)
while counter > 0:
    counter = tl.atomic_or(E_ptr + (h * Q_TILES + q_tile), 0)

q_rows   = q_tile * Q_TILE + tl.arange(0, Q_TILE)        # (Q_TILE,)
key_rows = tl.arange(0, S)                                # (S,)
d_cols   = tl.arange(0, D)                                # (D,)

# P[h, q_rows, :]  shape (Q_TILE, S)
p_ptrs = P_ptr + h * (S * S) + q_rows[:, None] * S + key_rows[None, :]
p      = tl.load(p_ptrs)

# V[h, key_rows, :]  shape (S, D)
v_ptrs = V_ptr + h * (S * D) + key_rows[:, None] * D + d_cols[None, :]
v      = tl.load(v_ptrs)

# P @ V -> (Q_TILE, D)
out    = tl.dot(p, v)

# Store O[h, q_rows, :]
o_ptrs = O_ptr + h * (S * D) + q_rows[:, None] * D + d_cols[None, :]
tl.store(o_ptrs, out)
"""


# ---------------------------------------------------------------------------
# IR construction
# ---------------------------------------------------------------------------


def build_attention_event_graph(n_heads: int, q_tiles: int) -> tuple[ModuleOp, GraphOp]:
    """Build the attention event.graph.

    Layout:
        compute_scores: tile_num = (n_heads * q_tiles,)
        apply_values:   tile_num = (n_heads * q_tiles,)
        E:              shape    = (n_heads * q_tiles,), wait_count = 1
    """
    n_events = n_heads * q_tiles
    block = Block()
    block.add_op(
        EventTensorOp.create(
            properties={
                "sym_name": StringAttr("E"),
                "event_type": EventTensorTypeAttr([n_events]),
                "wait_count": IntegerAttr(1, IntegerType(64)),
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("compute_scores"),
                "task_shape": ArrayAttr([IntegerAttr(n_events, IntegerType(64))]),
                "out_edges": ArrayAttr(
                    [EventCoordAttr("E", [str(k)], 1) for k in range(n_events)],
                ),
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("apply_values"),
                "task_shape": ArrayAttr([IntegerAttr(n_events, IntegerType(64))]),
                "in_edges": ArrayAttr(
                    [EventCoordAttr("E", [str(k)], 1) for k in range(n_events)],
                ),
            },
        ),
    )
    sm_count = max(1, min(n_events, 16))
    graph = GraphOp(
        sym_name="attention",
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
class CompiledAttentionMegakernel:
    kernel_name: str
    kernel_source: str
    kernel_callable: object
    lowering: MegakernelLoweringResult
    n_heads: int
    seq_len: int
    head_dim: int
    q_tiles: int
    q_tile_size: int
    sm_count: int
    max_qlen: int


def compile_attention_megakernel(
    n_heads: int = 4,
    seq_len: int = 64,
    head_dim: int = 32,
    q_tile_size: int = 16,
) -> CompiledAttentionMegakernel:
    """Compile the attention megakernel end-to-end."""
    if seq_len % q_tile_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by q_tile_size ({q_tile_size})")
    q_tiles = seq_len // q_tile_size

    mod, graph = build_attention_event_graph(n_heads, q_tiles)
    StaticMegakernelSchedule().run(mod)

    spec = MegakernelLoweringSpec(
        data_pointers=("Q_ptr", "K_ptr", "V_ptr", "P_ptr", "O_ptr"),
        constexpr_args=("S", "D", "Q_TILE", "Q_TILES", "INV_SQRT_D"),
        device_functions=(
            DeviceFunctionSpec(name="compute_scores", body_source=_COMPUTE_SCORES_BODY),
            DeviceFunctionSpec(name="apply_values", body_source=_APPLY_VALUES_BODY),
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

    sm_count = int(lowering.launch_config["grid"])
    max_qlen = max((len(q) for q in lowering.task_queue.values()), default=1)

    return CompiledAttentionMegakernel(
        kernel_name=lowering.kernel_name,
        kernel_source=lowering.kernel_source,
        kernel_callable=kernel_callable,
        lowering=lowering,
        n_heads=n_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        q_tiles=q_tiles,
        q_tile_size=q_tile_size,
        sm_count=sm_count,
        max_qlen=max_qlen,
    )


def _flatten_queue(
    compiled: CompiledAttentionMegakernel, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    queue = torch.zeros(
        (compiled.sm_count, compiled.max_qlen, 2),
        dtype=torch.int32,
        device=device,
    )
    lens = torch.zeros((compiled.sm_count,), dtype=torch.int32, device=device)
    for sm, entries in compiled.lowering.task_queue.items():
        for slot, (tid, kind) in enumerate(entries):
            task_id_int = int(tid.split(":")[1])
            queue[sm, slot, 0] = task_id_int
            queue[sm, slot, 1] = kind
        lens[sm] = len(entries)
    return queue, lens


def run_attention_megakernel(
    compiled: CompiledAttentionMegakernel,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Launch the emitted attention megakernel.  ``q/k/v`` shape: (H, S, D)."""
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise TypeError("emitted attention megakernel only supports float32 in ")
    expected = (compiled.n_heads, compiled.seq_len, compiled.head_dim)
    for name, t in (("Q", q), ("K", k), ("V", v)):
        if tuple(t.shape) != expected:
            raise ValueError(f"{name} shape {tuple(t.shape)} != expected {expected}")
        if not t.is_cuda:
            raise RuntimeError("attention megakernel requires CUDA tensors")

    device = q.device
    p = torch.zeros(
        (compiled.n_heads, compiled.seq_len, compiled.seq_len),
        dtype=torch.float32, device=device,
    )
    o = torch.zeros(
        (compiled.n_heads, compiled.seq_len, compiled.head_dim),
        dtype=torch.float32, device=device,
    )
    n_events = compiled.n_heads * compiled.q_tiles
    e = torch.full((n_events,), 1, dtype=torch.int32, device=device)

    queue, lens = _flatten_queue(compiled, device)

    inv_sqrt_d = 1.0 / (compiled.head_dim ** 0.5)

    compiled.kernel_callable[(compiled.sm_count,)](
        # data pointers (Q, K, V, P workspace, O output)
        q, k, v, p, o,
        # event pointer
        e,
        # queue + lens
        queue, lens,
        # constexpr args
        compiled.seq_len, compiled.head_dim, compiled.q_tile_size,
        compiled.q_tiles, inv_sqrt_d,
        # implicit constexprs
        compiled.sm_count, compiled.max_qlen,
        num_warps=compiled.lowering.launch_config["num_warps"],
        num_stages=compiled.lowering.launch_config["num_stages"],
    )
    torch.cuda.synchronize()

    if not bool(torch.all(e == 0)):
        raise RuntimeError(f"event counters did not drain: {e.tolist()}")

    return o


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """PyTorch eager reference -- ``F.scaled_dot_product_attention``."""
    # SDPA expects (B, H, S, D); we have (H, S, D), add a batch dim.
    return F.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        is_causal=False,
    ).squeeze(0)


__all__ = [
    "CompiledAttentionMegakernel",
    "build_attention_event_graph",
    "compile_attention_megakernel",
    "reference_attention",
    "run_attention_megakernel",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    # Llama-2-7B uses head_dim=128.  We use D=32 here so the test runs
    # quickly on any GPU; the megakernel structure is identical at any
    # head_dim that's a power of two.
    H, S, D = 4, 64, 32
    compiled = compile_attention_megakernel(n_heads=H, seq_len=S, head_dim=D, q_tile_size=16)
    print(f"Emitted attention megakernel: {compiled.kernel_name}")
    print(f"  H={H}, S={S}, D={D}, Q_TILES={compiled.q_tiles}, SM_COUNT={compiled.sm_count}")
    print(f"  source = {len(compiled.kernel_source)} chars")

    torch.manual_seed(0)
    q = torch.randn((H, S, D), dtype=torch.float32, device="cuda")
    k = torch.randn((H, S, D), dtype=torch.float32, device="cuda")
    v = torch.randn((H, S, D), dtype=torch.float32, device="cuda")

    got = run_attention_megakernel(compiled, q, k, v)
    ref = reference_attention(q, k, v)

    err_abs = (got - ref).abs().max().item()
    err_rel = ((got - ref).abs() / (ref.abs() + 1e-6)).max().item()
    print(f"max |got - ref|       = {err_abs:.3e}")
    print(f"max |got - ref|/|ref| = {err_rel:.3e}")
    assert err_abs < 1e-3, "attention megakernel output deviates from torch SDPA"
    print("PASS: emitted attention megakernel matches torch.nn.functional.scaled_dot_product_attention.")
