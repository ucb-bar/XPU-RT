"""Diamond-DAG conformance workload.

The smallest end-to-end stress test for the Phase-2/3/5 ETC pipeline:

       linear_a ─┐
                 │
  x ─┬─►        ├─► add ─► relu ─► y
     │          │
     └─► linear_b ─┘

Four `__device__` functions (``linear_a``, ``linear_b``, ``add``,
``relu``) connected by four event tensors:

    linear_a → ev_a → add (in: ev_a, ev_b)
    linear_b → ev_b /
    add → ev_add → relu → ev_done

The model is intentionally small (input ``[8, 64]`` → output ``[8, 32]``)
so a single tile per __device__ body is enough; one thread per output
element does the GEMM. Real workloads would tile, vectorise + use
Tensor Cores; for the diamond_dag conformance gate, correctness +
1-launch + atomics > 0 + 1.2× speedup is the bar — we don't need
peak-perf kernel writing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from compgen.runtime.event_tensor import EventTensor
from compgen.runtime.megakernel import (
    DeviceCall,
    EventEdge,
    MegakernelGraph,
)
from compgen.transforms.emit_cuda_megakernel import DeviceFunctionSource

# Model dimensions.
#
# Each linear is (B, IN) × (IN, OUT) → (B, OUT). The output is
# tiled into (B/TILE_M) × (OUT/TILE_N) sub-tiles; one DeviceCall per
# op + ``task_shape=(num_tiles,)`` lets the static scheduler fan
# many tile-tasks across many SMs (the paper's pattern), instead of
# each op being a single SM-bound monolith.
_BATCH = 64
_IN_DIM = 512
_OUT_DIM = 512
_TILE_M = 32
_TILE_N = 32
_TILE_K = 32

# Number of (32×32) output tiles per linear / add / relu task.
_NUM_TILES = (_BATCH // _TILE_M) * (_OUT_DIM // _TILE_N)
_TILES_PER_ROW = _OUT_DIM // _TILE_N


class _Diamond(nn.Module):
    """The diamond-DAG eager reference."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(_IN_DIM, _OUT_DIM, bias=False)
        self.b = nn.Linear(_IN_DIM, _OUT_DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a(x) + self.b(x)).relu()


@dataclass(frozen=True)
class Workload:
    """The structures :func:`_compile_and_evaluate` consumes."""

    model: nn.Module
    sample_inputs: tuple[torch.Tensor, ...]
    build_megakernel_graph: Callable[[nn.Module, tuple[torch.Tensor, ...]], MegakernelGraph]
    device_function_sources: dict[str, DeviceFunctionSource]
    user_buffer_layout: tuple[str, ...]


def build(*, dtype: str, num_gpus: int) -> Workload:
    """Build the diamond_dag workload for the given dtype.

    Args:
        dtype: ``"bf16"``, ``"fp16"``, ``"fp32"``. Lower-precision
            paths share the same eager model + same CUDA bodies; the
            harness handles dtype casting before launch.
        num_gpus: Single-GPU only. Multi-GPU TP variants live in the
            ``gemm_rs`` / ``ag_gemm`` factories (Phase 4b).

    Returns:
        :class:`Workload` ready for :func:`_compile_and_evaluate`.

    Raises:
        ValueError: ``num_gpus != 1`` (the diamond is single-GPU).
    """
    if num_gpus != 1:
        raise ValueError(f"diamond_dag is single-GPU; got num_gpus={num_gpus}")

    torch.manual_seed(0xD1A0)  # deterministic weights for goldens
    model = _Diamond()
    if dtype in ("bf16", "fp16"):
        model = model.to(getattr(torch, "bfloat16" if dtype == "bf16" else "float16"))
    sample_x = torch.randn(_BATCH, _IN_DIM)
    if dtype in ("bf16", "fp16"):
        sample_x = sample_x.to(getattr(torch, "bfloat16" if dtype == "bf16" else "float16"))

    return Workload(
        model=model,
        sample_inputs=(sample_x,),
        build_megakernel_graph=_build_diamond_graph,
        device_function_sources=_diamond_device_functions(),
        user_buffer_layout=("x", "wa", "wb", "ya", "yb", "yadd", "yout"),
    )


# ---------------------------------------------------------------------------
# MegakernelGraph factory — Event Tensor topology for the diamond
# ---------------------------------------------------------------------------


def _build_diamond_graph(model: nn.Module, sample_inputs: tuple[torch.Tensor, ...]) -> MegakernelGraph:
    """Build the tile-level event graph.

    The graph still has 4 :class:`DeviceCall`\\ s (one per logical
    op), but each carries ``task_shape=(_NUM_TILES,)`` so the
    scheduler enumerates ``_NUM_TILES`` distinct tasks per op — one
    per output tile. The event tensors are shape ``(_NUM_TILES,)``
    so each tile's edge has its own dedicated cell. Per-op tasks
    fan across many SMs; the diamond's per-tile critical path is
    ``max(linear_a_i, linear_b_i) + add_i + relu_i`` instead of the
    earlier ``max(linear_a, linear_b) + add + relu`` where each op
    was one giant single-SM task.

    With (B=64, OUT=512) and 32×32 tiles, ``_NUM_TILES = 32`` ⇒
    32 tile-tasks per op × 4 ops = 128 total tasks. cooperative-
    launch grid is still ``(sm_count, 1, 1)``; the schedule
    distributes the 128 tasks across the 188 SMs so most blocks
    actually do useful work.
    """
    del model, sample_inputs  # topology depends on dimensions, not weights
    ev_a = EventTensor((_NUM_TILES,), wait_count_default=1)
    ev_b = EventTensor((_NUM_TILES,), wait_count_default=1)
    ev_add = EventTensor((_NUM_TILES,), wait_count_default=1)
    ev_done = EventTensor((_NUM_TILES,), wait_count_default=1)

    # Each task's coord is its tile index; the EventEdge index_fn
    # passes it straight through, so tile_i's events are on cell
    # ``(i,)`` of each event tensor.
    same_cell = lambda c: (c[0],)  # noqa: E731

    calls = (
        DeviceCall(
            name="linear_a",
            body_fn=lambda c: None,
            task_shape=(_NUM_TILES,),
            out_edges=(EventEdge("ev_a", same_cell),),
        ),
        DeviceCall(
            name="linear_b",
            body_fn=lambda c: None,
            task_shape=(_NUM_TILES,),
            out_edges=(EventEdge("ev_b", same_cell),),
        ),
        DeviceCall(
            name="add_op",
            body_fn=lambda c: None,
            task_shape=(_NUM_TILES,),
            in_edges=(
                EventEdge("ev_a", same_cell),
                EventEdge("ev_b", same_cell),
            ),
            out_edges=(EventEdge("ev_add", same_cell),),
        ),
        DeviceCall(
            name="relu_op",
            body_fn=lambda c: None,
            task_shape=(_NUM_TILES,),
            in_edges=(EventEdge("ev_add", same_cell),),
            out_edges=(EventEdge("ev_done", same_cell),),
        ),
    )
    return MegakernelGraph(
        name="diamond_dag",
        calls=calls,
        event_tensors={
            "ev_a": ev_a,
            "ev_b": ev_b,
            "ev_add": ev_add,
            "ev_done": ev_done,
        },
        policy="static",
    )


# ---------------------------------------------------------------------------
# CUDA C++ device-function bodies
# ---------------------------------------------------------------------------


def _diamond_device_functions() -> dict[str, DeviceFunctionSource]:
    """One ``__device__`` body per task. Buffer layout matches
    :attr:`Workload.user_buffer_layout`:

        buffers[0] = x  (input,  fp32, [B, IN])
        buffers[1] = wa (weight a, fp32, [OUT, IN])
        buffers[2] = wb (weight b, fp32, [OUT, IN])
        buffers[3] = ya (intermediate, fp32, [B, OUT])
        buffers[4] = yb (intermediate, fp32, [B, OUT])
        buffers[5] = yadd (intermediate, fp32, [B, OUT])
        buffers[6] = yout (output, fp32, [B, OUT])

    Each body runs in a single 1024-thread block (32×32 grid). The
    block dim is set in the workload's :func:`build` so the schedule
    picks (32, 32, 1) — both GEMM bodies need the 2D thread index for
    shared-memory tiling, and the elementwise bodies happily use the
    same shape via a strided 1D loop.
    """
    common_dims = (
        f"const int B = {_BATCH};\n"
        f"const int IN = {_IN_DIM};\n"
        f"const int OUT = {_OUT_DIM};\n"
        f"const int TM = {_TILE_M}, TN = {_TILE_N}, TK = {_TILE_K};\n"
        f"const int TILES_PER_ROW = {_TILES_PER_ROW};\n"
    )

    # Each task body computes ONE 32×32 output tile, identified by
    # ``coord_x``. ``coord_x`` is the linearised tile index
    # (row_tile_idx * TILES_PER_ROW + col_tile_idx) — same row-major
    # convention the workload's static schedule produces. Single-tile
    # bodies are tiny (~5 µs each on Blackwell), letting the scheduler
    # fan all _NUM_TILES tile-tasks across distinct SMs.
    def _gemm_body(weight_buf: int, out_buf: int) -> str:
        return (
            common_dims
            + r"""
// Single-tile GEMM: this task computes the 32×32 output tile at
// (row_tile_idx, col_tile_idx) given by ``coord_x``.
const int row_tile_idx = coord_x / TILES_PER_ROW;
const int col_tile_idx = coord_x % TILES_PER_ROW;
const int row_start = row_tile_idx * TM;
const int col_start = col_tile_idx * TN;

const float *x = (const float *)buffers[0];
const float *w = (const float *)buffers[__WEIGHT_BUF__];
float       *y = (float *)buffers[__OUT_BUF__];

__shared__ float A_tile[32][33];
__shared__ float W_tile[32][33];

const int tx = threadIdx.x;
const int ty = threadIdx.y;

float acc = 0.0f;
for (int k_tile = 0; k_tile < IN; k_tile += TK) {
    int a_row = row_start + ty;
    int a_col = k_tile + tx;
    A_tile[ty][tx] = (a_row < B && a_col < IN)
        ? x[a_row * IN + a_col] : 0.0f;
    int w_n = col_start + tx;
    int w_k = k_tile + ty;
    W_tile[ty][tx] = (w_n < OUT && w_k < IN)
        ? w[w_n * IN + w_k] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < 32; ++kk) {
        acc = fmaf(A_tile[ty][kk], W_tile[kk][tx], acc);
    }
    __syncthreads();
}
int out_row = row_start + ty;
int out_col = col_start + tx;
if (out_row < B && out_col < OUT) {
    y[out_row * OUT + out_col] = acc;
}
""".replace("__WEIGHT_BUF__", str(weight_buf)).replace("__OUT_BUF__", str(out_buf))
        )

    # Elementwise bodies (add, relu) compute one element per thread
    # for their tile — block geometry (32, 32, 1) maps perfectly to
    # the 32×32 output tile.
    elementwise_prelude = common_dims + (
        "const int row_tile_idx = coord_x / TILES_PER_ROW;\n"
        "const int col_tile_idx = coord_x % TILES_PER_ROW;\n"
        "const int row = row_tile_idx * TM + threadIdx.y;\n"
        "const int col = col_tile_idx * TN + threadIdx.x;\n"
        "const int idx = row * OUT + col;\n"
        "const bool in_bounds = (row < B && col < OUT);\n"
    )

    return {
        "linear_a": DeviceFunctionSource(
            name="linear_a",
            # buffers[1] = wa, buffers[3] = ya
            body=_gemm_body(weight_buf=1, out_buf=3),
        ),
        "linear_b": DeviceFunctionSource(
            name="linear_b",
            # buffers[2] = wb, buffers[4] = yb
            body=_gemm_body(weight_buf=2, out_buf=4),
        ),
        "add_op": DeviceFunctionSource(
            name="add_op",
            body=elementwise_prelude
            + r"""
const float *ya = (const float *)buffers[3];
const float *yb = (const float *)buffers[4];
float       *yadd = (float *)buffers[5];
if (in_bounds) {
    yadd[idx] = ya[idx] + yb[idx];
}
""",
        ),
        "relu_op": DeviceFunctionSource(
            name="relu_op",
            body=elementwise_prelude
            + r"""
const float *yadd = (const float *)buffers[5];
float       *yout = (float *)buffers[6];
if (in_bounds) {
    float v = yadd[idx];
    yout[idx] = v > 0.0f ? v : 0.0f;
}
""",
        ),
    }
