"""decoder_layer conformance workload — Paper Fig. 11 reference.

**Scope (v1)**: the FFN portion of a transformer decoder layer.

    x → up_proj (d_model → d_ff) → relu → down_proj (d_ff → d_model)

The full paper Fig. 11 layer also has multi-head self-attention,
two LayerNorms, and two residual adds. **Multi-head attention is
deliberately out of scope for v1** — it adds 4 GEMMs (Q/K/V/O), a
softmax, and a causal mask, each with their own tile-task structure.
Landing the FFN-only path first lets us:

1. Validate the perf gate (≥1.2× speedup vs eager) on a real
   transformer-shape multi-GEMM workload.
2. Exercise the *cross-shape K-tile* dependency pattern — down_proj
   reads many up_proj tiles per K-iteration, which the diamond
   workload doesn't stress (each diamond add reads exactly one
   producer tile).

Once the FFN runs cleanly, the v2 expansion adds attention as a
separate set of DeviceCalls in front of the FFN block.

Topology (B = 64, d_model = 128, d_ff = 512, 32×32 tiles):

    up_proj:    task_shape=(_NUM_UP_TILES,)    # 32 tasks
    relu:       task_shape=(_NUM_UP_TILES,)    # 32 tasks
    down_proj:  task_shape=(_NUM_DOWN_TILES,)  # 8 tasks

    ev_up:      shape=(_NUM_UP_TILES,)         # 1 cell per up tile
    ev_relu:    shape=(_NUM_UP_TILES,)         # 1 cell per relu tile
    ev_done:    shape=(_NUM_DOWN_TILES,)       # 1 cell per down tile

    up_i        →  ev_up[i]
    relu_i      ← ev_up[i],   →  ev_relu[i]
    down_i      ← ev_relu[(i // _DOWN_TILES_PER_ROW) * _UP_TILES_PER_ROW + k]
                    for k in range(_UP_TILES_PER_ROW),
                  →  ev_done[i]

Each down task waits on ``_UP_TILES_PER_ROW`` relu cells (16 with
the default shape) — that's the K-axis dependency of the
``relu_out @ w_down`` GEMM, where the K dimension spans every tile
in the same row band of the relu output.
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

# Model dimensions — chosen so each tile-task is meaningful work and
# the cooperative-launch overhead amortises against real arithmetic,
# while staying small enough to fit comfortably in 188 SMs at 1
# block/SM occupancy.
_BATCH = 64
_D_MODEL = 128
_D_FF = 512
_TILE_M = 32
_TILE_N = 32
_TILE_K = 32

_UP_TILES_PER_ROW = _D_FF // _TILE_N  # 16
_DOWN_TILES_PER_ROW = _D_MODEL // _TILE_N  # 4
_NUM_UP_TILES = (_BATCH // _TILE_M) * _UP_TILES_PER_ROW  # 32
_NUM_DOWN_TILES = (_BATCH // _TILE_M) * _DOWN_TILES_PER_ROW  # 8


class _FfnBlock(nn.Module):
    """Reference FFN: x → up → relu → down. Same shapes as the full
    decoder layer's FFN sub-block."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(_D_MODEL, _D_FF, bias=False)
        self.down = nn.Linear(_D_FF, _D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


@dataclass(frozen=True)
class Workload:
    model: nn.Module
    sample_inputs: tuple[torch.Tensor, ...]
    build_megakernel_graph: Callable[[nn.Module, tuple[torch.Tensor, ...]], MegakernelGraph]
    device_function_sources: dict[str, DeviceFunctionSource]
    user_buffer_layout: tuple[str, ...]


def build(*, dtype: str, num_gpus: int) -> Workload:
    """Build the decoder_layer (v1: FFN-only) workload."""
    if num_gpus != 1:
        raise ValueError(f"decoder_layer is single-GPU; got num_gpus={num_gpus}")

    torch.manual_seed(0xDEC0DE)
    model = _FfnBlock()
    sample_x = torch.randn(_BATCH, _D_MODEL)
    if dtype in ("bf16", "fp16"):
        target_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        model = model.to(target_dtype)
        sample_x = sample_x.to(target_dtype)

    return Workload(
        model=model,
        sample_inputs=(sample_x,),
        build_megakernel_graph=_build_decoder_graph,
        device_function_sources=_decoder_device_functions(),
        user_buffer_layout=("x", "w_up", "w_down", "y_up", "y_relu", "y_out"),
    )


# ---------------------------------------------------------------------------
# MegakernelGraph factory
# ---------------------------------------------------------------------------


def _build_decoder_graph(model: nn.Module, sample_inputs: tuple[torch.Tensor, ...]) -> MegakernelGraph:
    del model, sample_inputs

    ev_up = EventTensor((_NUM_UP_TILES,), wait_count_default=1)
    ev_relu = EventTensor((_NUM_UP_TILES,), wait_count_default=1)
    ev_done = EventTensor((_NUM_DOWN_TILES,), wait_count_default=1)

    same_cell = lambda c: (c[0],)  # noqa: E731

    # down_tile_i depends on ev_relu cells in its row band: tile_i's
    # row_idx = i // _DOWN_TILES_PER_ROW; the k-th K-tile lives at
    # ev_relu[row_idx * _UP_TILES_PER_ROW + k].
    def _down_input_cell_for_k(k: int) -> Callable[[tuple[int, ...]], tuple[int, ...]]:
        return lambda c, _k=k: ((c[0] // _DOWN_TILES_PER_ROW) * _UP_TILES_PER_ROW + _k,)

    down_in_edges = tuple(EventEdge("ev_relu", _down_input_cell_for_k(k)) for k in range(_UP_TILES_PER_ROW))

    calls = (
        DeviceCall(
            name="up_proj",
            body_fn=lambda c: None,
            task_shape=(_NUM_UP_TILES,),
            out_edges=(EventEdge("ev_up", same_cell),),
        ),
        DeviceCall(
            name="relu_op",
            body_fn=lambda c: None,
            task_shape=(_NUM_UP_TILES,),
            in_edges=(EventEdge("ev_up", same_cell),),
            out_edges=(EventEdge("ev_relu", same_cell),),
        ),
        DeviceCall(
            name="down_proj",
            body_fn=lambda c: None,
            task_shape=(_NUM_DOWN_TILES,),
            in_edges=down_in_edges,
            out_edges=(EventEdge("ev_done", same_cell),),
        ),
    )
    return MegakernelGraph(
        name="decoder_layer",
        calls=calls,
        event_tensors={
            "ev_up": ev_up,
            "ev_relu": ev_relu,
            "ev_done": ev_done,
        },
        policy="static",
    )


# ---------------------------------------------------------------------------
# CUDA C++ device-function bodies
# ---------------------------------------------------------------------------


def _decoder_device_functions() -> dict[str, DeviceFunctionSource]:
    """Three bodies — two GEMMs with different (M, N, K) shapes and
    one elementwise relu. All use the (32, 32, 1) block geometry +
    fmaf accumulator pattern proven on diamond_dag.

    Buffer layout (matches :attr:`Workload.user_buffer_layout`):

        buffers[0] = x      (input,  fp32, [B, D_MODEL])
        buffers[1] = w_up   (weight, fp32, [D_FF,    D_MODEL])
        buffers[2] = w_down (weight, fp32, [D_MODEL, D_FF])
        buffers[3] = y_up   (intermediate, fp32, [B, D_FF])
        buffers[4] = y_relu (intermediate, fp32, [B, D_FF])
        buffers[5] = y_out  (output,       fp32, [B, D_MODEL])
    """
    common_dims = (
        f"const int B = {_BATCH};\n"
        f"const int D_MODEL = {_D_MODEL};\n"
        f"const int D_FF = {_D_FF};\n"
        f"const int TM = {_TILE_M}, TN = {_TILE_N}, TK = {_TILE_K};\n"
        f"const int UP_TILES_PER_ROW = {_UP_TILES_PER_ROW};\n"
        f"const int DOWN_TILES_PER_ROW = {_DOWN_TILES_PER_ROW};\n"
    )

    up_body = (
        common_dims
        + r"""
// up_proj: y_up = x @ w_up^T, x:[B,D_MODEL], w_up:[D_FF,D_MODEL].
// Per-tile: 32×32 output tile of y_up at coord_x.
const int row_tile_idx = coord_x / UP_TILES_PER_ROW;
const int col_tile_idx = coord_x % UP_TILES_PER_ROW;
const int row_start = row_tile_idx * TM;
const int col_start = col_tile_idx * TN;

const float *x    = (const float *)buffers[0];
const float *w_up = (const float *)buffers[1];
float       *y_up = (float *)buffers[3];

__shared__ float A_tile[32][33];
__shared__ float W_tile[32][33];

const int tx = threadIdx.x;
const int ty = threadIdx.y;

float acc = 0.0f;
for (int k_tile = 0; k_tile < D_MODEL; k_tile += TK) {
    int a_row = row_start + ty;
    int a_col = k_tile + tx;
    A_tile[ty][tx] = (a_row < B && a_col < D_MODEL)
        ? x[a_row * D_MODEL + a_col] : 0.0f;
    int w_n = col_start + tx;
    int w_k = k_tile + ty;
    W_tile[ty][tx] = (w_n < D_FF && w_k < D_MODEL)
        ? w_up[w_n * D_MODEL + w_k] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < 32; ++kk) {
        acc = fmaf(A_tile[ty][kk], W_tile[kk][tx], acc);
    }
    __syncthreads();
}
int out_row = row_start + ty;
int out_col = col_start + tx;
if (out_row < B && out_col < D_FF) {
    y_up[out_row * D_FF + out_col] = acc;
}
"""
    )

    relu_body = (
        common_dims
        + r"""
// relu: y_relu = max(0, y_up). One element per thread; the (32,32)
// block exactly covers a 32×32 output tile.
const int row_tile_idx = coord_x / UP_TILES_PER_ROW;
const int col_tile_idx = coord_x % UP_TILES_PER_ROW;
const int row = row_tile_idx * TM + threadIdx.y;
const int col = col_tile_idx * TN + threadIdx.x;
const int idx = row * D_FF + col;
if (row < B && col < D_FF) {
    const float *y_up   = (const float *)buffers[3];
    float       *y_relu = (float *)buffers[4];
    float v = y_up[idx];
    y_relu[idx] = v > 0.0f ? v : 0.0f;
}
"""
    )

    down_body = (
        common_dims
        + r"""
// down_proj: y_out = y_relu @ w_down^T, y_relu:[B,D_FF],
// w_down:[D_MODEL,D_FF]. Per-tile: 32×32 output tile of y_out at
// coord_x. K-axis spans the full D_FF (= 16 K-tiles), reading
// across the entire row band of y_relu.
const int row_tile_idx = coord_x / DOWN_TILES_PER_ROW;
const int col_tile_idx = coord_x % DOWN_TILES_PER_ROW;
const int row_start = row_tile_idx * TM;
const int col_start = col_tile_idx * TN;

const float *y_relu = (const float *)buffers[4];
const float *w_down = (const float *)buffers[2];
float       *y_out  = (float *)buffers[5];

__shared__ float A_tile[32][33];
__shared__ float W_tile[32][33];

const int tx = threadIdx.x;
const int ty = threadIdx.y;

float acc = 0.0f;
for (int k_tile = 0; k_tile < D_FF; k_tile += TK) {
    int a_row = row_start + ty;
    int a_col = k_tile + tx;
    A_tile[ty][tx] = (a_row < B && a_col < D_FF)
        ? y_relu[a_row * D_FF + a_col] : 0.0f;
    int w_n = col_start + tx;
    int w_k = k_tile + ty;
    W_tile[ty][tx] = (w_n < D_MODEL && w_k < D_FF)
        ? w_down[w_n * D_FF + w_k] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < 32; ++kk) {
        acc = fmaf(A_tile[ty][kk], W_tile[kk][tx], acc);
    }
    __syncthreads();
}
int out_row = row_start + ty;
int out_col = col_start + tx;
if (out_row < B && out_col < D_MODEL) {
    y_out[out_row * D_MODEL + out_col] = acc;
}
"""
    )

    return {
        "up_proj": DeviceFunctionSource(name="up_proj", body=up_body),
        "relu_op": DeviceFunctionSource(name="relu_op", body=relu_body),
        "down_proj": DeviceFunctionSource(name="down_proj", body=down_body),
    }
