"""gemm_reduce_scatter conformance workload — Paper Table 1 TP pattern.

Tensor-parallel matmul with row-parallel weights + ReduceScatter
output collation. Reference shape: ``(B=64, K=512, N=512)`` with
``world_size=2``.

Sharding:
- ``x``  is column-sharded across ranks: rank r has ``x_r`` of shape
  ``(B, K/R)``. Total of all ranks' ``x_r`` concatenated along axis 1
  recovers the full input.
- ``W``  is row-sharded across ranks: rank r has ``W_r`` of shape
  ``(K/R, N)``. Total along axis 0 recovers the full weight.

Local compute:
    ``y_partial_r = x_r @ W_r``         shape ``(B, N)``

After local compute, every rank holds a ``(B, N)`` partial sum.
Summing across ranks yields ``y_full = x @ W`` (since
``sum_r x_r @ W_r == x @ W`` by the standard column-row-shard
identity). The TP pattern then *scatters* the rows: rank r ends up
with ``y_full[r*B/R:(r+1)*B/R, :]`` of shape ``(B/R, N)``.

**v1 implementation**: each rank runs a single-GPU megakernel for
its local GEMM (32 tile-tasks across the (B, N) output, identical
in structure to decoder_layer's up_proj — just with ``K=K/R``),
then the harness orchestrates ``CudaCommGroup.allreduce_fp32_sum``
on the (B, N) partials, and each rank slices its row band. This
demonstrates the multi-GPU dispatch + NCCL primitive end-to-end on
real silicon. v2 will replace AllReduce + slice with a true
``ReduceScatter`` (less bandwidth) and integrate cross-rank Event
Tensor edges so the entire forward is one cooperative launch
across both ranks.
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

# Reference shape — chosen so each rank's local GEMM has meaningful
# arithmetic intensity (16 K-tiles × 32 fmaf per tile per thread)
# without the workload becoming so large that NCCL bootstrap
# dominates wallclock. Values match decoder_layer for cross-workload
# perf comparison.
_BATCH = 64
_K_TOTAL = 512
_N = 512
_TILE_M = 32
_TILE_N = 32
_TILE_K = 32

_TILES_PER_ROW = _N // _TILE_N  # 16
_NUM_TILES_PER_RANK = (_BATCH // _TILE_M) * _TILES_PER_ROW  # 32


class _RowParallelLinear(nn.Module):
    """Reference linear ``y = x @ W`` (no bias) for eager comparison.

    Single-rank semantics; the multi-rank sharding is handled by the
    harness's ``_workload_buffers`` branch when each rank gets its
    column-shard of x and row-shard of W.
    """

    def __init__(self) -> None:
        super().__init__()
        # nn.Linear stores W as (out, in) = (N, K_total). The per-rank
        # shard is W[:, r*K_total/R:(r+1)*K_total/R] which gives row-
        # shard semantics.
        self.linear = nn.Linear(_K_TOTAL, _N, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass(frozen=True)
class Workload:
    model: nn.Module
    sample_inputs: tuple[torch.Tensor, ...]
    build_megakernel_graph: Callable[[nn.Module, tuple[torch.Tensor, ...]], MegakernelGraph]
    device_function_sources: dict[str, DeviceFunctionSource]
    user_buffer_layout: tuple[str, ...]
    num_ranks: int

    # Multi-rank workloads need post-megakernel coordination across
    # ranks. The harness reads this flag to know it must drive an
    # AllReduce / ReduceScatter on the per-rank partial outputs.
    multi_rank_collective: str = "allreduce_sum"


def build(*, dtype: str, num_gpus: int) -> Workload:
    """Build the gemm_reduce_scatter workload.

    Args:
        dtype: ``"fp32"`` only in v1; ``"bf16"`` and ``"fp16"`` land
            once the bodies are fp16-aware.
        num_gpus: Must be ``2`` (or any even number that divides
            ``_BATCH`` and ``_K_TOTAL``). v1 hard-codes 2-rank.
    """
    if num_gpus != 2:
        raise ValueError(f"gemm_reduce_scatter v1 supports num_gpus=2 only; got {num_gpus}")
    if dtype not in ("fp32", "bf16", "fp16"):
        raise ValueError(f"unsupported dtype {dtype!r}")
    if dtype != "fp32":
        # bf16/fp16 needs body-side casting + tensor-core MMA. Filed
        # for the body-codegen workstream alongside the decoder_layer
        # perf-gate gap.
        raise NotImplementedError(
            f"gemm_reduce_scatter v1 supports fp32 only; {dtype!r} lands when tensor-core bodies do."
        )

    torch.manual_seed(0xC0CCA)
    model = _RowParallelLinear()
    sample_x = torch.randn(_BATCH, _K_TOTAL)

    return Workload(
        model=model,
        sample_inputs=(sample_x,),
        build_megakernel_graph=_build_per_rank_graph,
        device_function_sources=_per_rank_device_functions(),
        user_buffer_layout=("x_shard", "w_shard", "y_partial"),
        num_ranks=2,
        multi_rank_collective="allreduce_sum",
    )


# ---------------------------------------------------------------------------
# Per-rank megakernel graph factory
# ---------------------------------------------------------------------------


def _build_per_rank_graph(model: nn.Module, sample_inputs: tuple[torch.Tensor, ...]) -> MegakernelGraph:
    """Build the per-rank megakernel graph.

    Each rank's megakernel computes ``y_partial = x_shard @ w_shard``
    where ``x_shard`` is ``(B, K/R)`` and ``w_shard`` is ``(K/R, N)``.
    The graph has one DeviceCall (``gemm_local``) with
    ``task_shape=(_NUM_TILES_PER_RANK,)`` — 32 tile-tasks per rank.

    Cross-rank Event Tensor edges are NOT emitted in v1; the
    cross-rank coordination is the harness's
    ``CudaCommGroup.allreduce_fp32_sum`` call between rank
    megakernels. v2 replaces this with peer-mapped event tensors +
    a single cooperative launch across both ranks.
    """
    del model, sample_inputs

    # ``ev_done`` exists so the megakernel knows when each tile is
    # complete; the harness reads it back to confirm the launch
    # finished. Cross-rank summation happens via NCCL after all tiles
    # have notified.
    ev_done = EventTensor((_NUM_TILES_PER_RANK,), wait_count_default=1)

    same_cell = lambda c: (c[0],)  # noqa: E731

    calls = (
        DeviceCall(
            name="gemm_local",
            body_fn=lambda c: None,
            task_shape=(_NUM_TILES_PER_RANK,),
            out_edges=(EventEdge("ev_done", same_cell),),
        ),
    )
    return MegakernelGraph(
        name="gemm_reduce_scatter",
        calls=calls,
        event_tensors={"ev_done": ev_done},
        policy="static",
    )


# ---------------------------------------------------------------------------
# Per-rank device-function body
# ---------------------------------------------------------------------------


def _per_rank_device_functions() -> dict[str, DeviceFunctionSource]:
    """One body — a 32×32 output-tile GEMM with K=``K_TOTAL/R`` per rank.

    The body is shape-aware (uses ``K_LOCAL`` not ``K_TOTAL``) since
    each rank only sees its row-shard of the K dimension. Buffer
    layout (matches :attr:`Workload.user_buffer_layout`):

        buffers[0] = x_shard     fp32 [B, K_LOCAL]
        buffers[1] = w_shard     fp32 [N, K_LOCAL]   (row-shard of W^T)
        buffers[2] = y_partial   fp32 [B, N]
    """
    common_dims = (
        f"const int B = {_BATCH};\n"
        f"const int K_LOCAL = {_K_TOTAL // 2};\n"
        f"const int N = {_N};\n"
        f"const int TM = {_TILE_M}, TN = {_TILE_N}, TK = {_TILE_K};\n"
        f"const int TILES_PER_ROW = {_TILES_PER_ROW};\n"
    )

    body = (
        common_dims
        + r"""
const int row_tile_idx = coord_x / TILES_PER_ROW;
const int col_tile_idx = coord_x % TILES_PER_ROW;
const int row_start = row_tile_idx * TM;
const int col_start = col_tile_idx * TN;

const float *x_shard   = (const float *)buffers[0];
const float *w_shard   = (const float *)buffers[1];
float       *y_partial = (float *)buffers[2];

__shared__ float A_tile[32][33];
__shared__ float W_tile[32][33];

const int tx = threadIdx.x;
const int ty = threadIdx.y;

float acc = 0.0f;
for (int k_tile = 0; k_tile < K_LOCAL; k_tile += TK) {
    int a_row = row_start + ty;
    int a_col = k_tile + tx;
    A_tile[ty][tx] = (a_row < B && a_col < K_LOCAL)
        ? x_shard[a_row * K_LOCAL + a_col] : 0.0f;
    int w_n = col_start + tx;
    int w_k = k_tile + ty;
    W_tile[ty][tx] = (w_n < N && w_k < K_LOCAL)
        ? w_shard[w_n * K_LOCAL + w_k] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < 32; ++kk) {
        acc = fmaf(A_tile[ty][kk], W_tile[kk][tx], acc);
    }
    __syncthreads();
}
int out_row = row_start + ty;
int out_col = col_start + tx;
if (out_row < B && out_col < N) {
    y_partial[out_row * N + out_col] = acc;
}
"""
    )

    return {"gemm_local": DeviceFunctionSource(name="gemm_local", body=body)}
