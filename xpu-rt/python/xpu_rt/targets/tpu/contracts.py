"""TPU target-class Protocols (placeholder).

TPUs differ from GPUs structurally — they're tile-dataflow rather
than thread-block parallel. We expose a small Protocol surface
here and let v3/v4/v5 leaves specialize. No concrete vendor
implementation today; this exists so the matcher cascade has a
place to dispatch when a TPU target is registered (via MCP or
in-tree).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from compgen.transforms.emit_cuda_megakernel import DeviceFunctionSource


@runtime_checkable
class TpuBodyEmitter(Protocol):
    """Emit per-op body for the TPU's tile-dataflow execution.

    Output is HLO-like rather than CUDA C++. The matcher treats
    this as it does the GPU/CPU emitters — same op contract,
    different IR.
    """

    def preferred_tile_shape(self, *, op: str, dtype: str) -> tuple[int, int, int]:
        """TPU tile shapes are HBM-page-aligned (512×512 typical)."""
        ...

    def gemm(self, **kwargs: Any) -> DeviceFunctionSource: ...
    def elementwise(self, **kwargs: Any) -> DeviceFunctionSource: ...


@runtime_checkable
class TpuRuntime(Protocol):
    """XLA-side compile + dispatch."""

    def compile_source(self, **kwargs: Any) -> Any: ...
    def dispatch(self, **kwargs: Any) -> None: ...


@runtime_checkable
class TpuTopology(Protocol):
    """Cross-chip topology for multi-pod scaling. TPU-specific:
    GPU's cluster_dim doesn't apply here directly."""

    def num_chips(self) -> int: ...
    def slice_shape(self) -> tuple[int, ...]: ...
