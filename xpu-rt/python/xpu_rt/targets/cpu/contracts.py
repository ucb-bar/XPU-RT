"""CPU target-class Protocols.

CPU targets have a different shape than GPU because:

- No event tensors needed: intra-CPU sync is cheap (atomic adds
  on coherent caches), so the megakernel collapses to a serial
  task chain. The scheduling-overhead-per-task constant is
  effectively zero.
- No NVRTC / hipcc — the JIT path is a system compiler invocation
  + ``ctypes.CDLL`` of the resulting shared object.
- No tile shape selection for tensor cores — the body emitter
  picks loop unrolling + vector width based on the SIMD ISA.

So the GPU Protocols (``GpuProbe``, ``GpuRuntime``) don't apply
directly. CPU has its own narrower Protocol set.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from compgen.transforms.emit_cuda_megakernel import DeviceFunctionSource


@runtime_checkable
class CpuBodyEmitter(Protocol):
    """Emit per-op CPU body source (C++/AVX/etc.). The matcher
    treats this as it does the GPU body emitter — same per-op
    contract, different output language."""

    def preferred_tile_shape(self, *, op: str, dtype: str) -> tuple[int, int, int]:
        """CPU tile-shape hint. Typical: 32×32×32 for cache-line
        alignment on x86; 16×16×16 on ARM SVE."""
        ...

    def gemm(
        self,
        *,
        b_dim: int,
        k_dim: int,
        n_dim: int,
        n_tiles_per_row: int,
        x_buf: int,
        w_buf: int,
        out_buf: int,
        precision: str,
        tile_m: int,
        tile_n: int,
        tile_k: int,
    ) -> DeviceFunctionSource:
        """Emit a CPU GEMM body. Vendor-specific intrinsics
        (AVX-512, NEON, SVE) live in the leaf packages."""
        ...

    def elementwise(
        self,
        *,
        op: str,
        total_elems: int,
        in_bufs: tuple[int, ...],
        out_buf: int,
        tile_m: int,
        tile_n: int,
    ) -> DeviceFunctionSource: ...


@runtime_checkable
class CpuRuntime(Protocol):
    """JIT compile + dlopen + dispatch on CPU.

    No event-tensor / cooperative-launch concept — the megakernel
    on CPU is a single ``__global__``-equivalent C function that
    walks the task table sequentially. The cost model treats this
    as a near-zero scheduling overhead."""

    def compile_source(
        self,
        *,
        source: str,
        symbol_name: str,
        compile_flags: tuple[str, ...] = (),
    ) -> Any:
        """Returns a vendor-defined library handle (``ctypes.CDLL``
        of a built ``.so``)."""
        ...

    def dispatch(
        self,
        *,
        library_handle: Any,
        kernel_params: Any,
    ) -> None:
        """Call the loaded symbol with the marshalled params.
        Synchronous (CPU has no async dispatch concept here)."""
        ...
