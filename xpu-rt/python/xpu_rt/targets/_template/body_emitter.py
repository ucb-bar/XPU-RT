"""Template BodyEmitter — fill in for your target's per-op kernels.

Every emit returns a ``DeviceFunctionSource`` (universal IR) — the
matcher treats CUDA C, portable C, HIP, SPIR-V, HLO, etc. all
identically.
"""

from __future__ import annotations

from compgen.transforms.emit_cuda_megakernel import DeviceFunctionSource


class TemplateBodyEmitter:
    """Replace with ``YourArchBodyEmitter``."""

    def preferred_tile_shape(self, *, op: str, dtype: str) -> tuple[int, int, int]:
        """Tile shape this target wants for the given op + dtype.
        Used by the matcher's divisibility checks. NVIDIA Blackwell
        returns ``(64, 64, 16)`` for cuBLASDx; CPU returns
        ``(32, 32, 32)``."""
        del op, dtype
        return (32, 32, 32)

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
        """Emit your target's GEMM source. Buffer-index convention
        and tile-shape parameters come from the matcher; the emit
        is per-task."""
        raise NotImplementedError("Template — fill this in for your arch")

    def elementwise(
        self,
        *,
        op: str,
        total_elems: int,
        in_bufs: tuple[int, ...],
        out_buf: int,
        tile_m: int,
        tile_n: int,
    ) -> DeviceFunctionSource:
        """Emit elementwise body (relu, add, gelu, ...). Tile-aware
        loop pattern — each thread / SIMD lane covers
        ``ceil(tile_m * tile_n / lanes)`` elements."""
        raise NotImplementedError("Template — fill this in for your arch")
