"""CUDA-specific layout resolver.

Specializes layout encodings for NVIDIA GPU targets with tensor cores.
Maps generic tiled encodings to concrete pack specifications matching
MMA (matrix multiply-accumulate) hardware tile shapes.

This resolver would be contributed by the ``cuda_tile`` extension pack
via the ``tile_layout_generation`` aperture. For now, it is built-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compgen.ir.layout.attrs import PackSpecAttr


# MMA tile shapes for common GPU generations
_MMA_TILE_SHAPES: dict[str, dict[str, list[int]]] = {
    # A100 / H100 tensor core tiles
    "tensor_core": {
        "lhs": [128, 64],
        "rhs": [64, 128],
        "out": [128, 128],
    },
    # Fallback for generic CUDA
    "cuda_core": {
        "lhs": [64, 32],
        "rhs": [32, 64],
        "out": [64, 64],
    },
}


@dataclass(frozen=True)
class CudaLayoutResolver:
    """CUDA GPU layout resolver for tensor-core targets.

    Specializes tiled encodings into MMA-friendly pack specifications.
    Uses tensor-core tile shapes for matmul operands and falls back
    to cache-friendly tiles for other ops.
    """

    def specialize(
        self,
        encoding_str: str,
        target_caps: Any,
    ) -> PackSpecAttr | None:
        """Specialize a layout encoding for CUDA targets.

        Handles:
        - ``tiled_MxN``: Use MMA tile shapes for matmul operands.
        - ``rowmajor``/``colmajor``: No specialization needed.
        - Others: Return None (keep generic).
        """
        if not encoding_str.startswith("tiled"):
            return None

        # Determine if target has tensor cores
        has_tc = False
        if target_caps is not None:
            op_caps = getattr(target_caps, "op_capabilities", {})
            if op_caps:
                has_tc = True
            # Also check for tensor_core feature
            features = getattr(target_caps, "metadata", {}).get("features", [])
            if isinstance(features, (list, tuple)):
                has_tc = has_tc or "tensor_core" in features

        tile_family = "tensor_core" if has_tc else "cuda_core"
        tiles = _MMA_TILE_SHAPES.get(tile_family, _MMA_TILE_SHAPES["cuda_core"])

        # Parse tile dimensions from encoding string
        # Format: tiled_MxN (e.g., tiled_128x64)
        parts = encoding_str.replace("tiled_", "").split("x")
        if len(parts) >= 2:
            try:
                m, n = int(parts[0]), int(parts[1])
                return PackSpecAttr(
                    inner_tiles=[m, n],
                    outer_perm=[0, 1],
                    padding_value="zero",
                )
            except ValueError:
                pass

        # Default: use LHS tile shape
        inner = tiles["lhs"]
        return PackSpecAttr(
            inner_tiles=inner,
            outer_perm=[0, 1],
            padding_value="zero",
        )

    def materialize(self, specialized: PackSpecAttr) -> dict[str, Any]:
        """Return CUDA-specific materialization metadata."""
        inner = [
            a.value.data if hasattr(a, "value") else 0
            for a in specialized.inner_tiles.data
        ]
        perm = [
            a.value.data if hasattr(a, "value") else 0
            for a in specialized.outer_perm.data
        ]
        return {
            "inner_tiles": inner,
            "outer_perm": perm,
            "padding": specialized.padding_value.data,
            "backend": "cuda",
            "mma_compatible": True,
        }


__all__ = ["CudaLayoutResolver"]
