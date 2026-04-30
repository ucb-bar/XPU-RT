"""``CudaTileReferenceAdapter`` — in-tree reference for the cuda_tile dialect.

Compose ``lower_to_cuda_tile`` (lowering) + ``emit_cuda_tile_artifact``
(bundle) into a :class:`VendorDialectAdapter`. Auto-builds its
descriptor from the in-tree spec — callers don't need to load YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from compgen.extensions.vendor_dialect.adapter import LoweringResult, VendorDialectAdapter
from compgen.extensions.vendor_dialect.builtins.cuda_tile._descriptor import build_descriptor
from compgen.extensions.vendor_dialect.builtins.cuda_tile.bundle import emit_cuda_tile_artifact
from compgen.extensions.vendor_dialect.builtins.cuda_tile.lowering import lower_to_cuda_tile
from compgen.targets.backend import CompiledArtifact


class CudaTileReferenceAdapter(VendorDialectAdapter):
    """Reference cuda_tile adapter bundled with CompGen.

    See :mod:`compgen.extensions.vendor_dialect.builtins.cuda_tile`
    module docstring for the design rationale (in-tree mirror of the
    bridge-validated bwell-side ``compgen_cuda_tile`` package).

    This adapter has no kernel provider — the FFN matmul-relu-matmul
    template is hand-authored, not LLM-driven. Real op-family lowering
    via a kernel provider is the bwell-side track.
    """

    def __init__(self) -> None:
        super().__init__(build_descriptor())

    def capabilities(self) -> dict[str, Any]:
        base = super().capabilities()
        base.update(
            {
                "supported_op_types": ["linear", "relu", "ffn"],
                "supported_dtypes": ["fp32"],
                "target_archs": ["nvidia-blackwell", "sm_100"],
                "performance_profile": "reference-single-tile",
                "source": "in-tree-builtin",
                "in_tree_reference": True,
                "validated_against": "bridge#144",
            }
        )
        return base

    def lower_payload(
        self,
        payload_mlir: str,
        *,
        output_dir: str | Path,
        options: dict[str, Any] | None = None,
    ) -> LoweringResult:
        return lower_to_cuda_tile(
            payload_mlir,
            descriptor=self.descriptor,
            kernel_provider=self.kernel_provider(),
            output_dir=Path(output_dir),
            options=options or {},
        )

    def emit_artifact(
        self,
        lowering: LoweringResult,
        *,
        output_dir: str | Path,
        options: dict[str, Any] | None = None,
    ) -> CompiledArtifact:
        return emit_cuda_tile_artifact(
            lowering,
            descriptor=self.descriptor,
            output_dir=Path(output_dir),
            options=options or {},
        )


def make_adapter() -> CudaTileReferenceAdapter:
    """Construct a fresh reference adapter — convenient factory.

    Used by :mod:`compgen.extensions.vendor_dialect.builtins` for
    registration and by tests for isolated instances.
    """
    return CudaTileReferenceAdapter()


__all__ = ["CudaTileReferenceAdapter", "make_adapter"]
