"""Tiling stage template — apply data tiling decisions within dispatch groups.

For targets that need explicit tiling (GPU thread blocks, systolic arrays,
cache tiles).  Assigns ``compgen.tile_sizes`` attributes to compute-heavy ops.

Reuses: agent/env.py TileAction logic, ir/recipe/ops.py SetTileParams.
"""

from __future__ import annotations

from pathlib import Path

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.func import FuncOp, ReturnOp

from compgen.stages.base import CompilationStage, StageContract
from compgen.targets.schema import TargetProfile

TILE_SIZES_ATTR = "compgen.tile_sizes"


class TilingStage(CompilationStage):
    """Data tiling stage template.

    Shared passes assign default tile sizes based on op type.
    Target plugins override with hardware-optimal tile parameters.
    """

    @property
    def name(self) -> str:
        return "tiling"

    @property
    def description(self) -> str:
        return "Apply data tiling decisions within dispatch groups"

    def input_contract(self) -> StageContract:
        return StageContract(stage_name="tiling")

    def output_contract(self) -> StageContract:
        return StageContract(stage_name="tiling")

    def shared_passes(self, module: ModuleOp, target: TargetProfile) -> ModuleOp:
        """Assign default tile sizes to compute-heavy ops."""
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if not op.results:
                continue
            # Only tile linalg-like compute ops
            if op.name.startswith("linalg.") and TILE_SIZES_ATTR not in op.attributes:
                op.attributes[TILE_SIZES_ATTR] = StringAttr("default")
        return module

    def requirements_doc_path(self) -> Path:
        return Path(__file__).parent / "REQUIREMENTS_tiling.md"
