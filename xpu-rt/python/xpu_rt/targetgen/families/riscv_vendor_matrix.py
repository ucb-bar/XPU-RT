"""RISC-V vendor matrix extension family stack generator.

Produces: encoding → dispatch → tiling → matrix_lowering → codegen → bundle
6 stages — includes a matrix extension lowering step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.func import FuncOp, ReturnOp

from compgen.stages.bundle import BundleStage
from compgen.stages.dispatch import DispatchStage
from compgen.stages.encoding import EncodingStage
from compgen.stages.registry import TargetDialectStack
from compgen.stages.templates.codegen import CodegenStage
from compgen.stages.templates.lowering import LoweringStage
from compgen.stages.templates.tiling import TilingStage
from compgen.targetgen.hardware_spec import HardwareSpec
from compgen.targets.capability import CapabilitySpec
from compgen.targets.schema import TargetProfile


class MatrixLoweringPlugin:
    """Lower matrix ops to vendor extension instructions."""

    def __init__(self, spec: HardwareSpec) -> None:
        self._spec = spec

    @property
    def target_name(self) -> str:
        return self._spec.name

    @property
    def stage_name(self) -> str:
        return "matrix_lowering"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        pass

    def transform(self, module: ModuleOp) -> ModuleOp:
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if op.results and "matmul" in op.name.lower():
                op.attributes["compgen.matrix_ext"] = StringAttr("vendor_intrinsic")
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"matrix_lowering": "vendor_extension", "extensions": [e.name for e in self._spec.isa.extensions]}


def create_vendor_matrix_stack(spec: HardwareSpec, output_dir: str | None = None) -> TargetDialectStack:
    """Create RISC-V vendor matrix extension pipeline (6 stages)."""
    import tempfile

    bundle_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="matrix_bundle_"))
    return TargetDialectStack(
        target_name=spec.name,
        stages=[
            EncodingStage(),
            DispatchStage(),
            TilingStage(),
            LoweringStage("matrix_lowering", "Vendor matrix extension lowering"),
            CodegenStage(),
            BundleStage(output_dir=bundle_dir),
        ],
        plugins={"matrix_lowering": MatrixLoweringPlugin(spec)},
    )
