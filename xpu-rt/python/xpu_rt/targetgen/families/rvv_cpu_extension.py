"""RVV CPU extension family stack generator.

Produces: encoding → dispatch → tiling → codegen → bundle
Codegen targets LLVM RVV intrinsics.
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
from compgen.stages.templates.codegen import CODEGEN_BACKEND_ATTR, CodegenStage
from compgen.stages.templates.tiling import TILE_SIZES_ATTR, TilingStage
from compgen.targetgen.hardware_spec import HardwareSpec
from compgen.targets.capability import CapabilitySpec
from compgen.targets.schema import TargetProfile


class RvvTilingPlugin:
    """RVV tiling: tile to vector length."""

    def __init__(self, spec: HardwareSpec) -> None:
        self._vlen = spec.engine_geometry.vector_length_bits

    @property
    def target_name(self) -> str:
        return "rvv_cpu"

    @property
    def stage_name(self) -> str:
        return "tiling"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        pass

    def transform(self, module: ModuleOp) -> ModuleOp:
        vlen_elements = max(self._vlen // 32, 1)  # float32 elements per vector
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if op.results and op.name.startswith("linalg.") and TILE_SIZES_ATTR not in op.attributes:
                op.attributes[TILE_SIZES_ATTR] = StringAttr(str(vlen_elements))
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"tiling_strategy": f"rvv_vlen{self._vlen}"}


class RvvCodegenPlugin:
    """RVV codegen: target LLVM backend."""

    def __init__(self, spec: HardwareSpec) -> None:
        self._spec = spec

    @property
    def target_name(self) -> str:
        return "rvv_cpu"

    @property
    def stage_name(self) -> str:
        return "codegen"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        pass

    def transform(self, module: ModuleOp) -> ModuleOp:
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if op.results and CODEGEN_BACKEND_ATTR not in op.attributes:
                op.attributes[CODEGEN_BACKEND_ATTR] = StringAttr("llvm_rvv")
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"codegen_strategy": "llvm_rvv"}


def create_rvv_cpu_stack(spec: HardwareSpec, output_dir: str | Path) -> TargetDialectStack:
    """Create RVV CPU extension pipeline from spec.

    ``output_dir`` is mandatory; the bundle must land in a session-scoped
    path, never a volatile tempdir.
    """
    if output_dir is None:
        raise ValueError("create_rvv_cpu_stack requires output_dir; pass a session-scoped path")
    bundle_dir = Path(output_dir)
    return TargetDialectStack(
        target_name=spec.name,
        stages=[EncodingStage(), DispatchStage(), TilingStage(), CodegenStage(), BundleStage(output_dir=bundle_dir)],
        plugins={"tiling": RvvTilingPlugin(spec), "codegen": RvvCodegenPlugin(spec)},
    )
