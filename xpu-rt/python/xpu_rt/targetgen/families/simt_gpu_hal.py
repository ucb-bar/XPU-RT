"""SIMT GPU family stack generator.

Produces: encoding → dispatch → tiling → codegen → bundle
Same pattern as cuda_gpu.py but parameterized by HardwareSpec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp

from compgen.stages.bundle import BundleStage
from compgen.stages.dispatch import DispatchStage
from compgen.stages.encoding import EncodingStage
from compgen.stages.encoding.stage import ENCODING_ATTR
from compgen.stages.registry import TargetDialectStack
from compgen.stages.templates.codegen import CODEGEN_BACKEND_ATTR, CodegenStage
from compgen.stages.templates.tiling import TilingStage
from compgen.targetgen.hardware_spec import HardwareSpec
from compgen.targets.capability import CapabilitySpec
from compgen.targets.schema import TargetProfile


class GpuEncodingPlugin:
    """GPU encoding: prefer tiled layouts for tensor core ops."""

    def __init__(self, spec: HardwareSpec) -> None:
        self._spec = spec

    @property
    def target_name(self) -> str:
        return self._spec.name

    @property
    def stage_name(self) -> str:
        return "encoding"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        pass

    def transform(self, module: ModuleOp) -> ModuleOp:
        tile_str = "row_major"
        if self._spec.engine_geometry.tiles:
            dims = self._spec.engine_geometry.tiles[0].dimensions
            tile_str = f"tiled_{'x'.join(str(d) for d in dims)}"
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if any(isinstance(r.type, TensorType) for r in op.results):
                if "matmul" in op.name.lower():
                    op.attributes[ENCODING_ATTR] = StringAttr(tile_str)
                elif ENCODING_ATTR not in op.attributes:
                    op.attributes[ENCODING_ATTR] = StringAttr("row_major")
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"encoding_strategy": "gpu_tiled"}


class GpuCodegenPlugin:
    """GPU codegen: assign Triton backend."""

    def __init__(self, spec: HardwareSpec) -> None:
        self._spec = spec

    @property
    def target_name(self) -> str:
        return self._spec.name

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
                op.attributes[CODEGEN_BACKEND_ATTR] = StringAttr("triton")
        return module

    def get_artifacts(self) -> dict[str, Any]:
        return {"codegen_strategy": "gpu_triton"}


def create_gpu_stack(spec: HardwareSpec, output_dir: str | None = None) -> TargetDialectStack:
    """Create SIMT GPU compilation pipeline from spec."""
    import tempfile

    bundle_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="gpu_bundle_"))
    return TargetDialectStack(
        target_name=spec.name,
        stages=[EncodingStage(), DispatchStage(), TilingStage(), CodegenStage(), BundleStage(output_dir=bundle_dir)],
        plugins={"encoding": GpuEncodingPlugin(spec), "codegen": GpuCodegenPlugin(spec)},
    )
