"""RoCC accelerator family stack generator.

Produces: encoding → dispatch → tiling → accel_lowering → memory_plan → scheduling → bundle
7 stages — the deepest stack, with explicit memory planning and DMA scheduling.
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
from compgen.stages.templates.lowering import LoweringStage
from compgen.stages.templates.memory_plan import MemoryPlanStage
from compgen.stages.templates.scheduling import SchedulingStage
from compgen.stages.templates.tiling import TilingStage
from compgen.targetgen.hardware_spec import HardwareSpec
from compgen.targets.capability import CapabilitySpec
from compgen.targets.schema import TargetProfile


class RoccAccelLoweringPlugin:
    """Lower to accelerator dialect ops (mvin, mvout, compute)."""

    def __init__(self, spec: HardwareSpec) -> None:
        self._spec = spec

    @property
    def target_name(self) -> str:
        return self._spec.name

    @property
    def stage_name(self) -> str:
        return "accel_lowering"

    def configure(self, target: TargetProfile, capabilities: CapabilitySpec) -> None:
        pass

    def transform(self, module: ModuleOp) -> ModuleOp:
        # Mark ops with accelerator lowering decisions
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if op.results and "compgen.accel_op" not in op.attributes:
                if "matmul" in op.name.lower():
                    op.attributes["compgen.accel_op"] = StringAttr("compute")
                elif op.name.startswith("linalg."):
                    op.attributes["compgen.accel_op"] = StringAttr("fallback_cpu")
        return module

    def get_artifacts(self) -> dict[str, Any]:
        custom_instrs = list(self._spec.isa.custom_instructions.keys())
        return {"accel_lowering": "rocc", "custom_instructions": custom_instrs}


def create_rocc_stack(spec: HardwareSpec, output_dir: str | None = None) -> TargetDialectStack:
    """Create RoCC accelerator pipeline (7 stages)."""
    import tempfile

    bundle_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="rocc_bundle_"))
    return TargetDialectStack(
        target_name=spec.name,
        stages=[
            EncodingStage(),
            DispatchStage(),
            TilingStage(),
            LoweringStage("accel_lowering", "Accelerator dialect lowering"),
            MemoryPlanStage(),
            SchedulingStage(),
            BundleStage(output_dir=bundle_dir),
        ],
        plugins={"accel_lowering": RoccAccelLoweringPlugin(spec)},
    )
