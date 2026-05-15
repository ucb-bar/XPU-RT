"""Scheduling stage template — schedule dispatch groups to hardware.

For targets with complex scheduling (multi-device, async, pipeline
parallelism).  Assigns device placements and temporal schedules.

Reuses: solve/placement.py, solve/schedule.py, solve/memory.py.
"""

from __future__ import annotations

from pathlib import Path

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.func import FuncOp, ReturnOp

from compgen.stages.base import CompilationStage, StageContract
from compgen.targets.schema import TargetProfile

DEVICE_ATTR = "compgen.device"


class SchedulingStage(CompilationStage):
    """Dispatch scheduling stage template.

    Shared passes assign all ops to device 0 (single-device baseline).
    Target plugins use solvers for multi-device placement.
    """

    @property
    def name(self) -> str:
        return "scheduling"

    @property
    def description(self) -> str:
        return "Schedule dispatch groups to hardware devices and time slots"

    def input_contract(self) -> StageContract:
        return StageContract(stage_name="scheduling")

    def output_contract(self) -> StageContract:
        return StageContract(stage_name="scheduling")

    def shared_passes(self, module: ModuleOp, target: TargetProfile) -> ModuleOp:
        """Assign all ops to device 0 (baseline)."""
        for op in module.walk():
            if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            if op.results and DEVICE_ATTR not in op.attributes:
                op.attributes[DEVICE_ATTR] = StringAttr("device_0")
        return module

    def requirements_doc_path(self) -> Path:
        return Path(__file__).parent / "REQUIREMENTS_scheduling.md"
