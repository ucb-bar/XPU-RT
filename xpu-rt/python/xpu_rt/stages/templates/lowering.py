"""Generic lowering stage template — progressive dialect lowering.

For targets that need multi-step lowering (e.g., NPU Kernel → Schedule → ISA).
This template can be instantiated multiple times in a stack, each time with
a different name and plugin.

Reuses: ir/accel/lowering.py, ir/ukernel/lower.py.
"""

from __future__ import annotations

from pathlib import Path

from xdsl.dialects.builtin import ModuleOp

from compgen.stages.base import CompilationStage, StageContract
from compgen.targets.schema import TargetProfile


class LoweringStage(CompilationStage):
    """Generic progressive lowering stage.

    Instantiated with a custom name for each level in a target's
    dialect stack (e.g., "npu_kernel", "npu_schedule", "npu_isa").
    """

    def __init__(self, stage_name: str = "lowering", stage_description: str = "Progressive dialect lowering") -> None:
        super().__init__()
        self._stage_name = stage_name
        self._stage_description = stage_description

    @property
    def name(self) -> str:
        return self._stage_name

    @property
    def description(self) -> str:
        return self._stage_description

    def input_contract(self) -> StageContract:
        return StageContract(stage_name=self._stage_name)

    def output_contract(self) -> StageContract:
        return StageContract(stage_name=self._stage_name)

    def shared_passes(self, module: ModuleOp, target: TargetProfile) -> ModuleOp:
        """No shared passes — all lowering logic is target-specific."""
        return module

    def requirements_doc_path(self) -> Path:
        return Path(__file__).parent / "REQUIREMENTS_lowering.md"
