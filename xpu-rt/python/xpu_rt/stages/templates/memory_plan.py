"""Memory planning stage template — allocate buffers respecting device capacity.

For targets with explicit memory management (scratchpad, DMA, buffer pools).
Computes buffer lifetimes and allocation offsets.

Reuses: solve/memory.py.
"""

from __future__ import annotations

from pathlib import Path

from xdsl.dialects.builtin import ModuleOp

from compgen.stages.base import CompilationStage, StageContract
from compgen.targets.schema import TargetProfile


class MemoryPlanStage(CompilationStage):
    """Memory planning stage template.

    Shared passes compute buffer lifetimes.  Target plugins apply
    hardware-specific allocation strategies.
    """

    @property
    def name(self) -> str:
        return "memory_plan"

    @property
    def description(self) -> str:
        return "Plan memory allocation respecting device capacity"

    def input_contract(self) -> StageContract:
        return StageContract(stage_name="memory_plan")

    def output_contract(self) -> StageContract:
        return StageContract(stage_name="memory_plan")

    def shared_passes(self, module: ModuleOp, target: TargetProfile) -> ModuleOp:
        """No-op for shared passes; all logic is target-specific."""
        return module

    def requirements_doc_path(self) -> Path:
        return Path(__file__).parent / "REQUIREMENTS_memory.md"
