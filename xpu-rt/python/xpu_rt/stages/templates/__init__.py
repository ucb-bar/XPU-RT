"""Reusable stage templates for target dialect stacks.

These templates implement common compilation patterns that targets
can include in their stacks.  Each template provides shared infrastructure
and a plugin slot for target-specific behavior.
"""

from __future__ import annotations

from compgen.stages.templates.codegen import CodegenStage
from compgen.stages.templates.lowering import LoweringStage
from compgen.stages.templates.memory_plan import MemoryPlanStage
from compgen.stages.templates.scheduling import SchedulingStage
from compgen.stages.templates.tiling import TilingStage

__all__ = [
    "CodegenStage",
    "LoweringStage",
    "MemoryPlanStage",
    "SchedulingStage",
    "TilingStage",
]
