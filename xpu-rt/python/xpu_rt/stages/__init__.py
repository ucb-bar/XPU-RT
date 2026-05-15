"""Compilation stages framework for CompGen.

Provides the infrastructure for defining, sequencing, and executing
compilation stages with target-specific plugin support.  Each target
declares its own dialect stack (variable depth), and the registry
enforces contracts between adjacent stages.
"""

from __future__ import annotations

from compgen.stages.base import (
    CompilationStage,
    IRInvariant,
    StageContract,
    StageResult,
    TargetStagePlugin,
)
from compgen.stages.registry import PipelineResult, StageRegistry, TargetDialectStack

__all__ = [
    "CompilationStage",
    "IRInvariant",
    "PipelineResult",
    "StageContract",
    "StageRegistry",
    "StageResult",
    "TargetDialectStack",
    "TargetStagePlugin",
]
