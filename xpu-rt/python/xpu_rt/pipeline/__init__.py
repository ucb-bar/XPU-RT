"""CompGen end-to-end pipeline driver.

Entry point: :func:`compgen.pipeline.driver.compile_through_pipeline`.
Runs all 22 Wave 1-6 passes in the right order, returning an xDSL
``ModuleOp`` + an ``ExecutionPlan`` + a summary of every pass that
fired.

Modelled on hexagon-mlir's ``LinalgToLLVMPass.cpp`` pass-pipeline
orchestration.
"""

from __future__ import annotations

from compgen.pipeline.cache import PipelineCache, PipelineCacheStats
from compgen.pipeline.differential import DiffReport, compile_and_diff
from compgen.pipeline.driver import (
    PipelineResult,
    PipelineStageReport,
    compile_through_pipeline,
)

__all__ = [
    "DiffReport",
    "PipelineCache",
    "PipelineCacheStats",
    "PipelineResult",
    "PipelineStageReport",
    "compile_and_diff",
    "compile_through_pipeline",
]
