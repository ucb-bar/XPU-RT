"""Agentic compilation loop — LLM-driven iterative optimization.

Pipeline:
    1. Analyze: NetworkAnalyzer -> pattern clusters + bottlenecks
    2. Propose: LLM suggests optimization (via prompts/)
    3. Apply: Execute in CompilerEnv (validated before execution)
    4. Verify: Check correctness (structural + differential)
    5. Profile: Benchmark on real hardware (optional)
    6. Decide: Accept/reject based on cost improvement
    7. Refine: Feed results back to LLM for next iteration

Internal modules:
    - records:  IterationRecord, CompilationResult data classes
    - prompts:  prompt-context helpers derived from Observation/history
    - core:     AgenticCompilationLoop class itself
"""

from __future__ import annotations

from compgen.agent.loop.core import AgenticCompilationLoop
from compgen.agent.loop.phased import (
    DriveLoopResult,
    PhasedDriveLoop,
    PhasePolicy,
    PhaseRunSummary,
)
from compgen.agent.loop.records import CompilationResult, IterationRecord

__all__ = [
    "AgenticCompilationLoop",
    "CompilationResult",
    "DriveLoopResult",
    "IterationRecord",
    "PhasePolicy",
    "PhaseRunSummary",
    "PhasedDriveLoop",
]
