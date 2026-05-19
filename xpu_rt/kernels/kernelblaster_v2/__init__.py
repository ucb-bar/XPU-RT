"""KernelBlaster v2 — XPU-RT-native, contract-driven kernel generator.

Replaces the upstream KernelBlaster CUDA-RL loop with a pipeline that:

* takes a typed :class:`~xpu_rt.kernels.provider.KernelContract` as input
  (op family, archetype, dtypes, layout, target),
* reads the per-target
  :class:`~xpu_rt.memory.target_knowledge.TargetKnowledgeCard` to bring
  ISA / intrinsic / exemplar / lesson context into the prompt,
* drives a generic propose → evaluate → repair loop where the
  *generator* and the *evaluator* are both pluggable protocols,
* persists everything it learns (lessons.jsonl, strategies.json,
  exemplars/) back into the same Target Card so the next run starts
  from a richer base.

Two generator backends ship out of the box, mirroring the ingestion
router design:

* :class:`KernelGeneratorAgentFile` — Claude Code in the loop, no Gemini
  spend. Default for interactive flows driven from a skill.
* :class:`KernelGeneratorLLM` — Gemini-backed, gated by
  :func:`xpu_rt.observability.gemini_usage.check_pre_call` so the $100
  cumulative cap is respected.

The legacy CUDA path in
:mod:`xpu_rt.kernels.kernelblaster_adapter` is left untouched as an
opt-in escape hatch for users with an existing KB Docker image.
"""

from __future__ import annotations

from xpu_rt.kernels.kernelblaster_v2.agent_loop import (
    AgentLoopConfig,
    AgentLoopResult,
    Candidate,
    KernelBlasterV2,
)
from xpu_rt.kernels.kernelblaster_v2.contract_state import StateVector, derive_state
from xpu_rt.kernels.kernelblaster_v2.generators import (
    AgentFileBridge,
    KernelGenerator,
    KernelGeneratorAgentFile,
    KernelGeneratorLLM,
    KernelGeneratorMock,
    ProposeRequest,
    ProposeResponse,
)
from xpu_rt.kernels.kernelblaster_v2.lesson_writer import LessonWriter
from xpu_rt.kernels.kernelblaster_v2.prompt_builder import PromptBuilder
from xpu_rt.kernels.kernelblaster_v2.strategy_db import StrategyDB, StrategyEntry

__all__ = [
    "AgentFileBridge",
    "AgentLoopConfig",
    "AgentLoopResult",
    "Candidate",
    "KernelBlasterV2",
    "KernelGenerator",
    "KernelGeneratorAgentFile",
    "KernelGeneratorLLM",
    "KernelGeneratorMock",
    "LessonWriter",
    "PromptBuilder",
    "ProposeRequest",
    "ProposeResponse",
    "StateVector",
    "StrategyDB",
    "StrategyEntry",
    "derive_state",
]
