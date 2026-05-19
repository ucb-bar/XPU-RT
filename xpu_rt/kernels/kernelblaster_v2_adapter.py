"""Thin adapter that runs KernelBlaster v2 in-process.

The legacy :mod:`xpu_rt.kernels.kernelblaster_adapter` spawns the
upstream CUDA-only KernelBlaster as a subprocess. This v2 adapter is
the **in-process** path: it resolves the target's
:class:`~xpu_rt.memory.target_knowledge.TargetKnowledgeCard`, picks a
generator + evaluator, drives the agent loop, and returns a
:class:`~xpu_rt.kernels.provider.ProviderResult` directly. No
subprocess, no Docker, no shell scripts. The legacy adapter stays as
an opt-in escape hatch for users with an existing KB Docker image.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from xpu_rt.kernels.kernelblaster_v2 import (
    AgentLoopConfig,
    KernelBlasterV2,
    KernelGenerator,
    KernelGeneratorAgentFile,
    KernelGeneratorLLM,
)
from xpu_rt.kernels.kernelblaster_v2.evaluators import Evaluator, MockEvaluator
from xpu_rt.kernels.kernelblaster_v2.generators import AgentFileBridge
from xpu_rt.kernels.provider import KernelContract, ProviderResult
from xpu_rt.memory import target_knowledge as tk

logger = logging.getLogger(__name__)


class KernelBlasterV2Unavailable(RuntimeError):
    """Raised when KB v2 cannot service the contract (no card, etc.)."""


# Env vars matching the existing observability convention so users only
# learn one naming scheme. ``XPU_RT_KB_V2_MODE`` picks between
# ``"agent-file"`` (default in interactive Claude Code) and ``"llm-live"``.
ENV_MODE = "XPU_RT_KB_V2_MODE"
ENV_MODEL = "XPU_RT_KB_V2_MODEL"
MODE_AGENT_FILE = "agent-file"
MODE_LLM_LIVE = "llm-live"


@dataclass
class KernelBlasterV2Adapter:
    """Resolve the target card, drive the loop, return a ProviderResult.

    Both ``generator`` and ``evaluator`` are optional — when omitted the
    adapter picks defaults at run time so a single
    :class:`KernelBlasterV2Adapter()` works for both interactive and
    headless usage:

    * generator: ``RouterAgentFile`` if an :class:`AgentFileBridge` is
      supplied; otherwise ``RouterLLM`` when ``mode="llm-live"``;
      otherwise an explicit :class:`KernelBlasterV2Unavailable` so the
      caller is forced to think about which generator they want.
    * evaluator: ``MockEvaluator`` by default — the real evaluators
      (cross-compile + sim) come online with task #10.
    """

    bridge: AgentFileBridge | None = None
    mode: str = ""  # blank = read from env
    model: str = ""
    config: AgentLoopConfig | None = None
    generator: KernelGenerator | None = None
    evaluator: Evaluator | None = None

    # ----------------------------------------------------------- entrypoint

    def search_kernel(
        self,
        contract: KernelContract,
        *,
        target_id_override: str | None = None,
    ) -> ProviderResult:
        """Run one full agent-loop pass for ``contract``."""
        target_id = target_id_override or contract.target_name
        if not target_id:
            raise KernelBlasterV2Unavailable(
                "KB v2 requires a target_name on the KernelContract or an explicit override"
            )
        if not tk.exists(target_id):
            raise KernelBlasterV2Unavailable(
                f"no target knowledge card at {tk.target_dir(target_id)}; "
                f"run `/xpu-rt-target` or the ingestion pipeline first"
            )
        card = tk.load(target_id)
        generator = self.generator or self._resolve_generator()
        evaluator = self.evaluator or MockEvaluator()
        loop = KernelBlasterV2(
            card=card,
            generator=generator,
            evaluator=evaluator,
            config=self.config or AgentLoopConfig(),
        )
        result = loop.run(contract)
        return result.to_provider_result()

    def is_available(self, *, target_id: str) -> tuple[bool, str]:
        """Cheap pre-call check; returns ``(ok, reason)``."""
        if not target_id:
            return False, "target_id missing"
        if not tk.exists(target_id):
            return False, f"no knowledge card at {tk.target_dir(target_id)}"
        return True, "ok"

    # ------------------------------------------------------------ internal

    def _resolve_generator(self) -> KernelGenerator:
        mode = (self.mode or os.environ.get(ENV_MODE) or "").strip().lower()
        # Bridge wins if supplied — that's the explicit "Claude Code is
        # already in the loop" signal.
        if self.bridge is not None:
            return KernelGeneratorAgentFile(bridge=self.bridge)
        if mode in (MODE_AGENT_FILE, "agent_file", "agent"):
            raise KernelBlasterV2Unavailable(
                "agent-file mode requires an AgentFileBridge; "
                "construct KernelBlasterV2Adapter(bridge=...) before search_kernel"
            )
        if mode in (MODE_LLM_LIVE, "llm", "gemini"):
            model = self.model or os.environ.get(ENV_MODEL) or "gemini-2.5-flash"
            return KernelGeneratorLLM(model=model)
        # Default: prefer agent-file if we're in a Claude Code session, else
        # require an explicit choice.
        raise KernelBlasterV2Unavailable(
            "No generator chosen. Either pass bridge=AgentFileBridge(...) for "
            "the Claude Code path, or set XPU_RT_KB_V2_MODE=llm-live for Gemini."
        )


__all__ = [
    "ENV_MODE",
    "ENV_MODEL",
    "MODE_AGENT_FILE",
    "MODE_LLM_LIVE",
    "KernelBlasterV2Adapter",
    "KernelBlasterV2Unavailable",
]
