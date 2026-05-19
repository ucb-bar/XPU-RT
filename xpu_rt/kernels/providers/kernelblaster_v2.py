"""KernelBlaster v2 as a first-class :class:`KernelProvider`.

Sibling to the legacy :mod:`xpu_rt.kernels.providers.kernelblaster`
provider. Where the legacy provider gates on CUDA targets and shells
out to a Docker image, this v2 provider:

* gates on the existence of a per-target knowledge card,
* runs the propose → evaluate → repair loop in-process,
* prefers the Claude Code (agent-file) generator when a bridge is
  supplied via env or constructor, falls back to the budget-gated
  Gemini generator when explicitly opted in.

The two providers can coexist in the registry. When both could service
a contract, the v2 provider wins by ``priority`` because the in-process
path is dramatically faster and free under the agent-file mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from xpu_rt.kernels.kernelblaster_v2.generators import AgentFileBridge
from xpu_rt.kernels.kernelblaster_v2_adapter import (
    KernelBlasterV2Adapter,
    KernelBlasterV2Unavailable,
)
from xpu_rt.kernels.provider import (
    KernelContract,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)
from xpu_rt.memory import target_knowledge as tk

logger = logging.getLogger(__name__)


@dataclass
class KernelBlasterV2Provider:
    """v2 provider wired to :class:`KernelBlasterV2Adapter`."""

    # Above legacy KB (90) so the v2 path wins when both apply.
    priority: int = 95

    adapter: KernelBlasterV2Adapter | None = None
    _accumulated_knowledge: list[KnowledgeExport] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "kernelblaster_v2"

    def _resolve_adapter(self) -> KernelBlasterV2Adapter:
        if self.adapter is None:
            self.adapter = KernelBlasterV2Adapter()
        return self.adapter

    def accepts_contract(self, contract: KernelContract) -> bool:
        target_id = contract.target_name
        if not target_id:
            return False
        # Any target with a knowledge card is fair game.
        return tk.exists(target_id)

    def search(
        self,
        contract: KernelContract,
        budget: SearchBudget | None = None,
    ) -> ProviderResult:
        adapter = self._resolve_adapter()
        try:
            result = adapter.search_kernel(contract)
        except KernelBlasterV2Unavailable as exc:
            logger.info("kernelblaster_v2: declining contract (%s)", exc)
            return ProviderResult(found=False, metadata={"declined_reason": str(exc)})
        # Stash exports for later flushing.
        self._accumulated_knowledge.extend(result.knowledge_exports)
        return result

    def export_knowledge(self) -> list[KnowledgeExport]:
        out = list(self._accumulated_knowledge)
        self._accumulated_knowledge.clear()
        return out


__all__ = ["KernelBlasterV2Provider"]
