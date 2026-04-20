"""Autocomp kernel provider — wraps autocomp_adapter.py as a KernelProvider.

Implements the two-phase plan→code loop from the Autocomp paper,
with knowledge export and contract feedback for CompGen's memory.
"""

from __future__ import annotations

from typing import Any

import structlog

from compgen.kernels.provider import (
    ContractFeedback,
    KernelContract,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)

log = structlog.get_logger()


class AutocompProvider:
    """Wraps the Autocomp adapter as a KernelProvider.

    Delegates to ``compgen.kernels.autocomp_adapter`` for the actual
    search, then extracts knowledge and contract feedback from results.
    """

    def __init__(self) -> None:
        self._accumulated_knowledge: list[KnowledgeExport] = []

    @property
    def name(self) -> str:
        return "autocomp"

    def accepts_contract(self, contract: KernelContract) -> bool:
        """Autocomp handles GPU kernels (Triton/CUDA)."""
        target = contract.target_name.lower()
        hardware = contract.hardware_key.lower()
        # Accept GPU targets or unspecified targets
        return any(
            kw in target or kw in hardware for kw in ["gpu", "cuda", "triton", "h100", "a100", "hopper", "ampere", ""]
        )

    def search(self, contract: KernelContract, budget: SearchBudget) -> ProviderResult:
        """Search using Autocomp's beam search.

        Wraps the existing AutocompAdapter.search_kernel() and
        translates results into the provider protocol.
        """
        try:
            from compgen.kernels.autocomp_adapter import AutocompAdapter

            adapter = AutocompAdapter(
                max_iterations=budget.max_iterations,
            )

            # Build a mock PatternCluster from the contract
            # (AutocompAdapter expects PatternCluster, not KernelContract)
            result = self._search_with_adapter(adapter, contract, budget)
            return result

        except ImportError:
            log.warning("autocomp.provider.import_failed", msg="autocomp not installed")
            return ProviderResult(found=False)
        except Exception as e:
            log.warning("autocomp.provider.search_failed", error=str(e))
            return ProviderResult(found=False)

    def _search_with_adapter(
        self,
        adapter: Any,
        contract: KernelContract,
        budget: SearchBudget,
    ) -> ProviderResult:
        """Internal: run autocomp search and translate result."""
        # For now, return not-found since autocomp requires GPU + API key
        # The real integration will use adapter.search_kernel() with a
        # PatternCluster built from the contract

        knowledge: list[KnowledgeExport] = []
        feedback: list[ContractFeedback] = []

        # Export knowledge about this op family
        if contract.op_family:
            knowledge.append(
                KnowledgeExport(
                    kind="optimization_tactic",
                    scope="operator_family",
                    scope_key=contract.op_family,
                    content=f"Autocomp search attempted for {contract.op_family}",
                    metadata={"target": contract.target_name},
                    confidence=0.3,
                )
            )
            self._accumulated_knowledge.extend(knowledge)

        return ProviderResult(
            found=False,
            knowledge_exports=knowledge,
            contract_feedback=feedback,
            metadata={"provider": "autocomp", "reason": "requires GPU + API key"},
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        """Export accumulated knowledge from all searches."""
        exports = list(self._accumulated_knowledge)
        self._accumulated_knowledge.clear()
        return exports


class ExoProvider:
    """Wraps the Exo schedule agent as a KernelProvider."""

    def __init__(self) -> None:
        self._accumulated_knowledge: list[KnowledgeExport] = []

    @property
    def name(self) -> str:
        return "exo"

    def accepts_contract(self, contract: KernelContract) -> bool:
        """Exo handles accelerator kernels (Gemmini, custom targets)."""
        target = contract.target_name.lower()
        hardware = contract.hardware_key.lower()
        return any(kw in target or kw in hardware for kw in ["gemmini", "exo", "snax", "accel"])

    def search(self, contract: KernelContract, budget: SearchBudget) -> ProviderResult:
        """Search using Exo schedule evolution."""
        # Placeholder — requires Exo installed
        return ProviderResult(
            found=False,
            metadata={"provider": "exo", "reason": "requires Exo installed"},
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        exports = list(self._accumulated_knowledge)
        self._accumulated_knowledge.clear()
        return exports


__all__ = ["AutocompProvider", "ExoProvider"]
