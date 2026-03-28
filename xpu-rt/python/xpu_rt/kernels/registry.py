"""Kernel provider registry — dispatch contracts to providers.

The registry manages multiple kernel providers and mediates the
bidirectional communication between CompGen and providers:

1. Dispatch: route contracts to accepting providers
2. Knowledge ingestion: collect exports from all providers into memory
3. Contract evolution: apply provider feedback to update contracts
"""

from __future__ import annotations

from typing import Any

import structlog

from compgen.kernels.provider import (
    ContractFeedback,
    KernelContract,
    KernelProvider,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)

log = structlog.get_logger()


class ProviderRegistry:
    """Register and dispatch to kernel providers.

    Providers are tried in registration order. The first provider
    that accepts a contract and finds a result wins. Knowledge
    exports from ALL providers (even unsuccessful ones) are collected.
    """

    def __init__(self) -> None:
        self._providers: list[KernelProvider] = []
        self._accumulated_exports: list[KnowledgeExport] = []
        self._accumulated_feedback: list[ContractFeedback] = []

    def register(self, provider: KernelProvider) -> None:
        """Register a kernel provider."""
        self._providers.append(provider)
        log.info("provider.registered", name=provider.name)

    @property
    def provider_names(self) -> list[str]:
        """Names of all registered providers."""
        return [p.name for p in self._providers]

    def search(
        self,
        contract: KernelContract,
        budget: SearchBudget | None = None,
    ) -> ProviderResult:
        """Search for a kernel across all providers.

        Tries providers in registration order. Returns the first
        successful result. Accumulates knowledge exports and contract
        feedback from all providers that were tried.

        Args:
            contract: The kernel contract to fulfill.
            budget: Resource budget (defaults to standard budget).

        Returns:
            ProviderResult from the first successful provider,
            or an empty result if none succeed.
        """
        if budget is None:
            budget = SearchBudget()

        for provider in self._providers:
            if not provider.accepts_contract(contract):
                log.debug("provider.skip", name=provider.name, reason="does not accept contract")
                continue

            log.info("provider.search.start", name=provider.name, region=contract.region_id)

            try:
                result = provider.search(contract, budget)
            except Exception as e:
                log.warning("provider.search.error", name=provider.name, error=str(e))
                continue

            # Always collect knowledge and feedback
            self._accumulated_exports.extend(result.knowledge_exports)
            self._accumulated_feedback.extend(result.contract_feedback)

            if result.found:
                log.info(
                    "provider.search.found",
                    name=provider.name,
                    latency_us=result.latency_us,
                    speedup=result.speedup,
                )
                return result

            log.debug("provider.search.not_found", name=provider.name)

        return ProviderResult(found=False)

    def ingest_knowledge(self, memory: Any) -> int:
        """Collect all accumulated knowledge exports into CompilerMemory.

        Args:
            memory: CompilerMemory instance.

        Returns:
            Number of knowledge items ingested.
        """
        if not self._accumulated_exports:
            return 0

        exports = [
            {
                "kind": e.kind,
                "scope": e.scope,
                "scope_key": e.scope_key,
                "content": e.content,
                "summary": e.metadata.get("summary", ""),
                "confidence": e.confidence,
            }
            for e in self._accumulated_exports
        ]

        count = memory.ingest_provider_knowledge(
            provider_name="registry",
            exports=exports,
        )
        self._accumulated_exports.clear()
        return count

    def collect_feedback(self) -> list[ContractFeedback]:
        """Collect all accumulated contract feedback.

        Returns feedback and clears the internal buffer.
        Callers should apply this feedback to evolve contracts.
        """
        feedback = list(self._accumulated_feedback)
        self._accumulated_feedback.clear()
        return feedback

    def evolve_contract(
        self,
        contract: KernelContract,
        feedback: list[ContractFeedback] | None = None,
    ) -> KernelContract:
        """Apply contract feedback to produce an evolved contract.

        Args:
            contract: The original contract.
            feedback: Feedback to apply (uses accumulated if None).

        Returns:
            A new KernelContract with provider suggestions applied.
        """
        if feedback is None:
            feedback = self.collect_feedback()

        if not feedback:
            return contract

        # Build updated hints from feedback
        hints = dict(contract.provider_hints)
        for fb in feedback:
            hints[fb.field] = fb.suggested_value
            log.info(
                "contract.evolve",
                field=fb.field,
                from_=fb.current_value,
                to=fb.suggested_value,
                gain=fb.measured_gain,
            )

        # Apply layout/dtype changes directly if feedback is strong enough
        new_layout = contract.layout
        new_dtypes = contract.dtypes

        for fb in feedback:
            if fb.field == "layout" and fb.measured_gain > 0.1:
                new_layout = fb.suggested_value
            elif fb.field == "dtype" and fb.measured_gain > 0.1:
                new_dtypes = tuple(fb.suggested_value.split(","))

        return KernelContract(
            region_id=contract.region_id,
            op_family=contract.op_family,
            input_shapes=contract.input_shapes,
            output_shapes=contract.output_shapes,
            dtypes=new_dtypes,
            layout=new_layout,
            target_name=contract.target_name,
            hardware_key=contract.hardware_key,
            objective=contract.objective,
            constraints=contract.constraints,
            provider_hints=hints,
        )


__all__ = ["ProviderRegistry"]
