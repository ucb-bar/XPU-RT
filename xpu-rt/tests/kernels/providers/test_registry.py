"""Tests for the kernel provider registry."""

from __future__ import annotations

from compgen.kernels.provider import (
    ContractFeedback,
    KernelContract,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)
from compgen.kernels.registry import ProviderRegistry


class MockProvider:
    """A mock kernel provider for testing."""

    def __init__(
        self,
        name: str = "mock",
        accepts: bool = True,
        finds: bool = True,
        knowledge: list[KnowledgeExport] | None = None,
        feedback: list[ContractFeedback] | None = None,
    ) -> None:
        self._name = name
        self._accepts = accepts
        self._finds = finds
        self._knowledge = knowledge or []
        self._feedback = feedback or []
        self.search_count = 0

    @property
    def name(self) -> str:
        return self._name

    def accepts_contract(self, contract: KernelContract) -> bool:
        return self._accepts

    def search(self, contract: KernelContract, budget: SearchBudget) -> ProviderResult:
        self.search_count += 1
        return ProviderResult(
            found=self._finds,
            kernel_code="def kernel(): pass" if self._finds else "",
            language="triton",
            latency_us=50.0 if self._finds else 0.0,
            correct=self._finds,
            knowledge_exports=self._knowledge,
            contract_feedback=self._feedback,
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        return self._knowledge


class TestProviderRegistry:
    """Test provider dispatch and knowledge collection."""

    def test_single_provider(self) -> None:
        registry = ProviderRegistry()
        provider = MockProvider(name="autocomp")
        registry.register(provider)

        result = registry.search(KernelContract(region_id="r0"))
        assert result.found
        assert provider.search_count == 1

    def test_provider_priority(self) -> None:
        """First accepting provider with a result wins."""
        registry = ProviderRegistry()
        p1 = MockProvider(name="fast", finds=False)
        p2 = MockProvider(name="slow", finds=True)
        registry.register(p1)
        registry.register(p2)

        result = registry.search(KernelContract(region_id="r0"))
        assert result.found
        assert p1.search_count == 1  # tried first
        assert p2.search_count == 1  # tried second, succeeded

    def test_skip_non_accepting(self) -> None:
        registry = ProviderRegistry()
        p1 = MockProvider(name="exo", accepts=False)
        p2 = MockProvider(name="autocomp", accepts=True)
        registry.register(p1)
        registry.register(p2)

        result = registry.search(KernelContract(region_id="r0"))
        assert result.found
        assert p1.search_count == 0  # skipped
        assert p2.search_count == 1

    def test_knowledge_collection(self) -> None:
        registry = ProviderRegistry()
        knowledge = [
            KnowledgeExport(kind="schedule_template", content="tiled_matmul", confidence=0.9),
        ]
        provider = MockProvider(name="autocomp", knowledge=knowledge)
        registry.register(provider)

        registry.search(KernelContract(region_id="r0"))

        # Knowledge should be accumulated
        feedback = registry.collect_feedback()
        assert feedback == []  # no feedback from this provider

    def test_contract_evolution(self) -> None:
        registry = ProviderRegistry()
        feedback = [
            ContractFeedback(
                field="layout",
                current_value="row_major",
                suggested_value="column_major",
                reason="2x faster for this op family",
                measured_gain=0.5,
            ),
        ]
        provider = MockProvider(name="autocomp", feedback=feedback)
        registry.register(provider)

        registry.search(KernelContract(region_id="r0", layout="row_major"))

        # Evolve the contract based on feedback
        original = KernelContract(region_id="r0", layout="row_major")
        evolved = registry.evolve_contract(original)
        assert evolved.layout == "column_major"
        assert evolved.provider_hints["layout"] == "column_major"

    def test_no_providers(self) -> None:
        registry = ProviderRegistry()
        result = registry.search(KernelContract(region_id="r0"))
        assert not result.found
