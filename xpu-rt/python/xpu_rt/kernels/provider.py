"""Pluggable kernel provider protocol.

Kernel generators are NOT just autocomp. They are pluggable providers
(autocomp, KernelBlaster, KernelEvolve, future ones) that communicate
bidirectionally with CompGen's memory system:

1. CompGen sends a ``KernelContract`` (op, shapes, dtypes, layout, target)
2. Provider searches and returns a ``ProviderResult``
3. Provider exports ``KnowledgeExport``s (learned schedules, hardware rules)
4. Provider sends ``ContractFeedback`` (suggests contract modifications)
5. CompGen evolves contracts based on provider feedback

The contract and the generator are **codesigned** — the contract evolves
based on what generators discover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class KernelContract:
    """What CompGen asks a provider to generate.

    This is bidirectional — providers can suggest modifications via
    ContractFeedback, and CompGen updates future contracts accordingly.
    """

    region_id: str = ""
    op_family: str = ""
    input_shapes: tuple[tuple[int, ...], ...] = ()
    output_shapes: tuple[tuple[int, ...], ...] = ()
    dtypes: tuple[str, ...] = ()
    layout: str = "row_major"
    target_name: str = ""
    hardware_key: str = ""
    objective: str = "latency"
    constraints: dict[str, Any] = field(default_factory=dict)
    # Hints from previous provider feedback
    provider_hints: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchBudget:
    """Resource budget for a provider search."""

    max_iterations: int = 10
    max_time_ms: int = 60_000
    max_candidates: int = 50
    max_tokens: int = 100_000


@dataclass(frozen=True)
class KnowledgeExport:
    """What a provider learned that CompGen should know.

    Providers export their discoveries so CompGen's unified memory
    can reuse them for future tasks — even tasks routed to different
    providers.
    """

    kind: str = ""
    scope: str = "global"
    scope_key: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5


@dataclass(frozen=True)
class ContractFeedback:
    """Provider suggests contract modifications.

    Example: a provider discovers that column-major layout is 2x faster
    for this op family, and feeds that back. CompGen updates the
    KernelContract for future requests.
    """

    field: str = ""
    current_value: str = ""
    suggested_value: str = ""
    reason: str = ""
    measured_gain: float = 0.0


@dataclass(frozen=True)
class ProviderResult:
    """What a provider returns after search."""

    found: bool = False
    kernel_code: str = ""
    language: str = ""
    latency_us: float = 0.0
    correct: bool = False
    plan: str = ""
    speedup: float = 0.0
    iterations_used: int = 0
    total_candidates: int = 0
    knowledge_exports: list[KnowledgeExport] = field(default_factory=list)
    contract_feedback: list[ContractFeedback] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class KernelProvider(Protocol):
    """Protocol for pluggable kernel generation backends.

    Every kernel generator (autocomp, KernelBlaster, KernelEvolve,
    vendor tools, custom search) implements this protocol.
    """

    @property
    def name(self) -> str:
        """Provider name (e.g., 'autocomp', 'kernelblaster', 'exo')."""
        ...

    def accepts_contract(self, contract: KernelContract) -> bool:
        """Whether this provider can handle this contract.

        A provider may only support certain op families, targets,
        or hardware backends.
        """
        ...

    def search(self, contract: KernelContract, budget: SearchBudget) -> ProviderResult:
        """Search for a kernel matching the contract.

        The provider runs its internal search loop and returns the
        best result, along with knowledge exports and contract feedback.
        """
        ...

    def export_knowledge(self) -> list[KnowledgeExport]:
        """Export accumulated knowledge from this provider.

        Called after search to collect what the provider learned
        that CompGen's memory should store for future reuse.
        """
        ...


__all__ = [
    "ContractFeedback",
    "KernelContract",
    "KernelProvider",
    "KnowledgeExport",
    "ProviderResult",
    "SearchBudget",
]
