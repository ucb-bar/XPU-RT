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
class RuntimeCapabilities:
    """What math runtime the target makes available to emitted kernels.

    Mirrors the spec-level :class:`RuntimeMathSpec`. Surfaced to
    providers via ``KernelContract.runtime`` so a provider's
    ``accepts_contract`` / ``search`` can branch on what's reachable
    from the target's link unit:

    * ``runtime.has_libm`` → emit ``__builtin_expf`` / ``sqrtf`` / …
    * ``"mu_fexp" in runtime.intrinsics`` → cast to f16, use intrinsic.
    * neither → emit polynomial fallback (or reject).
    """

    has_libm: bool = False
    has_libc: bool = False
    intrinsics: tuple[str, ...] = ()


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
    # Target runtime capabilities (REQ-025) — what libm / intrinsics
    # the emitted kernel can call. Defaults to "nothing available"
    # so providers fail closed when the spec says nothing.
    runtime: RuntimeCapabilities = field(default_factory=RuntimeCapabilities)


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
class DispatchGeometry:
    """How a kernel wants to be launched on its target.

    Hand-tuned kernels know the right ``num_warps`` /
    ``threadblock_shape`` / ``grid_shape`` for a given input size; the
    consumer (pack composer / runtime ABI) consumes these to issue a
    matching dispatch. SIMT targets typically populate ``num_warps``
    and ``threadblock_shape``; CUDA-style backends populate ``grid_shape``.
    """

    num_warps: int = 1
    threadblock_shape: tuple[int, ...] = (1,)
    grid_shape: tuple[int, ...] = (1,)


@dataclass(frozen=True)
class ProviderResult:
    """What a provider returns after search.

    Attributes:
        found: True when the provider produced a usable kernel.
        kernel_code: Primary source text (also used as the
            ``index.json`` entry's primary file).
        language: Free-form language tag — drives the bundle file
            extension (``"triton"``, ``"cuda"``, ``"cpp"``, …).
        emit_mode: Shape of the emitted source.
            ``"compute_callback"`` (default): a function with a known
            signature that the consuming pack composer wraps with
            ``main`` / data / dispatch.
            ``"self_contained"``: a complete translation unit with
            its own ``main``, dispatch geometry, and data — the
            consumer concatenates and compiles without wrapping.
        dispatch_geometry: Optional dispatch shape. When set, the
            consumer's runtime call should honour these
            (``num_warps`` / threadblock / grid) instead of guessing.
        kernel_files: Optional multi-file bundle. When set, every
            entry is written into ``bundle/generated_kernels/<provider>/<op>/``
            as a directory; the canonical entry point is the file
            whose key matches ``<op>.<ext>``. Header files, helper
            ``.inc``s, lookup tables, etc. live here.
        expected_inputs: Optional symbol-layout map describing the
            input symbols the (typically self-contained) kernel
            expects. The bundle materializes these as a side
            ``<op>.data.h`` so the pack composer can ``#include`` it
            without inventing names. See REQ-016 for schema.
        latency_us / correct / plan / speedup / iterations_used /
        total_candidates: search-loop diagnostics.
        knowledge_exports / contract_feedback: bidirectional
        knowledge channel — see :class:`KnowledgeExport` /
        :class:`ContractFeedback`.
        metadata: free-form provider-specific dict.
    """

    found: bool = False
    kernel_code: str = ""
    language: str = ""
    emit_mode: str = "compute_callback"
    dispatch_geometry: DispatchGeometry | None = None
    kernel_files: dict[str, str] | None = None
    expected_inputs: dict[str, dict[str, Any]] | None = None
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

    Optional attributes (read with ``getattr(p, name, default)``):

    - ``priority: int`` (default 0) — when multiple providers accept
      the same contract, the highest-priority one wins. Lets a pack
      ship a hand-tuned fast path (priority=10) alongside a generic
      fallback (priority=0) without depending on entry-point load order.
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
    "DispatchGeometry",
    "KernelContract",
    "KernelProvider",
    "KnowledgeExport",
    "ProviderResult",
    "RuntimeCapabilities",
    "SearchBudget",
]
