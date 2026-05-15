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

    Phase D / adds two fields:

    * ``kind`` — typed feedback category. The typed-allowlist
      auto-applies entries whose ``kind`` is one of
      ``{layout_swap, dtype_widen, accumulator_widen,
      alignment_request, fast_math_opt_in}``. Empty ``kind`` is a
      backward-compatible signal — 's classifier infers it from
      ``field`` via a small heuristic.
    * ``applies_when`` — a free-text predicate the agent can read
      to decide whether the suggestion is conditional (e.g.
      ``"K >= 64"``). Currently informational; the predicate
      DSL can later evaluate it programmatically.
    """

    field: str = ""
    current_value: str = ""
    suggested_value: str = ""
    reason: str = ""
    measured_gain: float = 0.0
    kind: str = ""
    applies_when: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "reason": self.reason,
            "measured_gain": self.measured_gain,
            "kind": self.kind,
            "applies_when": self.applies_when,
        }


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


# ===========================================================================
# Phase D / two-stage provider protocol: bid → fulfill
# ===========================================================================
# Today's provider interface has one method that performs codegen
# (``search`` / ``ProviderResult``). The Phase D auction needs
# a *cheap* pre-codegen estimate so it can pick the top-K bidders to
# actually invoke. introduces ``BidPreview`` for that:
#   * Every provider may implement an optional ``bid(contract_v3)
#     -> BidPreview`` method.
#   * Legacy providers without ``bid()`` are bridged via
#     :func:`compgen.kernels.registry.compute_bid`, which returns a
#     low-confidence placeholder.
#   * The auction ranks by ``perf_estimate_us / confidence`` and sends
#     the top-K to ``fulfill()`` (today's ``search()`` semantics).
# Bid honesty is *unverified at bid time* — the auction trusts bids
# only as a ranking signal. The contract-driven verifier still
# runs on every fulfilled bid. Misleading bids waste top-K capacity,
# they do not let unsafe artifacts through.


class ProviderProtocolViolation(ValueError):
    """A provider returned a structurally-malformed BidPreview / ProviderResult.

    raises this from :func:`compgen.kernels.registry.compute_bid`
    when ``bid()`` returns a ``BidPreview`` with out-of-range fields
    (negative confidence, non-finite perf_estimate, contract_hash that
    disagrees with the canonical hash, etc.). The auction treats this
    as a fatal protocol violation for the offending provider — its bid
    is dropped, the rest of the auction proceeds.
    """


@dataclass(frozen=True)
class BidPreview:
    """A provider's cheap, pre-codegen bid on a :class:`KernelContractV3`.

    The provider declares (a) what it *thinks* it can deliver, (b) how
    confident it is, (c) how long ``fulfill()`` will take. The auction
     uses this to decide whether the provider gets to run at all.

    Attributes:
        provider_name: Mirrors :attr:`KernelProvider.name` so the
            BidPreview is self-describing on disk.
        contract_hash: The canonical hash of the contract this bid
            is for. Empty string is a sentinel — :func:`compute_bid`
            stamps the canonical hash if the provider didn't supply one.
            If the provider supplied a non-empty hash and it disagrees
            with the canonical hash, the bid is rejected as a protocol
            violation.
        perf_estimate_us: Estimated kernel latency (microseconds).
            ``+inf`` is the placeholder for "no estimate".
        confidence: Self-reported [0, 1]. 0 = pure placeholder; 1 =
            the provider is sure (e.g. cache hit). The auction
            multiplies this with ``perf_estimate`` for ranking.
        time_to_generate_s_estimate: How long ``fulfill()`` will take
            in wall-clock seconds. Auction uses this to short-circuit
            providers that won't fit the per-task budget.
        registers_used / occupancy / smem_bytes: Resource claims for
            informational ranking. Verifier may cross-check.
        rationale: One-line justification — surfaced in
            ``auction_report.json`` for tactician analysis.
        cache_hit: True iff the provider can serve from a verified
            cache, in which case ``fulfill()`` is essentially free.
            The auction uses this to prefer cached over fresh bids.
    """

    provider_name: str
    contract_hash: str = ""
    perf_estimate_us: float = float("inf")
    confidence: float = 0.0
    time_to_generate_s_estimate: float = 900.0
    registers_used: int = 0
    occupancy: float = 0.0
    smem_bytes: int = 0
    rationale: str = ""
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        # ``perf_estimate_us`` may be +inf — JSON cannot serialise that
        # losslessly; we use the string "+inf" sentinel that
        # :meth:`from_dict` reads back.
        import math

        perf = self.perf_estimate_us
        if math.isinf(perf):
            perf_repr: float | str = "+inf" if perf > 0 else "-inf"
        elif math.isnan(perf):
            perf_repr = "nan"
        else:
            perf_repr = float(perf)
        return {
            "provider_name": self.provider_name,
            "contract_hash": self.contract_hash,
            "perf_estimate_us": perf_repr,
            "confidence": self.confidence,
            "time_to_generate_s_estimate": self.time_to_generate_s_estimate,
            "registers_used": self.registers_used,
            "occupancy": self.occupancy,
            "smem_bytes": self.smem_bytes,
            "rationale": self.rationale,
            "cache_hit": self.cache_hit,
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> "BidPreview":
        perf_raw = body.get("perf_estimate_us", float("inf"))
        if isinstance(perf_raw, str):
            if perf_raw == "+inf":
                perf = float("inf")
            elif perf_raw == "-inf":
                perf = float("-inf")
            else:
                perf = float("nan")
        else:
            perf = float(perf_raw)
        return cls(
            provider_name=str(body.get("provider_name", "")),
            contract_hash=str(body.get("contract_hash", "")),
            perf_estimate_us=perf,
            confidence=float(body.get("confidence", 0.0)),
            time_to_generate_s_estimate=float(
                body.get("time_to_generate_s_estimate", 900.0)
            ),
            registers_used=int(body.get("registers_used", 0)),
            occupancy=float(body.get("occupancy", 0.0)),
            smem_bytes=int(body.get("smem_bytes", 0)),
            rationale=str(body.get("rationale", "")),
            cache_hit=bool(body.get("cache_hit", False)),
        )


def make_default_bid(
    *,
    provider_name: str,
    contract_hash: str,
    rationale: str = "no_bid_method",
) -> BidPreview:
    """Build a placeholder :class:`BidPreview` for a legacy provider that
    has no ``bid()`` method.

    Returns ``confidence=0.0`` so the auction treats the provider as a
    last-resort bidder; real providers must override ``bid()`` to
    surface non-zero confidence.
    """
    return BidPreview(
        provider_name=provider_name,
        contract_hash=contract_hash,
        perf_estimate_us=float("inf"),
        confidence=0.0,
        time_to_generate_s_estimate=900.0,
        rationale=rationale,
    )


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


# Phase D / providers may optionally implement::
#     def bid(self, contract: KernelContractV3) -> BidPreview: ...
# The Protocol above does NOT declare ``bid`` so legacy providers
# remain ``isinstance(p, KernelProvider)``-conformant. Use
# :func:`compgen.kernels.registry.compute_bid` to safely invoke it
# with a placeholder fallback.


__all__ = [
    "BidPreview",
    "ContractFeedback",
    "DispatchGeometry",
    "KernelContract",
    "KernelProvider",
    "KnowledgeExport",
    "ProviderProtocolViolation",
    "ProviderResult",
    "RuntimeCapabilities",
    "SearchBudget",
    "make_default_bid",
]
