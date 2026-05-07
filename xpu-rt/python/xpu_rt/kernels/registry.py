"""Kernel provider registry — dispatch contracts to providers.

The registry manages multiple kernel providers and mediates the
bidirectional communication between CompGen and providers:

1. Dispatch: route contracts to accepting providers
2. Knowledge ingestion: collect exports from all providers into memory
3. Contract evolution: apply provider feedback to update contracts

Phase D / M-55: ``applicable()`` exposes a static-metadata filter over
``KernelContractV3`` so the kernel-codegen pipeline (M-42) can log which
providers *could* bid before any provider methods are invoked. The
filter reads optional class-level attributes on each provider:

* ``applicable_targets: tuple[str, ...]`` — empty tuple = wildcard.
* ``applicable_archetypes: tuple[str, ...]`` — empty tuple = wildcard.

These attributes are read with ``getattr(p, name, ())`` so legacy
providers without them are treated as wildcard matches (today's
behaviour preserved exactly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from compgen.kernels.provider import (
    BidPreview,
    ContractFeedback,
    KernelContract,
    KernelProvider,
    KnowledgeExport,
    ProviderProtocolViolation,
    ProviderResult,
    SearchBudget,
    make_default_bid,
)

if TYPE_CHECKING:
    from compgen.kernels.contract_v3 import KernelContractV3

log = structlog.get_logger()


@dataclass(frozen=True)
class ProviderApplicability:
    """Why (or why not) a provider was deemed applicable for a contract.

    M-55 emits one of these per registered provider into
    ``04_kernel_codegen/registry_resolution.json`` so the M-57 auction
    has a stable, byte-deterministic record of which providers it
    considered for a given task.
    """

    provider_name: str
    source: str  # "in_tree" | "entry_point" | "user_path"
    priority: int
    applicable_targets: tuple[str, ...]
    applicable_archetypes: tuple[str, ...]
    matches_target: bool
    matches_archetype: bool
    match_reason: str

    @property
    def applicable(self) -> bool:
        return self.matches_target and self.matches_archetype

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "source": self.source,
            "priority": self.priority,
            "applicable_targets": list(self.applicable_targets),
            "applicable_archetypes": list(self.applicable_archetypes),
            "matches_target": self.matches_target,
            "matches_archetype": self.matches_archetype,
            "applicable": self.applicable,
            "match_reason": self.match_reason,
        }


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


    # ------------------------------------------------------------------
    # M-55 — static-metadata applicability over KernelContractV3
    # ------------------------------------------------------------------

    def applicable(
        self,
        contract_v3: KernelContractV3,
    ) -> list[ProviderApplicability]:
        """Static-metadata filter: which providers could bid on this V3 contract?

        This is a *pure* metadata match — no provider methods are
        invoked. M-56's ``bid()`` and M-57's ``fulfill()`` consume this
        list. Until then, the codegen pipeline calls this and writes
        ``04_kernel_codegen/registry_resolution.json`` for traceability;
        the actual codegen path (today: Claude-Code subagent via M-43
        commit) is unchanged.

        Args:
            contract_v3: The materialized V3 contract.

        Returns:
            One :class:`ProviderApplicability` per registered provider.
            Callers filter to ``.applicable`` for the bid list; the full
            list is logged so a non-match is auditable.
        """
        target_name = ""
        try:
            execution = contract_v3.orchestration.execution
            if execution is not None:
                target_name = execution.hardware.target_name
        except AttributeError:
            target_name = ""

        archetype_value = ""
        try:
            archetype_value = contract_v3.archetype.value
        except AttributeError:
            archetype_value = ""

        out: list[ProviderApplicability] = []
        for p in self._providers:
            applicable_targets = tuple(getattr(p, "applicable_targets", ()) or ())
            applicable_archetypes = tuple(getattr(p, "applicable_archetypes", ()) or ())
            priority = int(getattr(p, "priority", 0))
            source = str(getattr(p, "_compgen_source", "in_tree"))

            matches_target = (
                len(applicable_targets) == 0  # wildcard
                or target_name in applicable_targets
            )
            matches_archetype = (
                len(applicable_archetypes) == 0  # wildcard
                or archetype_value in applicable_archetypes
            )

            if matches_target and matches_archetype:
                if not applicable_targets and not applicable_archetypes:
                    reason = "wildcard"
                elif applicable_targets and applicable_archetypes:
                    reason = "target+archetype"
                elif applicable_targets:
                    reason = "target_only"
                else:
                    reason = "archetype_only"
            else:
                missing = []
                if not matches_target:
                    missing.append(f"target={target_name!r} not in {list(applicable_targets)}")
                if not matches_archetype:
                    missing.append(f"archetype={archetype_value!r} not in {list(applicable_archetypes)}")
                reason = "; ".join(missing)

            out.append(
                ProviderApplicability(
                    provider_name=p.name,
                    source=source,
                    priority=priority,
                    applicable_targets=applicable_targets,
                    applicable_archetypes=applicable_archetypes,
                    matches_target=matches_target,
                    matches_archetype=matches_archetype,
                    match_reason=reason,
                )
            )

        # Stable order: highest priority first, then by name.
        out.sort(key=lambda r: (-r.priority, r.provider_name))
        return out


# ===========================================================================
# M-56 — bid() invocation with legacy fallback
# ===========================================================================


def _validate_bid(bid: BidPreview, *, expected_hash: str) -> None:
    """Type-check a :class:`BidPreview` returned by a provider.

    Raises :class:`ProviderProtocolViolation` on any structural
    violation. Bid honesty (i.e. whether ``perf_estimate_us`` reflects
    reality) is not checked here — the auction trusts the bid only as
    a ranking signal; the contract-driven verifier (M-44) catches
    real-world divergence on the fulfilled artifact.
    """
    import math

    if not isinstance(bid, BidPreview):
        raise ProviderProtocolViolation(
            f"bid() must return BidPreview; got {type(bid).__name__}"
        )
    if not bid.provider_name:
        raise ProviderProtocolViolation("BidPreview.provider_name must not be empty")
    if math.isnan(bid.confidence) or bid.confidence < 0.0 or bid.confidence > 1.0:
        raise ProviderProtocolViolation(
            f"BidPreview.confidence must be in [0, 1]; got {bid.confidence!r}"
        )
    if math.isnan(bid.perf_estimate_us):
        raise ProviderProtocolViolation(
            "BidPreview.perf_estimate_us must not be NaN; use +inf for 'no estimate'"
        )
    if bid.perf_estimate_us < 0.0:
        raise ProviderProtocolViolation(
            f"BidPreview.perf_estimate_us must be non-negative; got {bid.perf_estimate_us!r}"
        )
    if bid.time_to_generate_s_estimate < 0.0 or math.isnan(
        bid.time_to_generate_s_estimate
    ):
        raise ProviderProtocolViolation(
            "BidPreview.time_to_generate_s_estimate must be non-negative finite"
        )
    if bid.registers_used < 0:
        raise ProviderProtocolViolation("BidPreview.registers_used must be non-negative")
    if bid.smem_bytes < 0:
        raise ProviderProtocolViolation("BidPreview.smem_bytes must be non-negative")
    if math.isnan(bid.occupancy) or bid.occupancy < 0.0 or bid.occupancy > 1.0:
        raise ProviderProtocolViolation(
            f"BidPreview.occupancy must be in [0, 1]; got {bid.occupancy!r}"
        )
    if bid.contract_hash and bid.contract_hash != expected_hash:
        raise ProviderProtocolViolation(
            f"BidPreview.contract_hash mismatch: "
            f"provider returned {bid.contract_hash!r}, expected {expected_hash!r}"
        )


def compute_bid(
    provider: KernelProvider,
    contract_v3: KernelContractV3,
    *,
    expected_hash: str | None = None,
) -> BidPreview:
    """Invoke ``provider.bid(contract_v3)`` with legacy fallback.

    Args:
        provider: The provider to query.
        contract_v3: The materialized V3 contract.
        expected_hash: Pre-computed canonical contract hash. If
            ``None``, computed via
            :func:`compgen.promotion.contract_hash.hash_contract`.
            Pass it in when calling for many providers in a row to
            avoid re-hashing.

    Returns:
        A validated :class:`BidPreview`. If the provider has no
        ``bid()`` method or returns ``None``, a placeholder bid is
        returned with ``confidence=0.0`` and ``rationale="no_bid_method"``.

    Raises:
        ProviderProtocolViolation: When the provider's ``bid()`` raises
            a typed protocol violation, or returns a malformed
            :class:`BidPreview`. Provider-internal exceptions of any
            other kind are caught and converted into a low-confidence
            placeholder bid (so a single buggy provider does not abort
            the auction).
    """
    if expected_hash is None:
        try:
            from compgen.promotion.contract_hash import hash_contract

            expected_hash = hash_contract(contract_v3)
        except Exception:  # noqa: BLE001
            expected_hash = ""

    bid_method = getattr(provider, "bid", None)
    if bid_method is None or not callable(bid_method):
        return make_default_bid(
            provider_name=provider.name,
            contract_hash=expected_hash or "",
            rationale="no_bid_method",
        )

    try:
        bid = bid_method(contract_v3)
    except ProviderProtocolViolation:
        # Typed protocol error from the provider itself — surface up.
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "provider.bid.error",
            provider=provider.name,
            error=f"{type(exc).__name__}: {exc}",
        )
        return make_default_bid(
            provider_name=provider.name,
            contract_hash=expected_hash or "",
            rationale=f"bid_raised:{type(exc).__name__}",
        )

    if bid is None:
        return make_default_bid(
            provider_name=provider.name,
            contract_hash=expected_hash or "",
            rationale="bid_returned_none",
        )

    _validate_bid(bid, expected_hash=expected_hash or bid.contract_hash)

    # Stamp the canonical hash if the provider didn't supply one.
    if not bid.contract_hash and expected_hash:
        from dataclasses import replace

        bid = replace(bid, contract_hash=expected_hash)

    return bid


def collect_bids(
    providers: list[KernelProvider],
    contract_v3: KernelContractV3,
) -> list[BidPreview]:
    """Run :func:`compute_bid` over a list of applicable providers.

    Hashes the contract once and reuses the canonical hash for every
    provider. Returns the bids in the input order; the auction (M-57)
    is responsible for ranking them.
    """
    try:
        from compgen.promotion.contract_hash import hash_contract

        expected_hash = hash_contract(contract_v3)
    except Exception:  # noqa: BLE001
        expected_hash = ""

    return [compute_bid(p, contract_v3, expected_hash=expected_hash) for p in providers]


def discover_default_providers() -> list[KernelProvider]:
    """Collect kernel providers from the entry-point plugin registry.

    Mirrors :func:`compgen.kernels.codegen_fallback._discover_providers`
    but exposed publicly. The returned providers carry a synthesised
    ``_compgen_source`` attribute (``"entry_point"``) so
    :meth:`ProviderRegistry.applicable` can attribute them.
    """
    try:
        from compgen.plugins import GROUP_KERNEL_PROVIDERS, discover_all, registry
    except Exception:  # noqa: BLE001
        return []

    discover_all()
    out: list[KernelProvider] = []
    for plugin in registry().get(GROUP_KERNEL_PROVIDERS):
        obj = plugin.object
        try:
            if isinstance(obj, type):
                instance = obj()
            elif callable(obj) and not all(
                hasattr(obj, m) for m in ("name", "accepts_contract", "search")
            ):
                instance = obj()
            else:
                instance = obj
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "registry.provider_instantiate_failed",
                plugin_name=plugin.name,
                error=str(exc),
            )
            continue
        try:
            object.__setattr__(instance, "_compgen_source", "entry_point")
        except Exception:  # noqa: BLE001
            pass
        out.append(instance)
    return out


def default_registry() -> ProviderRegistry:
    """Build the default :class:`ProviderRegistry` for Phase D.

    Registers (in priority order):

    1. ``CReferenceProvider`` (in-tree, deterministic cffi-C matmul
       baseline; always-on so the auction has at least one bidder for
       host_cpu matmul contracts).
    2. Entry-point providers via :func:`discover_default_providers`.

    The Claude-Code subagent path is not registered here — its bid is
    cache-aware (M-56) and is added explicitly by callers that want
    in-session codegen. Tests inject stubs via a fresh
    ``ProviderRegistry()`` rather than this default.
    """
    reg = ProviderRegistry()

    # In-tree baseline.
    try:
        from compgen.kernels.providers.c_reference import CReferenceProvider

        baseline = CReferenceProvider()
        try:
            object.__setattr__(baseline, "_compgen_source", "in_tree")
        except Exception:  # noqa: BLE001
            pass
        reg.register(baseline)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "registry.c_reference_load_failed",
            error=f"{type(exc).__name__}: {exc}",
        )

    for p in discover_default_providers():
        reg.register(p)
    return reg


__all__ = [
    "ProviderApplicability",
    "ProviderRegistry",
    "collect_bids",
    "compute_bid",
    "default_registry",
    "discover_default_providers",
]
