"""Legacy provider shim.

Wraps existing legacy providers from
:mod:`compgen.kernels.providers.*` so they satisfy the new
:class:`compgen.providers.kernel_provider.KernelProvider` ABC.

The legacy providers already implement:

* ``accepts_contract(contract)`` — analogous to ``can_bid`` gating.
* ``search(contract, budget)`` — analogous to ``propose``.
* Optionally ``bid(contract_v3)`` — direct map to ``can_bid``.

The shim translates between the two interfaces without modifying
the legacy modules. New adapters subclass
:class:`compgen.providers.kernel_provider.KernelProvider`
directly.
"""

from __future__ import annotations

from typing import Any

from compgen.kernels.provider import (
    BidPreview,
    KernelContract,
    SearchBudget,
    make_default_bid,
)
from compgen.providers.kernel_provider import (
    KernelCodegenRequest,
    KernelProvider,
)
from compgen.providers.provider_types import (
    ProviderCard,
    ProviderProbeResult,
)


class LegacyProviderAdapter(KernelProvider):
    """Adapts a legacy ``search()``-style provider to the new ABC.

    ``probe()`` is delegated to
    :func:`compgen.providers.provider_probe.probe_provider` using
    the provider's :class:`ProviderCard`. ``can_bid()`` maps to the
    legacy ``bid()`` if available, else returns a placeholder.
    ``propose()`` calls the legacy ``search()`` and wraps the
    result as a v1 :class:`ProviderResult`.
    """

    def __init__(self, *, card: ProviderCard, legacy_instance: Any) -> None:
        self.card = card
        self.provider_id = card.provider_id
        self._legacy = legacy_instance

    def probe(self) -> ProviderProbeResult:
        from compgen.providers.provider_probe import probe_provider
        return probe_provider(self.card)

    def can_bid(self, contract: Any, target: Any) -> BidPreview:
        bid_fn = getattr(self._legacy, "bid", None)
        if callable(bid_fn):
            try:
                preview = bid_fn(contract)
                if isinstance(preview, BidPreview):
                    return preview
            except Exception:
                # Legacy ``bid()`` raising is non-fatal; we fall back to
                # the placeholder. The auction treats confidence=0 as
                # "rank last".
                pass
        return make_default_bid(
            provider_name=self.provider_id,
            contract_hash="",
            rationale="legacy_provider_no_bid_method",
        )

    def propose(self, request: KernelCodegenRequest) -> Any:
        """Run the legacy ``search()`` and translate to v1.

        Imports the v1 producer lazily to avoid an import cycle —
        ``result_v1`` lands .
        """

        from compgen.providers.result_v1 import (
            ProviderResultV1,
            legacy_to_v1,
        )

        contract = request.contract
        budget = request.extras.get("budget") or SearchBudget()
        accepts = getattr(self._legacy, "accepts_contract", None)
        if callable(accepts):
            try:
                if not accepts(contract):
                    return ProviderResultV1(
                        schema_version="provider_result_v1",
                        task_id=request.task_id,
                        provider_id=self.provider_id,
                        target_id=getattr(request.target, "name", ""),
                        contract_hash=getattr(contract, "hardware_key", ""),
                        status="contract_rejected",
                        detail="legacy accepts_contract returned False",
                    )
            except Exception as exc:
                return ProviderResultV1(
                    schema_version="provider_result_v1",
                    task_id=request.task_id,
                    provider_id=self.provider_id,
                    target_id=getattr(request.target, "name", ""),
                    contract_hash=getattr(contract, "hardware_key", ""),
                    status="error",
                    detail=f"accepts_contract raised: {type(exc).__name__}: {exc}",
                )

        search = getattr(self._legacy, "search", None)
        if not callable(search):
            return ProviderResultV1(
                schema_version="provider_result_v1",
                task_id=request.task_id,
                provider_id=self.provider_id,
                target_id=getattr(request.target, "name", ""),
                contract_hash="",
                status="blocked",
                detail="legacy provider has no search() method",
            )

        try:
            legacy_result = search(contract, budget)
        except Exception as exc:
            return ProviderResultV1(
                schema_version="provider_result_v1",
                task_id=request.task_id,
                provider_id=self.provider_id,
                target_id=getattr(request.target, "name", ""),
                contract_hash="",
                status="error",
                detail=f"{type(exc).__name__}: {exc}",
            )

        return legacy_to_v1(
            legacy_result,
            task_id=request.task_id,
            provider_id=self.provider_id,
            target_id=getattr(request.target, "name", ""),
            contract_hash=getattr(contract, "hardware_key", "")
            or request.extras.get("contract_hash", ""),
            artifact_dir=request.artifact_dir,
        )


def wrap_legacy(card: ProviderCard, legacy_instance: Any) -> LegacyProviderAdapter:
    """Convenience constructor."""

    return LegacyProviderAdapter(card=card, legacy_instance=legacy_instance)
