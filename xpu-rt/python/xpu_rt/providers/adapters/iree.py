"""IREE dialect-provider shell.

card-level only; no kernel provider counterpart yet.
"""

from __future__ import annotations

from typing import Any

from compgen.dialects.dialect_provider_types import DialectProviderCard
from compgen.kernels.provider import BidPreview, make_default_bid
from compgen.providers.card_loader import iter_dialect_cards
from compgen.providers.kernel_provider import (
    KernelCodegenRequest,
    KernelProvider,
)
from compgen.providers.provider_types import ProviderProbeResult
from compgen.providers.provider_probe import probe_dialect_provider
from compgen.providers.result_v1 import SCHEMA_VERSION, ProviderResultV1


def _find_dialect_card(dialect_provider_id: str) -> DialectProviderCard:
    for c in iter_dialect_cards():
        if c.dialect_provider_id == dialect_provider_id:
            return c
    raise RuntimeError(
        f"no dialect card found for dialect_provider_id={dialect_provider_id!r}"
    )


class IreeDialectProvider(KernelProvider):
    """IREE dialect shell — card_only integration level."""

    provider_id = "iree"

    def __init__(self) -> None:
        self.card = _find_dialect_card("iree")

    def probe(self) -> ProviderProbeResult:
        return probe_dialect_provider(self.card)

    def can_bid(self, contract: Any, target: Any) -> BidPreview:
        return make_default_bid(
            provider_name="iree",
            contract_hash="",
            rationale="iree dialect shell: card_only integration level",
        )

    def propose(self, request: KernelCodegenRequest) -> ProviderResultV1:
        probe = self.probe()
        return ProviderResultV1(
            schema_version=SCHEMA_VERSION,
            task_id=request.task_id,
            provider_id="iree",
            target_id=getattr(request.target, "name", ""),
            contract_hash="",
            status="blocked",
            detail=(
                f"iree dialect at integration_level=card_only; "
                f"probe={probe.status}, reason={probe.blocked_reason}"
            ),
            claims={"adapter_kind": "dialect_card_only"},
        )
