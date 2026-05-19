"""StableHLO / MHLO dialect shell. card_only level."""

from __future__ import annotations

from typing import Any

from xpu_rt.kernels.provider import BidPreview, make_default_bid
from xpu_rt.providers.adapters.iree import _find_dialect_card
from xpu_rt.providers.kernel_provider import (
    KernelCodegenRequest,
    KernelProvider,
)
from xpu_rt.providers.provider_types import ProviderProbeResult
from xpu_rt.providers.provider_probe import probe_dialect_provider
from xpu_rt.providers.result_v1 import SCHEMA_VERSION, ProviderResultV1


class StableHloDialectProvider(KernelProvider):
    provider_id = "stablehlo"

    def __init__(self) -> None:
        self.card = _find_dialect_card("stablehlo")

    def probe(self) -> ProviderProbeResult:
        return probe_dialect_provider(self.card)

    def can_bid(self, contract: Any, target: Any) -> BidPreview:
        return make_default_bid(
            provider_name="stablehlo",
            contract_hash="",
            rationale="stablehlo dialect shell: card_only integration level",
        )

    def propose(self, request: KernelCodegenRequest) -> ProviderResultV1:
        probe = self.probe()
        return ProviderResultV1(
            schema_version=SCHEMA_VERSION,
            task_id=request.task_id,
            provider_id="stablehlo",
            target_id=getattr(request.target, "name", ""),
            contract_hash="",
            status="blocked",
            detail=(
                f"stablehlo dialect at integration_level=card_only; "
                f"probe={probe.status}, reason={probe.blocked_reason}"
            ),
            claims={"adapter_kind": "dialect_card_only"},
        )
