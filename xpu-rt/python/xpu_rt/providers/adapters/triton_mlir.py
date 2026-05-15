"""Triton-MLIR (TTIR / TTGIR) dialect shell."""

from __future__ import annotations

from typing import Any

from compgen.kernels.provider import BidPreview, make_default_bid
from compgen.providers.adapters.iree import _find_dialect_card
from compgen.providers.kernel_provider import (
    KernelCodegenRequest,
    KernelProvider,
)
from compgen.providers.provider_types import ProviderProbeResult
from compgen.providers.provider_probe import probe_dialect_provider
from compgen.providers.result_v1 import SCHEMA_VERSION, ProviderResultV1


class TritonMLIRDialectProvider(KernelProvider):
    provider_id = "triton_mlir"

    def __init__(self) -> None:
        self.card = _find_dialect_card("triton_mlir")

    def probe(self) -> ProviderProbeResult:
        return probe_dialect_provider(self.card)

    def can_bid(self, contract: Any, target: Any) -> BidPreview:
        return make_default_bid(
            provider_name="triton_mlir",
            contract_hash="",
            rationale="triton_mlir dialect shell (M-90)",
        )

    def propose(self, request: KernelCodegenRequest) -> ProviderResultV1:
        probe = self.probe()
        return ProviderResultV1(
            schema_version=SCHEMA_VERSION,
            task_id=request.task_id,
            provider_id="triton_mlir",
            target_id=getattr(request.target, "name", ""),
            contract_hash="",
            status="blocked",
            detail=(
                f"triton_mlir dialect shell; probe={probe.status}, "
                f"reason={probe.blocked_reason}"
            ),
            claims={"adapter_kind": "dialect_shell"},
        )
