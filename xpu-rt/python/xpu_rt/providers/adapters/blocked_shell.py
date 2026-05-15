"""Shared base for typed-blocked adapter shells.

For every provider whose real backend isn't implemented yet (or
whose toolchain isn't installed in this environment), the shell
adapter:

* implements the :class:`KernelProvider` ABC,
* delegates ``probe`` to the free-function
  ``probe_provider(card)``,
* declines every contract in ``can_bid()`` until a real backend
  lands,
* returns a typed ``status="blocked"`` v1 ``ProviderResult`` from
  ``propose()`` with the exact missing prerequisite, so the
  execution-evidence audit can record a ``blocked_proof.json``.

Concrete shells live in sibling modules and only need to subclass
:class:`BlockedShellAdapter` with the card id.
"""

from __future__ import annotations

from typing import Any

from compgen.kernels.provider import BidPreview, make_default_bid
from compgen.providers.card_loader import iter_provider_cards
from compgen.providers.kernel_provider import (
    KernelCodegenRequest,
    KernelProvider,
)
from compgen.providers.provider_probe import probe_provider
from compgen.providers.provider_types import ProviderCard, ProviderProbeResult
from compgen.providers.result_v1 import (
    SCHEMA_VERSION,
    ProviderResultV1,
)


class AdapterShellError(RuntimeError):
    """Raised when a shell adapter cannot locate its card."""


def _find_card(provider_id: str) -> ProviderCard:
    for c in iter_provider_cards():
        if c.provider_id == provider_id:
            return c
    raise AdapterShellError(f"no card found for provider_id={provider_id!r}")


class BlockedShellAdapter(KernelProvider):
    """Base class for typed-blocked provider shells.

    Subclasses set ``provider_id`` at class scope. Everything else
    flows from the shipped :class:`ProviderCard`.
    """

    #: Concrete subclasses override this.
    provider_id: str = ""

    def __init__(self) -> None:
        if not self.provider_id:
            raise AdapterShellError(
                f"{type(self).__name__} must set class attribute provider_id"
            )
        self.card = _find_card(self.provider_id)

    # ------------------------------------------------------------------
    # KernelProvider ABC
    # ------------------------------------------------------------------

    def probe(self) -> ProviderProbeResult:
        return probe_provider(self.card)

    def can_bid(self, contract: Any, target: Any) -> BidPreview:
        # Shells decline every contract — confidence 0.0 + clear
        # rationale so the auction ranks them last.
        return make_default_bid(
            provider_name=self.provider_id,
            contract_hash="",
            rationale=(
                f"{self.provider_id} adapter shell (M-90): no real "
                f"backend implementation; honestly blocked"
            ),
        )

    def propose(self, request: KernelCodegenRequest) -> ProviderResultV1:
        probe_result = self.probe()
        if probe_result.status == "available":
            # Card-wise the toolchain is present but the real backend
            # hasn't been built yet. Surface a clear, typed reason.
            return ProviderResultV1(
                schema_version=SCHEMA_VERSION,
                task_id=request.task_id,
                provider_id=self.provider_id,
                target_id=getattr(request.target, "name", "") or getattr(
                    request.target, "target_id", ""
                ),
                contract_hash=getattr(request.contract, "hardware_key", ""),
                status="blocked",
                detail=(
                    f"{self.provider_id} adapter shell (M-90): toolchain "
                    f"present but real codegen backend not yet implemented. "
                    f"Track follow-up under M-91 long-pole."
                ),
                claims={"adapter_kind": "blocked_shell"},
            )

        # Toolchain itself is missing — propagate the probe's typed
        # blocked_reason verbatim.
        return ProviderResultV1(
            schema_version=SCHEMA_VERSION,
            task_id=request.task_id,
            provider_id=self.provider_id,
            target_id=getattr(request.target, "name", "") or getattr(
                request.target, "target_id", ""
            ),
            contract_hash=getattr(request.contract, "hardware_key", ""),
            status="blocked",
            detail=(
                f"{self.provider_id} probe={probe_result.status}, "
                f"reason={probe_result.blocked_reason}, "
                f"missing={probe_result.detail!r}"
            ),
            claims={
                "adapter_kind": "blocked_shell",
                "probe_status": probe_result.status,
                "probe_blocked_reason": probe_result.blocked_reason,
                "probe_missing": probe_result.detail,
            },
        )
