"""New ``KernelProvider`` 3-method ABC.

This is the **spec'd** provider interface that every Phase F adapter
must satisfy:

* ``probe()`` → :class:`compgen.providers.provider_types.ProviderProbeResult`
* ``can_bid(contract, target)`` →
  :class:`compgen.kernels.provider.BidPreview`
* ``propose(request)`` → :class:`KernelCodegenRequest` →
  :class:`compgen.providers.result_v1.ProviderResult`

The legacy Protocol at :mod:`compgen.kernels.provider.KernelProvider`
stays in place for backward compatibility. New adapters subclass
:class:`KernelProvider` here; legacy adapters are wrapped via
:class:`compgen.providers.legacy_shim.LegacyProviderAdapter`.

Hard contract:

1. ``probe()`` is **structural** — no live LLM calls, no GPU
   workload. It only checks toolchain prereqs.
2. ``can_bid()`` is **cheap** — no codegen, no LLM. Used by the
   auction to rank providers.
3. ``propose()`` is the real work — codegen, optional autotuning,
   may take time.
4. **Providers never certify themselves.** The verifier owns the
   certificate path; ``ProviderResult.status`` of ``"generated"``
   is a *claim*, not a guarantee.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from compgen.kernels.provider import BidPreview
from compgen.providers.provider_types import ProviderProbeResult


@dataclass(frozen=True)
class KernelCodegenRequest:
    """Typed request handed to ``provider.propose()``.

    Carries the minimum a real adapter needs to emit a kernel:
    contract, target descriptor, artifact directory, and an optional
    task id for evidence-pack correlation.
    """

    task_id: str
    contract: Any
    target: Any
    artifact_dir: str
    extras: dict[str, Any] = field(default_factory=dict)


class KernelProvider(ABC):
    """Spec'd 3-method provider interface.

    Concrete subclasses live under
    :mod:`compgen.providers.adapters.*`. Legacy providers in
    :mod:`compgen.kernels.providers.*` are accessed through
    :class:`compgen.providers.legacy_shim.LegacyProviderAdapter`.
    """

    provider_id: str = ""

    @abstractmethod
    def probe(self) -> ProviderProbeResult:
        """Return the toolchain-availability status for this provider.

        Must not raise on missing toolchains; missing prereqs are
        encoded as :class:`ProviderProbeResult` with typed
        ``status`` + ``blocked_reason``.
        """

    @abstractmethod
    def can_bid(self, contract: Any, target: Any) -> BidPreview:
        """Cheap pre-codegen bid.

        Returns a :class:`BidPreview`. May return a placeholder
        bid (``confidence=0.0``) when the provider has no opinion
        but does not want to be silently dropped from the auction.
        """

    @abstractmethod
    def propose(self, request: KernelCodegenRequest) -> Any:
        """Produce a kernel artifact.

        Returns a v1 :class:`compgen.providers.result_v1.ProviderResult`
        . Adapters that can't fulfill ``request`` must return
        a result with ``status="blocked"`` carrying the typed
        ``blocked_reason`` — they must not raise.
        """
