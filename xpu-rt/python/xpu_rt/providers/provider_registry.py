"""spec'd path for the provider registry.

Combines the card loader, the :class:`KernelProvider`
ABC, and the adapter resolution into one importable
"registry" surface. The actual provider registry behavior is
spread across:

* :mod:`compgen.providers.card_loader` — discovers
  ``providers/cards/*.yaml``;
* :mod:`compgen.providers.adapters.base` — resolves
  ``card.entrypoint`` to a real class;
* :mod:`compgen.providers.legacy_shim` — wraps legacy classes;
* :mod:`compgen.providers.provider_probe` — probes toolchain
  readiness.

This module gives callers a single entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.providers.adapters.base import (
    AdapterResolutionError,
    resolve_provider_class,
)
from compgen.providers.card_loader import (
    iter_dialect_cards,
    iter_provider_cards,
    iter_target_cards,
    load_all_cards,
)
from compgen.providers.kernel_provider import KernelProvider
from compgen.providers.legacy_shim import LegacyProviderAdapter, wrap_legacy
from compgen.providers.provider_probe import probe_provider
from compgen.providers.provider_types import ProviderCard, ProviderProbeResult


@dataclass
class ProviderRegistry:
    """In-memory registry built from shipped + extension cards.

    The registry is **per-process** and **rebuilt on demand**.
    Each entry pairs the :class:`ProviderCard` with a lazily-
    instantiated adapter (the cached `KernelProvider` instance).
    """

    cards: dict[str, ProviderCard] = field(default_factory=dict)
    _instances: dict[str, KernelProvider | None] = field(default_factory=dict)

    def provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.cards.keys()))

    def card_for(self, provider_id: str) -> ProviderCard:
        return self.cards[provider_id]

    def instance(
        self,
        provider_id: str,
        *,
        construct_args: tuple = (),
        construct_kwargs: dict[str, Any] | None = None,
    ) -> KernelProvider:
        """Return a KernelProvider instance for ``provider_id``.

        Legacy classes are auto-wrapped via the shim. Caches
        the instance per registry; pass ``construct_kwargs`` for
        providers that require constructor arguments (e.g.
        ``claude_kernel``).
        """

        cached = self._instances.get(provider_id)
        if cached is not None:
            return cached
        card = self.cards[provider_id]
        cls = resolve_provider_class(card)
        inst = cls(*construct_args, **(construct_kwargs or {}))
        if not isinstance(inst, KernelProvider):
            inst = wrap_legacy(card, inst)
        self._instances[provider_id] = inst
        return inst

    def probe(self, provider_id: str) -> ProviderProbeResult:
        """Probe by card id without instantiating the adapter — fast."""

        return probe_provider(self.cards[provider_id])


def build_provider_registry(
    *,
    extra_cards: tuple[ProviderCard, ...] = (),
) -> ProviderRegistry:
    """Discover shipped provider cards + merge user-extension cards."""

    cards: dict[str, ProviderCard] = {}
    for c in iter_provider_cards():
        cards[c.provider_id] = c
    for c in extra_cards:
        cards[c.provider_id] = c
    return ProviderRegistry(cards=cards)


__all__ = [
    "AdapterResolutionError",
    "LegacyProviderAdapter",
    "ProviderRegistry",
    "build_provider_registry",
    "iter_dialect_cards",
    "iter_provider_cards",
    "iter_target_cards",
    "load_all_cards",
    "resolve_provider_class",
    "wrap_legacy",
]
