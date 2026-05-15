"""spec'd dialect-provider-registry path.

Re-exports the /card-driven dialect provider catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compgen.dialects.dialect_provider_types import DialectProviderCard
from compgen.providers.card_loader import iter_dialect_cards


@dataclass
class DialectProviderRegistry:
    cards: dict[str, DialectProviderCard] = field(default_factory=dict)

    def dialect_provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.cards.keys()))

    def card_for(self, dialect_provider_id: str) -> DialectProviderCard:
        return self.cards[dialect_provider_id]

    def by_dialect(self, dialect_name: str) -> tuple[DialectProviderCard, ...]:
        return tuple(c for c in self.cards.values() if c.dialect_name == dialect_name)


def build_dialect_provider_registry(
    *,
    extra_cards: tuple[DialectProviderCard, ...] = (),
) -> DialectProviderRegistry:
    cards: dict[str, DialectProviderCard] = {}
    for c in iter_dialect_cards():
        cards[c.dialect_provider_id] = c
    for c in extra_cards:
        cards[c.dialect_provider_id] = c
    return DialectProviderRegistry(cards=cards)


__all__ = [
    "DialectProviderCard",
    "DialectProviderRegistry",
    "build_dialect_provider_registry",
    "iter_dialect_cards",
]
