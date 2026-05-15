"""spec'd target-registry path.

Re-exports the /card-driven target catalog. The legacy
``compgen.targets.registry`` module is preserved for backward
compatibility with capture+lower; this module is the Phase F
extension surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compgen.providers.card_loader import iter_target_cards
from compgen.targets.target_types import TargetCard


@dataclass
class TargetRegistry:
    cards: dict[str, TargetCard] = field(default_factory=dict)

    def target_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.cards.keys()))

    def card_for(self, target_id: str) -> TargetCard:
        return self.cards[target_id]

    def by_family(self, family: str) -> tuple[TargetCard, ...]:
        return tuple(c for c in self.cards.values() if c.family == family)


def build_target_registry(
    *,
    extra_cards: tuple[TargetCard, ...] = (),
) -> TargetRegistry:
    cards: dict[str, TargetCard] = {}
    for c in iter_target_cards():
        cards[c.target_id] = c
    for c in extra_cards:
        cards[c.target_id] = c
    return TargetRegistry(cards=cards)


__all__ = [
    "TargetCard",
    "TargetRegistry",
    "build_target_registry",
    "iter_target_cards",
]
