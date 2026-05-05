"""Refinement monotonicity (M-33.3).

CompGen passes preserve refinement at one of four declared levels
(see :data:`compgen.passes.cards.REFINEMENT_KINDS`):

- ``bit_equality``  — output is bit-for-bit identical to eager
- ``tolerance_eps`` — output matches eager within an epsilon
- ``none``           — no correctness guarantee
- ``unknown``        — the pass has not characterized its behavior

These form a strict order: ``bit_equality > tolerance_eps > none``.
``unknown`` is treated as "weakest" because we cannot prove anything.

When a recipe applies multiple passes, the *claimable* refinement of
the recipe is the **weakest** preserves_refinement in the chain. A
recipe that claims ``bit_equality`` after applying any
``tolerance_eps`` pass is lying — that's the
:class:`RefinementMonotonicityViolation` failure mode.

Monotonicity here means: a recipe's claimable refinement can only
*decrease* across composition; it can never spontaneously increase
without an explicit ``relax_refinement`` op (out of scope for M-33).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from compgen.audit.errors import RefinementMonotonicityViolation
from compgen.passes.cards import REFINEMENT_KINDS, PassCard


class RefinementLevel(int, enum.Enum):
    """Total order on refinement strength.

    The integer value is *strength*: bigger = stronger guarantee.
    Comparison via ``<`` / ``>`` works as expected.
    """

    UNKNOWN = 0
    NONE = 1
    TOLERANCE_EPS = 2
    BIT_EQUALITY = 3

    @classmethod
    def from_string(cls, name: str) -> RefinementLevel:
        try:
            return _STRING_TO_LEVEL[name]
        except KeyError as exc:
            raise ValueError(
                f"refinement {name!r} not in {REFINEMENT_KINDS}"
            ) from exc

    def to_string(self) -> str:
        return _LEVEL_TO_STRING[self]


_STRING_TO_LEVEL: dict[str, RefinementLevel] = {
    "unknown": RefinementLevel.UNKNOWN,
    "none": RefinementLevel.NONE,
    "tolerance_eps": RefinementLevel.TOLERANCE_EPS,
    "bit_equality": RefinementLevel.BIT_EQUALITY,
}
_LEVEL_TO_STRING: dict[RefinementLevel, str] = {
    v: k for k, v in _STRING_TO_LEVEL.items()
}


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


def compute_claimable_refinement(
    applied_cards: Sequence[PassCard],
) -> RefinementLevel:
    """Return the weakest preserves_refinement across the chain.

    The empty chain has nothing constraining it; the strongest
    claim (``bit_equality``) is permitted because no pass has run
    that could weaken anything. (A recipe with zero passes is the
    identity — bit-equal to eager by construction.)
    """
    if not applied_cards:
        return RefinementLevel.BIT_EQUALITY
    weakest = RefinementLevel.BIT_EQUALITY
    for card in applied_cards:
        level = RefinementLevel.from_string(card.preserves_refinement)
        if level < weakest:
            weakest = level
    return weakest


def assert_refinement_chain_valid(
    applied_cards: Sequence[PassCard],
    claimed: str | RefinementLevel,
    *,
    recipe_id: str = "<unknown>",
) -> None:
    """Verify ``claimed`` does not exceed the chain's claimable refinement.

    Raises :class:`RefinementMonotonicityViolation` when a recipe claims
    a stronger refinement than the chain actually preserves.

    Equality is allowed (claim == claimable); a *weaker* claim is also
    fine — the recipe is voluntarily under-claiming.
    """
    if isinstance(claimed, str):
        claimed_level = RefinementLevel.from_string(claimed)
    else:
        claimed_level = claimed
    claimable = compute_claimable_refinement(applied_cards)
    if claimed_level > claimable:
        chain_summary = [
            (c.pass_id, c.preserves_refinement) for c in applied_cards
        ]
        raise RefinementMonotonicityViolation(
            f"recipe {recipe_id!r} claims refinement "
            f"{claimed_level.to_string()!r} but the applied chain "
            f"{chain_summary} preserves only "
            f"{claimable.to_string()!r}"
        )


# --------------------------------------------------------------------------- #
# Inspection helpers (used by validate_agent_decision_response)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RefinementChainReport:
    """Diagnostic record returned to the validator."""

    claimed: str
    claimable: str
    holds: bool
    chain: tuple[tuple[str, str], ...]  # (pass_id, preserves_refinement)
    detail: str = ""


def inspect_refinement_chain(
    applied_cards: Sequence[PassCard],
    claimed: str,
    *,
    recipe_id: str = "<unknown>",
) -> RefinementChainReport:
    """Non-raising variant: compute the chain status as data."""
    chain = tuple(
        (c.pass_id, c.preserves_refinement) for c in applied_cards
    )
    try:
        claimable = compute_claimable_refinement(applied_cards)
    except ValueError as exc:
        return RefinementChainReport(
            claimed=claimed,
            claimable="unknown",
            holds=False,
            chain=chain,
            detail=f"chain contains unknown refinement: {exc}",
        )
    try:
        claimed_level = RefinementLevel.from_string(claimed)
    except ValueError as exc:
        return RefinementChainReport(
            claimed=claimed,
            claimable=claimable.to_string(),
            holds=False,
            chain=chain,
            detail=f"claim is invalid: {exc}",
        )
    holds = claimed_level <= claimable
    detail = (
        ""
        if holds
        else (
            f"claim {claimed!r} exceeds claimable "
            f"{claimable.to_string()!r}"
        )
    )
    return RefinementChainReport(
        claimed=claimed,
        claimable=claimable.to_string(),
        holds=holds,
        chain=chain,
        detail=detail,
    )
