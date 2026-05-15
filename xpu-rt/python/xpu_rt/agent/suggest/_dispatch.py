"""Slot-name → suggester registration + dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from xdsl.dialects.builtin import ModuleOp

from compgen.agent.suggest._candidate import ProposalCandidate

SuggesterFn = Callable[..., list[ProposalCandidate]]


SUGGESTERS: dict[str, SuggesterFn] = {}


def register_suggester(slot_name: str) -> Callable[[SuggesterFn], SuggesterFn]:
    """Decorator that registers a suggester against ``slot_name``."""

    def _decorate(fn: SuggesterFn) -> SuggesterFn:
        SUGGESTERS[slot_name] = fn
        return fn

    return _decorate


def supported_slot_names() -> tuple[str, ...]:
    return tuple(sorted(SUGGESTERS.keys()))


def suggest(
    slot_name: str,
    *,
    recipe: ModuleOp,
    dossier: Any,
    target: Any,
    k: int = 5,
) -> list[ProposalCandidate]:
    """Dispatch to the registered suggester for ``slot_name``.

    Returns an empty list when no suggester is registered for the
    given slot (the caller can fall back to the slot's
    ``baseline_seed`` for a single-candidate default). Stamps
    ``slot_name`` on every candidate so its ``next_call`` hint is
    self-contained.
    """
    fn = SUGGESTERS.get(slot_name)
    if fn is None:
        return []
    try:
        out = fn(recipe=recipe, dossier=dossier, target=target, k=k)
    except Exception:  # noqa: BLE001
        # A suggester crashing must NEVER break the agent's flow.
        return []
    for c in out:
        c.slot_name = slot_name
    return out


__all__ = [
    "SUGGESTERS",
    "SuggesterFn",
    "register_suggester",
    "suggest",
    "supported_slot_names",
]
