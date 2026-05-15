"""Deterministic provider routing.

Given a ``contract_kind`` and a ``target_family``, returns an
ordered list of provider ids that *card-wise* claim support. The
router is **deterministic** — same inputs → same output across
runs — so the auction is reproducible and the architecture audit
can verify the routing decisions.

The router operates on **declared capability** (from the
:class:`ProviderCard`'s ``contract_kinds`` + ``target_families``).
It does NOT take live probe status into account; downstream
auction code intersects this with the probe results.
"""

from __future__ import annotations

from compgen.providers.card_loader import iter_provider_cards
from compgen.providers.provider_types import ProviderCard


# Per-contract-kind priority — providers within a kind are sorted
# by integration_level (promote → verify → generate → probe → card_only)
# with this hand-curated tie-breaker tail.
INTEGRATION_LEVEL_RANK = {
    "promote": 0,
    "verify": 1,
    "generate": 2,
    "probe": 3,
    "card_only": 4,
}

# When two cards have the same integration_level, prefer the one that
# also appears in this hand-curated preference list (lower index wins).
KIND_PREFERENCE: dict[str, tuple[str, ...]] = {
    "matmul": ("cffi_c", "triton", "cutlass_cute", "tilelang", "autocomp"),
    "pointwise": ("cffi_c", "triton", "tilelang", "autocomp"),
    "fused_region": ("triton", "tilelang", "cutlass_cute", "autocomp", "kernelblaster"),
    "attention": ("triton", "thunderkittens", "autocomp", "kernelblaster"),
    "softmax": ("triton", "cffi_c"),
    "reduction": ("triton", "cffi_c"),
    "quantized_matmul": ("bitblas", "cutlass_cute", "tilelang"),
    "conv": ("cffi_c", "triton"),
}


def _level_rank(card: ProviderCard) -> int:
    return INTEGRATION_LEVEL_RANK.get(card.integration_level, 99)


def _preference_index(provider_id: str, kind: str) -> int:
    prefs = KIND_PREFERENCE.get(kind, ())
    return prefs.index(provider_id) if provider_id in prefs else len(prefs)


def route_for(
    *,
    contract_kind: str,
    target_family: str,
    cards: tuple[ProviderCard, ...] | None = None,
) -> tuple[str, ...]:
    """Return the deterministic ordered list of provider ids.

    Filters to cards whose declared ``contract_kinds`` and
    ``target_families`` include the requested pair. Then sorts by
    ``integration_level`` rank, then by per-kind preference, then
    by alphabetical provider id (final tie-breaker).
    """

    pool = cards if cards is not None else tuple(iter_provider_cards())
    matched = [
        c
        for c in pool
        if contract_kind in c.contract_kinds
        and target_family in c.target_families
    ]
    matched.sort(
        key=lambda c: (
            _level_rank(c),
            _preference_index(c.provider_id, contract_kind),
            c.provider_id,
        )
    )
    return tuple(c.provider_id for c in matched)


def supported_kinds(
    cards: tuple[ProviderCard, ...] | None = None,
) -> tuple[str, ...]:
    """Closed set of contract kinds any shipped card claims to handle."""

    pool = cards if cards is not None else tuple(iter_provider_cards())
    out: set[str] = set()
    for c in pool:
        out.update(c.contract_kinds)
    return tuple(sorted(out))


def supported_target_families(
    cards: tuple[ProviderCard, ...] | None = None,
) -> tuple[str, ...]:
    pool = cards if cards is not None else tuple(iter_provider_cards())
    out: set[str] = set()
    for c in pool:
        out.update(c.target_families)
    return tuple(sorted(out))


def route_matrix(
    cards: tuple[ProviderCard, ...] | None = None,
) -> dict[tuple[str, str], tuple[str, ...]]:
    """Build a full {(kind, family): [provider_ids]} matrix.

    Useful for audit reports and the evidence-pack matrix CSV.
    """

    pool = cards if cards is not None else tuple(iter_provider_cards())
    out: dict[tuple[str, str], tuple[str, ...]] = {}
    for k in supported_kinds(pool):
        for f in supported_target_families(pool):
            ordered = route_for(contract_kind=k, target_family=f, cards=pool)
            if ordered:
                out[(k, f)] = ordered
    return out


__all__ = [
    "INTEGRATION_LEVEL_RANK",
    "KIND_PREFERENCE",
    "route_for",
    "route_matrix",
    "supported_kinds",
    "supported_target_families",
]
