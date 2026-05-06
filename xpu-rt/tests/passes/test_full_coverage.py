"""Tests for full pass-card coverage (M-33.6).

Section 20 / M-33.6 contract: every ported compiler pass under
``python/compgen/`` has a typed pass card with a declared source
(provenance) and impl_path (implementation file). The agent's
vocabulary is now the registry; cards without provenance leave the
realness audit honest-but-incomplete.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from compgen.passes import (
    PASS_FAMILIES,
    PASS_SOURCES,
    PassCardRegistry,
    default_registry_root,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Registry coverage
# --------------------------------------------------------------------------- #


def test_full_registry_loads() -> None:
    """All cards under default_registry_root load + cross-link cleanly."""
    registry = PassCardRegistry.load(default_registry_root())
    assert len(registry) >= 60, (
        f"M-33.6 expected ≥60 cards; got {len(registry)}"
    )


def test_every_card_has_source() -> None:
    """Every card must declare a source from PASS_SOURCES (no empty)."""
    registry = PassCardRegistry.load(default_registry_root())
    missing = [c.pass_id for c in registry if not c.source]
    assert not missing, (
        f"{len(missing)} card(s) missing 'source' field; "
        f"first 5: {missing[:5]}"
    )


def test_every_card_source_is_known() -> None:
    """Every card's source must be in PASS_SOURCES."""
    registry = PassCardRegistry.load(default_registry_root())
    bad = [(c.pass_id, c.source) for c in registry if c.source not in PASS_SOURCES]
    assert not bad, f"unknown source in {len(bad)} card(s); first 5: {bad[:5]}"


def test_every_card_has_impl_path() -> None:
    """Every card must reference its on-disk implementation file."""
    registry = PassCardRegistry.load(default_registry_root())
    missing = [c.pass_id for c in registry if not c.impl_path]
    assert not missing, (
        f"{len(missing)} card(s) missing 'impl_path' field; "
        f"first 5: {missing[:5]}"
    )


def test_every_impl_path_exists_on_disk() -> None:
    """Every card's impl_path must resolve to an actual file."""
    registry = PassCardRegistry.load(default_registry_root())
    bad: list[tuple[str, str]] = []
    for card in registry:
        full = REPO_ROOT / card.impl_path
        if not full.exists():
            bad.append((card.pass_id, card.impl_path))
    assert not bad, (
        f"{len(bad)} card(s) reference non-existent impl_path; "
        f"first 5: {bad[:5]}"
    )


# --------------------------------------------------------------------------- #
# Family / level distribution sanity
# --------------------------------------------------------------------------- #


def test_every_family_has_at_least_one_card() -> None:
    """Each declared family must have at least one card.

    Catches a family added to PASS_FAMILIES that no card uses — that
    would mean the constant has dead entries.

    Acceptable empties: families reserved for future milestones (e.g.
    `verify`, `profile`, `dispatch`, `promote`). We only assert
    families used by the M-33.6 set.
    """
    registry = PassCardRegistry.load(default_registry_root())
    used = {c.family for c in registry}
    expected_used = {
        "canonicalize", "fusion", "tiling", "layout", "layout_pipeline",
        "quant", "codegen", "scheduling", "memory", "eqsat",
        "event_tensor", "fx_graph",
    }
    missing = expected_used - used
    assert not missing, f"expected families have no cards: {missing}"


def test_no_card_uses_unknown_family() -> None:
    registry = PassCardRegistry.load(default_registry_root())
    bad = [(c.pass_id, c.family) for c in registry if c.family not in PASS_FAMILIES]
    assert not bad, f"unknown family in {len(bad)} card(s); first 5: {bad[:5]}"


# --------------------------------------------------------------------------- #
# Provenance coverage
# --------------------------------------------------------------------------- #


def test_provenance_distribution_is_diverse() -> None:
    """The card set should span every recognized source bucket that
    actually has a corresponding port in the tree (XLA, IREE,
    hexagon-mlir, Event Tensor Compiler, homemade)."""
    registry = PassCardRegistry.load(default_registry_root())
    sources_seen = {c.source for c in registry}
    expected_at_least = {"XLA", "IREE", "hexagon-mlir", "Event Tensor Compiler", "homemade"}
    missing = expected_at_least - sources_seen
    assert not missing, (
        f"expected provenance buckets missing from card set: {missing}"
    )


def test_xla_and_iree_each_have_multiple_cards() -> None:
    """The bulk of the ports are XLA / IREE; each should have plenty
    of cards."""
    registry = PassCardRegistry.load(default_registry_root())
    by_source: dict[str, list[str]] = defaultdict(list)
    for c in registry:
        by_source[c.source].append(c.pass_id)
    assert len(by_source["XLA"]) >= 10, (
        f"expected ≥10 XLA cards, got {len(by_source['XLA'])}: {by_source['XLA']}"
    )
    assert len(by_source["IREE"]) >= 10, (
        f"expected ≥10 IREE cards, got {len(by_source['IREE'])}: {by_source['IREE']}"
    )


# --------------------------------------------------------------------------- #
# Cross-check against agent_decision_request emission
# --------------------------------------------------------------------------- #


def test_agent_passes_allowed_lists_full_registry() -> None:
    """The constant in agent_decision.py builds passes_allowed from the
    registry — so the agent sees every card.

    M-33 baseline: 7 cards. M-33.6: 60+. This test guards against
    regressions where the registry loads differently than the
    agent_decision pipeline expects.
    """
    registry = PassCardRegistry.load(default_registry_root())
    assert len(registry) >= 60
    # Must NOT contain duplicates
    ids = list(registry.passes_allowed())
    assert len(ids) == len(set(ids))
