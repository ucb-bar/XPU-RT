"""Tests for compgen.passes.refinement."""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.audit.errors import RefinementMonotonicityViolation
from compgen.passes.cards import (
    PassCard, default_registry_root, load_card, resolve_card_path,
)
from compgen.passes.refinement import (
    RefinementChainReport,
    RefinementLevel,
    assert_refinement_chain_valid,
    compute_claimable_refinement,
    inspect_refinement_chain,
)


def _card(pass_id: str, refinement: str) -> PassCard:
    return PassCard(
        schema_version="pass_card_v1",
        pass_id=pass_id,
        display_name=pass_id,
        level="payload",
        family="tiling",
        reads=("a.json",),
        writes=("b.json",),
        preconditions=("x",),
        invalidates=("payload_summary",),
        preserves_refinement=refinement,
        verification=("structural",),
        cost="cheap",
        failure_modes=("x",),
    )


# --------------------------------------------------------------------------- #
# Total order
# --------------------------------------------------------------------------- #


def test_refinement_level_total_order() -> None:
    levels = [
        RefinementLevel.UNKNOWN,
        RefinementLevel.NONE,
        RefinementLevel.TOLERANCE_EPS,
        RefinementLevel.BIT_EQUALITY,
    ]
    for i, lvl in enumerate(levels):
        for j in range(i + 1, len(levels)):
            assert lvl < levels[j], f"{lvl} should be < {levels[j]}"


def test_refinement_round_trip() -> None:
    for s in ("bit_equality", "tolerance_eps", "none", "unknown"):
        lvl = RefinementLevel.from_string(s)
        assert lvl.to_string() == s


def test_refinement_invalid_string_raises() -> None:
    with pytest.raises(ValueError, match="refinement"):
        RefinementLevel.from_string("totally_made_up")


# --------------------------------------------------------------------------- #
# Chain composition
# --------------------------------------------------------------------------- #


def test_empty_chain_is_bit_equality() -> None:
    """Identity chain — no passes have run, so we can claim the strongest
    refinement (the recipe IS bit-equal to eager, trivially)."""
    assert compute_claimable_refinement([]) == RefinementLevel.BIT_EQUALITY


def test_two_bit_equality_chain_is_bit_equality() -> None:
    chain = [
        _card("set_tile_params", "bit_equality"),
        _card("fuse_producer_consumer", "bit_equality"),
    ]
    assert compute_claimable_refinement(chain) == RefinementLevel.BIT_EQUALITY


def test_bit_equality_then_tolerance_is_tolerance() -> None:
    chain = [
        _card("set_tile_params", "bit_equality"),
        _card("layout_layout", "tolerance_eps"),
    ]
    assert compute_claimable_refinement(chain) == RefinementLevel.TOLERANCE_EPS


def test_chain_with_unknown_collapses_to_unknown() -> None:
    chain = [
        _card("set_tile_params", "bit_equality"),
        _card("mystery_pass", "unknown"),
    ]
    assert compute_claimable_refinement(chain) == RefinementLevel.UNKNOWN


def test_chain_with_none_collapses_to_none() -> None:
    chain = [
        _card("set_tile_params", "bit_equality"),
        _card("approximation_pass", "none"),
    ]
    assert compute_claimable_refinement(chain) == RefinementLevel.NONE


# --------------------------------------------------------------------------- #
# Assertion (raising form)
# --------------------------------------------------------------------------- #


def test_claim_equal_to_claimable_passes() -> None:
    chain = [_card("a", "tolerance_eps")]
    assert_refinement_chain_valid(chain, "tolerance_eps")  # no raise


def test_under_claim_passes() -> None:
    """Claiming weaker than what's preserved is voluntarily under-claiming."""
    chain = [_card("a", "bit_equality")]
    assert_refinement_chain_valid(chain, "tolerance_eps")
    assert_refinement_chain_valid(chain, "none")


def test_over_claim_raises() -> None:
    chain = [_card("a", "tolerance_eps")]
    with pytest.raises(RefinementMonotonicityViolation, match="bit_equality"):
        assert_refinement_chain_valid(chain, "bit_equality", recipe_id="r1")


def test_over_claim_with_unknown_in_chain() -> None:
    chain = [
        _card("a", "bit_equality"),
        _card("b", "unknown"),
    ]
    with pytest.raises(RefinementMonotonicityViolation):
        assert_refinement_chain_valid(chain, "bit_equality")


def test_empty_chain_supports_bit_equality_claim() -> None:
    assert_refinement_chain_valid([], "bit_equality")


# --------------------------------------------------------------------------- #
# Inspection (non-raising form)
# --------------------------------------------------------------------------- #


def test_inspect_holds_when_claim_is_valid() -> None:
    chain = [_card("a", "bit_equality")]
    rpt = inspect_refinement_chain(chain, "bit_equality")
    assert isinstance(rpt, RefinementChainReport)
    assert rpt.holds
    assert rpt.claimable == "bit_equality"
    assert rpt.detail == ""


def test_inspect_does_not_raise_on_violation() -> None:
    chain = [_card("a", "tolerance_eps")]
    rpt = inspect_refinement_chain(chain, "bit_equality")
    assert not rpt.holds
    assert "exceeds claimable" in rpt.detail


def test_inspect_invalid_claim_returns_holds_false() -> None:
    chain = [_card("a", "bit_equality")]
    rpt = inspect_refinement_chain(chain, "totally_made_up")
    assert not rpt.holds


# --------------------------------------------------------------------------- #
# Real seed cards
# --------------------------------------------------------------------------- #


def test_seed_cards_chain_supports_bit_equality() -> None:
    """The two production seed cards both preserve bit_equality, so a
    recipe applying both can legitimately claim bit_equality."""
    set_tile = load_card(resolve_card_path("set_tile_params"))
    fuse = load_card(resolve_card_path("fuse_producer_consumer"))
    chain = [set_tile, fuse]
    assert_refinement_chain_valid(chain, "bit_equality")
    rpt = inspect_refinement_chain(chain, "bit_equality")
    assert rpt.holds
    assert rpt.chain == (
        ("set_tile_params", "bit_equality"),
        ("fuse_producer_consumer", "bit_equality"),
    )
