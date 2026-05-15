"""Tests for compgen.passes.scheduler."""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.audit.errors import (
    PairContractViolation,
    PassPlanInvalid,
    PhaseTransitionViolation,
)
from compgen.passes import (
    PASS_PHASES,
    PassCard,
    PassCardRegistry,
    PassPlanReport,
    PassPlanStep,
    assert_pass_plan_valid,
    default_phase_for_family,
    default_registry_root,
    inspect_pass_plan,
    phase_index,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _card(
    pass_id: str,
    *,
    family: str = "tiling",
    phase: str = "",
    requires_after: tuple[str, ...] = (),
    excludes: tuple[str, ...] = (),
    refinement: str = "bit_equality",
) -> PassCard:
    return PassCard(
        schema_version="pass_card_v1",
        pass_id=pass_id,
        display_name=pass_id,
        level="payload",
        family=family,
        reads=("a.json",),
        writes=("b.json",),
        preconditions=("x",),
        invalidates=("payload_summary",),
        preserves_refinement=refinement,
        verification=("structural",),
        cost="cheap",
        failure_modes=("x",),
        phase=phase,
        requires_after=requires_after,
        excludes=excludes,
    )


def _registry(*cards: PassCard) -> PassCardRegistry:
    reg = PassCardRegistry()
    for card in cards:
        reg.cards[card.pass_id] = card
    return reg


# --------------------------------------------------------------------------- #
# Phase utilities
# --------------------------------------------------------------------------- #


def test_pass_phases_match_spec() -> None:
    assert PASS_PHASES == (
        "canonicalize", "analyze", "optimize", "verify", "emit",
    )


def test_phase_index_strict_order() -> None:
    for i in range(len(PASS_PHASES) - 1):
        assert phase_index(PASS_PHASES[i]) < phase_index(PASS_PHASES[i + 1])


def test_default_phase_for_family() -> None:
    assert default_phase_for_family("canonicalize") == "canonicalize"
    assert default_phase_for_family("layout_pipeline") == "optimize"
    assert default_phase_for_family("codegen") == "emit"
    assert default_phase_for_family("eqsat") == "optimize"
    assert default_phase_for_family("event_tensor") == "emit"
    # Unknown family falls back to "optimize" (the most permissive)
    assert default_phase_for_family("totally_made_up") == "optimize"


def test_card_effective_phase_falls_back_to_family() -> None:
    card = _card("x", family="layout_pipeline")
    assert card.effective_phase() == "optimize"


def test_card_explicit_phase_overrides_family_default() -> None:
    card = _card("x", family="layout_pipeline", phase="canonicalize")
    assert card.effective_phase() == "canonicalize"


# --------------------------------------------------------------------------- #
# PassPlanStep round-trip
# --------------------------------------------------------------------------- #


def test_step_round_trip() -> None:
    step = PassPlanStep(
        pass_id="set_tile_params",
        region_id="matmul_0",
        candidate_id="tile_M16_N16_K16",
        rationale={"why": "smallest predicted runtime"},
    )
    assert PassPlanStep.from_dict(step.to_dict()) == step


def test_step_from_dict_missing_pass_id_raises() -> None:
    with pytest.raises(PassPlanInvalid, match="pass_id"):
        PassPlanStep.from_dict({"region_id": "x"})


# --------------------------------------------------------------------------- #
# Structural checks
# --------------------------------------------------------------------------- #


def test_empty_plan_is_valid() -> None:
    reg = _registry()
    report = inspect_pass_plan([], registry=reg)
    assert report.holds


def test_unknown_pass_id_fails_structural() -> None:
    reg = _registry(_card("known_pass"))
    plan = [PassPlanStep(pass_id="ghost_pass")]
    report = inspect_pass_plan(plan, registry=reg)
    assert not report.holds
    assert not report.structural_ok
    assert "ghost_pass" in report.structural_detail


def test_duplicate_step_fails_structural() -> None:
    reg = _registry(_card("p1"))
    plan = [
        PassPlanStep(pass_id="p1", region_id="r0", candidate_id="c0"),
        PassPlanStep(pass_id="p1", region_id="r0", candidate_id="c0"),
    ]
    report = inspect_pass_plan(plan, registry=reg)
    assert not report.structural_ok
    assert "duplicate" in report.structural_detail.lower()


def test_candidate_id_outside_allowlist_fails_structural() -> None:
    reg = _registry(_card("p1"))
    plan = [PassPlanStep(pass_id="p1", candidate_id="cand_x")]
    report = inspect_pass_plan(
        plan, registry=reg, candidate_ids_allowed=["cand_a"],
    )
    assert not report.structural_ok
    assert "candidate_id" in report.structural_detail


def test_candidate_id_in_allowlist_passes() -> None:
    reg = _registry(_card("p1"))
    plan = [PassPlanStep(pass_id="p1", candidate_id="cand_a")]
    report = inspect_pass_plan(
        plan, registry=reg, candidate_ids_allowed=["cand_a", "cand_b"],
    )
    assert report.structural_ok


# --------------------------------------------------------------------------- #
# Phase ordering
# --------------------------------------------------------------------------- #


def test_phase_ordering_canonical_to_emit_passes() -> None:
    reg = _registry(
        _card("canon", family="canonicalize"),
        _card("opt", family="tiling"),  # default phase: optimize
        _card("verify_pass", family="verify"),
        _card("emit_pass", family="codegen"),
    )
    plan = [
        PassPlanStep(pass_id="canon"),
        PassPlanStep(pass_id="opt"),
        PassPlanStep(pass_id="verify_pass"),
        PassPlanStep(pass_id="emit_pass"),
    ]
    report = inspect_pass_plan(plan, registry=reg)
    assert report.holds, report.detail


def test_phase_ordering_optimize_before_canonicalize_fails() -> None:
    reg = _registry(
        _card("canon", family="canonicalize"),
        _card("opt", family="tiling"),
    )
    plan = [
        PassPlanStep(pass_id="opt"),
        PassPlanStep(pass_id="canon"),
    ]
    report = inspect_pass_plan(plan, registry=reg)
    assert not report.phase_ok
    with pytest.raises(PhaseTransitionViolation, match="phase"):
        assert_pass_plan_valid(plan, registry=reg)


def test_phase_ordering_same_phase_in_any_order_passes() -> None:
    """Two optimize-phase passes can run in either order."""
    reg = _registry(
        _card("opt_a", family="tiling"),
        _card("opt_b", family="fusion"),
    )
    plan_ab = [PassPlanStep(pass_id="opt_a"), PassPlanStep(pass_id="opt_b")]
    plan_ba = [PassPlanStep(pass_id="opt_b"), PassPlanStep(pass_id="opt_a")]
    assert inspect_pass_plan(plan_ab, registry=reg).holds
    assert inspect_pass_plan(plan_ba, registry=reg).holds


# --------------------------------------------------------------------------- #
# requires_after pair contracts
# --------------------------------------------------------------------------- #


def test_requires_after_satisfied_passes() -> None:
    reg = _registry(
        _card("p1", requires_after=("p2",)),
        _card("p2"),
    )
    plan = [PassPlanStep(pass_id="p1"), PassPlanStep(pass_id="p2")]
    report = inspect_pass_plan(plan, registry=reg)
    assert report.holds


def test_requires_after_missing_pass_fails() -> None:
    reg = _registry(_card("p1", requires_after=("p2",)), _card("p2"))
    plan = [PassPlanStep(pass_id="p1")]  # p2 missing
    report = inspect_pass_plan(plan, registry=reg)
    assert not report.requires_after_ok
    with pytest.raises(PairContractViolation, match="p2"):
        assert_pass_plan_valid(plan, registry=reg)


def test_requires_after_wrong_order_fails() -> None:
    """If p1 requires p2 to follow, but p2 appears before p1, fail."""
    reg = _registry(_card("p1", requires_after=("p2",)), _card("p2"))
    plan = [PassPlanStep(pass_id="p2"), PassPlanStep(pass_id="p1")]
    report = inspect_pass_plan(plan, registry=reg)
    assert not report.requires_after_ok


# --------------------------------------------------------------------------- #
# excludes pair contracts
# --------------------------------------------------------------------------- #


def test_excludes_violation_fails() -> None:
    reg = _registry(_card("p1", excludes=("p2",)), _card("p2"))
    plan = [PassPlanStep(pass_id="p1"), PassPlanStep(pass_id="p2")]
    report = inspect_pass_plan(plan, registry=reg)
    assert not report.excludes_ok
    with pytest.raises(PairContractViolation, match="excludes"):
        assert_pass_plan_valid(plan, registry=reg)


def test_excludes_no_other_in_plan_passes() -> None:
    reg = _registry(_card("p1", excludes=("p2",)), _card("p2"))
    plan = [PassPlanStep(pass_id="p1")]
    report = inspect_pass_plan(plan, registry=reg)
    assert report.holds


# --------------------------------------------------------------------------- #
# Real seed-card registry roundtrip
# --------------------------------------------------------------------------- #


def test_real_registry_loads_phases() -> None:
    """Every card under default_registry_root has a resolvable phase."""
    reg = PassCardRegistry.load(default_registry_root())
    for card in reg:
        assert card.effective_phase() in PASS_PHASES, card.pass_id


def test_real_registry_phases_present_includes_optimize_and_emit() -> None:
    """The 60-card registry covers at least the optimize + emit phases."""
    reg = PassCardRegistry.load(default_registry_root())
    phases = reg.phases_present()
    assert "optimize" in phases
    assert "emit" in phases


def test_real_seed_plan_is_valid() -> None:
    """A canonicalize → optimize → emit plan over real cards is valid."""
    reg = PassCardRegistry.load(default_registry_root())
    plan = [
        PassPlanStep(pass_id="simplify_while_loop"),  # canonicalize
        PassPlanStep(pass_id="set_tile_params"),  # optimize
        PassPlanStep(pass_id="alias_io_buffers"),  # emit
    ]
    report = inspect_pass_plan(plan, registry=reg)
    assert report.holds, report.detail


def test_real_seed_plan_wrong_order_fails() -> None:
    """Putting an emit pass before an optimize pass on real cards
    raises PhaseTransitionViolation."""
    reg = PassCardRegistry.load(default_registry_root())
    plan = [
        PassPlanStep(pass_id="alias_io_buffers"),  # emit
        PassPlanStep(pass_id="set_tile_params"),  # optimize
    ]
    with pytest.raises(PhaseTransitionViolation):
        assert_pass_plan_valid(plan, registry=reg)
