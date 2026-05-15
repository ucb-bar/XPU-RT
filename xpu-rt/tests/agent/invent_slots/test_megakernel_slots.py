"""Tests for the ETC megakernel + scheduling-policy invent-slots."""

from __future__ import annotations

from compgen.agent.invent_slots.registrar import _SLOT_SPECS, register_invent_slots
from compgen.agent.invent_slots.seeds import (
    propose_megakernel_synthesis_seed,
    propose_scheduling_policy_seed,
)
from compgen.llm.registry import Registry

# ---------------------------------------------------------------------------
# Spec presence / shape
# ---------------------------------------------------------------------------


def test_megakernel_synthesis_slot_is_registered_in_phase_4() -> None:
    spec = next(s for s in _SLOT_SPECS if s["name"] == "propose_megakernel_synthesis")
    assert spec["phase"] == 4
    assert spec["output_op"] == "recipe.propose_megakernel_synthesis"
    assert spec["autocomp_cost_impact"] == "very_high"


def test_scheduling_policy_slot_is_registered_in_phase_4() -> None:
    spec = next(s for s in _SLOT_SPECS if s["name"] == "propose_scheduling_policy")
    assert spec["phase"] == 4
    assert spec["output_op"] == "recipe.propose_scheduling_policy"


def test_register_invent_slots_idempotent_for_megakernel_slots() -> None:
    reg = Registry()
    first = set(register_invent_slots(reg))
    second = set(register_invent_slots(reg))
    assert "propose_megakernel_synthesis" in first
    assert "propose_scheduling_policy" in first
    assert second == set(), "second call should not re-register anything"


# ---------------------------------------------------------------------------
# Megakernel synthesis seed
# ---------------------------------------------------------------------------


def test_megakernel_seed_with_regions_produces_named_megakernel() -> None:
    seed = propose_megakernel_synthesis_seed(
        candidate_regions=["region_mm", "region_rs"],
        inter_region_edges=[{"shape": [4], "wait_count": 2}],
        task_shape=[4],
    )
    chosen = seed["chosen"]
    assert chosen["megakernel_name"] == "mk_region_mm_region_rs"
    assert chosen["fused_region_refs"] == ["region_mm", "region_rs"]
    assert chosen["event_tensor_decls"][0]["shape"] == [4]
    assert chosen["event_tensor_decls"][0]["wait_count"] == 2
    assert chosen["task_partition"] == {"region_mm": [4], "region_rs": [4]}
    assert seed["select_vs_invent"] == "invent"


def test_megakernel_seed_without_regions_still_returns_payload() -> None:
    """Empty candidate set -> structurally valid payload that the gate
    will reject; we don't crash so the LLM gets a chance to override."""
    seed = propose_megakernel_synthesis_seed()
    assert seed["chosen"]["megakernel_name"] == "mk_unspecified"
    assert seed["chosen"]["fused_region_refs"] == []


def test_megakernel_seed_supports_explicit_name_override() -> None:
    seed = propose_megakernel_synthesis_seed(
        candidate_regions=["a", "b"],
        megakernel_name="qwen3_decode_step",
    )
    assert seed["chosen"]["megakernel_name"] == "qwen3_decode_step"


# ---------------------------------------------------------------------------
# Scheduling-policy seed
# ---------------------------------------------------------------------------


def test_policy_seed_defaults_to_static_when_no_data_dep_edges() -> None:
    seed = propose_scheduling_policy_seed(sm_count=108)
    assert seed["chosen"]["policy"] == "static"
    assert seed["chosen"]["sm_count"] == 108
    assert seed["chosen"]["early_push"] is False
    assert seed["chosen"]["dynamic_features"] == []


def test_policy_seed_picks_dynamic_when_data_dep_edges_present() -> None:
    seed = propose_scheduling_policy_seed(
        sm_count=108,
        has_data_dependent_edges=True,
        data_dependent_edges=["topk", "exp_indptr"],
    )
    assert seed["chosen"]["policy"] == "dynamic"
    assert seed["chosen"]["dynamic_features"] == ["topk", "exp_indptr"]
