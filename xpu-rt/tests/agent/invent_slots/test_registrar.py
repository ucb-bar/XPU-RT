"""Tests for the invent-slot registrar + seed callables."""

from __future__ import annotations

import pytest
from compgen.agent.invent_slots import seeds as seed_mod
from compgen.agent.invent_slots.registrar import register_invent_slots
from compgen.llm.registry import Registry

SLOT_NAMES = (
    "propose_layout_plan",
    "propose_fusion",
    "propose_peephole_pattern",
    "propose_numerics_plan",
    "propose_dequant_fusion",
    # Phase 5 invent-slots (P15, wave 5)
    "propose_buffer_lifetime_plan",
    "propose_rematerialization_plan",
    # Phase 4 ETC megakernel invent-slots (Event Tensor Compiler integration)
    "propose_megakernel_synthesis",
    "propose_scheduling_policy",
)


def test_registrar_registers_all_slots() -> None:
    r = Registry()
    registered = register_invent_slots(r)
    assert set(registered) == set(SLOT_NAMES)
    # Correct phase assignments per the plan
    assert r.lookup_invent_slot("propose_layout_plan", phase=3) is not None
    assert r.lookup_invent_slot("propose_fusion", phase=3) is not None
    assert r.lookup_invent_slot("propose_peephole_pattern", phase=2) is not None
    assert r.lookup_invent_slot("propose_numerics_plan", phase=2) is not None
    assert r.lookup_invent_slot("propose_dequant_fusion", phase=2) is not None
    assert r.lookup_invent_slot("propose_buffer_lifetime_plan", phase=5) is not None
    assert r.lookup_invent_slot("propose_rematerialization_plan", phase=5) is not None
    assert r.lookup_invent_slot("propose_megakernel_synthesis", phase=4) is not None
    assert r.lookup_invent_slot("propose_scheduling_policy", phase=4) is not None


def test_registrar_is_idempotent() -> None:
    r = Registry()
    register_invent_slots(r)
    registered_2 = register_invent_slots(r)
    assert registered_2 == []  # second call: nothing new


def test_slots_are_not_stubs() -> None:
    r = Registry()
    register_invent_slots(r)
    for name in SLOT_NAMES:
        slot = r.lookup_invent_slot(name)
        assert slot is not None
        assert slot.is_stub is False


@pytest.mark.parametrize(
    "seed_fn_name",
    [
        "propose_layout_plan_seed",
        "propose_fusion_seed",
        "propose_peephole_pattern_seed",
        "propose_numerics_plan_seed",
        "propose_dequant_fusion_seed",
        "propose_buffer_lifetime_plan_seed",
        "propose_rematerialization_plan_seed",
        "propose_megakernel_synthesis_seed",
        "propose_scheduling_policy_seed",
    ],
)
def test_every_seed_produces_structural_ok_payload(seed_fn_name: str) -> None:
    seed_fn = getattr(seed_mod, seed_fn_name)
    payload = seed_fn()
    # Structural contract requires these keys
    assert "chosen" in payload
    assert "select_vs_invent" in payload
    assert payload["select_vs_invent"] in ("select", "invent")
    # candidates is non-empty list
    assert isinstance(payload["candidates"], list)
    assert len(payload["candidates"]) >= 1


def test_slot_verify_accepts_its_own_seed() -> None:
    r = Registry()
    register_invent_slots(r)
    for name in SLOT_NAMES:
        slot = r.lookup_invent_slot(name)
        seed_payload = slot.propose_baseline()
        # Default composite (structural-only when no ref_fn/got_fn)
        gate_result = slot.verify(seed_payload)
        assert gate_result["status"] == "accepted", f"{name} seed should pass structural"
        assert "gate_trace" in gate_result["details"]


def test_global_registry_picks_up_slots_on_import() -> None:
    import compgen.agent.invent_slots  # noqa: F401
    from compgen.llm import get_registry

    r = get_registry()
    for name in SLOT_NAMES:
        slot = r.lookup_invent_slot(name)
        assert slot is not None, f"{name} not registered in global registry"
