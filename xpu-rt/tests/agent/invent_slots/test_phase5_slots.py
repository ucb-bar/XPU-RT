"""Tests for the Phase 5 invent-slots (P15)."""

from __future__ import annotations

import compgen.agent.invent_slots  # noqa: F401
from compgen.agent.invent_slots.seeds import (
    propose_buffer_lifetime_plan_seed,
    propose_rematerialization_plan_seed,
)
from compgen.agent.invent_slots.registrar import register_invent_slots
from compgen.llm.registry import Registry


def test_buffer_lifetime_seed_shape() -> None:
    payload = propose_buffer_lifetime_plan_seed()
    assert "chosen" in payload
    assert payload["chosen"]["coloring_policy"] in ("first_fit", "greedy", "min_peak")
    assert payload["select_vs_invent"] == "invent"


def test_remat_seed_respects_memory_budget() -> None:
    payload = propose_rematerialization_plan_seed(memory_budget_bytes=4096)
    assert payload["chosen"]["memory_budget_bytes"] == 4096


def test_phase5_slots_registered_in_fresh_registry() -> None:
    r = Registry()
    names = register_invent_slots(r)
    assert "propose_buffer_lifetime_plan" in names
    assert "propose_rematerialization_plan" in names
    slot = r.lookup_invent_slot("propose_buffer_lifetime_plan", phase=5)
    assert slot is not None
    assert slot.is_stub is False


def test_phase5_slot_seed_passes_default_gate() -> None:
    r = Registry()
    register_invent_slots(r)
    for name in ("propose_buffer_lifetime_plan", "propose_rematerialization_plan"):
        slot = r.lookup_invent_slot(name, phase=5)
        seed = slot.propose_baseline()
        gate_result = slot.verify(seed)
        assert gate_result["status"] == "accepted"
