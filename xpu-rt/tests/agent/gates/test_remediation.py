"""Unit tests for the gate remediation catalogue.

Every known ``reason`` string must yield a non-None
``remediation_hint``; unknown reasons must return ``None`` (not a
fabricated hint).
"""

from __future__ import annotations

import pytest
import torch
from compgen.agent.gates import (
    composite_gate,
    differential_gate,
    structural_gate,
)
from compgen.agent.gates._remediation import (
    add_remediation,
    known_reasons,
)


def test_known_reasons_all_produce_hints() -> None:
    for reason in known_reasons():
        result = {
            "status": "rejected",
            "details": {"reason": reason},
        }
        add_remediation(result, slot_name="test_slot")
        hint = result["details"]["remediation_hint"]
        assert hint is not None, f"no hint for reason={reason!r}"
        assert isinstance(hint, str) and len(hint) > 20


def test_unknown_reason_returns_none_not_fabricated() -> None:
    result = {
        "status": "rejected",
        "details": {"reason": "some_reason_we_have_never_heard_of"},
    }
    add_remediation(result)
    # Intentionally None — hallucinating a hint would mislead the LLM.
    assert result["details"]["remediation_hint"] is None


def test_accepted_results_are_not_touched() -> None:
    before = {"status": "accepted", "details": {"kind": "xdsl_module"}}
    after = add_remediation(dict(before))
    assert after["details"] == before["details"]


def test_slot_name_surfaces_in_hint() -> None:
    result = {
        "status": "rejected",
        "details": {"reason": "missing_required_keys"},
    }
    add_remediation(result, slot_name="propose_fusion")
    assert "propose_fusion" in result["details"]["remediation_hint"]


def test_numerical_drift_synthesised_from_comparisons() -> None:
    """Differential gate signals drift via comparisons[].passed=False."""
    result = {
        "status": "rejected",
        "details": {
            "comparisons": [
                {
                    "index": 0,
                    "passed": False,
                    "max_abs_error": 1.2,
                    "max_rel_error": 0.08,
                },
                {
                    "index": 1,
                    "passed": True,
                    "max_abs_error": 1e-6,
                    "max_rel_error": 1e-6,
                },
            ],
        },
    }
    add_remediation(result)
    assert result["details"]["reason"] == "numerical_drift"
    hint = result["details"]["remediation_hint"]
    assert hint is not None
    assert "atol" in hint or "tolerance" in hint.lower()
    worst = result["details"].get("worst_comparison")
    assert worst is not None
    assert worst["index"] == 0
    assert worst["max_abs_error"] == pytest.approx(1.2)


def test_composite_gate_populates_remediation_on_rejection() -> None:
    """End-to-end: composite_gate → structural rejects → hint surfaces."""
    result = composite_gate(
        {},  # missing chosen/select_vs_invent → structural rejects
        gates=[structural_gate],
        slot_name="my_slot",
    )
    assert result["status"] == "rejected"
    hint = result["details"]["remediation_hint"]
    assert hint is not None
    assert "my_slot" in hint


def test_composite_gate_accepted_no_hint() -> None:
    result = composite_gate(
        {"chosen": {}, "select_vs_invent": "invent"},
        gates=[structural_gate],
    )
    assert result["status"] == "accepted"
    # On accepted, we do not inject a hint key.
    assert "remediation_hint" not in result["details"]


def test_composite_gate_numerical_drift_chain() -> None:
    """differential gate with mismatching tensors → drift hint surfaces."""
    result = composite_gate(
        {"chosen": {}, "select_vs_invent": "invent"},
        gates=[structural_gate, differential_gate],
        ref_fn=lambda: torch.zeros(4),
        got_fn=lambda: torch.ones(4),
    )
    assert result["status"] == "rejected"
    hint = result["details"]["remediation_hint"]
    assert hint is not None


def test_deferred_hint_fires_on_missing_context() -> None:
    """Differential gate deferred when ctx lacks ref_fn/got_fn."""
    result = composite_gate(
        {"chosen": {}, "select_vs_invent": "invent"},
        gates=[differential_gate],
    )
    assert result["status"] == "deferred"
    hint = result["details"].get("remediation_hint")
    # Deferred-reason hint path ('differential gate requires ...').
    assert hint is not None and "ref_fn" in hint
