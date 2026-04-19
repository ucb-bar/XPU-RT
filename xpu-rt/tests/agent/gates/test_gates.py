"""Tests for the structural / differential / composite gates."""

from __future__ import annotations

import pytest
import torch

from compgen.agent.gates import (
    composite_gate,
    differential_gate,
    structural_gate,
)


def test_structural_accepts_well_formed_proposal() -> None:
    r = structural_gate({"chosen": {"x": 1}, "select_vs_invent": "invent"})
    assert r["status"] == "accepted"


def test_structural_rejects_missing_chosen() -> None:
    r = structural_gate({"select_vs_invent": "invent"})
    assert r["status"] == "rejected"
    assert "chosen" in r["details"]["missing"]


def test_structural_rejects_bad_select_vs_invent() -> None:
    r = structural_gate({"chosen": {}, "select_vs_invent": "bogus"})
    assert r["status"] == "rejected"


def test_structural_accepts_xdsl_module() -> None:
    from xdsl.dialects.builtin import ModuleOp

    r = structural_gate({}, module=ModuleOp([]))
    assert r["status"] == "accepted"
    assert r["details"]["kind"] == "xdsl_module"


def test_differential_accepts_matching_tensors() -> None:
    t = torch.zeros(4)
    r = differential_gate(
        {},
        ref_fn=lambda: t.clone(),
        got_fn=lambda: t.clone(),
    )
    assert r["status"] == "accepted"


def test_differential_rejects_mismatch() -> None:
    r = differential_gate(
        {},
        ref_fn=lambda: torch.zeros(4),
        got_fn=lambda: torch.ones(4),
    )
    assert r["status"] == "rejected"


def test_differential_deferred_without_context() -> None:
    r = differential_gate({})
    assert r["status"] == "deferred"


def test_composite_short_circuits_on_first_rejection() -> None:
    r = composite_gate(
        {},   # missing chosen/select_vs_invent → structural rejects
        gates=[structural_gate, differential_gate],
    )
    assert r["status"] == "rejected"
    trace = r["details"]["gate_trace"]
    # Only structural ran (short-circuit)
    assert len(trace) == 1
    assert trace[0]["gate"] == "structural_gate"


def test_composite_fail_fast_false_runs_all() -> None:
    r = composite_gate(
        {},   # structural rejects
        gates=[structural_gate, differential_gate],
        fail_fast=False,
    )
    assert r["status"] == "rejected"
    assert len(r["details"]["gate_trace"]) == 2


def test_composite_accepts_when_all_accept() -> None:
    r = composite_gate(
        {"chosen": {}, "select_vs_invent": "select"},
        gates=[structural_gate],
    )
    assert r["status"] == "accepted"
