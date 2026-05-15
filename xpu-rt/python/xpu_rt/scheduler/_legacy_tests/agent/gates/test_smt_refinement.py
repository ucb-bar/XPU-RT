"""Tests for smt_refinement_gate (opt-in)."""

from __future__ import annotations

from unittest.mock import patch

from compgen.agent.gates import smt_refinement_gate
from xdsl.dialects.builtin import ModuleOp


def test_deferred_when_not_required() -> None:
    r = smt_refinement_gate({})
    assert r["status"] == "deferred"
    assert "not requested" in r["details"]["reason"]


def test_deferred_when_required_but_modules_missing() -> None:
    r = smt_refinement_gate({}, require_smt=True)
    assert r["status"] == "deferred"


def test_accepts_identical_modules() -> None:
    # validate_translation short-circuits on identical modules → valid
    m = ModuleOp([])
    r = smt_refinement_gate(
        {},
        require_smt=True,
        source_module=m,
        target_module=m,
    )
    assert r["status"] == "accepted"


def test_rejects_on_translation_failure() -> None:
    """If validate_translation raises, the gate rejects cleanly."""
    from compgen.agent.gates import smt_refinement as sr

    def _boom(*a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("z3 blew up")

    with patch(
        "compgen.ir.semantic.translation_validation.validate_translation",
        side_effect=_boom,
    ):
        m1 = ModuleOp([])
        m2 = ModuleOp([])
        r = sr.smt_refinement_gate(
            {},
            require_smt=True,
            source_module=m1,
            target_module=m2,
        )
        assert r["status"] == "rejected"
