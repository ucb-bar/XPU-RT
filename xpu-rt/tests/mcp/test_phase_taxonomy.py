"""H2 — phase taxonomy + ``enter_phase`` tool + phase-scoped gating.

Coverage:

1. ``PHASES`` is a closed enum + all transitions in
   ``LEGAL_TRANSITIONS`` reference known phases.
2. ``is_legal_transition`` honours forward-only by default; ``unsafe``
   opens backward / cross-phase paths.
3. ``is_tool_allowed_in_phase`` handles glob patterns + exact matches.
4. The ``enter_phase`` tool accepts a legal transition and updates
   ``session.current_phase``.
5. Illegal transitions return ``blocked_reason="illegal_transition"``.
6. Unknown phases return ``blocked_reason="unknown_phase"``.
7. With ``COMPGEN_STRICT_PHASE_GATING=1`` set + a phase active,
   ``dispatch_tool`` refuses tools not in the phase allowlist with
   ``blocked_reason="phase_violation"``.
8. Without the env flag, dispatch is unchanged even when the session
   has a ``current_phase``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from compgen.mcp.phase_taxonomy import (
    LEGAL_TRANSITIONS,
    PHASES,
    PHASE_BUNDLE_EMIT,
    PHASE_GRAPH_ANALYSIS,
    PHASE_RECIPE_AUTHORING,
    PHASE_SESSION_INIT,
    is_known_phase,
    is_legal_transition,
    is_tool_allowed_in_phase,
)
from compgen.mcp.server import dispatch_tool
from compgen.mcp.session import SessionManager
from compgen.mcp.tools.lifecycle import enter_phase


# ----------------------------------------------------------------------
# Closed enum + transition matrix
# ----------------------------------------------------------------------


def test_phases_closed_enum_known() -> None:
    assert is_known_phase(PHASE_SESSION_INIT)
    assert is_known_phase(PHASE_BUNDLE_EMIT)
    assert not is_known_phase("not_a_phase")


def test_legal_transitions_only_reference_known_phases() -> None:
    for src, dests in LEGAL_TRANSITIONS.items():
        assert src in PHASES
        for d in dests:
            assert d in PHASES


def test_forward_transition_is_legal() -> None:
    assert is_legal_transition(
        from_phase=PHASE_SESSION_INIT, to_phase=PHASE_GRAPH_ANALYSIS
    )


def test_backward_transition_blocked_without_unsafe() -> None:
    assert not is_legal_transition(
        from_phase=PHASE_RECIPE_AUTHORING, to_phase=PHASE_SESSION_INIT
    )


def test_backward_transition_allowed_with_unsafe() -> None:
    assert is_legal_transition(
        from_phase=PHASE_RECIPE_AUTHORING,
        to_phase=PHASE_SESSION_INIT,
        unsafe=True,
    )


def test_entering_first_phase_from_none_is_legal() -> None:
    assert is_legal_transition(from_phase=None, to_phase=PHASE_SESSION_INIT)


def test_unknown_target_phase_is_illegal() -> None:
    assert not is_legal_transition(
        from_phase=PHASE_SESSION_INIT, to_phase="not_a_phase"
    )


# ----------------------------------------------------------------------
# Per-tool allowlist
# ----------------------------------------------------------------------


def test_allowlist_handles_exact_match() -> None:
    assert is_tool_allowed_in_phase("open_target", PHASE_SESSION_INIT)


def test_allowlist_handles_glob_pattern() -> None:
    # ``list_*`` covers list_packaged_examples + many list_* tools
    assert is_tool_allowed_in_phase("list_packaged_examples", PHASE_SESSION_INIT)


def test_allowlist_refuses_unrelated_tool() -> None:
    assert not is_tool_allowed_in_phase("apply_recipe", PHASE_SESSION_INIT)


def test_allowlist_returns_true_when_no_phase_set() -> None:
    """Backwards-compat: phase=None disables enforcement."""

    assert is_tool_allowed_in_phase("anything", None)


# ----------------------------------------------------------------------
# enter_phase tool
# ----------------------------------------------------------------------


def test_enter_phase_accepts_legal_transition(tmp_path: Path) -> None:
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    s = sm.open()
    r = enter_phase(sm, target_phase=PHASE_SESSION_INIT, session_id=s.session_id)
    assert r["ok"]
    assert r["to_phase"] == PHASE_SESSION_INIT
    r2 = enter_phase(
        sm, target_phase=PHASE_GRAPH_ANALYSIS, session_id=s.session_id
    )
    assert r2["ok"]
    assert s.current_phase == PHASE_GRAPH_ANALYSIS


def test_enter_phase_refuses_illegal_transition(tmp_path: Path) -> None:
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    s = sm.open()
    enter_phase(sm, target_phase=PHASE_SESSION_INIT, session_id=s.session_id)
    r = enter_phase(
        sm, target_phase=PHASE_BUNDLE_EMIT, session_id=s.session_id
    )
    assert r["ok"] is False
    assert r["blocked_reason"] == "illegal_transition"
    # Phase unchanged after refusal.
    assert s.current_phase == PHASE_SESSION_INIT


def test_enter_phase_refuses_unknown_phase(tmp_path: Path) -> None:
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    s = sm.open()
    r = enter_phase(sm, target_phase="not_a_phase", session_id=s.session_id)
    assert r["ok"] is False
    assert r["blocked_reason"] == "unknown_phase"


def test_enter_phase_unsafe_allows_backward(tmp_path: Path) -> None:
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    s = sm.open()
    enter_phase(sm, target_phase=PHASE_SESSION_INIT, session_id=s.session_id)
    enter_phase(sm, target_phase=PHASE_GRAPH_ANALYSIS, session_id=s.session_id)
    r = enter_phase(
        sm,
        target_phase=PHASE_SESSION_INIT,
        session_id=s.session_id,
        unsafe=True,
    )
    assert r["ok"]
    assert s.current_phase == PHASE_SESSION_INIT


# ----------------------------------------------------------------------
# Dispatch-level phase gating (env-flag gated)
# ----------------------------------------------------------------------


def test_dispatch_phase_gating_blocks_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMPGEN_STRICT_PHASE_GATING", "1")
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    s = sm.open()
    enter_phase(sm, target_phase=PHASE_SESSION_INIT, session_id=s.session_id)

    def _handler(sm_arg: Any, **_: Any) -> dict[str, Any]:
        return {"ok": True, "called": True}

    tool_by_name = {
        "apply_recipe": {
            "name": "apply_recipe",
            "phase": "recipe",
            "handler": _handler,
        },
    }
    result = dispatch_tool(
        "apply_recipe",
        {"session_id": s.session_id},
        sm=sm,
        tool_by_name=tool_by_name,
        recorder=None,
    )
    assert result["ok"] is False
    assert result["blocked_reason"] == "phase_violation"


def test_dispatch_phase_gating_off_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COMPGEN_STRICT_PHASE_GATING", raising=False)
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    s = sm.open()
    enter_phase(sm, target_phase=PHASE_SESSION_INIT, session_id=s.session_id)

    def _handler(sm_arg: Any, **_: Any) -> dict[str, Any]:
        return {"ok": True, "called": True}

    tool_by_name = {
        "apply_recipe": {
            "name": "apply_recipe",
            "phase": "recipe",
            "handler": _handler,
        },
    }
    result = dispatch_tool(
        "apply_recipe",
        {"session_id": s.session_id},
        sm=sm,
        tool_by_name=tool_by_name,
        recorder=None,
    )
    # Handler runs even though phase wouldn't allow it.
    assert result["ok"] is True
    assert result["called"] is True
