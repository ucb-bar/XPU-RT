"""H3 — capability tokens + per-tool gating.

Coverage:

1. ``CAPABILITY_TOKENS`` is a closed enum; ``is_known_token`` rejects
   unknowns.
2. ``missing_capabilities`` returns empty when the tool has no
   requirements (read-only default).
3. ``missing_capabilities`` returns the right missing tokens when the
   session is under-provisioned.
4. ``missing_capabilities`` returns ``role_mismatch`` when the caller
   role is wrong.
5. With strict env flag set, dispatch refuses ``override_decision``
   when the operator_override token is absent.
6. With strict env flag set, ``apply_recipe`` is refused without
   the ``agent_role`` token.
7. With strict env flag set, capabilities present pass.
8. With the flag off, the gating is bypassed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from compgen.mcp.capabilities import (
    CAPABILITY_AGENT_ROLE,
    CAPABILITY_OPERATOR_OVERRIDE,
    CAPABILITY_TOKENS,
    ROLE_AGENT,
    ROLE_KERNEL_PROVIDER,
    is_known_role,
    is_known_token,
    missing_capabilities,
)
from compgen.mcp.server import dispatch_tool
from compgen.mcp.session import SessionManager


def test_capability_tokens_closed_enum() -> None:
    assert is_known_token(CAPABILITY_OPERATOR_OVERRIDE)
    assert not is_known_token("not_a_token")


def test_roles_closed_enum() -> None:
    assert is_known_role(ROLE_AGENT)
    assert is_known_role(ROLE_KERNEL_PROVIDER)
    assert not is_known_role("admin")


def test_capability_tokens_match_section_11_dream_3() -> None:
    """The closed enum is exactly the 7 tokens from the plan."""

    assert len(CAPABILITY_TOKENS) == 7


def test_missing_capabilities_read_only_default() -> None:
    """Tools without requirements pass through."""

    missing, role_mismatch = missing_capabilities(
        tool_name="list_phase_tools",
        session_caps=frozenset(),
        caller_role=ROLE_AGENT,
    )
    assert not missing
    assert role_mismatch is None


def test_missing_capabilities_reports_missing_tokens() -> None:
    missing, role_mismatch = missing_capabilities(
        tool_name="override_decision",
        session_caps=frozenset(),
        caller_role=ROLE_AGENT,
    )
    assert CAPABILITY_OPERATOR_OVERRIDE in missing
    assert role_mismatch is None


def test_missing_capabilities_reports_role_mismatch() -> None:
    missing, role_mismatch = missing_capabilities(
        tool_name="register_kernel_result",
        session_caps=frozenset(),  # no provider tokens either
        caller_role=ROLE_AGENT,  # wrong role
    )
    assert role_mismatch == ROLE_KERNEL_PROVIDER
    assert missing  # also missing provider token


# ----------------------------------------------------------------------
# Dispatch-level
# ----------------------------------------------------------------------


def test_dispatch_blocks_override_decision_without_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMPGEN_STRICT_CAPABILITIES", "1")
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    s = sm.open()

    def _handler(sm_arg: Any, **_: Any) -> dict[str, Any]:
        return {"ok": True, "called": True}

    tool_by_name = {
        "override_decision": {
            "name": "override_decision",
            "phase": "decision",
            "handler": _handler,
        },
    }
    result = dispatch_tool(
        "override_decision",
        {"session_id": s.session_id},
        sm=sm,
        tool_by_name=tool_by_name,
        recorder=None,
    )
    assert result["ok"] is False
    assert result["blocked_reason"] == "capability_missing"
    assert CAPABILITY_OPERATOR_OVERRIDE in result["missing_tokens"]


def test_dispatch_allows_when_token_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMPGEN_STRICT_CAPABILITIES", "1")
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    s = sm.open()
    s.capabilities = frozenset({CAPABILITY_OPERATOR_OVERRIDE})

    def _handler(sm_arg: Any, **_: Any) -> dict[str, Any]:
        return {"ok": True, "called": True}

    tool_by_name = {
        "override_decision": {
            "name": "override_decision",
            "phase": "decision",
            "handler": _handler,
        },
    }
    result = dispatch_tool(
        "override_decision",
        {"session_id": s.session_id},
        sm=sm,
        tool_by_name=tool_by_name,
        recorder=None,
    )
    assert result["ok"] is True
    assert result["called"] is True


def test_dispatch_capabilities_off_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COMPGEN_STRICT_CAPABILITIES", raising=False)
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    s = sm.open()
    s.capabilities = frozenset()  # no tokens

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
    # Gating off: handler runs.
    assert result["ok"] is True
    assert result["called"] is True


def test_apply_recipe_requires_agent_role_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMPGEN_STRICT_CAPABILITIES", "1")
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    s = sm.open()
    s.capabilities = frozenset()  # missing agent_role

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
    assert result["blocked_reason"] == "capability_missing"
    assert CAPABILITY_AGENT_ROLE in result["missing_tokens"]
