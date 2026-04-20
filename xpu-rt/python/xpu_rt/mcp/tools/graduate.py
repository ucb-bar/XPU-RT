"""MCP tool: in-session promotion of authored tools.

Wraps :func:`compgen.agent.self_extension.graduate.promote_authored_tools`
with a session-scoped lower threshold so an agent can iterate on a
freshly-authored tool and graduate it after a couple of in-session
trials. Cross-session graduation (the higher 5-pass / 2-workloads /
2-targets bar) still requires the standalone path.

Trials counted: every entry in the session's authored-trials JSONL
log (under ``$COMPGEN_SESSION_DIR/<sid>/authored_trials.jsonl`` by
default; falls back to the process-wide log when no session-scoped
file exists).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from compgen.agent.self_extension._index import snapshot_authored_index
from compgen.agent.self_extension.graduate import promote_authored_tools
from compgen.agent.self_extension.trials import default_trial_log_path
from compgen.mcp.session import SessionManager

log = structlog.get_logger()


def _session_trial_log_path(scratch_dir: Path) -> Path:
    """Per-session trial log; falls back to the global one when absent."""
    candidate = scratch_dir / "authored_trials.jsonl"
    if candidate.exists():
        return candidate
    return default_trial_log_path()


def promote_in_session_authored_tools(
    sm: SessionManager,
    *,
    session_id: str,
    min_passes: int = 2,
    log_path: str | None = None,
) -> dict[str, Any]:
    """Try to promote every authored tool with ``>= min_passes`` in-session
    trial passes (no cross-workload / cross-target requirement).

    Mutates the session's driver registry only — the process-wide
    cross-session graduation state file is untouched, so the same tool
    can later graduate cross-session under the higher bar.
    """
    session = sm.get(session_id)
    driver = session.require_driver()
    assert driver.registry is not None

    if log_path is not None:
        path = Path(log_path).expanduser()
    else:
        path = _session_trial_log_path(session.scratch_dir)

    report = promote_authored_tools(
        driver.registry,
        authored_index=snapshot_authored_index(),
        log_path=path,
        min_passes_session=int(min_passes),
    )
    log.info(
        "mcp.promote_in_session",
        session_id=session_id,
        candidates_found=report.candidates_found,
        new_tools=[t["tool_name"] for t in report.new_tools_registered],
        log_path=str(path),
    )
    return {
        "ok": True,
        "session_id": session_id,
        "trials_scanned": report.trials_scanned,
        "candidates_found": report.candidates_found,
        "candidates_already_applied": report.candidates_already_applied,
        "new_tools_registered": list(report.new_tools_registered),
        "errors": list(report.errors),
        "log_path": str(path),
        "min_passes": int(min_passes),
    }


GRADUATE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "promote_in_session_authored_tools",
        "description": (
            "Promote authored tools with >= min_passes in-session "
            "trials into the session's driver registry. Lower threshold "
            "than cross-session graduation; lets the agent iterate on a "
            "newly-authored tool quickly without polluting the persistent "
            "cross-session state file."
        ),
        "phase": "transform",
        "handler": promote_in_session_authored_tools,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "min_passes": {"type": "integer", "default": 2},
                "log_path": {"type": "string"},
            },
            "required": ["session_id"],
        },
    },
]


__all__ = ["GRADUATE_TOOLS", "promote_in_session_authored_tools"]
