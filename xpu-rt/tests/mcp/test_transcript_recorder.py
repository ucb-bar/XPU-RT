"""Tests for the MCP session transcript recorder + server dispatch middleware."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.mcp.server import dispatch_tool
from compgen.mcp.session import SessionManager
from compgen.mcp.transcript import (
    ENV_VAR,
    McpTranscriptRecorder,
    TRANSCRIPT_FILENAME,
    default_session_root,
)


@pytest.fixture()
def recorder(tmp_path: Path) -> McpTranscriptRecorder:
    return McpTranscriptRecorder(root=tmp_path / "sessions")


def _read_transcript(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- McpTranscriptRecorder ----------------------------------------------------


def test_recorder_writes_one_line_per_record(recorder: McpTranscriptRecorder) -> None:
    recorder.record(
        tool="open_target",
        args={"spec_path": "/tmp/x.yaml"},
        result={"ok": True},
        session_id="abc",
        duration_ms=12.3,
    )
    recorder.record(
        tool="register_pack",
        args={"pack": "my_pack"},
        result={"ok": True, "pack_name": "my_pack"},
        session_id="abc",
        duration_ms=4.2,
    )

    path = recorder.transcript_path("abc")
    records = _read_transcript(path)
    assert len(records) == 2
    assert records[0]["tool"] == "open_target"
    assert records[1]["tool"] == "register_pack"
    assert records[0]["duration_ms"] >= 0
    assert recorder.count("abc") == 2


def test_recorder_summarizes_oversized_results(recorder: McpTranscriptRecorder) -> None:
    huge_result = {"blob": "x" * 10_000}
    recorder.record(
        tool="view_recipe",
        args={},
        result=huge_result,
        session_id="s1",
        duration_ms=1.0,
    )
    records = _read_transcript(recorder.transcript_path("s1"))
    assert records[0]["result"].get("summary") is True
    assert "sha256" in records[0]["result"]
    assert records[0]["result"]["bytes"] > 4096


def test_recorder_error_field_propagates(recorder: McpTranscriptRecorder) -> None:
    recorder.record(
        tool="compile",
        args={"session_id": "s"},
        result={"ok": False, "error": "boom"},
        session_id="s",
        duration_ms=1.5,
        error="RuntimeError: boom",
    )
    records = _read_transcript(recorder.transcript_path("s"))
    assert records[0]["error"] == "RuntimeError: boom"


def test_recorder_disabled_writes_nothing(tmp_path: Path) -> None:
    rec = McpTranscriptRecorder(root=tmp_path / "sessions", enabled=False)
    rec.record(
        tool="open_target", args={}, result={"ok": True}, session_id="s", duration_ms=1.0
    )
    assert not rec.transcript_path("s").exists()


def test_recorder_session_isolation(recorder: McpTranscriptRecorder) -> None:
    recorder.record(
        tool="open_target", args={}, result={"ok": True}, session_id="a", duration_ms=1.0
    )
    recorder.record(
        tool="open_target", args={}, result={"ok": True}, session_id="b", duration_ms=1.0
    )
    assert recorder.count("a") == 1
    assert recorder.count("b") == 1
    assert (recorder.root / "a" / TRANSCRIPT_FILENAME).exists()
    assert (recorder.root / "b" / TRANSCRIPT_FILENAME).exists()


def test_default_session_root_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "my_sessions"))
    assert default_session_root() == tmp_path / "my_sessions"


def test_default_session_root_cwd_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert default_session_root(cwd=tmp_path) == tmp_path / "sessions"


# --- dispatch_tool middleware -------------------------------------------------


def _tools_catalogue() -> dict[str, dict]:
    # Minimal in-test tool catalogue
    def _echo(sm, *, value: str, session_id: str | None = None) -> dict:
        return {"ok": True, "echoed": value, "session_id": session_id}

    def _boom(sm, *, reason: str = "bad", session_id: str | None = None) -> dict:
        raise RuntimeError(reason)

    return {
        "echo": {
            "name": "echo",
            "description": "echo tool",
            "handler": _echo,
            "input_schema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
        "boom": {
            "name": "boom",
            "description": "always raises",
            "handler": _boom,
            "input_schema": {"type": "object"},
        },
    }


def test_dispatch_tool_records_success(tmp_path: Path) -> None:
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    rec = McpTranscriptRecorder(root=tmp_path / "sessions")
    catalogue = _tools_catalogue()

    session = sm.open("sess1")
    result = dispatch_tool(
        "echo",
        {"value": "hello", "session_id": session.session_id},
        sm=sm,
        tool_by_name=catalogue,
        recorder=rec,
    )
    assert result["ok"] is True
    assert result["echoed"] == "hello"
    records = _read_transcript(rec.transcript_path("sess1"))
    assert len(records) == 1
    assert records[0]["tool"] == "echo"
    assert records[0]["args"]["value"] == "hello"
    assert records[0]["error"] is None
    assert records[0]["duration_ms"] >= 0.0


def test_dispatch_tool_records_error(tmp_path: Path) -> None:
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    rec = McpTranscriptRecorder(root=tmp_path / "sessions")
    catalogue = _tools_catalogue()

    result = dispatch_tool(
        "boom",
        {"reason": "expected", "session_id": "sess2"},
        sm=sm,
        tool_by_name=catalogue,
        recorder=rec,
    )
    assert result["ok"] is False
    assert "expected" in result["error"]
    records = _read_transcript(rec.transcript_path("sess2"))
    assert records[0]["error"] is not None
    assert "expected" in records[0]["error"]


def test_dispatch_tool_unknown_tool(tmp_path: Path) -> None:
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    rec = McpTranscriptRecorder(root=tmp_path / "sessions")
    catalogue = _tools_catalogue()

    result = dispatch_tool(
        "nosuchtool",
        {"session_id": "sess3"},
        sm=sm,
        tool_by_name=catalogue,
        recorder=rec,
    )
    assert result["ok"] is False
    assert "Unknown tool" in result["error"]
    records = _read_transcript(rec.transcript_path("sess3"))
    assert records[0]["tool"] == "nosuchtool"
