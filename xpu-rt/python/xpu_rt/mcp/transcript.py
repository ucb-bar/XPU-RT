"""JSONL transcript recorder for MCP tool calls.

Every call dispatched through :mod:`compgen.mcp.server` is appended to a
per-session ``transcript.jsonl`` so an external observer can audit what
an agent did. Independent from :class:`compgen.llm.recorder.ToolCallRecorder`
— the two record different layers (raw MCP JSON-RPC vs. Recipe-IR
invent-slot semantics) and intentionally use disjoint schemas.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENV_VAR = "COMPGEN_SESSION_DIR"
DEFAULT_ROOT_DIR = "sessions"
TRANSCRIPT_FILENAME = "transcript.jsonl"

# Large results (e.g. full Recipe IR views) are replaced by a summary to
# keep transcripts observable.
_RESULT_SUMMARY_BYTES_THRESHOLD = 4096


def default_session_root(cwd: str | Path | None = None) -> Path:
    """Return the root directory under which session transcripts are written.

    Resolution:
      1. ``$COMPGEN_SESSION_DIR`` if set.
      2. ``<cwd>/sessions`` otherwise (cwd defaults to the current process cwd).
    """

    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        return Path(env).expanduser()
    base = Path(cwd) if cwd is not None else Path.cwd()
    return base / DEFAULT_ROOT_DIR


def _serialize(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return repr(obj)


def _summarize_result(result: Any) -> Any:
    """Return either the raw result or a summary if the result is oversized."""

    blob = _serialize(result)
    if len(blob) <= _RESULT_SUMMARY_BYTES_THRESHOLD:
        try:
            return json.loads(blob)
        except (TypeError, ValueError):
            return blob
    return {
        "summary": True,
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
        "head": blob[:256],
    }


@dataclass
class McpTranscriptRecorder:
    """Append MCP tool-call records to ``<root>/<session_id>/transcript.jsonl``."""

    root: Path
    enabled: bool = True
    _counts: dict[str, int] = field(default_factory=dict)

    def transcript_path(self, session_id: str) -> Path:
        return self.root / session_id / TRANSCRIPT_FILENAME

    def record(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        result: Any,
        session_id: str,
        duration_ms: float,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Append one record. Returns the serialized record for tests."""

        record: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id,
            "tool": tool,
            "args": args,
            "result": _summarize_result(result),
            "duration_ms": round(float(duration_ms), 3),
            "error": error,
        }
        if not self.enabled:
            return record
        self._counts[session_id] = self._counts.get(session_id, 0) + 1
        path = self.transcript_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return record

    def count(self, session_id: str) -> int:
        return self._counts.get(session_id, 0)

    @classmethod
    def from_env(cls, *, enabled: bool = True) -> "McpTranscriptRecorder":
        return cls(root=default_session_root(), enabled=enabled)


__all__ = [
    "DEFAULT_ROOT_DIR",
    "ENV_VAR",
    "McpTranscriptRecorder",
    "TRANSCRIPT_FILENAME",
    "default_session_root",
]
