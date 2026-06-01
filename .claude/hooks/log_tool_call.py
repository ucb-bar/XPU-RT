#!/usr/bin/env python3
"""PreToolUse hook (matcher *): append each tool call to .claude/logs/tool_calls.jsonl.

A lightweight, always-available behaviour trace (complements OTel). Never blocks.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone


def _root():
    return os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _summary(tool, ti):
    if tool == "Bash":
        return (ti.get("command", "") or "")[:200]
    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        return ti.get("file_path") or ti.get("notebook_path") or ""
    return ti.get("file_path") or ti.get("pattern") or ti.get("path") or ""


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    root = _root()
    d = os.path.join(root, ".claude", "logs")
    os.makedirs(d, exist_ok=True)
    rec = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "session_id": payload.get("session_id"),
        "tool": payload.get("tool_name", ""),
        "detail": _summary(payload.get("tool_name", ""), payload.get("tool_input", {}) or {}),
    }
    with open(os.path.join(d, "tool_calls.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
