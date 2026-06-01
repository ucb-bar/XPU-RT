#!/usr/bin/env python3
"""Stop/SessionEnd hook: roll the tool-call log into a per-session summary.

Writes .claude/logs/session_<id>.json with tool-call counts and files touched.
Never blocks.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone


def _root():
    return os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}
    root = _root()
    logs = os.path.join(root, ".claude", "logs")
    sid = payload.get("session_id") or "session"
    calls = []
    p = os.path.join(logs, "tool_calls.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                calls.append(json.loads(line))
            except Exception:
                pass
    by_tool = Counter(c["tool"] for c in calls if c.get("tool"))
    touched = sorted({c["detail"] for c in calls
                      if c.get("tool") in ("Edit", "Write", "MultiEdit") and c.get("detail")})
    os.makedirs(logs, exist_ok=True)
    summary = {
        "session_id": sid,
        "ended_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool_calls_total": len(calls),
        "tool_calls_by_name": dict(by_tool),
        "files_touched": touched,
    }
    with open(os.path.join(logs, f"session_{sid}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
