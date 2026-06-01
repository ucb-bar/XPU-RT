#!/usr/bin/env python3
"""PreToolUse hook: enforce a per-task file-edit scope (inert by default).

Reads the proposed Edit/Write target and checks it against an OPTIONAL policy at
.claude/edit_policy.json:

    {"allow": ["xpu-rt/scheduler*.py"], "deny": ["zephyr-chipyard-sw/**"]}

Semantics:
  - no edit_policy.json present  => allow everything (this hook is INERT by
    default, so it never changes behaviour unless a task opts in by writing one)
  - if "allow" is non-empty, ONLY matching paths may be edited
  - any path matching "deny" is blocked (deny wins)

A block is signalled with exit code 2 + a stderr message (surfaced to the model).
Use it to scope an agent to e.g. only the scheduler/cost-model files during a
targeted change.
"""
from __future__ import annotations

import fnmatch
import json
import os
import sys

EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}


def _root():
    return os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    if payload.get("tool_name") not in EDIT_TOOLS:
        return 0
    ti = payload.get("tool_input", {}) or {}
    path = ti.get("file_path") or ti.get("notebook_path") or ""
    if not path:
        return 0

    root = _root()
    policy_path = os.path.join(root, ".claude", "edit_policy.json")
    if not os.path.exists(policy_path):
        return 0  # inert
    try:
        policy = json.load(open(policy_path))
    except Exception:
        return 0
    try:
        rel = os.path.relpath(os.path.abspath(path), root)
    except Exception:
        rel = path
    allow = policy.get("allow", [])
    deny = policy.get("deny", [])

    def matches(globs):
        return any(fnmatch.fnmatch(rel, g) for g in globs)

    if matches(deny):
        print(f"[deny_forbidden_edits] BLOCKED {rel}: matches a deny pattern", file=sys.stderr)
        return 2
    if allow and not matches(allow):
        print(f"[deny_forbidden_edits] BLOCKED {rel}: outside allowed scope {allow}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
