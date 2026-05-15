#!/usr/bin/env -S uv run python
"""Fresh-agent harness CLI.

Subcommands::

    uv run python scripts/dev/run_fresh_agent_harness.py list
    uv run python scripts/dev/run_fresh_agent_harness.py describe <task_id>
    uv run python scripts/dev/run_fresh_agent_harness.py run-baseline <task_id> --out <run_dir>
    uv run python scripts/dev/run_fresh_agent_harness.py grade <task_id> --run-dir <run_dir>

``run-baseline`` is the CI-runnable path that proves a task is solvable
by deterministic means; ``grade`` is the post-hoc check the T6
gate consumes. A real fresh-Claude session runs *between* these two —
the harness does not spawn the agent itself.

Exit code 0 iff the requested operation succeeded; non-zero with a
typed JSON error payload on stdout otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compgen.audit.fresh_agent_grading import (
    FreshAgentTaskError,
    grade,
    list_task_ids,
    load_task,
    run_baseline,
)


def _emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2))


def cmd_list(args: argparse.Namespace) -> int:
    ids = list_task_ids()
    _emit({"status": "ok", "task_ids": ids})
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    try:
        task = load_task(args.task_id)
    except FreshAgentTaskError as exc:
        _emit({"status": "error", "error_type": "task_load_failed", "message": str(exc)})
        return 3
    payload = task.to_dict()
    payload["task_md"] = (task.task_md.read_text(encoding="utf-8"))
    _emit({"status": "ok", "task": payload})
    return 0


def cmd_run_baseline(args: argparse.Namespace) -> int:
    try:
        task = load_task(args.task_id)
    except FreshAgentTaskError as exc:
        _emit({"status": "error", "error_type": "task_load_failed", "message": str(exc)})
        return 3
    if task.baseline is None:
        _emit(
            {
                "status": "error",
                "error_type": "no_baseline",
                "task_id": task.task_id,
                "message": "task has no deterministic baseline; populate run_dir manually then run 'grade'",
            }
        )
        return 4
    proc = run_baseline(task, run_dir=args.out)
    if proc.returncode != 0:
        _emit(
            {
                "status": "error",
                "error_type": "baseline_failed",
                "task_id": task.task_id,
                "exit_code": proc.returncode,
                "stderr": proc.stderr.strip()[-2000:],
            }
        )
        return 5
    # Auto-grade after baseline so callers get a single-shot CI gate.
    result = grade(task, run_dir=args.out)
    payload = {"status": "ok" if result.passed else "error", "grading": result.to_dict()}
    _emit(payload)
    return 0 if result.passed else 1


def cmd_grade(args: argparse.Namespace) -> int:
    try:
        task = load_task(args.task_id)
    except FreshAgentTaskError as exc:
        _emit({"status": "error", "error_type": "task_load_failed", "message": str(exc)})
        return 3
    result = grade(task, run_dir=args.run_dir)
    payload = {"status": "ok" if result.passed else "error", "grading": result.to_dict()}
    _emit(payload)
    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=None,
        help="Override the fresh-agent tasks directory (defaults to .rcg-artifacts/fresh_agent_tasks).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="List every registered fresh-agent task.")
    sp.set_defaults(fn=cmd_list)

    sp = sub.add_parser("describe", help="Print a task's full body + task.md.")
    sp.add_argument("task_id", type=str)
    sp.set_defaults(fn=cmd_describe)

    sp = sub.add_parser(
        "run-baseline",
        help="Run the task's deterministic baseline into --out and grade.",
    )
    sp.add_argument("task_id", type=str)
    sp.add_argument("--out", type=Path, required=True)
    sp.set_defaults(fn=cmd_run_baseline)

    sp = sub.add_parser("grade", help="Grade a run directory against a task.")
    sp.add_argument("task_id", type=str)
    sp.add_argument("--run-dir", type=Path, required=True)
    sp.set_defaults(fn=cmd_grade)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
