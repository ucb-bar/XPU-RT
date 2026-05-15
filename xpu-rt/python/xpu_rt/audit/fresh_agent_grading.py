"""Fresh-agent task + grading.

A *fresh-agent task* is the operational contract that lets us prove a
brand-new Claude Code session can drive a CompGen tool to completion
without any conversational context. Each task lives in its own
directory under ``.rcg-artifacts/fresh_agent_tasks/<task_id>/`` and
must contain:

* ``task.md`` — the instructions the fresh agent reads.
* ``allowed_tools.json`` — closed list of MCP/CLI tool ids the agent
  may invoke. The fresh-agent harness rejects any task whose
  ``allowed_tools`` references a tool that does not exist.
* ``expected_artifacts.json`` — file paths (relative to the run
  directory) the agent must produce, with optional shape constraints
  (each entry may declare ``min_bytes``, ``json_required_keys``, or
  ``contains``).
* ``grading_script.py`` — a *deterministic* Python script that takes
  the run directory as its first argument and writes
  ``grading_result.json`` to it. The script is run by the harness
  (not by the agent) and is byte-deterministic — same artifacts in,
  same grading result out.
* Optional ``baseline.json`` — a deterministic baseline the harness
  can execute to populate the run directory without spawning a real
  Claude session. Used for CI gates and for the T6 audit.

The harness itself never edits source. The T6 gate consumes
``grading_result.json`` and refuses to promote a ToolCard past T6
until ``passed=true``.

Hard rules:

1. ``grading_script.py`` exit code 0 implies ``passed=true``; any
   non-zero exit OR a missing ``grading_result.json`` is recorded as
   ``passed=false`` with the reason ``grading_script_did_not_complete``.
2. The grader never modifies the task directory.
3. Every task lives in *exactly one* directory under
   ``.rcg-artifacts/fresh_agent_tasks/``. No nested tasks; no
   per-tool-id wildcard discovery.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

FRESH_AGENT_TASKS_ROOT_REL: Final[Path] = Path(".rcg-artifacts") / "fresh_agent_tasks"

GRADING_VIOLATION_KINDS: Final[tuple[str, ...]] = (
    "missing_artifact",
    "artifact_too_small",
    "artifact_missing_required_key",
    "artifact_missing_substring",
    "grading_script_did_not_complete",
    "grading_script_emitted_invalid_result",
    "allowed_tool_unknown",
    "task_directory_missing",
    "task_directory_incomplete",
)


@dataclass(frozen=True)
class FreshAgentTask:
    """In-memory handle on a fresh-agent task package."""

    task_id: str
    task_dir: Path
    task_md: Path
    allowed_tools: tuple[str, ...]
    expected_artifacts: tuple[dict[str, Any], ...]
    grading_script: Path
    baseline: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_dir": str(self.task_dir),
            "task_md": str(self.task_md),
            "allowed_tools": list(self.allowed_tools),
            "expected_artifacts": [dict(a) for a in self.expected_artifacts],
            "grading_script": str(self.grading_script),
            "baseline": dict(self.baseline) if self.baseline else None,
        }


@dataclass(frozen=True)
class GradingViolation:
    """One failure encountered by the grader."""

    kind: str
    detail: str
    artifact: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in GRADING_VIOLATION_KINDS:
            raise ValueError(
                f"unknown grading violation {self.kind!r}; "
                f"must be one of {GRADING_VIOLATION_KINDS}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail, "artifact": self.artifact}


@dataclass(frozen=True)
class GradingResult:
    """Typed grading outcome for one fresh-agent run."""

    task_id: str
    run_dir: str
    passed: bool
    reason: str
    violations: tuple[GradingViolation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_dir": self.run_dir,
            "passed": self.passed,
            "reason": self.reason,
            "violations": [v.to_dict() for v in self.violations],
        }


class FreshAgentTaskError(ValueError):
    """A task directory is malformed or incomplete."""


def fresh_agent_tasks_root(repo_root: Path | None = None) -> Path:
    base = repo_root or Path(__file__).resolve().parents[3]
    return base / FRESH_AGENT_TASKS_ROOT_REL


def load_task(
    task_id: str,
    *,
    repo_root: Path | None = None,
    known_tool_ids: tuple[str, ...] | None = None,
) -> FreshAgentTask:
    """Load a fresh-agent task from disk.

    Raises :class:`FreshAgentTaskError` if the directory is missing
    required files or if ``allowed_tools.json`` references an unknown
    tool_id (when ``known_tool_ids`` is supplied — typically the
    output of :func:`compgen.tools.iter_tool_cards`).
    """

    root = fresh_agent_tasks_root(repo_root)
    task_dir = root / task_id
    if not task_dir.is_dir():
        raise FreshAgentTaskError(
            f"fresh-agent task directory {task_dir} does not exist"
        )

    task_md = task_dir / "task.md"
    allowed_path = task_dir / "allowed_tools.json"
    expected_path = task_dir / "expected_artifacts.json"
    grading_path = task_dir / "grading_script.py"
    baseline_path = task_dir / "baseline.json"

    for required in (task_md, allowed_path, expected_path, grading_path):
        if not required.is_file():
            raise FreshAgentTaskError(
                f"task {task_id!r}: missing required file {required.name}"
            )

    allowed_raw = json.loads(allowed_path.read_text(encoding="utf-8"))
    if not isinstance(allowed_raw, list):
        raise FreshAgentTaskError(
            f"task {task_id!r}: allowed_tools.json must be a list of tool ids"
        )
    allowed_tools = tuple(str(t) for t in allowed_raw)

    if known_tool_ids is not None:
        unknown = [t for t in allowed_tools if t not in known_tool_ids]
        if unknown:
            raise FreshAgentTaskError(
                f"task {task_id!r}: allowed_tools references unknown tools {unknown}"
            )

    expected_raw = json.loads(expected_path.read_text(encoding="utf-8"))
    if not isinstance(expected_raw, list):
        raise FreshAgentTaskError(
            f"task {task_id!r}: expected_artifacts.json must be a list"
        )
    expected_artifacts: tuple[dict[str, Any], ...] = tuple(
        dict(item) if isinstance(item, dict) else {"path": str(item)}
        for item in expected_raw
    )
    for entry in expected_artifacts:
        if "path" not in entry:
            raise FreshAgentTaskError(
                f"task {task_id!r}: expected_artifacts entry missing 'path' key"
            )

    baseline: dict[str, Any] | None = None
    if baseline_path.is_file():
        baseline_body = json.loads(baseline_path.read_text(encoding="utf-8"))
        if not isinstance(baseline_body, dict):
            raise FreshAgentTaskError(
                f"task {task_id!r}: baseline.json must be a JSON object"
            )
        baseline = baseline_body

    return FreshAgentTask(
        task_id=task_id,
        task_dir=task_dir,
        task_md=task_md,
        allowed_tools=allowed_tools,
        expected_artifacts=expected_artifacts,
        grading_script=grading_path,
        baseline=baseline,
    )


def list_task_ids(repo_root: Path | None = None) -> list[str]:
    """Return every known fresh-agent task id, alphabetically."""

    root = fresh_agent_tasks_root(repo_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _check_expected_artifact(run_dir: Path, entry: dict[str, Any]) -> list[GradingViolation]:
    out: list[GradingViolation] = []
    rel = entry["path"]
    path = run_dir / rel
    if not path.is_file():
        out.append(
            GradingViolation(
                kind="missing_artifact",
                detail=f"required artifact {rel} not produced under {run_dir}",
                artifact=rel,
            )
        )
        return out
    min_bytes = entry.get("min_bytes")
    if isinstance(min_bytes, int):
        size = path.stat().st_size
        if size < min_bytes:
            out.append(
                GradingViolation(
                    kind="artifact_too_small",
                    detail=f"{rel} is {size} bytes; expected ≥ {min_bytes}",
                    artifact=rel,
                )
            )
    required_keys = entry.get("json_required_keys")
    if isinstance(required_keys, list) and required_keys:
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            out.append(
                GradingViolation(
                    kind="artifact_missing_required_key",
                    detail=f"{rel} did not parse as JSON: {exc}",
                    artifact=rel,
                )
            )
            return out
        missing = [k for k in required_keys if not (isinstance(body, dict) and k in body)]
        if missing:
            out.append(
                GradingViolation(
                    kind="artifact_missing_required_key",
                    detail=f"{rel} JSON missing keys {missing}",
                    artifact=rel,
                )
            )
    substrings = entry.get("contains")
    if isinstance(substrings, list) and substrings:
        body_text = path.read_text(encoding="utf-8", errors="replace")
        missing = [s for s in substrings if str(s) not in body_text]
        if missing:
            out.append(
                GradingViolation(
                    kind="artifact_missing_substring",
                    detail=f"{rel} missing substrings {missing}",
                    artifact=rel,
                )
            )
    return out


def grade(
    task: FreshAgentTask,
    *,
    run_dir: Path,
    write_result_json: bool = True,
) -> GradingResult:
    """Grade a run directory against a fresh-agent task.

    The grader does two independent passes:

    1. Verify every entry in ``expected_artifacts`` against the run
       directory (presence + optional shape constraints).
    2. Execute ``grading_script.py`` as a subprocess with the run
       directory as ``argv[1]``. The script is responsible for any
       semantic checks the file-shape constraints cannot express.
       The grader collects its exit code + the optional
       ``grading_result.json`` it writes.

    Both passes must succeed for ``passed=true``. The final
    ``GradingResult`` is also written to
    ``run_dir/grading_result.json`` so the T6 gate can pick it
    up without re-running the grader.
    """

    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    violations: list[GradingViolation] = list(
        v for entry in task.expected_artifacts for v in _check_expected_artifact(run_dir, entry)
    )

    # Execute the grading script. We use the same Python interpreter
    # the harness is running under so import paths resolve identically.
    env = {**os.environ}
    completed = subprocess.run(
        [sys.executable, str(task.grading_script), str(run_dir)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        violations.append(
            GradingViolation(
                kind="grading_script_did_not_complete",
                detail=(
                    f"grading_script.py exited {completed.returncode}; "
                    f"stderr={completed.stderr.strip()[:400]!r}"
                ),
            )
        )

    # If the script wrote a result.json, surface its violations too.
    script_result_path = run_dir / "grading_script_result.json"
    if script_result_path.is_file():
        try:
            body = json.loads(script_result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            violations.append(
                GradingViolation(
                    kind="grading_script_emitted_invalid_result",
                    detail=f"grading_script_result.json malformed: {exc}",
                )
            )
        else:
            for v in body.get("violations", []) or []:
                if isinstance(v, dict) and v.get("kind") in GRADING_VIOLATION_KINDS:
                    violations.append(
                        GradingViolation(
                            kind=str(v["kind"]),
                            detail=str(v.get("detail", "")),
                            artifact=(str(v["artifact"]) if v.get("artifact") else None),
                        )
                    )

    passed = not violations
    reason = "all expected artifacts present and grading_script.py exited 0" if passed else (
        f"{len(violations)} violation(s); see violations list"
    )
    result = GradingResult(
        task_id=task.task_id,
        run_dir=str(run_dir),
        passed=passed,
        reason=reason,
        violations=tuple(violations),
    )
    if write_result_json:
        serialized = json.dumps(result.to_dict(), sort_keys=True, indent=2) + "\n"
        (run_dir / "grading_result.json").write_text(serialized, encoding="utf-8")
        # Sidecar in the task directory so the T6 gate can find
        # the most recent grading result without knowing the run_dir.
        # We never overwrite a passing result with a later failure —
        # the audit cares whether the task has *ever* been graded clean.
        sidecar = task.task_dir / "last_grading_result.json"
        prior_passed = False
        if sidecar.is_file():
            try:
                prior = json.loads(sidecar.read_text(encoding="utf-8"))
                prior_passed = bool(prior.get("passed"))
            except json.JSONDecodeError:
                prior_passed = False
        if result.passed or not prior_passed:
            sidecar.write_text(serialized, encoding="utf-8")
    return result


def run_baseline(task: FreshAgentTask, *, run_dir: Path) -> subprocess.CompletedProcess:
    """Execute the task's deterministic baseline into ``run_dir``.

    The baseline is the *no-LLM* path that proves the task is solvable
    by deterministic means; the fresh-agent harness uses it for CI
    gates. If a task has no baseline, raises :class:`FreshAgentTaskError`.

    The baseline body shape (``baseline.json``):

    ::

        {
          "command": ["scripts/dev/probe_extension_providers.py", "--out", "${run_dir}"],
          "env": {"COMPGEN_FOO": "bar"},
          "timeout_s": 120
        }

    The literal ``${run_dir}`` token in any argv element is replaced
    with the resolved run directory; the script never substitutes
    other tokens (no surprise expansion).
    """

    if task.baseline is None:
        raise FreshAgentTaskError(f"task {task.task_id!r} has no baseline")
    cmd = list(task.baseline.get("command") or [])
    if not cmd:
        raise FreshAgentTaskError(
            f"task {task.task_id!r}: baseline.command must be a non-empty list"
        )
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_cmd: list[str] = []
    for tok in cmd:
        s = str(tok)
        resolved_cmd.append(s.replace("${run_dir}", str(run_dir)))
    if resolved_cmd[0].endswith(".py"):
        # Relative-to-repo Python script. Run with the harness interpreter.
        repo_root = Path(__file__).resolve().parents[3]
        resolved_cmd = [sys.executable, str(repo_root / resolved_cmd[0]), *resolved_cmd[1:]]
    timeout = float(task.baseline.get("timeout_s") or 120.0)
    env = {**os.environ, **{str(k): str(v) for k, v in (task.baseline.get("env") or {}).items()}}
    return subprocess.run(
        resolved_cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )


__all__ = [
    "GRADING_VIOLATION_KINDS",
    "FreshAgentTask",
    "FreshAgentTaskError",
    "GradingResult",
    "GradingViolation",
    "fresh_agent_tasks_root",
    "grade",
    "list_task_ids",
    "load_task",
    "run_baseline",
]
