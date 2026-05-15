"""Tests for :mod:`compgen.audit.fresh_agent_grading`.

Coverage:

Positive:
* ``list_task_ids`` returns the shipped ``probe_providers_v1`` task.
* ``load_task`` round-trips through a synthesised task directory.
* ``grade`` on a clean run directory (artifacts present + grading
  script exits 0) returns ``passed=true`` and writes
  ``grading_result.json``.
* ``run_baseline`` produces a runnable command list with
  ``${run_dir}`` substituted; the shipped probe-providers baseline
  runs end-to-end and the grader returns ``passed=true``.

Negative controls (one per GRADING_VIOLATION_KIND we can synthesise):
* ``missing_artifact`` — expected file absent from run_dir.
* ``artifact_too_small`` — file present but below ``min_bytes``.
* ``artifact_missing_required_key`` — JSON missing a required top-level key.
* ``artifact_missing_substring`` — text artifact missing required substring.
* ``grading_script_did_not_complete`` — grading_script exits non-zero.
* ``grading_script_emitted_invalid_result`` — script writes malformed result.
* ``allowed_tool_unknown`` — task's allowed_tools references a tool not
  in the registry.
* ``task_directory_missing`` — load_task on nonexistent dir.
* ``task_directory_incomplete`` — load_task on dir missing a required file.

Hard real-hardware check (skipped under ``COMPGEN_SKIP_BASELINE=1`` so
CI without the probe environment still passes):
* The shipped probe-providers baseline runs the actual probe and
  the grader confirms 7 real artifacts produced under the run_dir.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest
from compgen.audit.fresh_agent_grading import (
    GRADING_VIOLATION_KINDS,
    FreshAgentTaskError,
    grade,
    list_task_ids,
    load_task,
    run_baseline,
)

# ---------- Helpers --------------------------------------------------


def _write_task(
    base: Path,
    task_id: str,
    *,
    task_md: str = "# test task\nsay hi.\n",
    allowed_tools: list[str] | None = None,
    expected_artifacts: list[dict] | None = None,
    grading_script_body: str | None = None,
    baseline: dict | None = None,
) -> Path:
    task_dir = base / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.md").write_text(task_md, encoding="utf-8")
    (task_dir / "allowed_tools.json").write_text(
        json.dumps(allowed_tools or []), encoding="utf-8"
    )
    (task_dir / "expected_artifacts.json").write_text(
        json.dumps(expected_artifacts or []), encoding="utf-8"
    )
    (task_dir / "grading_script.py").write_text(
        grading_script_body
        or textwrap.dedent(
            """\
            import json, sys
            from pathlib import Path
            (Path(sys.argv[1]) / "grading_script_result.json").write_text(
                json.dumps({"passed": True, "violations": []}), encoding="utf-8"
            )
            sys.exit(0)
            """
        ),
        encoding="utf-8",
    )
    if baseline is not None:
        (task_dir / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    return task_dir


# ---------- Positive -------------------------------------------------


def test_shipped_probe_providers_task_discoverable():
    ids = list_task_ids()
    assert "probe_providers_v1" in ids


def test_load_synthesised_task(tmp_path, monkeypatch):
    """``load_task`` reads a freshly authored task directory."""

    fake_root = tmp_path / ".rcg-artifacts" / "fresh_agent_tasks"
    fake_root.mkdir(parents=True)
    _write_task(fake_root, "test_basic_v1", expected_artifacts=[{"path": "out.txt"}])
    monkeypatch.setattr(
        "compgen.audit.fresh_agent_grading.fresh_agent_tasks_root",
        lambda repo_root=None: fake_root,
    )
    task = load_task("test_basic_v1")
    assert task.task_id == "test_basic_v1"
    assert task.expected_artifacts == ({"path": "out.txt"},)


def test_grade_clean_run(tmp_path, monkeypatch):
    fake_root = tmp_path / ".rcg-artifacts" / "fresh_agent_tasks"
    fake_root.mkdir(parents=True)
    _write_task(
        fake_root,
        "clean_v1",
        expected_artifacts=[{"path": "hello.txt", "min_bytes": 1, "contains": ["hi"]}],
    )
    monkeypatch.setattr(
        "compgen.audit.fresh_agent_grading.fresh_agent_tasks_root",
        lambda repo_root=None: fake_root,
    )
    task = load_task("clean_v1")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "hello.txt").write_text("hi there\n", encoding="utf-8")
    result = grade(task, run_dir=run_dir)
    assert result.passed, result.violations
    rj = json.loads((run_dir / "grading_result.json").read_text(encoding="utf-8"))
    assert rj["passed"] is True
    assert rj["task_id"] == "clean_v1"


def test_violation_kinds_enum_is_closed():
    assert len(set(GRADING_VIOLATION_KINDS)) == len(GRADING_VIOLATION_KINDS)


# ---------- Negative controls ----------------------------------------


def test_missing_artifact(tmp_path, monkeypatch):
    fake_root = tmp_path / ".rcg-artifacts" / "fresh_agent_tasks"
    fake_root.mkdir(parents=True)
    _write_task(
        fake_root, "missing_v1", expected_artifacts=[{"path": "nope.txt"}]
    )
    monkeypatch.setattr(
        "compgen.audit.fresh_agent_grading.fresh_agent_tasks_root",
        lambda repo_root=None: fake_root,
    )
    task = load_task("missing_v1")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = grade(task, run_dir=run_dir)
    assert not result.passed
    kinds = {v.kind for v in result.violations}
    assert "missing_artifact" in kinds


def test_artifact_too_small(tmp_path, monkeypatch):
    fake_root = tmp_path / ".rcg-artifacts" / "fresh_agent_tasks"
    fake_root.mkdir(parents=True)
    _write_task(
        fake_root,
        "small_v1",
        expected_artifacts=[{"path": "tiny.txt", "min_bytes": 1000}],
    )
    monkeypatch.setattr(
        "compgen.audit.fresh_agent_grading.fresh_agent_tasks_root",
        lambda repo_root=None: fake_root,
    )
    task = load_task("small_v1")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "tiny.txt").write_text("hi\n", encoding="utf-8")
    result = grade(task, run_dir=run_dir)
    kinds = {v.kind for v in result.violations}
    assert "artifact_too_small" in kinds


def test_artifact_missing_required_key(tmp_path, monkeypatch):
    fake_root = tmp_path / ".rcg-artifacts" / "fresh_agent_tasks"
    fake_root.mkdir(parents=True)
    _write_task(
        fake_root,
        "miss_key_v1",
        expected_artifacts=[
            {"path": "doc.json", "json_required_keys": ["foo", "bar"]}
        ],
    )
    monkeypatch.setattr(
        "compgen.audit.fresh_agent_grading.fresh_agent_tasks_root",
        lambda repo_root=None: fake_root,
    )
    task = load_task("miss_key_v1")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "doc.json").write_text(json.dumps({"foo": 1}), encoding="utf-8")
    result = grade(task, run_dir=run_dir)
    kinds = {v.kind for v in result.violations}
    assert "artifact_missing_required_key" in kinds


def test_artifact_missing_substring(tmp_path, monkeypatch):
    fake_root = tmp_path / ".rcg-artifacts" / "fresh_agent_tasks"
    fake_root.mkdir(parents=True)
    _write_task(
        fake_root,
        "miss_sub_v1",
        expected_artifacts=[{"path": "report.md", "contains": ["required_phrase"]}],
    )
    monkeypatch.setattr(
        "compgen.audit.fresh_agent_grading.fresh_agent_tasks_root",
        lambda repo_root=None: fake_root,
    )
    task = load_task("miss_sub_v1")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "report.md").write_text("nothing useful here", encoding="utf-8")
    result = grade(task, run_dir=run_dir)
    kinds = {v.kind for v in result.violations}
    assert "artifact_missing_substring" in kinds


def test_grading_script_did_not_complete(tmp_path, monkeypatch):
    fake_root = tmp_path / ".rcg-artifacts" / "fresh_agent_tasks"
    fake_root.mkdir(parents=True)
    _write_task(
        fake_root,
        "crash_v1",
        expected_artifacts=[],
        grading_script_body="import sys; sys.exit(7)\n",
    )
    monkeypatch.setattr(
        "compgen.audit.fresh_agent_grading.fresh_agent_tasks_root",
        lambda repo_root=None: fake_root,
    )
    task = load_task("crash_v1")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = grade(task, run_dir=run_dir)
    kinds = {v.kind for v in result.violations}
    assert "grading_script_did_not_complete" in kinds


def test_grading_script_emitted_invalid_result(tmp_path, monkeypatch):
    fake_root = tmp_path / ".rcg-artifacts" / "fresh_agent_tasks"
    fake_root.mkdir(parents=True)
    _write_task(
        fake_root,
        "bad_emit_v1",
        expected_artifacts=[],
        grading_script_body=textwrap.dedent(
            """\
            import sys
            from pathlib import Path
            (Path(sys.argv[1]) / "grading_script_result.json").write_text("{not json")
            sys.exit(0)
            """
        ),
    )
    monkeypatch.setattr(
        "compgen.audit.fresh_agent_grading.fresh_agent_tasks_root",
        lambda repo_root=None: fake_root,
    )
    task = load_task("bad_emit_v1")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = grade(task, run_dir=run_dir)
    kinds = {v.kind for v in result.violations}
    assert "grading_script_emitted_invalid_result" in kinds


def test_task_directory_missing(tmp_path, monkeypatch):
    fake_root = tmp_path / "empty"
    fake_root.mkdir()
    monkeypatch.setattr(
        "compgen.audit.fresh_agent_grading.fresh_agent_tasks_root",
        lambda repo_root=None: fake_root,
    )
    with pytest.raises(FreshAgentTaskError, match="does not exist"):
        load_task("does_not_exist_v1")


def test_task_directory_incomplete(tmp_path, monkeypatch):
    fake_root = tmp_path / ".rcg-artifacts" / "fresh_agent_tasks"
    fake_root.mkdir(parents=True)
    bad_dir = fake_root / "incomplete_v1"
    bad_dir.mkdir()
    (bad_dir / "task.md").write_text("hello", encoding="utf-8")
    # deliberately missing allowed_tools.json + expected_artifacts.json + grading_script.py
    monkeypatch.setattr(
        "compgen.audit.fresh_agent_grading.fresh_agent_tasks_root",
        lambda repo_root=None: fake_root,
    )
    with pytest.raises(FreshAgentTaskError, match="missing required file"):
        load_task("incomplete_v1")


def test_allowed_tool_unknown(tmp_path, monkeypatch):
    fake_root = tmp_path / ".rcg-artifacts" / "fresh_agent_tasks"
    fake_root.mkdir(parents=True)
    _write_task(
        fake_root, "unknown_tool_v1", allowed_tools=["does_not_exist_tool_xyz"]
    )
    monkeypatch.setattr(
        "compgen.audit.fresh_agent_grading.fresh_agent_tasks_root",
        lambda repo_root=None: fake_root,
    )
    with pytest.raises(FreshAgentTaskError, match="unknown tools"):
        load_task("unknown_tool_v1", known_tool_ids=("compgen_echo",))


# ---------- Real-hardware path --------------------------------------


@pytest.mark.skipif(
    os.environ.get("COMPGEN_SKIP_BASELINE") == "1",
    reason="COMPGEN_SKIP_BASELINE=1 set; skipping the real-hardware baseline run",
)
def test_probe_providers_baseline_runs_and_grades_clean(tmp_path):
    """Run the shipped probe-providers baseline on real hardware.

    This is the canonical success: a brand-new run produces all
    seven probe artifacts and the grader confirms typed shape across
    every JSON file + the CSV matrix header.
    """

    task = load_task("probe_providers_v1")
    run_dir = tmp_path / "probe_run"
    proc = run_baseline(task, run_dir=run_dir)
    assert proc.returncode == 0, proc.stderr
    result = grade(task, run_dir=run_dir)
    assert result.passed, result.violations
    rj = json.loads((run_dir / "grading_result.json").read_text(encoding="utf-8"))
    assert rj["passed"] is True
    # Sanity: the probe produced a non-empty providers list.
    body = json.loads((run_dir / "provider_status.json").read_text(encoding="utf-8"))
    assert isinstance(body["providers"], list) and len(body["providers"]) > 0
