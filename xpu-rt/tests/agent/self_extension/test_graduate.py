"""Tests for :mod:`compgen.agent.self_extension.graduate`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.agent.self_extension.authored_tool import (
    AuthoredTool,
    AuthoredToolSource,
    authored_tool_key,
)
from compgen.agent.self_extension.graduate import (
    AuthoredGraduationReport,
    promote_authored_tools,
)
from compgen.agent.self_extension.trials import TrialScenario, run_trial
from compgen.llm.registry import Registry


_TOOL_SRC = """
def run(x):
    return {"x_squared": x * x}
"""


def _authored() -> AuthoredTool:
    return AuthoredTool(
        name="square_tool",
        phase=3,
        source=AuthoredToolSource(source=_TOOL_SRC, entry_name="run"),
        description="returns x squared",
        args_schema=({"name": "x", "dtype": "int", "description": "x"},),
        result_schema={"dtype": "dict", "description": "x^2"},
    )


def _populate_trials(
    tool: AuthoredTool, log: Path, *,
    passes_per_pair: int = 3,
    pairs: list[tuple[str, str]] | None = None,
) -> None:
    pairs = pairs or [("w1", "t1"), ("w2", "t2")]

    def _scorer(_):
        return True, 1.0

    for w, t in pairs:
        for _ in range(passes_per_pair):
            run_trial(
                tool,
                TrialScenario(
                    workload=w, target=t, scorer=_scorer, kwargs={"x": 3},
                ),
                log_path=log,
            )


def test_no_trials_yields_empty_report(tmp_path: Path) -> None:
    reg = Registry()
    report = promote_authored_tools(reg, log_path=tmp_path / "trials.jsonl")
    assert report.trials_scanned == 0
    assert not report.new_tools_registered


def test_graduation_promotes_tool(tmp_path: Path) -> None:
    log = tmp_path / "trials.jsonl"
    tool = _authored()
    _populate_trials(tool, log, passes_per_pair=3)  # 6 passes across 2 pairs

    index = {authored_tool_key(tool): tool}
    reg = Registry()
    report = promote_authored_tools(
        reg, authored_index=index, log_path=log,
        min_passes=5, min_workloads=2, min_targets=2,
    )
    assert report.candidates_found == 1
    assert len(report.new_tools_registered) == 1

    registered = reg.lookup_tool("square_tool__authored")
    assert registered is not None
    assert not registered.is_stub
    call_result = registered.invoke(x=4)
    assert call_result["status"] == "ok"
    assert call_result["value"] == {"x_squared": 16}


def test_graduation_is_idempotent(tmp_path: Path) -> None:
    log = tmp_path / "trials.jsonl"
    tool = _authored()
    _populate_trials(tool, log, passes_per_pair=3)

    index = {authored_tool_key(tool): tool}
    reg = Registry()
    promote_authored_tools(reg, authored_index=index, log_path=log)
    second = promote_authored_tools(reg, authored_index=index, log_path=log)
    assert second.candidates_already_applied == 1
    assert not second.new_tools_registered


def test_below_passes_threshold_not_graduated(tmp_path: Path) -> None:
    log = tmp_path / "trials.jsonl"
    tool = _authored()
    # 4 passes across 2 pairs — under the default 5.
    _populate_trials(tool, log, passes_per_pair=2)
    index = {authored_tool_key(tool): tool}
    reg = Registry()
    report = promote_authored_tools(reg, authored_index=index, log_path=log)
    assert report.candidates_found == 0


def test_below_workload_threshold_not_graduated(tmp_path: Path) -> None:
    log = tmp_path / "trials.jsonl"
    tool = _authored()
    # 6 passes but only one workload -> under min_workloads=2.
    _populate_trials(tool, log, passes_per_pair=6, pairs=[("w1", "t1")])
    index = {authored_tool_key(tool): tool}
    reg = Registry()
    report = promote_authored_tools(reg, authored_index=index, log_path=log)
    assert report.candidates_found == 0


def test_missing_index_entry_records_error(tmp_path: Path) -> None:
    log = tmp_path / "trials.jsonl"
    tool = _authored()
    _populate_trials(tool, log, passes_per_pair=3)
    reg = Registry()
    report = promote_authored_tools(reg, authored_index={}, log_path=log)
    # Cleared thresholds but no source in the index — skip + record.
    assert report.candidates_found == 1
    assert report.errors


def test_new_digest_reclears_thresholds_independently(tmp_path: Path) -> None:
    """A revised authored source has a distinct digest and is a fresh candidate."""
    log = tmp_path / "trials.jsonl"

    v1 = _authored()
    _populate_trials(v1, log, passes_per_pair=3)

    v2 = AuthoredTool(
        name="square_tool",  # same name
        phase=3,
        source=AuthoredToolSource(
            source=_TOOL_SRC + "\n# rev 2\n",  # different digest
            entry_name="run",
        ),
    )
    # Only 2 passes for v2 — under the threshold.
    _populate_trials(v2, log, passes_per_pair=1)

    index = {authored_tool_key(v1): v1, authored_tool_key(v2): v2}
    reg = Registry()
    report = promote_authored_tools(reg, authored_index=index, log_path=log)
    # Only v1 clears.
    assert len(report.new_tools_registered) == 1
    assert report.new_tools_registered[0]["source_digest"] == v1.source.digest


def test_malformed_jsonl_does_not_crash(tmp_path: Path) -> None:
    log = tmp_path / "trials.jsonl"
    # Write a valid and an invalid line.
    tool = _authored()
    _populate_trials(tool, log, passes_per_pair=3)
    log.write_text("not json\n" + log.read_text())
    reg = Registry()
    report = promote_authored_tools(
        reg, authored_index={authored_tool_key(tool): tool}, log_path=log,
    )
    assert isinstance(report, AuthoredGraduationReport)
    assert len(report.new_tools_registered) == 1
