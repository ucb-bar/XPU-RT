"""Tests for :mod:`compgen.agent.self_extension.trials`."""

from __future__ import annotations

import json
from pathlib import Path

from compgen.agent.self_extension.authored_tool import (
    AuthoredTool,
    AuthoredToolSource,
)
from compgen.agent.self_extension.trials import TrialScenario, run_trial


def _pass_scorer(_):
    return True, 1.0


def _fail_scorer(_):
    return False, 0.0


def _bad_scorer(_):
    raise RuntimeError("scorer bug")


_TOOL_SRC = """
def run(x):
    return x + 1
"""


def _tool() -> AuthoredTool:
    return AuthoredTool(
        name="my_authored_tool",
        phase=3,
        source=AuthoredToolSource(source=_TOOL_SRC, entry_name="run"),
        description="demo authored tool",
        args_schema=({"name": "x", "dtype": "int", "description": "x"},),
        result_schema={"dtype": "int", "description": "x+1"},
    )


def test_run_trial_records_pass(tmp_path: Path) -> None:
    log = tmp_path / "trials.jsonl"
    scenario = TrialScenario(
        workload="w1",
        target="t1",
        scorer=_pass_scorer,
        kwargs={"x": 2},
        name="plus_one",
    )
    trial = run_trial(_tool(), scenario, session_id="s1", log_path=log)
    assert trial.passed
    assert trial.workload == "w1"
    line = log.read_text().strip().splitlines()[-1]
    record = json.loads(line)
    assert record["tool_name"] == "my_authored_tool"
    assert record["passed"] is True


def test_run_trial_records_fail(tmp_path: Path) -> None:
    log = tmp_path / "trials.jsonl"
    scenario = TrialScenario(
        workload="w1",
        target="t1",
        scorer=_fail_scorer,
        kwargs={"x": 2},
    )
    trial = run_trial(_tool(), scenario, log_path=log)
    assert not trial.passed


def test_run_trial_bad_scorer_counts_as_fail(tmp_path: Path) -> None:
    log = tmp_path / "trials.jsonl"
    scenario = TrialScenario(
        workload="w1",
        target="t1",
        scorer=_bad_scorer,
        kwargs={"x": 2},
    )
    trial = run_trial(_tool(), scenario, log_path=log)
    assert not trial.passed
    assert trial.error and "scorer_raised" in trial.error


def test_run_trial_authored_tool_raises_counts_as_fail(tmp_path: Path) -> None:
    tool = AuthoredTool(
        name="blows_up",
        phase=3,
        source=AuthoredToolSource(
            source="def run():\n    raise ValueError('nope')\n",
        ),
    )
    log = tmp_path / "trials.jsonl"
    scenario = TrialScenario(
        workload="w1",
        target="t1",
        scorer=_pass_scorer,
    )
    trial = run_trial(tool, scenario, log_path=log)
    assert not trial.passed
    assert trial.violation_count >= 1


def test_trial_log_writes_one_line_per_call(tmp_path: Path) -> None:
    log = tmp_path / "trials.jsonl"
    scenario = TrialScenario(
        workload="w",
        target="t",
        scorer=_pass_scorer,
        kwargs={"x": 1},
    )
    run_trial(_tool(), scenario, log_path=log)
    run_trial(_tool(), scenario, log_path=log)
    assert len(log.read_text().splitlines()) == 2
