"""End-to-end: register an authored tool, log enough passing trials,
trigger session-start graduation, observe the live registry."""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.agent.self_extension import (
    AuthoredTool,
    AuthoredToolSource,
    TrialScenario,
    clear_authored_index,
    promote_authored_tools,
    register_authored_tool,
    run_trial,
)
from compgen.agent.self_extension._index import snapshot_authored_index
from compgen.llm.registry import Registry

_TOOL = AuthoredTool(
    name="reg_int_tool",
    phase=3,
    source=AuthoredToolSource(
        source="def run(n):\n    return {'doubled': n * 2}\n",
        entry_name="run",
    ),
    description="doubles n",
    args_schema=({"name": "n", "dtype": "int", "description": "n"},),
    result_schema={"dtype": "dict", "description": "n*2"},
)


def _scorer(_):
    return True, 1.0


@pytest.fixture(autouse=True)
def _reset_index():
    clear_authored_index()
    yield
    clear_authored_index()


def test_registered_index_is_visible_to_promotion(monkeypatch, tmp_path: Path) -> None:
    log = tmp_path / "trials.jsonl"
    register_authored_tool(_TOOL)

    for w in ("alpha", "beta"):
        for t in ("cpu", "gpu"):
            for _ in range(2):
                run_trial(
                    _TOOL,
                    TrialScenario(workload=w, target=t, scorer=_scorer, kwargs={"n": 7}),
                    log_path=log,
                )

    reg = Registry()
    report = promote_authored_tools(
        reg,
        authored_index=snapshot_authored_index(),
        log_path=log,
    )
    assert report.candidates_found >= 1
    assert reg.lookup_tool("reg_int_tool__authored") is not None


def test_session_start_graduation_disabled_by_env(monkeypatch, tmp_path: Path) -> None:
    """When the disable env var is set, registrar must not graduate.

    The conftest already sets it for the whole session, so this is a
    smoke test that the registrar honours it.
    """
    monkeypatch.setenv("COMPGEN_DISABLE_AUTHORED_GRADUATION", "1")
    from compgen.agent.invent_slots.registrar import register_invent_slots

    reg = Registry()
    register_invent_slots(reg)
    # No authored tools should appear because the loop didn't run.
    assert reg.lookup_tool("reg_int_tool__authored") is None
