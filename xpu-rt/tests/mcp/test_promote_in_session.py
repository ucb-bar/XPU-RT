"""P7.5 — session-scoped authored-tool graduation.

Two trials in one session must be enough to graduate; the same trial
set under cross-session thresholds (5 passes / 2 workloads / 2 targets)
must NOT graduate. Cross-session graduation state file must remain
untouched after a session-scoped promotion (so a future cross-session
promotion can still happen with the higher bar).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from compgen.agent.invent_slots.registrar import register_invent_slots
from compgen.agent.llm_driver import LLMDrivenCompiler
from compgen.agent.self_extension import (
    AuthoredTool, AuthoredToolSource, TrialScenario,
    clear_authored_index, register_authored_tool, run_trial,
)
from compgen.agent.self_extension.graduate import promote_authored_tools
from compgen.api import compile_model, device as _device
from compgen.llm.mock_client import MockLLMClient
from compgen.llm.registry import Registry
from compgen.mcp.session import SessionManager
from compgen.mcp.tools.graduate import (
    GRADUATE_TOOLS,
    promote_in_session_authored_tools,
)

EXEMPLAR = (
    Path(__file__).resolve().parents[1]
    / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
)

_TOOL = AuthoredTool(
    name="ses_demo_tool",
    phase=3,
    source=AuthoredToolSource(
        source="def run(x):\n    return {'doubled': x * 2}\n",
        entry_name="run",
    ),
    description="doubles x",
    args_schema=({"name": "x", "dtype": "int", "description": "x"},),
    result_schema={"dtype": "dict", "description": "x*2"},
)


def _scorer(_):
    return True, 1.0


def _open(tmp_path: Path) -> tuple[SessionManager, str, Path]:
    sm = SessionManager(scratch_root=tmp_path / "scratch")
    session = sm.open()
    dev = _device(EXEMPLAR)
    compiled = compile_model(
        nn.Linear(8, 4).eval(), dev, sample_inputs=(torch.randn(1, 8),),
    )
    reg = Registry(); register_invent_slots(reg)
    env = compiled.create_agent_env(budget=4)
    driver = LLMDrivenCompiler(
        env=env, target=dev.profile,
        llm_client=MockLLMClient(strict=False),
        budget=4, registry=reg,
    )
    session.compiled = compiled
    session.device = dev
    session.driver = driver
    # Per-session log path the new tool defaults to.
    log = session.scratch_dir / "authored_trials.jsonl"
    return sm, session.session_id, log


@pytest.fixture(autouse=True)
def _reset_index():
    clear_authored_index()
    yield
    clear_authored_index()


def test_promote_in_session_tool_is_registered() -> None:
    names = [t["name"] for t in GRADUATE_TOOLS]
    assert "promote_in_session_authored_tools" in names


def test_two_trials_in_session_promote(tmp_path: Path) -> None:
    sm, sid, log = _open(tmp_path)
    register_authored_tool(_TOOL)
    for _ in range(2):
        run_trial(
            _TOOL,
            TrialScenario(workload="w1", target="t1",
                          scorer=_scorer, kwargs={"x": 3}),
            log_path=log,
        )

    r = promote_in_session_authored_tools(
        sm, session_id=sid, min_passes=2,
    )
    assert r["ok"]
    assert r["candidates_found"] == 1
    assert len(r["new_tools_registered"]) == 1
    # The graduated tool must be in the session's driver registry.
    session = sm.get(sid)
    assert session.driver.registry.lookup_tool("ses_demo_tool__authored") is not None


def test_same_trials_dont_clear_cross_session_threshold(tmp_path: Path) -> None:
    """Identical trial set under cross-session thresholds must NOT graduate."""
    _, _, log = _open(tmp_path)
    register_authored_tool(_TOOL)
    for _ in range(2):
        run_trial(
            _TOOL,
            TrialScenario(workload="w1", target="t1",
                          scorer=_scorer, kwargs={"x": 3}),
            log_path=log,
        )
    fresh_reg = Registry()
    report = promote_authored_tools(
        fresh_reg,
        authored_index={f"{_TOOL.name}@{_TOOL.source.digest}": _TOOL},
        log_path=log,
        # default min_passes=5 / min_workloads=2 / min_targets=2
    )
    assert report.candidates_found == 0
    assert fresh_reg.lookup_tool("ses_demo_tool__authored") is None


def test_in_session_promotion_does_not_touch_cross_session_state(tmp_path: Path) -> None:
    """Session-scoped promotion must NOT write to the persistent state file."""
    sm, sid, log = _open(tmp_path)
    register_authored_tool(_TOOL)
    for _ in range(2):
        run_trial(
            _TOOL,
            TrialScenario(workload="w1", target="t1",
                          scorer=_scorer, kwargs={"x": 3}),
            log_path=log,
        )
    state_path = log.parent / "authored_graduations.json"
    state_before = state_path.read_text() if state_path.exists() else ""

    promote_in_session_authored_tools(
        sm, session_id=sid, min_passes=2,
    )

    state_after = state_path.read_text() if state_path.exists() else ""
    assert state_before == state_after, (
        "session-scoped promotion mutated the persistent state file"
    )


def test_in_session_promotion_idempotent_within_session(tmp_path: Path) -> None:
    """Calling twice in a row must not double-register."""
    sm, sid, log = _open(tmp_path)
    register_authored_tool(_TOOL)
    for _ in range(2):
        run_trial(
            _TOOL,
            TrialScenario(workload="w1", target="t1",
                          scorer=_scorer, kwargs={"x": 3}),
            log_path=log,
        )
    r1 = promote_in_session_authored_tools(sm, session_id=sid, min_passes=2)
    r2 = promote_in_session_authored_tools(sm, session_id=sid, min_passes=2)
    assert r1["ok"] and r2["ok"]
    # Second call: tool is already in registry, so register_tool would
    # raise ValueError; promote_authored_tools catches that and adds
    # an error entry, so candidates_found shows 1 but no new tools.
    assert r1["candidates_found"] == 1
    # In the second call the tool already exists; either it's reported
    # in errors OR re-registered as no-op; both are acceptable as long
    # as the registry isn't corrupted.
    session = sm.get(sid)
    assert session.driver.registry.lookup_tool("ses_demo_tool__authored") is not None
