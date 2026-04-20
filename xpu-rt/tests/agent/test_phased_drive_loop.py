"""Tests for compgen.agent.loop.phased (P2 skeleton)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compgen.agent.loop import PhasedDriveLoop
from compgen.llm import (
    InventSlot,
    Registry,
    Tool,
    ToolArg,
    ToolCallRecorder,
    ToolResult,
)


@pytest.fixture
def registry() -> Registry:
    r = Registry()

    def _echo_impl(**kwargs):
        return {"status": "ok", "echoed": kwargs}

    r.register_tool(
        Tool(
            name="echo_tool",
            phase=2,
            kind="tool",
            wraps_pass="stub",
            autocomp_cost_impact="low",
            args=(ToolArg("region", "region_ref", "region"),),
            result=ToolResult("ok", "ok"),
            description="echo",
            impl=_echo_impl,
            stub=False,
        )
    )

    def _gate(proposal, **ctx):
        return {
            "status": "accepted" if proposal.get("chosen") else "rejected",
            "details": {},
        }

    def _seed(**kw):
        return {
            "candidates": [{"plan": "seeded"}],
            "chosen": {"plan": "seeded"},
            "seed_source": "default",
        }

    r.register_invent_slot(
        InventSlot(
            name="propose_thing",
            phase=3,
            input_schema="inp",
            output_op="recipe.x",
            gate="stub",
            autocomp_cost_impact="high",
            description="test slot",
            baseline_seed=_seed,
            gate_impl=_gate,
            stub=False,
        )
    )

    return r


def test_tool_call_recorded(registry: Registry, tmp_path: Path) -> None:
    rec = ToolCallRecorder(log_path=tmp_path / "t.jsonl")
    loop = PhasedDriveLoop(registry=registry, recorder=rec)

    def policy(phase, r, ctx):
        if phase == 2:
            return [("echo_tool", {"region": "r0"})]
        return []

    result = loop.run(phases=[2], policy=policy)
    assert result.total_tool_calls == 1
    assert result.total_invent_calls == 0

    lines = (tmp_path / "t.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["name"] == "echo_tool"
    assert parsed["kind"] == "tool_call"
    assert parsed["select_vs_invent"] == "select"


def test_invent_slot_runs_baseline_seed_when_requested(registry: Registry, tmp_path: Path) -> None:
    rec = ToolCallRecorder(log_path=tmp_path / "t.jsonl")
    loop = PhasedDriveLoop(registry=registry, recorder=rec)

    def policy(phase, r, ctx):
        if phase == 3:
            return [("propose_thing", {"use_baseline_seed": True})]
        return []

    result = loop.run(phases=[3], policy=policy)
    assert result.total_invent_calls == 1
    summary = result.phase_summaries[0]
    assert summary.invent_calls[0]["status"] == "accepted"
    assert summary.rejected_invents == 0

    parsed = json.loads((tmp_path / "t.jsonl").read_text().strip())
    assert parsed["select_vs_invent"] == "invent"
    assert parsed["gate_result"]["status"] == "accepted"


def test_invent_slot_rejection_counted(registry: Registry, tmp_path: Path) -> None:
    rec = ToolCallRecorder(log_path=tmp_path / "t.jsonl")
    loop = PhasedDriveLoop(registry=registry, recorder=rec)

    def policy(phase, r, ctx):
        if phase == 3:
            # Pass a bad proposal (no chosen) — gate rejects
            return [("propose_thing", {"proposal": {}})]
        return []

    result = loop.run(phases=[3], policy=policy)
    assert result.total_invent_calls == 1
    assert result.phase_summaries[0].rejected_invents == 1


def test_unknown_name_recorded_as_not_found(registry: Registry, tmp_path: Path) -> None:
    rec = ToolCallRecorder(log_path=tmp_path / "t.jsonl")
    loop = PhasedDriveLoop(registry=registry, recorder=rec)

    def policy(phase, r, ctx):
        return [("ghost_tool", {})]

    result = loop.run(phases=[2], policy=policy)
    entries = result.phase_summaries[0].tool_calls
    assert len(entries) == 1
    assert entries[0]["status"] == "not_found"


def test_tool_exception_recorded(registry: Registry, tmp_path: Path) -> None:
    def _boom(**kw):
        raise RuntimeError("boom")

    registry.register_tool(
        Tool(
            name="boom_tool",
            phase=2,
            kind="tool",
            wraps_pass="stub",
            autocomp_cost_impact="low",
            args=(),
            result=ToolResult("ok", "ok"),
            description="boom",
            impl=_boom,
            stub=False,
        )
    )

    rec = ToolCallRecorder(log_path=tmp_path / "t.jsonl")
    loop = PhasedDriveLoop(registry=registry, recorder=rec)

    def policy(phase, r, ctx):
        return [("boom_tool", {})]

    result = loop.run(phases=[2], policy=policy)
    entry = result.phase_summaries[0].tool_calls[0]
    assert entry["status"] == "error"


def test_multi_phase_ordering(registry: Registry, tmp_path: Path) -> None:
    rec = ToolCallRecorder(log_path=tmp_path / "t.jsonl")
    loop = PhasedDriveLoop(registry=registry, recorder=rec)

    def policy(phase, r, ctx):
        if phase == 2:
            return [("echo_tool", {"region": "phase2_r"})]
        if phase == 3:
            return [("propose_thing", {"use_baseline_seed": True})]
        return []

    result = loop.run(phases=[2, 3], policy=policy)
    assert [s.phase for s in result.phase_summaries] == [2, 3]
    assert result.total_tool_calls == 1
    assert result.total_invent_calls == 1

    lines = (tmp_path / "t.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_no_recorder_ok(registry: Registry) -> None:
    loop = PhasedDriveLoop(registry=registry, recorder=None)

    def policy(phase, r, ctx):
        return [("echo_tool", {"region": "r0"})]

    result = loop.run(phases=[2], policy=policy)
    assert result.transcript_path is None
    assert result.total_tool_calls == 1
