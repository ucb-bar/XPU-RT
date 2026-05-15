"""Tests for compgen.llm.recorder.ToolCallRecorder (P13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compgen.llm import ToolCallRecorder


@pytest.fixture
def recorder(tmp_path: Path) -> ToolCallRecorder:
    return ToolCallRecorder(log_path=tmp_path / "transcript.jsonl")


def test_single_record_roundtrip(recorder: ToolCallRecorder) -> None:
    rec = recorder.record(
        phase=2,
        name="raise_special_ops",
        kind="tool_call",
        args={"region": "r0", "library": ["softmax"]},
        result={"status": "ok"},
        elapsed_ms=5,
    )
    assert rec.phase == 2
    assert rec.name == "raise_special_ops"
    assert rec.kind == "tool_call"
    assert rec.elapsed_ms == 5
    assert recorder.total_calls == 1

    lines = recorder.log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["phase"] == 2
    assert parsed["args"]["region"] == "r0"


def test_recorder_distinguishes_before_after_hashes(recorder: ToolCallRecorder) -> None:
    rec = recorder.record(
        phase=3,
        name="propose_fusion",
        kind="invent_proposal",
        select_vs_invent="invent",
        args={"region": "r0"},
        result={"candidates": 2},
        before={"op_count": 10},
        after={"op_count": 3},
        gate_result={"status": "accepted", "details": {}},
        elapsed_ms=50,
    )
    assert rec.recipe_ir_diff["before_hash"].startswith("sha256:")
    assert rec.recipe_ir_diff["after_hash"].startswith("sha256:")
    assert rec.recipe_ir_diff["before_hash"] != rec.recipe_ir_diff["after_hash"]


def test_recorder_stable_hash() -> None:
    tr = ToolCallRecorder(log_path=Path("/tmp/unused_stable_hash.jsonl"), enabled=False)
    h1 = tr.hash_ir({"a": 1, "b": 2})
    h2 = tr.hash_ir({"b": 2, "a": 1})
    assert h1 == h2, "hash_ir should be stable across dict key order"


def test_recorder_disabled_does_not_write(tmp_path: Path) -> None:
    p = tmp_path / "noop.jsonl"
    tr = ToolCallRecorder(log_path=p, enabled=False)
    tr.record(phase=2, name="x", result={}, elapsed_ms=0)
    assert not p.exists()


def test_recorder_appends_multiple(recorder: ToolCallRecorder) -> None:
    for i in range(3):
        recorder.record(
            phase=2,
            name=f"tool_{i}",
            kind="tool_call",
            args={"i": i},
            result={"ok": True},
            elapsed_ms=i,
        )
    lines = recorder.log_path.read_text().strip().split("\n")
    assert len(lines) == 3
    assert recorder.total_calls == 3
    # Spot-check ordering preserved
    parsed = [json.loads(line) for line in lines]
    assert [p["name"] for p in parsed] == ["tool_0", "tool_1", "tool_2"]


def test_invent_proposal_with_gate_result_recorded(recorder: ToolCallRecorder) -> None:
    rec = recorder.record(
        phase=3,
        name="propose_layout_plan",
        kind="invent_proposal",
        select_vs_invent="invent",
        args={"region": "r0"},
        result={"plan_candidates": 2},
        gate_result={"status": "rejected", "details": {"reason": "tile unaligned"}},
        elapsed_ms=12,
    )
    assert rec.gate_result["status"] == "rejected"
    parsed = json.loads(recorder.log_path.read_text().strip())
    assert parsed["gate_result"]["status"] == "rejected"
