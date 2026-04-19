"""Tests for :mod:`compgen.promotion.cross_session`.

Builds synthetic ``tools.jsonl`` transcripts for two sessions that
share one accepted invent-proposal across >=2 workloads and >=2
targets, then asserts that :func:`promote_pending_graduations`
materialises a new :class:`Tool` entry the next time the registry is
accessed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from compgen.llm.registry import Registry
from compgen.promotion.cross_session import (
    CrossSessionGraduationReport,
    promote_pending_graduations,
    report_to_dict,
)


def _jsonl_entry(
    *, slot_name: str, workload: str, target: str,
    chosen: dict[str, Any] | None = None, session_idx: int = 0,
    llm_turn_id: str | None = None,
) -> str:
    """Build one ToolCallRecorder-style JSONL line."""
    chosen = chosen or {"kind": "demo_fusion", "rank": 1}
    entry = {
        "phase": 3,
        "llm_turn_id": llm_turn_id or f"turn_{session_idx}",
        "kind": "invent_proposal",
        "name": slot_name,
        "args": {"workload": workload, "target": target},
        "result": {"status": "accepted", "chosen": chosen},
        "select_vs_invent": "invent",
        "recipe_ir_diff": {"before_hash": "", "after_hash": "", "op_delta": []},
        "gate_result": {"status": "accepted", "details": {}},
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_ms": 5,
    }
    return json.dumps(entry)


def _setup_transcripts(root: Path, slot_name: str = "my_fusion") -> None:
    """Write two tools.jsonl files across two workloads/two targets."""
    root.mkdir(parents=True, exist_ok=True)
    chosen = {"kind": "demo_fusion", "rank": 1}

    # Session 1: workload=distilbert, target=cuda_a100
    s1 = root / "sess_a"
    s1.mkdir(parents=True, exist_ok=True)
    (s1 / "tools.jsonl").write_text("\n".join([
        _jsonl_entry(slot_name=slot_name, workload="distilbert", target="cuda_a100",
                     chosen=chosen, session_idx=1),
    ]) + "\n")

    # Session 2: workload=phi2, target=npu_xyz — same chosen payload.
    s2 = root / "sess_b"
    s2.mkdir(parents=True, exist_ok=True)
    (s2 / "tools.jsonl").write_text("\n".join([
        _jsonl_entry(slot_name=slot_name, workload="phi2", target="npu_xyz",
                     chosen=chosen, session_idx=2),
    ]) + "\n")


def test_no_transcripts_yields_empty_report(tmp_path: Path) -> None:
    reg = Registry()
    report = promote_pending_graduations(reg, transcripts_root=tmp_path)
    assert report.transcripts_scanned == 0
    assert report.requests_found == 0
    assert not report.new_tools_registered


def test_graduation_materialises_tool(tmp_path: Path) -> None:
    _setup_transcripts(tmp_path)
    reg = Registry()
    report = promote_pending_graduations(reg, transcripts_root=tmp_path)
    assert report.transcripts_scanned == 2
    assert report.requests_found == 1
    assert len(report.new_tools_registered) == 1
    g = report.new_tools_registered[0]
    assert g.slot_name == "my_fusion"
    assert g.tool_name == "my_fusion__graduated"
    assert set(g.workloads_proven) == {"distilbert", "phi2"}
    assert set(g.targets_proven) == {"cuda_a100", "npu_xyz"}

    # The graduated tool is registered and callable.
    tool = reg.lookup_tool("my_fusion__graduated")
    assert tool is not None
    assert not tool.is_stub
    result = tool.invoke(ctx={})
    assert result["status"] == "graduated"
    assert result["chosen"]["kind"] == "demo_fusion"


def test_graduation_is_idempotent(tmp_path: Path) -> None:
    _setup_transcripts(tmp_path)
    reg = Registry()
    promote_pending_graduations(reg, transcripts_root=tmp_path)
    second = promote_pending_graduations(reg, transcripts_root=tmp_path)
    # Second call sees the state file and skips.
    assert second.requests_found == 1
    assert second.requests_already_applied == 1
    assert not second.new_tools_registered


def test_below_threshold_not_graduated(tmp_path: Path) -> None:
    """Only one workload + target -> under the default threshold."""
    (tmp_path / "sess").mkdir()
    (tmp_path / "sess" / "tools.jsonl").write_text(
        _jsonl_entry(slot_name="edge_only", workload="w1", target="t1") + "\n"
    )
    reg = Registry()
    report = promote_pending_graduations(reg, transcripts_root=tmp_path)
    assert report.requests_found == 0
    assert not report.new_tools_registered


def test_report_to_dict_is_json_safe(tmp_path: Path) -> None:
    _setup_transcripts(tmp_path)
    reg = Registry()
    report = promote_pending_graduations(reg, transcripts_root=tmp_path)
    d = report_to_dict(report)
    json.dumps(d)                         # must not raise
    assert d["requests_found"] == 1


def test_malformed_jsonl_does_not_crash(tmp_path: Path) -> None:
    (tmp_path / "sess").mkdir()
    (tmp_path / "sess" / "tools.jsonl").write_text(
        "this is not json\n"
        + _jsonl_entry(slot_name="ok", workload="w1", target="t1") + "\n"
    )
    reg = Registry()
    # Must not raise, and must still return a coherent report.
    report = promote_pending_graduations(reg, transcripts_root=tmp_path)
    assert isinstance(report, CrossSessionGraduationReport)
