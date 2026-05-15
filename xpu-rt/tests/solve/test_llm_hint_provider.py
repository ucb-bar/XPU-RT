"""Tests for the LLM hint provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.solve.llm_hint_provider import (
    get_memory_hints,
    read_llm_hints_from_file,
    write_llm_hint_request,
)
from compgen.solve.memory_planner import (
    BufferSpec,
    MemoryPlanInput,
    TierCapacity,
)
from compgen.solve.solver_hints import MemoryHints


def _plan_input() -> MemoryPlanInput:
    return MemoryPlanInput(
        buffers=(
            BufferSpec("weights", 64 * 1024 * 1024, 0, 100, ("host", "scratchpad")),
            BufferSpec("activations", 1 * 1024 * 1024, 5, 6, ("scratchpad", "host")),
        ),
        tier_capacities=(
            TierCapacity("scratchpad", 512 * 1024 * 1024),
            TierCapacity("host", 4 * 1024 * 1024 * 1024),
        ),
    )


def test_default_mode_returns_rule_based():
    hints = get_memory_hints(_plan_input())
    assert hints.source == "rule_based"


def test_llm_file_mode_loads_real_document(tmp_path: Path):
    doc = {
        "schema_version": "memory_hints_v1",
        "source": "llm",
        "tier_hints": [
            {"buffer_id": "weights", "tier_id": "host", "confidence": 0.95, "reason": "large"},
        ],
        "offset_warm_start": [],
        "stage_partition": [],
        "symmetry_classes": [],
        "confidence_summary": {"tier_hints_fraction": 0.5},
    }
    path = tmp_path / "llm_hints.json"
    path.write_text(json.dumps(doc))
    hints = get_memory_hints(_plan_input(), mode="llm_file", llm_hint_path=path)
    assert hints.source == "llm"
    assert len(hints.tier_hints) == 1
    assert hints.tier_hints[0].buffer_id == "weights"


def test_llm_file_mode_falls_back_when_missing(tmp_path: Path):
    """Missing file must degrade to rule_based honestly, not crash."""

    hints = get_memory_hints(
        _plan_input(), mode="llm_file",
        llm_hint_path=tmp_path / "does_not_exist.json",
    )
    assert hints.source == "rule_based"


def test_llm_file_mode_falls_back_on_malformed_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{ broken json")
    hints = get_memory_hints(
        _plan_input(), mode="llm_file", llm_hint_path=path,
    )
    assert hints.source == "rule_based"


def test_merged_mode_combines_rule_based_and_llm(tmp_path: Path):
    """Rule-based + LLM merge: LLM overrides on conflict
    (higher confidence wins)."""

    doc = {
        "schema_version": "memory_hints_v1",
        "source": "llm",
        "tier_hints": [
            {"buffer_id": "weights", "tier_id": "scratchpad", "confidence": 0.99,
             "reason": "LLM thinks scratchpad is better"},
        ],
        "offset_warm_start": [],
        "stage_partition": [],
        "symmetry_classes": [],
        "confidence_summary": {},
    }
    path = tmp_path / "llm_hints.json"
    path.write_text(json.dumps(doc))
    merged = get_memory_hints(
        _plan_input(), mode="merged", llm_hint_path=path,
    )
    assert merged.source.startswith("merged:")
    # LLM wins for ``weights`` (higher confidence than rule_based 0.85).
    by_id = {h.buffer_id: h for h in merged.tier_hints}
    assert by_id["weights"].tier_id == "scratchpad"
    assert by_id["weights"].confidence == pytest.approx(0.99)


def test_write_llm_hint_request_serializes_problem(tmp_path: Path):
    out = tmp_path / "req.json"
    write_llm_hint_request(_plan_input(), out_path=out)
    body = json.loads(out.read_text())
    assert body["schema_version"] == "memory_hint_request_v1"
    assert len(body["buffers"]) == 2
    assert {b["buffer_id"] for b in body["buffers"]} == {"weights", "activations"}
    assert "prompt" in body


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown hint mode"):
        get_memory_hints(_plan_input(), mode="magic")


def test_read_returns_none_on_missing_file(tmp_path: Path):
    assert read_llm_hints_from_file(tmp_path / "nope.json") is None


def test_read_returns_none_on_non_dict_json(tmp_path: Path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]")
    assert read_llm_hints_from_file(path) is None
