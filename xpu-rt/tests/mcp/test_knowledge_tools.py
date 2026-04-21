"""Tests for ``compgen.mcp.tools.knowledge``.

Locks in:
  * record_lesson with explicit scope or target derives the right path
  * record_lesson rejects bad category / stage values
  * query_knowledge walks the target's scope chain and returns lessons
  * narrow filters (op_family, topic) intersect correctly
  * get_context_brief returns a non-empty prompt-friendly brief when
    matching lessons exist
"""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.mcp.session import SessionManager
from compgen.mcp.tools.knowledge import (
    KNOWLEDGE_TOOLS,
    get_context_brief,
    query_knowledge,
    record_lesson,
)
from compgen.memory.knowledge import (
    KnowledgeStore, set_shared_store,
)


@pytest.fixture
def isolated_store(tmp_path: Path):
    s = KnowledgeStore(root=tmp_path / "knowledge")
    set_shared_store(s)
    yield s
    set_shared_store(None)


@pytest.fixture
def sm(tmp_path: Path) -> SessionManager:
    s = SessionManager(scratch_root=tmp_path / "compgen_mcp")
    s.open(session_id="sess1")
    return s


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_knowledge_tools_registered_with_expected_names() -> None:
    names = {t["name"] for t in KNOWLEDGE_TOOLS}
    assert names == {"record_lesson", "query_knowledge", "get_context_brief"}


# ---------------------------------------------------------------------------
# record_lesson
# ---------------------------------------------------------------------------


def test_record_lesson_with_target_uses_most_specific_scope(sm, isolated_store) -> None:
    out = record_lesson(
        sm, session_id="sess1",
        category="perf", summary="Ampere mma 16x16x16 fp16 matmul peaks at 90% of theoretical TFLOPS",
        target="cuda-a100",
        stage="kernel-gen", op_family="matmul", topic="tile-selection",
        tags=["mma", "fp16"],
    )
    assert out["ok"]
    # cuda-a100's most-specific scope is backends/gpu/nvidia/ampere
    assert out["scope"] == "backends/gpu/nvidia/ampere"
    assert out["lesson_id"]


def test_record_lesson_with_explicit_scope(sm, isolated_store) -> None:
    out = record_lesson(
        sm, session_id="sess1",
        category="design",
        summary="Always promote DRAM→SCRATCHPAD before MEGA fusion",
        scope="general",
    )
    assert out["ok"] and out["scope"] == "general"


def test_record_lesson_requires_target_or_scope(sm, isolated_store) -> None:
    out = record_lesson(
        sm, session_id="sess1",
        category="perf", summary="x",
    )
    assert out["ok"] is False
    assert "scope" in out["error"]


def test_record_lesson_rejects_invalid_category(sm, isolated_store) -> None:
    out = record_lesson(
        sm, session_id="sess1",
        category="bogus", summary="x", target="cuda-a100",
    )
    assert out["ok"] is False
    assert "category" in out["error"]


def test_record_lesson_rejects_invalid_stage(sm, isolated_store) -> None:
    out = record_lesson(
        sm, session_id="sess1",
        category="perf", summary="x", target="cuda-a100",
        stage="not-a-stage",
    )
    assert out["ok"] is False
    assert "stage" in out["error"]


# ---------------------------------------------------------------------------
# query_knowledge
# ---------------------------------------------------------------------------


def test_query_walks_scope_chain_and_returns_lessons(sm, isolated_store) -> None:
    record_lesson(
        sm, session_id="sess1",
        category="design", summary="general cross-target lesson",
        scope="general",
    )
    record_lesson(
        sm, session_id="sess1",
        category="perf", summary="ampere-specific tile size",
        target="cuda-a100",
    )
    out = query_knowledge(sm, session_id="sess1", target="cuda-a100")
    assert out["ok"]
    summaries = [l["summary"] for l in out["lessons"]]
    assert "general cross-target lesson" in summaries
    assert "ampere-specific tile size" in summaries
    assert out["scope_chain"][-1] == "general"


def test_query_with_narrow_filters_intersects(sm, isolated_store) -> None:
    record_lesson(
        sm, session_id="sess1",
        category="perf", summary="matmul-tile lesson",
        target="cuda-a100",
        op_family="matmul", topic="tile-selection",
    )
    record_lesson(
        sm, session_id="sess1",
        category="perf", summary="softmax fusion lesson",
        target="cuda-a100",
        op_family="softmax", topic="fusion-decision",
    )
    out = query_knowledge(
        sm, session_id="sess1", target="cuda-a100",
        op_family="matmul", topic="tile-selection",
    )
    summaries = [l["summary"] for l in out["lessons"]]
    assert "matmul-tile lesson" in summaries
    assert "softmax fusion lesson" not in summaries


# ---------------------------------------------------------------------------
# get_context_brief
# ---------------------------------------------------------------------------


def test_get_context_brief_returns_non_empty_when_lessons_match(sm, isolated_store) -> None:
    record_lesson(
        sm, session_id="sess1",
        category="recipe", summary="Triton matmul on Turing: 64x64x16 + num_warps=4",
        target="cuda-titan-rtx",
        stage="kernel-gen", op_family="matmul", topic="tile-selection",
    )
    out = get_context_brief(
        sm, session_id="sess1",
        target="cuda-titan-rtx", stage="kernel-gen",
        op_family="matmul", topic="tile-selection",
    )
    assert out["ok"]
    assert out["brief"]    # non-empty
    assert "64x64x16" in out["brief"] or "Turing" in out["brief"] or "matmul" in out["brief"]
