"""Tests for the hierarchical knowledge store.

Locks in:
  * scope-chain resolution from target name + profile
  * narrow queries (stage / op_family / topic) actually narrow
  * context_brief renders only relevant lessons
  * seed-lessons installer is idempotent
"""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.memory.knowledge import (
    KnowledgeStore,
    Lesson,
    scope_chain_for_target,
)
from compgen.memory import seed_lessons


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(root=tmp_path / "knowledge")


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------


def test_scope_chain_for_known_target() -> None:
    chain = scope_chain_for_target("cuda-a100")
    assert chain[0] == "backends/gpu/nvidia/ampere"
    assert "general" in chain
    assert chain[-1] == "general"


def test_scope_chain_for_unknown_target_falls_back_by_prefix() -> None:
    assert scope_chain_for_target("cuda-something-new")[0] == "backends/gpu/nvidia/general"
    assert scope_chain_for_target("rocm-mi400")[0] == "backends/gpu/amd/general"
    assert scope_chain_for_target("cpu-foobar")[0] == "backends/cpu/general"
    assert scope_chain_for_target("totally-unknown")[0] == "general"


# ---------------------------------------------------------------------------
# Lesson add + read round-trip
# ---------------------------------------------------------------------------


def test_add_then_query_round_trips(store: KnowledgeStore) -> None:
    l = Lesson(
        scope="general", category="design",
        summary="opt-in instrumentation",
        stage="instrumentation", topic="profiling",
        tags=("profiling",),
    )
    store.add(l)
    out = store.query(["general"])
    assert len(out) == 1
    assert out[0].summary == "opt-in instrumentation"
    assert out[0].id.startswith("lesson_")
    assert out[0].timestamp


def test_lesson_rejects_invalid_category() -> None:
    with pytest.raises(ValueError, match="category"):
        Lesson(scope="general", category="bogus", summary="x")


def test_lesson_rejects_invalid_stage() -> None:
    with pytest.raises(ValueError, match="stage"):
        Lesson(scope="general", category="design", summary="x", stage="bogus-stage")


# ---------------------------------------------------------------------------
# Containerised queries — actually narrow
# ---------------------------------------------------------------------------


def _populate_test_lessons(store: KnowledgeStore) -> None:
    # Note: this lesson is intentionally topic="" so it matches *any* topic filter
    # (semantics: empty topic = universal applicability).
    store.add(Lesson(scope="general", category="design",
                     summary="general design lesson",
                     stage="any"))
    store.add(Lesson(scope="general", category="recipe",
                     summary="general matmul recipe",
                     stage="kernel-gen", op_family="matmul",
                     topic="tile-selection", tags=("matmul",)))
    store.add(Lesson(scope="backends/gpu/nvidia/turing", category="limit",
                     summary="turing-specific matmul limit",
                     stage="kernel-gen", op_family="matmul",
                     topic="perf-ceiling", tags=("matmul", "turing")))
    store.add(Lesson(scope="backends/gpu/nvidia/turing", category="recipe",
                     summary="turing-specific softmax recipe",
                     stage="kernel-gen", op_family="softmax",
                     topic="tile-selection", tags=("softmax", "turing")))


def test_query_by_op_family_excludes_unrelated_ops(store: KnowledgeStore) -> None:
    _populate_test_lessons(store)
    matmul_lessons = store.query(
        scope_chain_for_target("test-gpu-simt"),  # → turing chain
        op_family="matmul",
    )
    summaries = {l.summary for l in matmul_lessons}
    # Both general matmul + turing matmul should be in
    assert "general matmul recipe" in summaries
    assert "turing-specific matmul limit" in summaries
    # The softmax lesson must NOT be in the matmul query
    assert "turing-specific softmax recipe" not in summaries
    # The general design lesson IS in (op_family-empty matches any op)
    assert "general design lesson" in summaries


def test_query_by_stage_filters_correctly(store: KnowledgeStore) -> None:
    _populate_test_lessons(store)
    kernel_gen = store.query(
        scope_chain_for_target("cuda-titan-rtx"),
        stage="kernel-gen",
    )
    summaries = {l.summary for l in kernel_gen}
    # The "any" lesson is in (any-stage matches every stage filter)
    assert "general design lesson" in summaries
    # The kernel-gen lessons are in
    assert "general matmul recipe" in summaries
    assert "turing-specific matmul limit" in summaries

    # A profile-stage query must EXCLUDE the kernel-gen-specific ones
    profiling = store.query(
        scope_chain_for_target("cuda-titan-rtx"),
        stage="instrumentation",
    )
    summaries_p = {l.summary for l in profiling}
    assert "general design lesson" in summaries_p           # any-stage ✓
    assert "general matmul recipe" not in summaries_p       # kernel-gen ✗
    assert "turing-specific matmul limit" not in summaries_p


def test_query_by_topic_narrows_to_one_topic(store: KnowledgeStore) -> None:
    _populate_test_lessons(store)
    tile = store.query(
        scope_chain_for_target("cuda-titan-rtx"),
        topic="tile-selection",
    )
    summaries = {l.summary for l in tile}
    assert "general matmul recipe" in summaries          # tile-selection ✓
    assert "turing-specific softmax recipe" in summaries  # tile-selection ✓
    assert "turing-specific matmul limit" not in summaries  # perf-ceiling ✗
    assert "general design lesson" in summaries           # topic="" matches all


def test_intersected_filters_narrow_to_one_lesson(store: KnowledgeStore) -> None:
    _populate_test_lessons(store)
    out = store.query(
        scope_chain_for_target("cuda-titan-rtx"),
        stage="kernel-gen",
        op_family="matmul",
        topic="tile-selection",
    )
    summaries = {l.summary for l in out}
    # Only the general matmul recipe matches all three filters precisely.
    assert summaries == {"general matmul recipe", "general design lesson"}


def test_limit_caps_returned_lessons(store: KnowledgeStore) -> None:
    _populate_test_lessons(store)
    out = store.query(scope_chain_for_target("cuda-titan-rtx"), limit=2)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# context_brief — markdown render for prompt injection
# ---------------------------------------------------------------------------


def test_context_brief_renders_only_relevant_lessons(store: KnowledgeStore) -> None:
    _populate_test_lessons(store)
    brief = store.context_brief(
        "cuda-titan-rtx",
        stage="kernel-gen", op_family="matmul", topic="tile-selection",
    )
    assert "general matmul recipe" in brief
    assert "turing-specific softmax recipe" not in brief    # different op_family
    # Header carries the filter values
    assert "cuda-titan-rtx" in brief
    assert "kernel-gen" in brief
    assert "matmul" in brief
    assert "tile-selection" in brief


def test_context_brief_returns_empty_when_no_match(store: KnowledgeStore) -> None:
    # Nothing populated → brief is empty string.
    brief = store.context_brief("cuda-a100", stage="kernel-gen", op_family="foobar")
    assert brief == ""


# ---------------------------------------------------------------------------
# Seed installer
# ---------------------------------------------------------------------------


def test_seed_install_is_idempotent(store: KnowledgeStore) -> None:
    n1 = seed_lessons.install(store)
    assert n1 > 0
    n2 = seed_lessons.install(store)
    assert n2 == 0          # nothing new added on second run


def test_seed_lessons_cover_known_scopes(store: KnowledgeStore) -> None:
    seed_lessons.install(store)
    scopes = set(store.list_scopes())
    # Spot check — we expect the seed to have populated multiple
    # specific arch scopes plus the generic ones.
    assert "general" in scopes
    assert "backends/gpu/general" in scopes
    assert "backends/gpu/nvidia/general" in scopes
    assert "backends/gpu/nvidia/turing" in scopes
    assert "backends/gpu/nvidia/ampere" in scopes
    assert "drivers/cuda/general" in scopes
