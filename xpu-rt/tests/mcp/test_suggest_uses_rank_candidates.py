"""Tests for the G6 wire-in: mcp/tools/suggest reorders candidates
through the P3.3 rank_candidates primitive.

Coverage:

* When suggest is called with candidates from the underlying _suggest,
  the reordering it produces is a *permutation* of the input list —
  no candidate invented, none silently dropped (the headline P3.0
  invariant).
* Under COMPGEN_DISABLE_LLM=1 the order matches the deterministic
  fallback (descending expected_impact + alphabetical tiebreak).
* An empty candidate list passes through unchanged (no crash).

This test exercises the wire-in directly against the underlying
``rank_candidates`` primitive — the full MCP integration (which
requires a live SessionManager + driver + compiled session) is
covered indirectly by the existing tests/agent/suggest tests.
"""

from __future__ import annotations

import pytest
from compgen.agent.primitives.rank_candidates import rank_candidates
from compgen.agent.suggest._candidate import ProposalCandidate


def _cand(rationale: str, impact: float) -> ProposalCandidate:
    return ProposalCandidate(
        chosen={"id": rationale.replace(" ", "_")},
        rationale=rationale,
        expected_impact=impact,
    )


@pytest.fixture(autouse=True)
def _disable_llm(monkeypatch):
    """Run under the deterministic-fallback regime so the test never
    depends on a live LLM connection."""

    monkeypatch.setenv("COMPGEN_DISABLE_LLM", "1")
    yield


def test_rank_candidates_returns_permutation():
    """The primitive's contract — invariant the suggest wire-in
    relies on: the returned ranking is a permutation of the input."""

    candidates = [
        _cand("zeta", 0.5),
        _cand("alpha", 0.9),
        _cand("mu", 0.3),
    ]
    result = rank_candidates(candidates)
    returned_indices = sorted(r["index"] for r in result["ranking"])
    assert returned_indices == list(range(len(candidates)))


def test_rank_candidates_fallback_matches_legacy_sort_order():
    """The fallback orders by descending impact + alphabetical rationale.
    This is the order the legacy _suggest already produced — the wire-in
    preserves it bit-for-bit when no LLM is available."""

    candidates = [
        _cand("zeta", 0.5),
        _cand("alpha", 0.9),
        _cand("mu", 0.5),
    ]
    result = rank_candidates(candidates)
    rationales = [r["rationale"] for r in result["ranking"]]
    # 0.9 wins; then the two 0.5s break alphabetically (mu < zeta).
    assert rationales == ["alpha", "mu", "zeta"]


def test_rank_candidates_empty_input():
    result = rank_candidates([])
    assert result["ranking"] == []
    assert result["n_candidates"] == 0


def test_suggest_module_imports_rank_candidates():
    """Smoke-test: the wire-in does not import-fail. Importing the
    suggest module must succeed even with the primitive registered."""

    from compgen.mcp.tools import suggest  # noqa: F401

    assert callable(suggest.suggest_proposals)


def test_suggest_wire_in_preserves_permutation():
    """End-to-end: simulate the candidate-reordering step of suggest
    by calling rank_candidates exactly as suggest.py:65-92 does,
    then verify the resulting permutation."""

    candidates = [
        _cand("low_impact", 0.1),
        _cand("mid_impact", 0.5),
        _cand("high_impact", 0.9),
    ]
    ranking = rank_candidates(list(candidates))
    order = [int(r["index"]) for r in ranking.get("ranking", [])]
    assert len(order) == len(candidates)
    assert set(order) == set(range(len(candidates)))
    reordered = [candidates[i] for i in order]
    # High impact first.
    assert reordered[0].rationale == "high_impact"
    assert reordered[-1].rationale == "low_impact"
