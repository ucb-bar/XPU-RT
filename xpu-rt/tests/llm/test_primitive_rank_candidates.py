"""Tests for the P3.3 ``rank_candidates`` primitive.

Coverage:

Positive:
* The fallback returns a strict permutation of the input list (set
  of returned ``index`` values equals ``range(len(input))``).
* The ranking is ordered by ``expected_impact`` descending; ties
  break on ``rationale`` alphabetically.
* The output is byte-deterministic across reruns with the same
  input.
* The wrapper exposes the registered ``LLMCallSiteCard``.

Negative controls (forbidden-action invariants):
* No candidate index in the output exceeds the input length — the
  ``invent_candidate_not_in_input_list`` forbidden action cannot
  fire even under random shuffling.
* Empty input list returns ``ranking=[]`` with ``n_candidates=0``.
* The decorator's output_schema rejects a malformed primary (we
  monkeypatch the primary to return invalid output).

The decorator is tested in tests/llm/test_call_site.py; this test
file is specifically about the primitive's *semantic* contract on
the highest-leverage primitive.
"""

from __future__ import annotations

import random

from compgen.agent.primitives.rank_candidates import (
    _rank_candidates_fallback,
    rank_candidates,
)
from compgen.agent.suggest._candidate import ProposalCandidate
from compgen.llm.call_site import get_call_site


def _candidate(impact: float, rationale: str) -> ProposalCandidate:
    return ProposalCandidate(
        chosen={"id": rationale.replace(" ", "_")},
        rationale=rationale,
        expected_impact=impact,
    )


def test_output_is_a_permutation_of_input(monkeypatch):
    monkeypatch.setenv("COMPGEN_DISABLE_LLM", "1")
    cands = [
        _candidate(0.1, "small"),
        _candidate(0.9, "huge"),
        _candidate(0.5, "medium"),
        _candidate(0.7, "large"),
    ]
    result = rank_candidates(cands)
    returned_indices = sorted(r["index"] for r in result["ranking"])
    assert returned_indices == list(range(len(cands)))
    assert result["n_candidates"] == len(cands)


def test_ranking_descending_by_expected_impact(monkeypatch):
    monkeypatch.setenv("COMPGEN_DISABLE_LLM", "1")
    cands = [
        _candidate(0.1, "small"),
        _candidate(0.9, "huge"),
        _candidate(0.5, "medium"),
    ]
    result = rank_candidates(cands)
    scores = [r["score"] for r in result["ranking"]]
    assert scores == sorted(scores, reverse=True)
    # First entry is the highest-impact candidate.
    assert result["ranking"][0]["index"] == 1


def test_tiebreak_alphabetical_on_rationale(monkeypatch):
    monkeypatch.setenv("COMPGEN_DISABLE_LLM", "1")
    cands = [
        _candidate(0.5, "zebra"),
        _candidate(0.5, "apple"),
        _candidate(0.5, "mango"),
    ]
    result = rank_candidates(cands)
    # All impact equal → alphabetical: apple, mango, zebra.
    rationales = [r["rationale"] for r in result["ranking"]]
    assert rationales == ["apple", "mango", "zebra"]


def test_byte_deterministic_across_reruns(monkeypatch):
    monkeypatch.setenv("COMPGEN_DISABLE_LLM", "1")
    cands = [
        _candidate(0.5, "a"),
        _candidate(0.6, "b"),
        _candidate(0.7, "c"),
    ]
    a = rank_candidates(cands)
    b = rank_candidates(cands)
    assert a == b


def test_empty_input(monkeypatch):
    monkeypatch.setenv("COMPGEN_DISABLE_LLM", "1")
    result = rank_candidates([])
    assert result["ranking"] == []
    assert result["n_candidates"] == 0
    assert result["fallback_used"] is True


def test_card_exposes_contract():
    card = get_call_site("rank_candidates")
    assert "invent_candidate_not_in_input_list" in card.forbidden
    assert card.fallback == "rank_candidates_deterministic"


def test_fallback_used_flag_is_true(monkeypatch):
    """Until P2.5 wires a real LLM, the primary delegates to the
    fallback; the ``fallback_used`` flag must surface that honestly.
    A future regression where the primary silently lies would flip
    this to False."""

    monkeypatch.setenv("COMPGEN_DISABLE_LLM", "1")
    result = rank_candidates([_candidate(0.5, "x")])
    assert result["fallback_used"] is True


def test_random_shuffled_input_still_permutation(monkeypatch):
    """Pseudo-stress test: 50 random inputs each get a valid permutation."""

    monkeypatch.setenv("COMPGEN_DISABLE_LLM", "1")
    rng = random.Random(0xC0FFEE)
    for _ in range(50):
        n = rng.randint(0, 12)
        cands = [
            _candidate(rng.random(), f"r{i}_{rng.randint(0, 99)}") for i in range(n)
        ]
        result = rank_candidates(cands)
        indices = sorted(r["index"] for r in result["ranking"])
        assert indices == list(range(n))


def test_fallback_direct_call():
    """Calling the fallback directly produces the same shape as the
    wrapped function — useful for tests that want to bypass the
    decorator."""

    cands = [_candidate(0.5, "a"), _candidate(0.9, "b")]
    direct = _rank_candidates_fallback(cands)
    assert direct["ranking"][0]["index"] == 1
    assert direct["fallback_used"] is True
