"""Tests for the P2.2 pre-commit CostPreview module.

Coverage:

Positive:
* All-legal candidate set: the lowest static-cost entry has empty
  ``dominated_by`` and is the only survivor; higher-cost entries get
  the lower-cost ids as their dominators.
* A blocked candidate is never reported as dominating anyone (even if
  its static cost is the lowest); it never gets dominators of its own
  (the LLM must see its typed blocked state intact).
* ``delta_surrogate`` and ``confidence`` start ``None`` and become
  populated when the optional dicts are supplied.
* ``survivors`` filters to legal + non-dominated rows.
* Output is byte-deterministic across reruns.

Negative controls:
* Duplicate candidate_id raises CostPreviewError.
* Unknown legality value raises at CandidateInput construction.
* candidates not being a list raises.
* Empty candidate_id raises.
* Tied static costs do NOT silently prefer one over the other (no
  dominance between ties).
"""

from __future__ import annotations

import json

import pytest
from compgen.agent.cost_preview import (
    LEGALITY_VALUES,
    CandidateInput,
    CostPreviewError,
    compute_cost_previews,
    survivors,
)


def _cand(cid: str, cost: float, legality: str = "ok") -> CandidateInput:
    return CandidateInput(candidate_id=cid, delta_static=cost, legality=legality)


# ---------- Positive --------------------------------------------------


def test_lowest_cost_legal_is_sole_survivor():
    previews = compute_cost_previews(
        [_cand("a", 10.0), _cand("b", 5.0), _cand("c", 20.0)]
    )
    by_id = {p.candidate_id: p for p in previews}
    assert by_id["b"].dominated_by == ()
    assert by_id["a"].dominated_by == ("b",)
    assert by_id["c"].dominated_by == ("b", "a")
    surv = survivors(previews)
    assert [p.candidate_id for p in surv] == ["b"]


def test_blocked_candidate_never_dominates_or_is_dominated():
    """A blocked candidate with the lowest static cost must not silence
    other legal candidates; equally, it must not be silenced itself."""

    previews = compute_cost_previews(
        [
            _cand("blocked_low", 0.5, legality="blocked"),
            _cand("ok_a", 10.0),
            _cand("ok_b", 5.0),
        ]
    )
    by_id = {p.candidate_id: p for p in previews}
    # The blocked entry stays visible to the LLM.
    assert by_id["blocked_low"].dominated_by == ()
    assert by_id["blocked_low"].is_survivor is False  # blocked, not a survivor
    # The legal entries dominate each other correctly.
    assert by_id["ok_a"].dominated_by == ("ok_b",)
    # The legal entries are NOT dominated by the blocked one.
    assert "blocked_low" not in by_id["ok_a"].dominated_by


def test_surrogate_and_confidence_propagate():
    surrogate = {"a": 0.07, "b": 0.05}
    conf = {"a": 0.8, "b": 0.95}
    previews = compute_cost_previews(
        [_cand("a", 10.0), _cand("b", 5.0)],
        surrogate_deltas=surrogate,
        confidence_by_id=conf,
    )
    by_id = {p.candidate_id: p for p in previews}
    assert by_id["a"].delta_surrogate == 0.07
    assert by_id["b"].delta_surrogate == 0.05
    assert by_id["a"].confidence == 0.8


def test_surrogate_unset_yields_none():
    previews = compute_cost_previews([_cand("a", 1.0)])
    assert previews[0].delta_surrogate is None
    assert previews[0].confidence is None


def test_byte_deterministic():
    cands = [_cand("a", 10.0), _cand("b", 5.0), _cand("c", 20.0)]
    a = compute_cost_previews(cands)
    b = compute_cost_previews(cands)
    sa = json.dumps([p.to_dict() for p in a], sort_keys=True)
    sb = json.dumps([p.to_dict() for p in b], sort_keys=True)
    assert sa == sb


def test_legality_values_closed_enum():
    assert set(LEGALITY_VALUES) == {"ok", "blocked", "unknown"}


def test_tied_costs_no_dominance():
    """Tied static costs are an honest no-dominance — the LLM picks
    between them by other axes (rationale, target_feature)."""

    previews = compute_cost_previews([_cand("a", 5.0), _cand("b", 5.0)])
    by_id = {p.candidate_id: p for p in previews}
    assert by_id["a"].dominated_by == ()
    assert by_id["b"].dominated_by == ()
    assert {p.candidate_id for p in survivors(previews)} == {"a", "b"}


def test_unknown_legality_is_not_a_survivor():
    previews = compute_cost_previews(
        [_cand("u", 1.0, legality="unknown"), _cand("ok", 2.0)]
    )
    by_id = {p.candidate_id: p for p in previews}
    assert by_id["u"].is_survivor is False
    # The unknown entry is NOT a dominator of the legal one.
    assert "u" not in by_id["ok"].dominated_by


def test_empty_candidate_set():
    assert compute_cost_previews([]) == []
    assert survivors([]) == []


# ---------- Negative controls ----------------------------------------


def test_duplicate_candidate_id_rejected():
    with pytest.raises(CostPreviewError, match="unique"):
        compute_cost_previews([_cand("a", 1.0), _cand("a", 2.0)])


def test_unknown_legality_rejected():
    with pytest.raises(CostPreviewError, match="legality"):
        CandidateInput(candidate_id="a", delta_static=1.0, legality="rejected")


def test_empty_candidate_id_rejected():
    with pytest.raises(CostPreviewError, match="non-empty"):
        CandidateInput(candidate_id="", delta_static=1.0)


def test_candidates_not_list_rejected():
    with pytest.raises(CostPreviewError, match="must be a list"):
        compute_cost_previews({"a": 1.0})  # type: ignore[arg-type]
