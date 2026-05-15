"""Tests for the P2.1 token-economical state views.

Covers:

canonical_view:
* Pure: same input → same output (byte-deterministic).
* Bounded: raises CanonicalViewBudgetError when the summary would
  exceed the byte cap.
* Unplanned session yields ``global_objective='unplanned'``.

focus_chunk:
* Returns the full per-region dossier on a known region id.
* Raises FocusChunkNotFoundError on an unknown id (no silent fallback).

diff_since:
* Detects added/removed/changed region entries.
* Plan version bump surfaces a typed entry.
* Empty diff when ``before == after``.
* Unknown diff kind rejected at construction.
"""

from __future__ import annotations

import json

import pytest
from compgen.agent.views import (
    CANONICAL_VIEW_BYTE_BUDGET,
    CanonicalViewBudgetError,
    DiffEntry,
    canonical_view,
    diff_since,
    focus_chunk,
)
from compgen.agent.views.focus_chunk import FocusChunkNotFoundError


def _state(**overrides):
    state = {
        "session_id": "ses_1",
        "plan": {
            "plan_version": 1,
            "global_objective": "minimize_p50_latency",
            "region_partition": [
                {"region_id": "017", "tactic": "fuse", "fallback_ladder": ["fuse", "tile_only"]},
                {"region_id": "023", "tactic": "megakernel", "fallback_ladder": ["megakernel", "fused_chain"]},
            ],
        },
        "regions": [
            {
                "region_id": "017",
                "open_decision_sites": ["site_a", "site_b"],
                "last_verdict": "rejected",
                "last_reason": "scratchpad_overflow",
                "ops": ["matmul", "softmax"],
                "candidate_set": [{"id": "tile_128"}, {"id": "tile_64"}],
                "contract_envelope": {"dtype": "fp16"},
            },
            {
                "region_id": "023",
                "open_decision_sites": [],
                "last_verdict": None,
                "last_reason": None,
                "ops": ["fused_attention"],
                "candidate_set": [],
                "contract_envelope": {"dtype": "fp16"},
            },
        ],
    }
    state.update(overrides)
    return state


# ---------- canonical_view -------------------------------------------


def test_canonical_view_deterministic():
    a = canonical_view(_state())
    b = canonical_view(_state())
    assert a.to_dict() == b.to_dict()


def test_canonical_view_planned_session():
    v = canonical_view(_state())
    assert v.global_objective == "minimize_p50_latency"
    assert v.plan_version == 1
    by_region = {r.region_id: r for r in v.rows}
    assert by_region["017"].current_tactic == "fuse"
    assert by_region["017"].last_verdict == "rejected"
    assert by_region["017"].last_reason == "scratchpad_overflow"


def test_canonical_view_unplanned_session():
    v = canonical_view({"session_id": "ses_unplanned", "regions": [
        {"region_id": "r", "open_decision_sites": [], "last_verdict": None, "last_reason": None}
    ]})
    assert v.global_objective == "unplanned"
    assert v.plan_version == 0


def test_canonical_view_byte_budget_enforced():
    """A massive region list must trigger the budget guard."""

    huge = _state()
    huge["regions"] = [
        {"region_id": f"reg_{i:04d}", "open_decision_sites": ["s"], "last_verdict": "rejected", "last_reason": "x" * 200}
        for i in range(200)
    ]
    huge["plan"]["region_partition"] = [
        {"region_id": f"reg_{i:04d}", "tactic": "t"} for i in range(200)
    ]
    with pytest.raises(CanonicalViewBudgetError):
        canonical_view(huge)


def test_canonical_view_byte_size_matches_serialised_length():
    v = canonical_view(_state())
    serialised = json.dumps(v.to_dict(), sort_keys=True, separators=(",", ":"))
    assert v.byte_size == len(serialised.encode("utf-8"))
    # to_dict_with_metadata re-exposes the cached byte_size for
    # callers that want both the canonical body and its size.
    meta = v.to_dict_with_metadata()
    assert meta["byte_size"] == v.byte_size


def test_canonical_view_default_budget_constant():
    assert CANONICAL_VIEW_BYTE_BUDGET >= 1024


# ---------- focus_chunk ----------------------------------------------


def test_focus_chunk_known_region():
    chunk = focus_chunk(_state(), region_id="017")
    assert chunk.region_id == "017"
    assert chunk.ops == ("matmul", "softmax")
    assert len(chunk.candidate_set) == 2
    assert chunk.contract_envelope["dtype"] == "fp16"


def test_focus_chunk_unknown_region_raises():
    with pytest.raises(FocusChunkNotFoundError, match="region_id"):
        focus_chunk(_state(), region_id="zzz")


def test_focus_chunk_serialises():
    chunk = focus_chunk(_state(), region_id="017")
    body = chunk.to_dict()
    assert body["region_id"] == "017"
    assert isinstance(body["candidate_set"], list)


# ---------- diff_since -----------------------------------------------


def test_diff_since_empty_when_unchanged():
    s = _state()
    assert diff_since(s, s) == []


def test_diff_since_detects_plan_version_bump():
    before = _state()
    after = _state()
    after["plan"]["plan_version"] = 2
    diff = diff_since(before, after)
    pointers = [e.pointer for e in diff]
    assert "plan.plan_version" in pointers
    pv_entry = next(e for e in diff if e.pointer == "plan.plan_version")
    assert pv_entry.before == 1
    assert pv_entry.after == 2


def test_diff_since_detects_region_added():
    before = _state()
    after = _state()
    after["regions"].append(
        {"region_id": "099", "open_decision_sites": [], "last_verdict": None, "last_reason": None}
    )
    diff = diff_since(before, after)
    pointers = [(e.pointer, e.kind) for e in diff]
    assert ("region.099", "added") in pointers


def test_diff_since_detects_region_removed():
    before = _state()
    after = _state()
    after["regions"] = [r for r in after["regions"] if r["region_id"] != "023"]
    diff = diff_since(before, after)
    pointers = [(e.pointer, e.kind) for e in diff]
    assert ("region.023", "removed") in pointers


def test_diff_since_detects_field_change():
    before = _state()
    after = _state()
    for r in after["regions"]:
        if r["region_id"] == "017":
            r["last_verdict"] = "accepted"
    diff = diff_since(before, after)
    pointers = [(e.pointer, e.kind) for e in diff]
    assert ("region.017.last_verdict", "changed") in pointers


def test_diff_entry_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown diff kind"):
        DiffEntry(pointer="x", kind="bogus", before=None, after=None)
