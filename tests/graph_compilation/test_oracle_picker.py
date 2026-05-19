"""Tests for the calibrated oracle picker in `_select_greedy`.

When a calibration dict is provided (via the
`XPU_RT_CALIBRATION_DIR` env var at the caller, or directly as the
`calibration` kwarg), `_select_greedy` overrides its static priority
order with the empirical-min-latency pick. This is the closure path
for the `cost_model_uncalibrated_across_decisions` caveat.

These tests use synthetic decision_sites + candidate_actions inputs
so they exercise the selector logic directly without running the
full pipeline.
"""

from __future__ import annotations

from typing import Any

from xpu_rt.graph_compilation.recipe_planning import _select_greedy


def _site(site_id: str, priority: int, region_id: str, kind: str,
          candidate_ids: list[str]) -> dict[str, Any]:
    return {
        "site_id": site_id,
        "priority": priority,
        "region_id": region_id,
        "kind": kind,
        "candidate_ids": candidate_ids,
    }


def _cand(cid: str, region_id: str, kind: str,
          static_cost: float, legal: bool = True,
          boundary: bool = False) -> dict[str, Any]:
    return {
        "candidate_id": cid,
        "region_id": region_id,
        "kind": kind,
        "legality": {"ok": legal, "reason": "" if legal else "not legal"},
        "cost_preview": {
            "static_relative_cost": static_cost,
            "boundary_required": boundary,
        },
    }


# --------------------------------------------------------------------------- #
# No calibration: behaves like the existing greedy.
# --------------------------------------------------------------------------- #


def test_no_calibration_picks_static_priority() -> None:
    sites = {"sites": [
        _site("site_tile_matmul_0", 1, "matmul_0", "tiling",
              ["cand_a", "cand_b"]),
    ]}
    cands = {"candidates": [
        _cand("cand_a", "matmul_0", "set_tile_params", static_cost=0.8),
        _cand("cand_b", "matmul_0", "set_tile_params", static_cost=1.0),
    ]}
    sel, _trace, primary = _select_greedy(sites, cands, set(), None)
    assert sel is not None
    # cand_a has lower static_cost; greedy picks it.
    assert sel["candidate_id"] == "cand_a"
    assert "calibrated" not in primary.lower()


# --------------------------------------------------------------------------- #
# Calibration override: empirical min wins regardless of static cost.
# --------------------------------------------------------------------------- #


def test_calibration_overrides_static_priority() -> None:
    sites = {"sites": [
        _site("site_tile_matmul_0", 1, "matmul_0", "tiling",
              ["cand_a", "cand_b"]),
    ]}
    cands = {"candidates": [
        _cand("cand_a", "matmul_0", "set_tile_params", static_cost=0.8),
        _cand("cand_b", "matmul_0", "set_tile_params", static_cost=1.0),
    ]}
    # cand_b is empirically faster despite higher static cost.
    calibration = {"cand_a": 150.0, "cand_b": 100.0}
    sel, trace, primary = _select_greedy(
        sites, cands, set(), calibration,
    )
    assert sel is not None
    assert sel["candidate_id"] == "cand_b"
    assert "calibrated_pick_override" in primary
    # The trace records the selected candidate with the override reason.
    chosen_trace = [t for t in trace if t.decision == "selected"]
    assert len(chosen_trace) == 1
    assert chosen_trace[0].candidate_id == "cand_b"
    assert "calibrated_pick_override" in chosen_trace[0].reason


# --------------------------------------------------------------------------- #
# Calibration partial: only some candidates have empirical data.
# --------------------------------------------------------------------------- #


def test_calibration_uses_only_covered_candidates() -> None:
    """When the calibration covers a subset of legal candidates,
    pick the empirical min among the covered. Candidates not in the
    calibration are skipped — they're untrusted by definition."""
    sites = {"sites": [
        _site("site", 1, "r", "tiling", ["cand_a", "cand_b", "cand_c"]),
    ]}
    cands = {"candidates": [
        _cand("cand_a", "r", "set_tile_params", static_cost=0.5),
        _cand("cand_b", "r", "set_tile_params", static_cost=0.8),
        _cand("cand_c", "r", "set_tile_params", static_cost=1.0),
    ]}
    # Only cand_b and cand_c have calibration data.
    calibration = {"cand_b": 90.0, "cand_c": 80.0}
    sel, _trace, primary = _select_greedy(
        sites, cands, set(), calibration,
    )
    assert sel is not None
    # cand_c is the empirical min among the calibrated subset.
    assert sel["candidate_id"] == "cand_c"
    assert "calibrated_pick_override" in primary


def test_calibration_falls_back_when_no_covered_candidate_is_legal() -> None:
    """When none of the legal candidates in a site appear in the
    calibration, fall back to the static greedy tier-sort."""
    sites = {"sites": [
        _site("site", 1, "r", "tiling", ["cand_a", "cand_b"]),
    ]}
    cands = {"candidates": [
        _cand("cand_a", "r", "set_tile_params", static_cost=0.5),
        _cand("cand_b", "r", "set_tile_params", static_cost=1.0),
    ]}
    # Calibration is for a completely different candidate.
    calibration = {"cand_unrelated": 50.0}
    sel, _trace, primary = _select_greedy(
        sites, cands, set(), calibration,
    )
    assert sel is not None
    # Falls back to static: cand_a has lower static_cost.
    assert sel["candidate_id"] == "cand_a"
    assert "calibrated_pick_override" not in primary


# --------------------------------------------------------------------------- #
# Cross-site behavior: oracle is applied per-site.
# --------------------------------------------------------------------------- #


def test_calibration_picks_globally_across_sites() -> None:
    """The oracle override is **cross-site**: it considers all legal
    candidates across all sites and picks the globally-min empirical
    latency. This bypasses greedy's "first legal site wins" priority
    order — which the calibration audit revealed to be the dominant
    source of regret on merlin_mlp_wide (priority-1 matmul_0 commits
    before priority-2 matmul_1 even though matmul_1 is empirically
    faster)."""
    sites = {"sites": [
        _site("site_p1", 1, "r1", "tiling", ["cand_a", "cand_b"]),
        _site("site_p2", 2, "r2", "fusion", ["cand_c", "cand_d"]),
    ]}
    cands = {"candidates": [
        _cand("cand_a", "r1", "set_tile_params", static_cost=0.8),
        _cand("cand_b", "r1", "set_tile_params", static_cost=1.0),
        _cand("cand_c", "r2", "fuse_producer_consumer", static_cost=0.5),
        _cand("cand_d", "r2", "fuse_producer_consumer", static_cost=0.6),
    ]}
    calibration = {
        "cand_a": 100.0, "cand_b": 50.0,
        "cand_c": 200.0, "cand_d": 30.0,  # globally smallest
    }
    sel, _, primary = _select_greedy(sites, cands, set(), calibration)
    assert sel is not None
    # Cross-site: cand_d (priority-2 site) wins because its empirical
    # latency is the lowest across all calibrated candidates.
    assert sel["candidate_id"] == "cand_d"
    assert sel["region_id"] == "r2"
    assert "calibrated_pick_override (cross-site)" in primary
