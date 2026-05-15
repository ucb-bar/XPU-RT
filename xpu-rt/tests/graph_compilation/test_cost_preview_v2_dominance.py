"""Tests for the P2.2 dominance-prune wire-in (G1).

Coverage:

* ``_annotate_dominance`` adds ``dominated_by`` + ``is_survivor`` to
  every input dict, sourced from the P2.2 module's pure dominance
  rule (lowest-cost legal candidate survives; ties never dominate).
* Blocked candidates are never marked as a dominator OR as dominated —
  the LLM must see their typed state intact.
* ``_emit_dominance_report`` writes a typed sidecar with the
  expected schema_version + per-row dominance facts.
* The wire-in is *additive*: the existing cost_preview fields
  (relative_cost, legality_ok, features, candidate_id) are preserved
  byte-for-byte.
* Duplicate candidate_id and missing candidate_id are tolerated
  without raising (the audit upstream catches malformed inputs).
"""

from __future__ import annotations

import json

from compgen.graph_compilation.cost_preview_v2 import (
    _annotate_dominance,
    _emit_dominance_report,
)


def _cp(cid: str, cost: float, ok: bool = True, **extra) -> dict:
    """Minimal cost_preview dict shape mirroring the production schema."""

    body = {
        "candidate_id": cid,
        "relative_cost": cost,
        "legality_ok": ok,
        "features": {},
    }
    body.update(extra)
    return body


# ---------- _annotate_dominance --------------------------------------


def test_lowest_cost_legal_is_sole_survivor():
    cps = [
        _cp("a", 1.5),
        _cp("b", 0.8),
        _cp("c", 2.0),
    ]
    _annotate_dominance(cps)
    by_id = {cp["candidate_id"]: cp for cp in cps}
    assert by_id["b"]["is_survivor"] is True
    assert by_id["b"]["dominated_by"] == []
    assert by_id["a"]["is_survivor"] is False
    assert by_id["a"]["dominated_by"] == ["b"]
    assert by_id["c"]["dominated_by"] == ["b", "a"]


def test_blocked_candidate_never_dominates_or_is_dominated():
    """A blocked candidate with the lowest cost cannot silence the
    legal candidates — they must remain visible to the LLM."""

    cps = [
        _cp("blocked_low", 0.4, ok=False),
        _cp("legal_high", 2.0, ok=True),
        _cp("legal_best", 1.0, ok=True),
    ]
    _annotate_dominance(cps)
    by_id = {cp["candidate_id"]: cp for cp in cps}
    assert by_id["blocked_low"]["dominated_by"] == []
    assert by_id["blocked_low"]["is_survivor"] is False
    assert by_id["legal_best"]["is_survivor"] is True
    assert "blocked_low" not in by_id["legal_high"]["dominated_by"]


def test_tied_costs_do_not_dominate_each_other():
    cps = [_cp("a", 1.0), _cp("b", 1.0)]
    _annotate_dominance(cps)
    by_id = {cp["candidate_id"]: cp for cp in cps}
    assert by_id["a"]["dominated_by"] == []
    assert by_id["b"]["dominated_by"] == []
    assert by_id["a"]["is_survivor"] is True
    assert by_id["b"]["is_survivor"] is True


def test_existing_fields_preserved_byte_for_byte():
    """The wire-in is purely additive: original cost_preview fields
    must round-trip unchanged."""

    cps = [_cp("a", 1.5, ok=True, baseline_static_latency_us=100.0, confidence=0.9)]
    _annotate_dominance(cps)
    assert cps[0]["candidate_id"] == "a"
    assert cps[0]["relative_cost"] == 1.5
    assert cps[0]["legality_ok"] is True
    assert cps[0]["baseline_static_latency_us"] == 100.0
    assert cps[0]["confidence"] == 0.9
    assert "dominated_by" in cps[0]
    assert "is_survivor" in cps[0]


def test_empty_input_no_crash():
    cps: list[dict] = []
    _annotate_dominance(cps)
    assert cps == []


def test_missing_candidate_id_tolerated():
    """A malformed entry without candidate_id must not crash the
    annotator — the audit upstream catches schema violations."""

    cps = [{"relative_cost": 1.0, "legality_ok": True, "features": {}}]
    _annotate_dominance(cps)
    assert cps[0]["dominated_by"] == []
    assert cps[0]["is_survivor"] is False


def test_duplicate_candidate_id_tolerated():
    """Duplicate ids inside one cost_preview list must not raise; only
    the first occurrence participates in the dominance computation."""

    cps = [_cp("a", 1.5), _cp("a", 0.5), _cp("b", 1.0)]
    _annotate_dominance(cps)
    # The second 'a' is skipped by the annotator (defensive); the
    # production validator catches the duplicate upstream.
    by_id_first = cps[0]  # first 'a'
    assert "dominated_by" in by_id_first
    assert "is_survivor" in by_id_first


# ---------- _emit_dominance_report -----------------------------------


def test_emit_dominance_report_writes_typed_sidecar(tmp_path):
    cps = [
        _cp("a", 1.5),
        _cp("b", 0.8),
        _cp("c", 2.0),
        _cp("d", 0.4, ok=False),
    ]
    _annotate_dominance(cps)
    out_path = _emit_dominance_report(tmp_path, cps)
    assert out_path.is_file()
    body = json.loads(out_path.read_text(encoding="utf-8"))
    assert body["schema_version"] == "cost_preview_dominance_report_v1"
    assert body["summary"]["survivors"] == 1   # b
    assert body["summary"]["dominated"] == 2   # a, c
    assert body["summary"]["blocked"] == 1     # d
    assert body["summary"]["total"] == 4
    rows_by_id = {r["candidate_id"]: r for r in body["rows"]}
    assert rows_by_id["b"]["is_survivor"] is True
    assert rows_by_id["a"]["dominated_by"] == ["b"]
    assert rows_by_id["d"]["legality_ok"] is False


def test_emit_creates_parent_dir(tmp_path):
    """The sidecar's parent dir is auto-created if absent."""

    out = _emit_dominance_report(tmp_path / "nested" / "ga", [_cp("a", 1.0)])
    assert out.is_file()
    assert out.parent.name == "ga"


# ---------- Integration with run_cost_preview_v2 ---------------------


def test_run_cost_preview_v2_writes_dominance_report(tmp_path):
    """End-to-end: drive run_cost_preview_v2 on a synthetic fixture
    and assert the new dominance_report.json is emitted next to
    cost_preview_v2.json."""

    from compgen.graph_compilation.cost_preview_v2 import run_cost_preview_v2

    # Build a minimal run_dir with the required inputs.
    run_dir = tmp_path / "run"
    ga = run_dir / "02_graph_analysis"
    ga.mkdir(parents=True)
    (ga / "region_map.json").write_text(
        json.dumps({"model_id": "test_model", "regions": []}), encoding="utf-8"
    )
    (ga / "candidate_actions.json").write_text(
        json.dumps({"model_id": "test_model", "candidates": []}),
        encoding="utf-8",
    )
    (ga / "graph_dossier_v2.json").write_text(
        json.dumps({"region_dossiers": {}}), encoding="utf-8"
    )
    (ga / "graph_dossier_v3.json").write_text(
        json.dumps({"regions": [], "source": {}}), encoding="utf-8"
    )
    (ga / "llm_graph_view.json").write_text(
        json.dumps({"regions": []}), encoding="utf-8"
    )

    run_cost_preview_v2(run_dir)

    assert (ga / "cost_preview_v2.json").is_file()
    assert (ga / "dominance_report.json").is_file()
    body = json.loads((ga / "dominance_report.json").read_text(encoding="utf-8"))
    assert body["schema_version"] == "cost_preview_dominance_report_v1"
    assert body["summary"]["total"] == 0  # empty fixture
