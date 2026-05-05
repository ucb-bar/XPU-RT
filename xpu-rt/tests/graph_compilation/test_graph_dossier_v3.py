"""Tests for M-10B Graph Dossier V3 unified agent view.

Cross-checks the four emitted artifacts against the existing
graph-analysis and recipe-planning artifacts. Read-only against the
suite results directories produced by:

- ``results/graph_compilation/dossier_v3_suite/`` — full pipeline,
  ``--stop-after differential-verification``.
- ``results/graph_compilation/dossier_v3_postlowering_suite/`` — partial
  pipeline, ``--stop-after post-lowering-verification`` (used for
  V3R007 M-08-fallback test).

Negative tests use ``tmp_path`` to mutate a copy of a real run dir and
re-call the builder.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from compgen.graph_compilation.graph_dossier_v3 import (
    _validate_v3,
    build_graph_dossier_v3,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_M09 = REPO_ROOT / "results" / "graph_compilation" / "dossier_v3_suite"
SUITE_M08 = REPO_ROOT / "results" / "graph_compilation" / "dossier_v3_postlowering_suite"

_CANONICAL = (
    "tiny_mlp", "tiny_attention", "tiny_conv_block",
    "proxy_vlm", "proxy_vla", "custom_unsupported_op",
)
# These five select a real candidate; custom_unsupported_op selects a
# kernel-contract recipe (still selected_candidate_count==1 but no real
# transformed payload).
_TRANSFORM_LIKE = (
    "tiny_mlp", "tiny_attention", "tiny_conv_block", "proxy_vlm", "proxy_vla",
)


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _need_suites() -> None:
    for s in (SUITE_M09, SUITE_M08):
        if not s.is_dir():
            pytest.skip(
                f"fixture suite missing: {s}; "
                f"run `compgen.graph_compilation run-suite` first"
            )


# --------------------------------------------------------------------------- #
# Positive tests over the canonical 6-model M-09 suite
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", _CANONICAL)
def test_all_six_canonical_models_emit_four_artifacts(model: str) -> None:
    _need_suites()
    ga = SUITE_M09 / model / "02_graph_analysis"
    for name in (
        "graph_dossier_v3.json",
        "graph_dossier_v3.mlir",
        "graph_dossier_v3_validation.json",
        "llm_graph_view.json",
    ):
        assert (ga / name).exists(), f"{model}: missing {name}"


@pytest.mark.parametrize("model", _CANONICAL)
def test_validation_overall_pass_for_all_six(model: str) -> None:
    _need_suites()
    v = _read(SUITE_M09 / model / "02_graph_analysis" / "graph_dossier_v3_validation.json")
    assert v["overall"] == "pass", (
        f"{model}: validation failed; checks={v['checks']}"
    )


@pytest.mark.parametrize("model", _CANONICAL)
def test_llm_graph_view_contains_zero_illegal_candidates(model: str) -> None:
    _need_suites()
    cas = _read(SUITE_M09 / model / "02_graph_analysis" / "candidate_actions.json")
    legality = {
        c["candidate_id"]: (c.get("legality") or {}).get("ok", False)
        for c in cas["candidates"]
    }
    view = _read(SUITE_M09 / model / "02_graph_analysis" / "llm_graph_view.json")
    for r in view["regions"]:
        for c in r["legal_candidates"]:
            cid = c["candidate_id"]
            assert legality.get(cid, False), (
                f"{model}: candidate {cid} appears in llm_graph_view but is "
                f"illegal in candidate_actions.json"
            )


@pytest.mark.parametrize("model", _TRANSFORM_LIKE)
def test_selected_candidate_traces_to_recipe_op_and_obligation_status(
    model: str,
) -> None:
    _need_suites()
    d = _read(SUITE_M09 / model / "02_graph_analysis" / "graph_dossier_v3.json")
    sel_regions = [r for r in d["regions"] if r.get("selected") is not None]
    assert len(sel_regions) >= 1, f"{model}: no selected candidate in v3"
    sel = sel_regions[0]["selected"]
    assert sel["candidate_id"], f"{model}: empty selected candidate_id"
    assert sel["recipe_op_id"], f"{model}: empty recipe_op_id"
    assert sel["obligation_id"], f"{model}: empty obligation_id"
    ostat = sel["obligation_status"]
    assert ostat is not None, f"{model}: missing obligation_status"
    assert ostat["status"], f"{model}: empty obligation_status.status"


def test_obligation_status_source_is_m09_when_m09_ran() -> None:
    _need_suites()
    d = _read(SUITE_M09 / "tiny_mlp" / "02_graph_analysis" / "graph_dossier_v3.json")
    assert d["summary"]["obligation_status_source_stage"] == "differential_verification"
    sel = next(r["selected"] for r in d["regions"] if r["selected"] is not None)
    assert sel["obligation_status"]["source_stage"] == "differential_verification"
    # M-09 status carries `discharged`; M-08 does not.
    assert "discharged" in sel["obligation_status"]


def test_obligation_status_source_is_m08_when_m09_absent() -> None:
    _need_suites()
    d = _read(SUITE_M08 / "tiny_mlp" / "02_graph_analysis" / "graph_dossier_v3.json")
    assert d["summary"]["obligation_status_source_stage"] == "post_lowering"
    sel = next(r["selected"] for r in d["regions"] if r["selected"] is not None)
    assert sel["obligation_status"]["source_stage"] == "post_lowering"
    # M-08 v1 schema does NOT have a discharged list; v3 must not invent one.
    assert "discharged" not in sel["obligation_status"]


@pytest.mark.parametrize("model", _CANONICAL)
def test_bounded_llm_graph_view_sizes(model: str) -> None:
    _need_suites()
    view = _read(SUITE_M09 / model / "02_graph_analysis" / "llm_graph_view.json")
    cap_r = view["budget"]["max_visible_regions"]
    cap_c = view["budget"]["max_candidates_per_region"]
    assert len(view["regions"]) <= cap_r
    for r in view["regions"]:
        assert len(r["legal_candidates"]) <= cap_c
    d = _read(SUITE_M09 / model / "02_graph_analysis" / "graph_dossier_v3.json")
    expected_truncated = len(d["regions"]) > cap_r
    assert bool(view["budget"]["truncated"]) == expected_truncated


def test_v3_idempotent_rerun_byte_identical(tmp_path: Path) -> None:
    """Two consecutive builder calls on the same run dir produce the
    same artifact bytes (modulo the meta.generated_at_utc block, which
    is excluded by computing SHA over the body only)."""
    _need_suites()
    src = SUITE_M09 / "tiny_mlp"
    dst = tmp_path / "tiny_mlp"
    shutil.copytree(src, dst)

    def _hash_body(path: Path) -> str:
        body = json.loads(path.read_text(encoding="utf-8"))
        body.pop("meta", None)
        import hashlib
        return hashlib.sha256(
            json.dumps(body, sort_keys=True).encode("utf-8")
        ).hexdigest()

    r1 = build_graph_dossier_v3(dst)
    j1 = _hash_body(r1.json_path)
    v1 = _hash_body(r1.validation_path)
    l1 = _hash_body(r1.llm_view_path)
    m1 = r1.mlir_path.read_bytes()

    r2 = build_graph_dossier_v3(dst)
    j2 = _hash_body(r2.json_path)
    v2 = _hash_body(r2.validation_path)
    l2 = _hash_body(r2.llm_view_path)
    m2 = r2.mlir_path.read_bytes()

    assert j1 == j2
    assert v1 == v2
    assert l1 == l2
    assert m1 == m2


# --------------------------------------------------------------------------- #
# Negative tests (mutate a copy + re-invoke validator directly)
# --------------------------------------------------------------------------- #


def _load_inputs_for(run_dir: Path) -> dict:
    ga = run_dir / "02_graph_analysis"
    rp = run_dir / "03_recipe_planning"
    return {
        "region_map": _read(ga / "region_map.json"),
        "decision_sites": _read(ga / "decision_sites.json"),
        "candidate_actions": _read(ga / "candidate_actions.json"),
        "candidate_selection": _read(rp / "candidate_selection.json"),
        "semantic_obligations": _read(rp / "semantic_obligations.json"),
        "dossier": _read(ga / "graph_dossier_v3.json"),
        "llm_view": _read(ga / "llm_graph_view.json"),
    }


def test_dangling_region_reference_fails_validation(tmp_path: Path) -> None:
    _need_suites()
    src = SUITE_M09 / "tiny_mlp"
    inputs = _load_inputs_for(src)
    inputs["dossier"]["regions"].append(
        {
            "region_id": "fabricated_region_id",
            "kind": "matmul",
            "module_id": "fake",
            "source_classification": "decomposed_structured",
            "fx_nodes": [],
            "estimated": {},
            "facts": {"dossier_ref": None, "cost": None,
                      "numerical_sensitivity_summary": None,
                      "working_set_summary": None,
                      "legality_constraints": []},
            "decision_sites": [],
            "legal_candidates": [],
            "illegal_candidates": [],
            "selected": None,
        }
    )
    v = _validate_v3(
        dossier=inputs["dossier"], llm_view=inputs["llm_view"],
        region_map=inputs["region_map"],
        decision_sites=inputs["decision_sites"],
        candidate_actions=inputs["candidate_actions"],
        candidate_selection=inputs["candidate_selection"],
        semantic_obligations=inputs["semantic_obligations"],
    )
    assert v["overall"] == "fail"
    v3r001 = next(c for c in v["checks"] if c["id"] == "V3R001")
    assert v3r001["status"] == "fail"
    assert any("fabricated_region_id" in d for d in v3r001["details"])


def test_candidate_not_in_candidate_actions_fails_validation(tmp_path: Path) -> None:
    _need_suites()
    src = SUITE_M09 / "tiny_mlp"
    inputs = _load_inputs_for(src)
    sel_region = next(
        r for r in inputs["dossier"]["regions"] if r.get("selected") is not None
    )
    sel_region["legal_candidates"].append(
        {
            "candidate_id": "cand_fabricated_xyz",
            "site_id": sel_region["selected"]["site_id"],
            "kind": "set_tile_params",
            "label": "fake",
            "cost_preview": {},
            "recipe_delta": [],
            "legality_reason": "",
        }
    )
    v = _validate_v3(
        dossier=inputs["dossier"], llm_view=inputs["llm_view"],
        region_map=inputs["region_map"],
        decision_sites=inputs["decision_sites"],
        candidate_actions=inputs["candidate_actions"],
        candidate_selection=inputs["candidate_selection"],
        semantic_obligations=inputs["semantic_obligations"],
    )
    assert v["overall"] == "fail"
    v3r003 = next(c for c in v["checks"] if c["id"] == "V3R003")
    assert v3r003["status"] == "fail"
    assert any("cand_fabricated_xyz" in d for d in v3r003["details"])


def test_selected_illegal_candidate_fails_validation(tmp_path: Path) -> None:
    _need_suites()
    src = SUITE_M09 / "tiny_mlp"
    inputs = _load_inputs_for(src)
    # Find a real illegal candidate from candidate_actions and point selection at it.
    illegal_cand = next(
        c for c in inputs["candidate_actions"]["candidates"]
        if (c.get("legality") or {}).get("ok") is False
    )
    sel_region = next(
        r for r in inputs["dossier"]["regions"]
        if r.get("selected") is not None
    )
    sel_region["selected"]["candidate_id"] = illegal_cand["candidate_id"]
    # candidate_selection.selected_candidate_id stays the original; this
    # produces a V3R005 disagreement which is the right failure mode.
    v = _validate_v3(
        dossier=inputs["dossier"], llm_view=inputs["llm_view"],
        region_map=inputs["region_map"],
        decision_sites=inputs["decision_sites"],
        candidate_actions=inputs["candidate_actions"],
        candidate_selection=inputs["candidate_selection"],
        semantic_obligations=inputs["semantic_obligations"],
    )
    assert v["overall"] == "fail"
    v3r005 = next(c for c in v["checks"] if c["id"] == "V3R005")
    assert v3r005["status"] == "fail"


# --------------------------------------------------------------------------- #
# Module-isolation guard
# --------------------------------------------------------------------------- #


def test_no_compiler_core_imports_in_v3_module() -> None:
    """v3 is read-only aggregation. Asserts the module does not import
    from compiler-core packages that are explicitly out of scope per
    M-10B's hard non-goals."""
    src = (
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "graph_dossier_v3.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "from compgen.ir",
        "import compgen.ir",
        "from compgen.capture",
        "import compgen.capture",
        "from compgen.pipeline",
        "import compgen.pipeline",
        "from runtime.bundle_emit",
    )
    for pat in forbidden:
        assert pat not in src, f"v3 module must not import: {pat}"
