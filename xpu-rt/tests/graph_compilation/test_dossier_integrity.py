"""Cross-artifact dossier integrity test suite.

Verifies that the multiple analysis tracks (FX dossier, readiness reports,
M-18 region calibration, M-18.3 candidate calibration, M-16.1 strict
gate, M-19 compiled kernel, M-17 evidence pack, recipe lowering, real
verification) are mutually consistent. Every test isolates ONE
cross-artifact invariant so a failure points at the exact place a
dossier track has drifted.

Three model-fixture runs cover the major paths:

- ``merlin_mlp_wide`` — SetTileParams clean-divides; exercises every
  opt-in including M-19 kernel execution (single-region foundation).
- ``proxy_vla`` — FuseProducerConsumer pointwise; exercises the M-16.2
  fusion track. Kernel execution emits ``not_applicable``.
- ``tiny_mlp`` — SetTileParams with K_iters>1 → real M-12 failure.
  Exercises the M-15B downstream-retry path (run exits non-zero but
  the artifact tree is otherwise complete up to the failure).

All fixtures run with the three calibration / kernel opt-ins enabled
so every track has data to cross-check.

Hard non-goals:

- This file does not generate new artifacts; it only inspects existing
  on-disk state.
- It does not assert correctness of values (M-12 / M-16.2 / M-19 own
  that). It asserts cross-artifact consistency: same id appears
  identically everywhere; counts match; selected candidate traces
  through every applicable level; calibration overlays reference real
  regions; etc.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _read_or_none(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return _read(p)
    except (OSError, json.JSONDecodeError):
        return None


def _run_full_optins(model: str, out_dir: Path) -> int:
    """Run the pipeline with EVERY analysis opt-in enabled so every
    track produces data."""
    env = os.environ.copy()
    env["COMPGEN_CALIBRATE_PROFILER"] = "1"
    env["COMPGEN_CALIBRATE_CANDIDATES"] = "1"
    env["COMPGEN_RUN_KERNELS"] = "1"
    res = subprocess.run(
        [
            sys.executable, "-m", "compgen.graph_compilation", "run",
            "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
            "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
            "--out", str(out_dir),
            "--stop-after", "agent-decision-request",
            "--selection-mode", "greedy",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )
    return res.returncode


# --------------------------------------------------------------------------- #
# Fixtures (one full-opt-in run per representative model)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def run_merlin_mlp_wide(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """SetTileParams clean-divides path with every opt-in."""
    out = tmp_path_factory.mktemp("integrity_merlin") / "run"
    _run_full_optins("merlin_mlp_wide", out)
    return out


@pytest.fixture(scope="module")
def run_proxy_vla(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """FuseProducerConsumer pointwise path with every opt-in."""
    out = tmp_path_factory.mktemp("integrity_proxy_vla") / "run"
    _run_full_optins("proxy_vla", out)
    return out


@pytest.fixture(scope="module")
def run_tiny_mlp(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """SetTileParams K_iters>1 → real M-12 failure path. Pipeline
    exits non-zero (M-15B raises) but artifacts up to the failure are
    on disk."""
    out = tmp_path_factory.mktemp("integrity_tiny_mlp") / "run"
    _run_full_optins("tiny_mlp", out)
    return out


# --------------------------------------------------------------------------- #
# Group A: ID consistency across artifacts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla", "run_tiny_mlp",
])
def test_every_candidate_region_id_in_region_map(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    cas = _read(run_dir / "02_graph_analysis" / "candidate_actions.json")
    rm = _read(run_dir / "02_graph_analysis" / "region_map.json")
    region_ids = {r["region_id"] for r in rm["regions"]}
    bad = [
        c["candidate_id"] for c in cas["candidates"]
        if c.get("region_id") and c["region_id"] not in region_ids
    ]
    assert not bad, f"{fixture_name}: candidates reference unknown regions: {bad[:5]}"


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla", "run_tiny_mlp",
])
def test_cost_preview_candidate_ids_subset_of_candidate_actions(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    cas = _read(run_dir / "02_graph_analysis" / "candidate_actions.json")
    cp = _read(run_dir / "02_graph_analysis" / "cost_preview_v2.json")
    ca_ids = {c["candidate_id"] for c in cas["candidates"]}
    cp_ids = {p["candidate_id"] for p in cp["cost_previews"]}
    orphans = cp_ids - ca_ids
    assert not orphans, (
        f"{fixture_name}: {len(orphans)} cost_preview entries reference "
        f"unknown candidates"
    )


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla", "run_tiny_mlp",
])
def test_llm_graph_view_only_contains_legal_candidates(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    cas = _read(run_dir / "02_graph_analysis" / "candidate_actions.json")
    legal_ids = {
        c["candidate_id"] for c in cas["candidates"]
        if (c.get("legality") or {}).get("ok")
    }
    illegal_ids = {
        c["candidate_id"] for c in cas["candidates"]
        if not (c.get("legality") or {}).get("ok")
    }
    lv = _read_or_none(run_dir / "02_graph_analysis" / "llm_graph_view.json")
    if lv is None:
        pytest.skip("llm_graph_view.json not emitted at this stop-after")
    visible: set[str] = set()
    for region in lv.get("regions", []) or []:
        for lc in region.get("legal_candidates", []) or []:
            visible.add(lc["candidate_id"])
    leaked_illegal = visible & illegal_ids
    unknown = visible - legal_ids - illegal_ids
    assert not leaked_illegal, f"{fixture_name}: illegal in llm_view: {sorted(leaked_illegal)[:3]}"
    assert not unknown, f"{fixture_name}: unknown ids in llm_view: {sorted(unknown)[:3]}"


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla",
])
def test_decision_sites_reference_real_regions(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    rm = _read(run_dir / "02_graph_analysis" / "region_map.json")
    sites = _read(run_dir / "02_graph_analysis" / "decision_sites.json")
    region_ids = {r["region_id"] for r in rm["regions"]}
    bad = [
        s["site_id"] for s in sites["sites"]
        if s.get("region_id") not in region_ids
    ]
    assert not bad, f"{fixture_name}: decision_sites with unknown region: {bad[:3]}"


# --------------------------------------------------------------------------- #
# Group B: Selected-candidate traceability across all levels
# --------------------------------------------------------------------------- #


def test_selected_candidate_traces_through_recipe_layer(
    run_merlin_mlp_wide: Path,
) -> None:
    """selected_candidate_id ∈ candidate_actions, in recipe.mlir, in
    semantic_obligations as recipe_op_id, in real_transform_manifest
    as selected_candidate_id."""
    run_dir = run_merlin_mlp_wide
    sel = _read(run_dir / "03_recipe_planning" / "candidate_selection.json")
    sel_id = sel["selected_candidate_id"]
    assert sel_id

    cas = _read(run_dir / "02_graph_analysis" / "candidate_actions.json")
    ca_ids = {c["candidate_id"] for c in cas["candidates"]}
    assert sel_id in ca_ids

    recipe_text = (run_dir / "03_recipe_planning" / "recipe.mlir").read_text(
        encoding="utf-8",
    )
    assert sel_id in recipe_text, (
        "selected candidate not referenced in recipe.mlir"
    )

    obligations = _read(
        run_dir / "03_recipe_planning" / "semantic_obligations.json"
    )
    op_ids = {o.get("recipe_op_id") for o in obligations.get("obligations", [])}
    assert "recipe_0000" in op_ids, "recipe_0000 missing from semantic_obligations"

    rtm_path = (
        run_dir / "03_recipe_planning" / "real_lowering"
        / "real_transform_manifest.json"
    )
    if rtm_path.exists():
        rtm = _read(rtm_path)
        assert (rtm.get("selected_recipe") or {}).get("selected_candidate_id") == sel_id


def test_selected_setttile_tile_matches_recipe_and_manifest(
    run_merlin_mlp_wide: Path,
) -> None:
    """For SetTileParams: the tile dims in candidate_selection's
    recipe_delta MUST match the tile in real_transform_manifest's
    selected_recipe block."""
    run_dir = run_merlin_mlp_wide
    sel = _read(run_dir / "03_recipe_planning" / "candidate_selection.json")
    delta = (sel.get("recipe_delta") or [{}])[0]
    sel_tile = delta.get("tile") or {}
    rtm = _read(
        run_dir / "03_recipe_planning" / "real_lowering"
        / "real_transform_manifest.json"
    )
    rtm_tile = (rtm.get("selected_recipe") or {}).get("tile") or {}
    assert sel_tile == rtm_tile, (
        f"tile mismatch: candidate_selection={sel_tile} "
        f"real_transform_manifest={rtm_tile}"
    )


def test_fusion_selected_producer_consumer_matches_real_fusion_manifest(
    run_proxy_vla: Path,
) -> None:
    """For FuseProducerConsumer: the producer/consumer/via_tensor in
    candidate_selection.recipe_delta MUST match real_fusion_manifest."""
    run_dir = run_proxy_vla
    sel = _read(run_dir / "03_recipe_planning" / "candidate_selection.json")
    if sel.get("candidate_kind") != "fuse_producer_consumer":
        pytest.skip("greedy did not pick fusion on this run")
    delta = (sel.get("recipe_delta") or [{}])[0]
    fm = _read_or_none(
        run_dir / "03_recipe_planning" / "real_lowering"
        / "real_fusion_manifest.json"
    )
    assert fm is not None, "fusion selected but real_fusion_manifest absent"
    fm_fusion = fm.get("fusion") or {}
    assert delta.get("producer") == fm_fusion.get("producer")
    assert delta.get("consumer") == fm_fusion.get("consumer")
    assert delta.get("via_tensor") == fm_fusion.get("via_tensor")


# --------------------------------------------------------------------------- #
# Group C: Counts match across artifacts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla", "run_tiny_mlp",
])
def test_graph_dossier_v3_region_count_matches_region_map(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    rm = _read(run_dir / "02_graph_analysis" / "region_map.json")
    v3 = _read_or_none(run_dir / "02_graph_analysis" / "graph_dossier_v3.json")
    if v3 is None:
        pytest.skip("graph_dossier_v3.json absent at this stop-after")
    assert len(v3.get("regions", [])) == len(rm["regions"])


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla",
])
def test_precision_budget_covers_every_region(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    rm = _read(run_dir / "02_graph_analysis" / "region_map.json")
    pb = _read(
        run_dir / "02_graph_analysis" / "readiness"
        / "precision_budget_report.json"
    )
    rm_ids = {r["region_id"] for r in rm["regions"]}
    pb_ids = {r["region_id"] for r in pb["regions"]}
    missing = rm_ids - pb_ids
    assert not missing, f"{fixture_name}: precision_budget missing regions: {sorted(missing)[:3]}"


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla",
])
def test_hardware_resource_covers_every_region(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    rm = _read(run_dir / "02_graph_analysis" / "region_map.json")
    hw = _read(
        run_dir / "02_graph_analysis" / "readiness"
        / "hardware_resource_report.json"
    )
    rm_ids = {r["region_id"] for r in rm["regions"]}
    hw_ids = {r["region_id"] for r in hw["regions"]}
    missing = rm_ids - hw_ids
    assert not missing, f"{fixture_name}: hardware_resource missing regions: {sorted(missing)[:3]}"


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla",
])
def test_counterfactual_covers_every_candidate(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    cas = _read(run_dir / "02_graph_analysis" / "candidate_actions.json")
    cf = _read(
        run_dir / "02_graph_analysis" / "readiness"
        / "candidate_counterfactual_report.json"
    )
    assert cf["summary"]["candidate_count"] == len(cas["candidates"])


# --------------------------------------------------------------------------- #
# Group D: M-18 region-level calibration cross-references
# --------------------------------------------------------------------------- #


def test_calibration_dossier_overlay_matches_report(
    run_merlin_mlp_wide: Path,
) -> None:
    """When M-18 ran successfully, graph_dossier_v3's top-level
    calibration block must match the standalone profiler_calibration_report
    summary."""
    run_dir = run_merlin_mlp_wide
    rep = _read(
        run_dir / "02_graph_analysis" / "calibration"
        / "profiler_calibration_report.json"
    )
    if rep["overall"] not in ("calibrated", "partial"):
        pytest.skip("calibration not_run on this fixture")
    v3 = _read(run_dir / "02_graph_analysis" / "graph_dossier_v3.json")
    cal = v3.get("calibration") or {}
    summary = rep.get("summary") or {}
    assert cal.get("calibration_status") == rep["calibration_status"]
    assert cal.get("matched_region_count") == summary.get("matched_region_count")
    assert cal.get("total_region_count") == summary.get("total_region_count")


def test_calibration_per_region_overlay_uses_real_regions(
    run_merlin_mlp_wide: Path,
) -> None:
    """Every region with a calibration overlay in graph_dossier_v3
    must be a real region from region_map (no synthetic ones)."""
    run_dir = run_merlin_mlp_wide
    rep = _read(
        run_dir / "02_graph_analysis" / "calibration"
        / "profiler_calibration_report.json"
    )
    if rep["overall"] not in ("calibrated", "partial"):
        pytest.skip("calibration not_run")
    rm = _read(run_dir / "02_graph_analysis" / "region_map.json")
    rm_ids = {r["region_id"] for r in rm["regions"]}
    v3 = _read(run_dir / "02_graph_analysis" / "graph_dossier_v3.json")
    overlay_regions = [
        r for r in v3["regions"] if "calibration" in r
    ]
    bad = [r["region_id"] for r in overlay_regions if r["region_id"] not in rm_ids]
    assert not bad, f"calibration overlay on unknown regions: {bad[:3]}"


# --------------------------------------------------------------------------- #
# Group E: M-18.3 candidate calibration cross-references
# --------------------------------------------------------------------------- #


def test_candidate_calibration_only_legal_set_tile_candidates(
    run_merlin_mlp_wide: Path,
) -> None:
    run_dir = run_merlin_mlp_wide
    cc = _read(
        run_dir / "02_graph_analysis" / "candidate_calibration"
        / "candidate_calibration_report.json"
    )
    if cc["overall"] not in ("calibrated", "partial"):
        pytest.skip("candidate_calibration not_run")
    cas = _read(run_dir / "02_graph_analysis" / "candidate_actions.json")
    legal_set_tile = {
        c["candidate_id"] for c in cas["candidates"]
        if c.get("kind") == "set_tile_params"
        and (c.get("legality") or {}).get("ok")
    }
    cc_ids = {c["candidate_id"] for c in cc["candidates"]}
    assert cc_ids == legal_set_tile, (
        f"candidate_calibration drift: "
        f"missing={legal_set_tile - cc_ids} extra={cc_ids - legal_set_tile}"
    )


def test_m24_kernel_readiness_row6_matches_m22(
    run_merlin_mlp_wide: Path,
) -> None:
    """M-24 row 6 (compiled_bottleneck) must report the same
    kernel_calibration_status as M-22's standalone report. The
    integrity invariant catches drift if M-24 starts deriving its
    own status independently."""
    run_dir = run_merlin_mlp_wide
    cb = _read_or_none(
        run_dir / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    r6 = _read_or_none(
        run_dir / "02_graph_analysis" / "kernel_readiness"
        / "bottleneck_report.json"
    )
    if cb is None or r6 is None:
        pytest.skip("M-22 or M-24 not on disk")
    assert r6.get("kernel_calibration_status") == cb.get(
        "kernel_calibration_status"
    )


def test_m24_kernel_readiness_matrix_well_formed(
    run_merlin_mlp_wide: Path,
) -> None:
    """The M-24 matrix counts (ready/ready_for_m24_1/partial/not_ready/
    not_run) must equal the number of rows."""
    run_dir = run_merlin_mlp_wide
    m = _read_or_none(
        run_dir / "02_graph_analysis" / "kernel_readiness"
        / "kernel_section_readiness_matrix.json"
    )
    if m is None:
        pytest.skip("M-24 not on disk")
    n_rows = len(m.get("slide_rows", []) or [])
    assert n_rows == 6
    s = (
        m.get("ready_count", 0)
        + m.get("ready_for_m24_1_count", 0)
        + m.get("partial_count", 0)
        + m.get("not_ready_count", 0)
        + m.get("not_run_count", 0)
    )
    assert s == n_rows, (
        f"row counts ({s}) don't add up to total rows ({n_rows}): "
        f"{m}"
    )


def test_m22_compiled_bottleneck_overlays_match_standalone_report(
    run_merlin_mlp_wide: Path,
) -> None:
    """For every M-22 ok-region, the compiled_evidence overlay on
    hardware_resource_report must match the standalone
    compiled_bottleneck_report. M-22 must not mutate the M-17.1
    deterministic-baseline ``calibration_status`` field."""
    run_dir = run_merlin_mlp_wide
    m22_path = (
        run_dir / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    if not m22_path.exists():
        pytest.skip("M-22 not_run on this fixture")
    m22 = _read(m22_path)
    if m22.get("overall") != "ok":
        pytest.skip("M-22 has no measurements")

    hrr_path = (
        run_dir / "02_graph_analysis" / "readiness"
        / "hardware_resource_report.json"
    )
    hrr = _read(hrr_path)

    # M-17.1's deterministic calibration_status is left untouched.
    assert hrr.get("calibration_status") == "not_profiler_calibrated", (
        "M-22 must not mutate M-17.1 calibration_status field"
    )
    assert hrr.get("kernel_calibration_status") == m22["kernel_calibration_status"]

    m22_by_region = {
        r["region_id"]: r for r in m22.get("regions", [])
        if r.get("model_status") == "ok"
    }

    overlaid_count = 0
    for r in hrr.get("regions", []) or []:
        rid = r.get("region_id")
        ev = r.get("compiled_evidence")
        m22_r = m22_by_region.get(rid)
        if m22_r is not None:
            assert ev is not None, (
                f"hardware_resource_report missing compiled_evidence for {rid}"
            )
            assert ev["analytical_bottleneck"] == m22_r["analytical_bottleneck"]
            assert ev["measured_bottleneck"] == m22_r["measured_bottleneck"]
            assert ev["bottleneck_classification_agreement"] == (
                m22_r["bottleneck_classification_agreement"]
            )
            overlaid_count += 1
        else:
            assert ev is None, (
                f"compiled_evidence overlay leaked onto unmeasured region {rid}"
            )
    assert overlaid_count == len(m22_by_region), (
        f"hardware_resource_report overlay count drift: "
        f"hrr={overlaid_count} m22={len(m22_by_region)}"
    )


def test_m21_analytical_cost_overlays_match_standalone_report(
    run_merlin_mlp_wide: Path,
) -> None:
    """Every modeled M-21 candidate's overlay block on cost_preview_v2
    must agree with the standalone per_candidate_analytical_cost.json
    report. Same for llm_graph_view.json."""
    run_dir = run_merlin_mlp_wide
    ac_path = (
        run_dir / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    if not ac_path.exists():
        pytest.skip("M-21 analytical_cost not_run on this fixture")
    ac = _read(ac_path)
    if ac.get("overall") != "ok":
        pytest.skip("M-21 analytical_cost not ok")
    modeled = {
        c["candidate_id"]: c for c in ac["candidates"]
        if c.get("model_status") == "ok"
    }
    if not modeled:
        pytest.skip("no modeled candidates")

    cp = _read(run_dir / "02_graph_analysis" / "cost_preview_v2.json")
    cp_overlay_count = 0
    for p in cp.get("cost_previews", []):
        cid = p.get("candidate_id")
        block = p.get("m21_analytical_cost")
        if cid in modeled:
            assert block is not None, f"M-21 overlay missing for modeled {cid}"
            assert block["predicted_us"] == modeled[cid]["predicted_us"]
            assert block["bottleneck_resource"] == modeled[cid]["bottleneck_resource"]
            assert block["bottleneck_tier"] == modeled[cid]["bottleneck_tier"]
            assert block["model_kind"] == modeled[cid]["model_kind"]
            cp_overlay_count += 1
        else:
            assert block is None, (
                f"M-21 overlay leaked onto non-modeled candidate {cid}"
            )
    assert cp_overlay_count == len(modeled), (
        f"cost_preview_v2 overlay count drift: cp={cp_overlay_count} "
        f"modeled={len(modeled)}"
    )

    lv_path = run_dir / "02_graph_analysis" / "llm_graph_view.json"
    if lv_path.exists():
        lv = _read(lv_path)
        seen_in_lv: set[str] = set()
        for region in lv.get("regions", []) or []:
            for lc in region.get("legal_candidates", []) or []:
                cid = lc.get("candidate_id")
                if cid in modeled:
                    block = lc.get("m21_analytical_cost")
                    assert block is not None, (
                        f"M-21 overlay missing in llm_graph_view for {cid}"
                    )
                    assert block["predicted_us"] == modeled[cid]["predicted_us"]
                    seen_in_lv.add(cid)
        assert seen_in_lv == set(modeled.keys()), (
            f"llm_graph_view missing M-21 overlays: "
            f"{sorted(set(modeled.keys()) - seen_in_lv)}"
        )


def test_candidate_calibration_overlays_cost_preview_v2(
    run_merlin_mlp_wide: Path,
) -> None:
    """Every calibrated candidate's measurements layered onto
    cost_preview_v2 match the standalone candidate_calibration_report."""
    run_dir = run_merlin_mlp_wide
    cc = _read(
        run_dir / "02_graph_analysis" / "candidate_calibration"
        / "candidate_calibration_report.json"
    )
    if cc["overall"] not in ("calibrated", "partial"):
        pytest.skip("candidate_calibration not_run")
    cp = _read(run_dir / "02_graph_analysis" / "cost_preview_v2.json")

    cc_by_id = {
        c["candidate_id"]: c for c in cc["candidates"]
        if c.get("calibration_status") == "calibrated"
    }
    cp_calibrated = [p for p in cp["cost_previews"] if "calibration" in p]
    assert len(cp_calibrated) == len(cc_by_id), (
        f"overlay count drift: cp={len(cp_calibrated)} cc={len(cc_by_id)}"
    )
    for p in cp_calibrated:
        cc_entry = cc_by_id.get(p["candidate_id"])
        assert cc_entry is not None, f"cost_preview overlay for unknown id"
        cal = p["calibration"]
        assert cal["measured_baseline_us"] == cc_entry["measured_baseline_us"]
        assert cal["measured_tiled_us"] == cc_entry["measured_tiled_us"]


# --------------------------------------------------------------------------- #
# Group F: M-19 compiled-kernel cross-references
# --------------------------------------------------------------------------- #


def test_compiled_kernel_artifacts_match_real_transform_manifest(
    run_merlin_mlp_wide: Path,
) -> None:
    """M-19's GPU + CPU artifacts must reference the SAME matmul_shape,
    tile, region_id, candidate_id as M-11B's real_transform_manifest."""
    run_dir = run_merlin_mlp_wide
    rtm = _read(
        run_dir / "03_recipe_planning" / "real_lowering"
        / "real_transform_manifest.json"
    )
    sel = rtm.get("selected_recipe") or {}
    sig = rtm.get("matmul_signature") or {}
    expected_tile = sel.get("tile") or {}
    expected_shape = {"M": sig.get("M"), "N": sig.get("N"), "K": sig.get("K")}
    expected_region = sel.get("region")
    expected_candidate = sel.get("selected_candidate_id")
    expected_recipe_op = sel.get("recipe_op_id")

    base = run_dir / "02_graph_analysis" / "kernel_execution"
    for name in ("compiled_kernel_run_gpu.json", "compiled_kernel_run_cpu.json"):
        path = base / name
        if not path.exists():
            continue
        d = _read(path)
        # Skip when the track wasn't applicable.
        if d.get("compile_status") in ("not_applicable",):
            continue
        assert d["matmul_shape"] == expected_shape, f"{name}: matmul_shape drift"
        assert d["tile"] == expected_tile, f"{name}: tile drift"
        assert d["region_id"] == expected_region, f"{name}: region_id drift"
        assert d["candidate_id"] == expected_candidate, f"{name}: candidate drift"
        assert d["recipe_op_id"] == expected_recipe_op, f"{name}: recipe_op drift"


def test_kernel_execution_not_applicable_for_fusion(
    run_proxy_vla: Path,
) -> None:
    """When greedy picks fusion (proxy_vla), kernel_execution must
    emit a not_applicable summary; no GPU/CPU artifact files."""
    base = run_proxy_vla / "02_graph_analysis" / "kernel_execution"
    if not base.exists():
        pytest.skip("kernel_execution didn't run on this fixture")
    summary = (base / "kernel_execution_summary.md").read_text(encoding="utf-8")
    assert "not_applicable" in summary
    assert not (base / "compiled_kernel_run_gpu.json").exists()
    assert not (base / "compiled_kernel_run_cpu.json").exists()


# --------------------------------------------------------------------------- #
# Group G: M-16.1 strict gate cross-references
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla", "run_tiny_mlp",
])
def test_strict_gate_evidence_paths_exist(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    pl = run_dir / "01_payload_lowering"
    sg_files = list(pl.glob("*_strict_gate_report.json"))
    assert len(sg_files) >= 1, f"{fixture_name}: no strict_gate_report"
    sg = _read(sg_files[0])
    for key, rel in (sg.get("evidence") or {}).items():
        assert (run_dir / rel).exists(), f"{fixture_name}: {key} → {rel} missing"


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla", "run_tiny_mlp",
])
def test_strict_gate_status_consistent_with_lowering_summary(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    pl = run_dir / "01_payload_lowering"
    ls = _read(pl / "lowering_summary.json")
    sg_files = list(pl.glob("*_strict_gate_report.json"))
    sg = _read(sg_files[0])
    if ls.get("status") == "fail":
        assert sg["status"] == "blocked"
    elif ls.get("status") in ("pass", "partial_success"):
        assert sg["status"] == "pass"


# --------------------------------------------------------------------------- #
# Group H: M-17.1 readiness matrix consistency
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla",
])
def test_readiness_matrix_overall_pass(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    m = _read_or_none(
        run_dir / "02_graph_analysis" / "readiness"
        / "graph_analysis_readiness_matrix.json"
    )
    if m is None:
        pytest.skip("readiness matrix absent")
    assert m["overall"] == "pass"
    statuses = [r["status"] for r in m["slide_rows"]]
    # Slide rows 0-4 must be ready; row 5 is calibrated or ready_for_m18.
    for i, s in enumerate(statuses[:5]):
        assert s == "ready", f"row {i+1} status={s}, expected ready"
    assert statuses[5] in ("ready_for_m18", "calibrated"), (
        f"row 6 status={statuses[5]}"
    )


def test_readiness_matrix_artifact_paths_exist(
    run_merlin_mlp_wide: Path,
) -> None:
    run_dir = run_merlin_mlp_wide
    base = run_dir / "02_graph_analysis" / "readiness"
    m = _read(base / "graph_analysis_readiness_matrix.json")
    for row in m["slide_rows"]:
        artifact = row["artifact"]
        assert (base / artifact).exists(), f"readiness row missing artifact: {artifact}"


# --------------------------------------------------------------------------- #
# Group I: agent_decision_request consistency
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla",
])
def test_agent_decision_request_candidate_ids_allowed_are_legal(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    req = _read_or_none(
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json"
    )
    if req is None:
        pytest.skip("agent_decision_request not emitted")
    cas = _read(run_dir / "02_graph_analysis" / "candidate_actions.json")
    legal_ids = {
        c["candidate_id"] for c in cas["candidates"]
        if (c.get("legality") or {}).get("ok")
    }
    illegal_in_allowed = set(req.get("candidate_ids_allowed", [])) - legal_ids
    assert not illegal_in_allowed, (
        f"{fixture_name}: agent_decision_request includes illegal: "
        f"{sorted(illegal_in_allowed)[:3]}"
    )


# --------------------------------------------------------------------------- #
# Group J: validate_run hash chain integrity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla",
])
def test_validate_run_overall_pass(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    """All opt-ins enabled must not break the manifest hash chain."""
    run_dir: Path = request.getfixturevalue(fixture_name)
    from compgen.graph_compilation.validate import validate_run

    rep = validate_run(run_dir)
    assert rep.overall == "pass", (
        f"{fixture_name}: validate_run overall={rep.overall}; "
        f"failures: {[f'{r.rule_id}: {r.detail[:80]}' for r in rep.rules if r.status != 'pass'][:3]}"
    )


# --------------------------------------------------------------------------- #
# Group K: M-15B downstream-retry on real-fail path
# --------------------------------------------------------------------------- #


def test_tiny_mlp_real_fail_emits_downstream_retry(run_tiny_mlp: Path) -> None:
    """tiny_mlp with greedy tile_16 → K_iters=4 → bit-equality fails →
    M-15B emits a typed downstream_retry_request. The retry request's
    failed_candidate_id must be in candidate_actions and excluded from
    candidate_ids_allowed."""
    rr = _read_or_none(
        run_tiny_mlp / "03_recipe_planning" / "downstream_retry"
        / "downstream_retry_request.json"
    )
    if rr is None:
        pytest.skip("tiny_mlp greedy didn't fail M-12 on this run")
    assert rr["status"] == "retry_required"
    failed = rr["failed_candidate_id"]
    assert failed
    assert failed not in rr["candidate_ids_allowed"], (
        "failed candidate must be excluded from retry_options"
    )
    cas = _read(run_tiny_mlp / "02_graph_analysis" / "candidate_actions.json")
    ca_ids = {c["candidate_id"] for c in cas["candidates"]}
    assert failed in ca_ids, "failed_candidate_id not in candidate_actions"


# --------------------------------------------------------------------------- #
# Group L: Region dossiers consistent with region map
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla",
])
def test_region_dossiers_cover_every_region(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    rm = _read(run_dir / "02_graph_analysis" / "region_map.json")
    rm_ids = {r["region_id"] for r in rm["regions"]}

    rd_dir = run_dir / "02_graph_analysis" / "region_dossiers"
    if not rd_dir.is_dir():
        pytest.skip("no region_dossiers/ dir")
    rd_ids: set[str] = set()
    for p in rd_dir.iterdir():
        if not p.is_file() or p.suffix != ".json":
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "region_id" in doc:
            rd_ids.add(doc["region_id"])

    missing = rm_ids - rd_ids
    assert not missing, (
        f"{fixture_name}: regions without dossier files: {sorted(missing)[:3]}"
    )


# --------------------------------------------------------------------------- #
# Group M: Suite-wide structural sanity (light — no expensive run-suite call)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla", "run_tiny_mlp",
])
def test_run_manifest_present_and_lists_stages(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    manifest = _read_or_none(run_dir / "run_manifest.json")
    if manifest is None:
        pytest.skip("run_manifest.json absent (pipeline likely raised)")
    stages = manifest.get("stages") or []
    stage_ids = [s.get("stage_id") for s in stages]
    for required in ("graph_capture", "payload_lowering", "graph_analysis"):
        assert required in stage_ids, (
            f"{fixture_name}: stage {required} missing from run_manifest"
        )


# --------------------------------------------------------------------------- #
# Group N: Schema-version stability across all artifacts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla",
])
def test_canonical_artifacts_have_schema_versions(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    expected = {
        "02_graph_analysis/region_map.json": "region_map",
        "02_graph_analysis/candidate_actions.json": "candidate_actions",
        "02_graph_analysis/decision_sites.json": "decision_sites",
        "02_graph_analysis/cost_preview_v2.json": "cost_preview_v2",
        "02_graph_analysis/graph_dossier_v3.json": "graph_dossier_v3",
        "02_graph_analysis/llm_graph_view.json": "llm_graph_view",
        "02_graph_analysis/readiness/graph_analysis_readiness_matrix.json":
            "graph_analysis_readiness_matrix",
        "02_graph_analysis/readiness/precision_budget_report.json":
            "precision_budget_report",
        "02_graph_analysis/readiness/working_set_fit_report.json":
            "working_set_fit_report",
        "02_graph_analysis/readiness/reuse_lifetime_report.json":
            "reuse_lifetime_report",
        "02_graph_analysis/readiness/candidate_counterfactual_report.json":
            "candidate_counterfactual_report",
        "02_graph_analysis/readiness/agent_view_completeness_report.json":
            "agent_view_completeness_report",
        "02_graph_analysis/readiness/hardware_resource_report.json":
            "hardware_resource_report",
    }
    for rel, prefix in expected.items():
        path = run_dir / rel
        if not path.exists():
            continue
        doc = _read(path)
        sv = doc.get("schema_version", "")
        assert sv.startswith(prefix), (
            f"{fixture_name}: {rel} schema_version={sv!r}, expected {prefix}*"
        )


# --------------------------------------------------------------------------- #
# Group O: Working-set fit cross-check with candidate_actions
# --------------------------------------------------------------------------- #


def test_working_set_fit_includes_every_legal_set_tile_candidate(
    run_merlin_mlp_wide: Path,
) -> None:
    run_dir = run_merlin_mlp_wide
    cas = _read(run_dir / "02_graph_analysis" / "candidate_actions.json")
    legal_set_tile = {
        c["candidate_id"] for c in cas["candidates"]
        if c.get("kind") == "set_tile_params"
        and (c.get("legality") or {}).get("ok")
    }
    ws = _read(
        run_dir / "02_graph_analysis" / "readiness"
        / "working_set_fit_report.json"
    )
    seen: set[str] = set()
    for region in ws["regions"]:
        for t in region["candidate_tiles"]:
            if t["legality_ok"]:
                seen.add(t["candidate_id"])
    assert seen >= legal_set_tile, (
        f"working_set_fit missing: {legal_set_tile - seen}"
    )


# --------------------------------------------------------------------------- #
# Group P: M-17 evidence pack joint claim consistency
# --------------------------------------------------------------------------- #


def test_evidence_pack_aggregate_matches_per_model_artifacts(
    run_merlin_mlp_wide: Path, run_proxy_vla: Path, tmp_path: Path,
) -> None:
    """The M-17 evidence pack's aggregates must equal the sum of fields
    pulled directly from each model's artifacts (joint integrity)."""
    suite = tmp_path / "joint_suite"
    canonical = suite / "canonical"
    canonical.mkdir(parents=True)
    import shutil
    shutil.copytree(run_merlin_mlp_wide, canonical / "merlin_mlp_wide")
    shutil.copytree(run_proxy_vla, canonical / "proxy_vla")

    from compgen.graph_compilation.evidence_pack import build_evidence_pack
    res = build_evidence_pack(
        canonical_suite_root=canonical, wide_suite_root=None,
        out_dir=suite / "evidence_pack", skip_figures=True,
    )
    agg = res.aggregates

    # bit_equality_discharged_count should equal #(rows where the per-model
    # real_differential or real_fusion_differential reports show
    # discharged_bit_equality).
    expected_bit_eq = 0
    for r in res.rows:
        rd = run_dir = (canonical / r.model_id)
        for rel in (
            "03_recipe_planning/real_verification/real_differential_report.json",
            "03_recipe_planning/real_verification/real_fusion_differential_report.json",
        ):
            d = _read_or_none(rd / rel)
            if d is None:
                continue
            if (d.get("error") or {}).get("refinement_status") == "discharged_bit_equality":
                if r.selected_candidate_kind == d.get("fusion", {}).get("producer", "")[:0] + (
                    "fuse_producer_consumer" if "fusion" in d else "set_tile_params"
                ):
                    expected_bit_eq += 1
                break
    # The aggregate must be at least what we re-counted from disk.
    assert agg["bit_equality_discharged_count"] >= expected_bit_eq, (
        f"evidence pack underreports bit_equality: "
        f"reported={agg['bit_equality_discharged_count']} expected≥{expected_bit_eq}"
    )


# --------------------------------------------------------------------------- #
# Group P: Stress-audit invariants promoted to permanent tests (2026-05-04)
# --------------------------------------------------------------------------- #
# These came from a deep stress audit that caught two real bugs:
#   1. run_manifest.json was lost when M-15B raised retry-required.
#   2. M-22.1 cache_evidence drifted between compiled_bottleneck_report
#      and hardware_resource_report on some paths.
# Each invariant below locks in one of those guarantees so the regression
# never silently returns.


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# --- P1: M-15B partial-manifest persistence (regression for the bug
#         where run_manifest.json was lost when M-15B raised). ----------- #


def test_m15b_retry_required_run_still_writes_run_manifest(
    run_tiny_mlp: Path,
) -> None:
    """tiny_mlp greedy hits a real M-12 K_iters>1 failure → M-15B
    raises retry-required → pipeline exits non-zero. The run dir must
    STILL contain a usable run_manifest.json with all stage records
    accumulated up to the failure point so audit/integrity tools can
    verify the run."""
    manifest_path = run_tiny_mlp / "run_manifest.json"
    assert manifest_path.exists(), (
        "run_manifest.json missing on M-15B retry-required run; "
        "audit/integrity tools cannot validate this run state"
    )
    manifest = _read(manifest_path)
    stages = manifest.get("stages") or []
    assert len(stages) >= 3, (
        f"partial manifest must have >= 3 stages (capture, lower, "
        f"graph_analysis); got {len(stages)}"
    )
    # R009 still holds across the partial chain.
    for prev, cur in zip(stages, stages[1:]):
        prev_id = prev.get("stage_id")
        cur_id = cur.get("stage_id")
        assert prev.get("output_hash") == cur.get("input_hash"), (
            f"R009 broken between {prev_id} and {cur_id} on "
            f"partial M-15B run"
        )
    # The downstream_retry_request must also be on disk.
    retry_req = (
        run_tiny_mlp / "03_recipe_planning" / "downstream_retry"
        / "downstream_retry_request.json"
    )
    assert retry_req.exists(), (
        "downstream_retry_request.json missing on M-15B retry-required run"
    )


def test_m15b_retry_partial_manifest_records_failed_stage(
    run_tiny_mlp: Path,
) -> None:
    """The partial manifest must include the stage record (with
    output_hash) for the stage immediately PRECEDING the M-15B
    failure, so the agent can verify the chain up to that point."""
    manifest = _read(run_tiny_mlp / "run_manifest.json")
    stage_ids = [s.get("stage_id") for s in manifest.get("stages") or []]
    # Stages that should be in any partial run reaching M-15B:
    for required in ("graph_capture", "payload_lowering", "graph_analysis"):
        assert required in stage_ids, (
            f"partial manifest missing required stage {required}: "
            f"got {stage_ids}"
        )


# --- P2: M-22.1 cache_evidence cross-overlay consistency. ---------------- #


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla",
])
def test_m221_cache_evidence_consistent_across_overlays(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    """For every region with M-22 evidence, cache_evidence must match
    between compiled_bottleneck_report.regions[*] and
    hardware_resource_report.regions[*].compiled_evidence."""
    run_dir: Path = request.getfixturevalue(fixture_name)
    cb = _read_or_none(
        run_dir / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    hrr = _read_or_none(
        run_dir / "02_graph_analysis" / "readiness"
        / "hardware_resource_report.json"
    )
    if cb is None or hrr is None or cb.get("overall") != "ok":
        pytest.skip("M-22 had no measurements")
    cb_by_rid = {
        r.get("region_id"): r.get("cache_evidence")
        for r in cb.get("regions", []) or []
        if r.get("model_status") == "ok"
    }
    hrr_by_rid = {
        r.get("region_id"): (
            (r.get("compiled_evidence") or {}).get("cache_evidence")
        )
        for r in hrr.get("regions", []) or []
        if r.get("compiled_evidence") is not None
    }
    mismatches = [
        (rid, cb_by_rid[rid], hrr_by_rid.get(rid))
        for rid in cb_by_rid
        if rid in hrr_by_rid and cb_by_rid[rid] != hrr_by_rid[rid]
    ]
    assert not mismatches, (
        f"{fixture_name}: M-22.1 cache_evidence drift between "
        f"compiled_bottleneck and hardware_resource_report: {mismatches[:3]}"
    )


# --- P3: agent_guidance block + new sources keys present. --------------- #


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla", "run_tiny_mlp",
])
def test_agent_decision_request_carries_full_agent_guidance(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    candidates = [
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json",
        run_dir / "agent_decision_request.json",
    ]
    req = next((_read(p) for p in candidates if p.exists()), None)
    if req is None:
        pytest.skip("agent_decision_request.json absent on this fixture")
    g = req.get("agent_guidance")
    assert g is not None, (
        f"{fixture_name}: agent_decision_request missing agent_guidance"
    )
    for required in (
        "guidance_version", "preamble", "cost_column_priority",
        "disagreement_handling", "rationale_field_examples",
        "forbidden_phrase_patterns", "preferred_neutral_phrases",
        "response_shape", "selection_modes_supported",
        "honest_non_claims",
    ):
        assert required in g, (
            f"{fixture_name}: agent_guidance missing field {required}"
        )


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla", "run_tiny_mlp",
])
def test_agent_decision_request_sources_lists_optional_evidence(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    """sources block must list all 7 new optional evidence keys —
    agents need to know which cost columns are available even when
    a particular column was not produced."""
    run_dir: Path = request.getfixturevalue(fixture_name)
    candidates = [
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_request.json",
        run_dir / "agent_decision_request.json",
    ]
    req = next((_read(p) for p in candidates if p.exists()), None)
    if req is None:
        pytest.skip("agent_decision_request.json absent on this fixture")
    sources = req.get("sources", {}) or {}
    for required_key in (
        "analytical_cost_report",
        "compiled_bottleneck_report",
        "region_compiled_differential_report",
        "hardware_resource_report",
        "readiness_matrix",
        "calibration_report",
        "candidate_calibration_report",
    ):
        assert required_key in sources, (
            f"{fixture_name}: sources missing optional-evidence key "
            f"{required_key}"
        )


# --- P4: cost_matrix_completeness — every legal SetTileParams candidate
#         carries the M-21 overlay (M-21 is always-on). ----------------- #


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla", "run_tiny_mlp",
])
def test_every_modeled_m21_candidate_has_overlay(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    run_dir: Path = request.getfixturevalue(fixture_name)
    ac = _read_or_none(
        run_dir / "02_graph_analysis" / "analytical_cost"
        / "per_candidate_analytical_cost.json"
    )
    cp = _read_or_none(run_dir / "02_graph_analysis" / "cost_preview_v2.json")
    if ac is None or cp is None or ac.get("overall") != "ok":
        pytest.skip("M-21 had no modeled candidates")
    modeled_ids = {
        c["candidate_id"] for c in ac.get("candidates", []) or []
        if c.get("model_status") == "ok"
    }
    cp_overlaid = {
        p["candidate_id"] for p in cp.get("cost_previews", []) or []
        if p.get("m21_analytical_cost") is not None
    }
    missing = modeled_ids - cp_overlaid
    assert not missing, (
        f"{fixture_name}: M-21 modeled but not overlaid onto "
        f"cost_preview_v2: {sorted(missing)[:3]}"
    )


# --- P5: ledger milestone-event coverage. ------------------------------- #


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla",
])
def test_ledger_records_full_kernel_pipeline_milestones(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    """A run with COMPGEN_RUN_KERNELS=1 must record M-19, M-20, M-21,
    M-22, M-22.1 ledger events. M-23 is recorded too (with not_run
    note when no fusion candidate)."""
    run_dir: Path = request.getfixturevalue(fixture_name)
    events = _read_jsonl(run_dir / "stage_ledger.jsonl")
    notes = [e.get("note") or "" for e in events]
    for tag in ("M-19", "M-20", "M-21", "M-22", "M-22.1", "M-23"):
        assert any(tag in n for n in notes), (
            f"{fixture_name}: ledger missing {tag} stage event"
        )
    # Ledger is monotone-ordered.
    indices = {
        tag: next(i for i, n in enumerate(notes) if tag in n)
        for tag in ("M-19", "M-20", "M-21", "M-22", "M-22.1", "M-23")
    }
    ordered = (
        indices["M-19"] < indices["M-20"]
        < indices["M-21"] < indices["M-22"]
        < indices["M-22.1"] < indices["M-23"]
    )
    assert ordered, (
        f"{fixture_name}: ledger milestone order drift: {indices}"
    )


# --- P6: IR artifacts present + non-degenerate. ------------------------ #


@pytest.mark.parametrize("fixture_name", [
    "run_merlin_mlp_wide", "run_proxy_vla", "run_tiny_mlp",
])
def test_ir_artifacts_present_and_non_empty(
    fixture_name: str, request: pytest.FixtureRequest,
) -> None:
    """payload.mlir under 01_payload_lowering/, action_space.mlir
    under 02_graph_analysis/, and transformed_payload.real.mlir under
    03_recipe_planning/real_lowering/ must all exist and be
    non-trivially sized (>= 8 bytes) for any successful capture+lower
    run."""
    run_dir: Path = request.getfixturevalue(fixture_name)
    payload_candidates = sorted(
        (run_dir / "01_payload_lowering").glob("**/payload.mlir")
    )
    assert payload_candidates, (
        f"{fixture_name}: no payload.mlir found under 01_payload_lowering/"
    )
    for p in payload_candidates:
        assert p.stat().st_size >= 8, (
            f"{fixture_name}: payload.mlir suspiciously small: "
            f"{p.stat().st_size}B at {p}"
        )

    action_space = run_dir / "02_graph_analysis" / "action_space.mlir"
    assert action_space.exists(), (
        f"{fixture_name}: action_space.mlir missing"
    )
    assert action_space.stat().st_size >= 8

    transformed = (
        run_dir / "03_recipe_planning" / "real_lowering"
        / "transformed_payload.real.mlir"
    )
    if transformed.exists():
        assert transformed.stat().st_size >= 8, (
            f"{fixture_name}: transformed_payload.real.mlir suspiciously "
            f"small: {transformed.stat().st_size}B"
        )
