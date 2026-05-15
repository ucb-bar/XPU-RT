"""Tests Cost Preview V2.

Cross-checks the cost-preview-v2 artifact against on-disk inputs and
verifies target/tile sensitivity, confidence ordering, and verification
evidence grounding.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from compgen.graph_compilation.cost_preview_v2 import (
    _DEFAULT_TARGET,
    _TargetProfile,
    _build_candidate_cost_preview,
    _matmul_baseline_cost,
    _matmul_tiled_cost,
    run_cost_preview_v2,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE = REPO_ROOT / "results" / "graph_compilation" / "cost_preview_v2_suite"
WIDE = REPO_ROOT / "results" / "graph_compilation" / "wide_cost_preview_v2_suite"

_CANONICAL = (
    "tiny_mlp", "tiny_attention", "tiny_conv_block",
    "proxy_vlm", "proxy_vla", "custom_unsupported_op",
)


def _need_canonical() -> None:
    if not SUITE.is_dir():
        pytest.skip(
            f"fixture suite missing: {SUITE}; run "
            f"`compgen.graph_compilation run-suite --stop-after "
            f"cost-preview-v2` first"
        )


def _need_wide() -> None:
    if not WIDE.is_dir():
        pytest.skip(f"wide fixture suite missing: {WIDE}")


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Suite-wide positive checks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", _CANONICAL)
def test_artifact_shape(model: str) -> None:
    _need_canonical()
    ga = SUITE / model / "02_graph_analysis"
    for name in (
        "cost_preview_v2.json",
        "cost_preview_v2_validation.json",
        "cost_preview_v2_summary.md",
    ):
        assert (ga / name).exists(), f"{model}: missing {name}"


@pytest.mark.parametrize("model", _CANONICAL)
def test_validation_overall_pass(model: str) -> None:
    _need_canonical()
    v = _read(
        SUITE / model / "02_graph_analysis" / "cost_preview_v2_validation.json"
    )
    assert v["overall"] == "pass", v["checks"]


@pytest.mark.parametrize("model", _CANONICAL)
def test_every_legal_candidate_has_cost_preview_v2(model: str) -> None:
    _need_canonical()
    cas = _read(SUITE / model / "02_graph_analysis" / "candidate_actions.json")
    cp = _read(SUITE / model / "02_graph_analysis" / "cost_preview_v2.json")
    legal_ids = {
        c["candidate_id"] for c in cas["candidates"]
        if (c.get("legality") or {}).get("ok") is True
    }
    cp_ids = {p["candidate_id"] for p in cp["cost_previews"]}
    assert legal_ids.issubset(cp_ids), (
        f"{model}: legal candidates without cost_preview_v2: "
        f"{legal_ids - cp_ids}"
    )


def test_merlin_mlp_wide_records_real_transform_verified() -> None:
    _need_wide()
    cp = _read(
        WIDE / "merlin_mlp_wide" / "02_graph_analysis" / "cost_preview_v2.json"
    )
    selected = _read(
        WIDE / "merlin_mlp_wide" / "03_recipe_planning"
        / "candidate_selection.json"
    )
    sel_id = selected["selected_candidate_id"]
    sel_cp = next(p for p in cp["cost_previews"] if p["candidate_id"] == sel_id)
    assert sel_cp["features"]["real_transform_verified"] is True
    assert sel_cp["confidence"] >= 0.75
    assert sel_cp["evidence"]["real_differential_report"] is not None


def test_graph_dossier_v3_has_cost_preview_v2_inlined() -> None:
    _need_wide()
    v3 = _read(
        WIDE / "merlin_mlp_wide" / "02_graph_analysis" / "graph_dossier_v3.json"
    )
    found = False
    for region in v3["regions"]:
        for c in region["legal_candidates"]:
            if "cost_preview_v2" in c:
                found = True
                cp = c["cost_preview_v2"]
                assert "relative_cost" in cp
                assert "confidence" in cp
                assert "features" in cp
    assert found, "no legal candidate carries cost_preview_v2 in graph_dossier_v3"


def test_llm_graph_view_sorted_by_relative_cost() -> None:
    _need_wide()
    llm = _read(
        WIDE / "merlin_mlp_wide" / "02_graph_analysis" / "llm_graph_view.json"
    )
    saw_sorted = False
    for region in llm["regions"]:
        candidates = region["legal_candidates"]
        if len(candidates) < 2:
            continue
        costs = [
            (c.get("cost_preview_v2") or {}).get("relative_cost", 1.0)
            for c in candidates
        ]
        # Each region's candidates are sorted ascending by relative_cost.
        assert costs == sorted(costs), (
            f"region {region['region_id']!r} legal_candidates not sorted: {costs}"
        )
        if len(set(costs)) >= 2:
            saw_sorted = True
    assert saw_sorted, (
        "no region had ≥2 distinct relative_costs; sort cannot be verified"
    )


def test_llm_graph_view_does_not_expose_illegal_candidates() -> None:
    _need_wide()
    cas = _read(
        WIDE / "merlin_mlp_wide" / "02_graph_analysis" / "candidate_actions.json"
    )
    illegal_ids = {
        c["candidate_id"] for c in cas["candidates"]
        if (c.get("legality") or {}).get("ok") is False
    }
    llm = _read(
        WIDE / "merlin_mlp_wide" / "02_graph_analysis" / "llm_graph_view.json"
    )
    for region in llm["regions"]:
        for c in region["legal_candidates"]:
            assert c["candidate_id"] not in illegal_ids, (
                f"illegal candidate {c['candidate_id']!r} leaked into "
                f"llm_graph_view"
            )


# --------------------------------------------------------------------------- #
# Cost-model unit tests (target / tile sensitivity)
# --------------------------------------------------------------------------- #


def test_changing_target_peak_compute_changes_estimated_latency() -> None:
    """Compute-bound matmul: doubling peak_compute halves latency."""
    base = _DEFAULT_TARGET
    fast = _TargetProfile(
        target_id="fast_cpu",
        peak_compute_flops=base.peak_compute_flops * 2,
        peak_bandwidth_bytes_per_sec=base.peak_bandwidth_bytes_per_sec,
        scratchpad_bytes=base.scratchpad_bytes,
        l2_bytes=base.l2_bytes,
        l3_bytes=base.l3_bytes,
    )
    base_us, _ = _matmul_baseline_cost(M=64, N=64, K=64, target=base)
    fast_us, _ = _matmul_baseline_cost(M=64, N=64, K=64, target=fast)
    assert fast_us < base_us


def test_changing_target_bandwidth_changes_memory_bound_latency() -> None:
    """Memory-bound case: doubling bandwidth roughly halves the
    memory-time portion of a small-flops, large-bytes op."""
    base = _DEFAULT_TARGET
    fast = _TargetProfile(
        target_id="fast_mem_cpu",
        peak_compute_flops=base.peak_compute_flops,
        peak_bandwidth_bytes_per_sec=base.peak_bandwidth_bytes_per_sec * 2,
        scratchpad_bytes=base.scratchpad_bytes,
        l2_bytes=base.l2_bytes,
        l3_bytes=base.l3_bytes,
    )
    # An elementwise-like operation: high bytes, low flops.
    from compgen.graph_compilation.cost_preview_v2 import _roofline_latency_us
    base_us, _ = _roofline_latency_us(
        flops=1024.0, bytes_moved=10_000_000,
        working_set_bytes=10_000_000, target=base,
    )
    fast_us, _ = _roofline_latency_us(
        flops=1024.0, bytes_moved=10_000_000,
        working_set_bytes=10_000_000, target=fast,
    )
    assert fast_us < base_us


def test_changing_tile_dimensions_changes_estimate() -> None:
    """Different tile sizes produce different costs (loop overhead +
    working-set tier selection)."""
    target = _DEFAULT_TARGET
    cost_16, _ = _matmul_tiled_cost(
        M=128, N=128, K=128, tM=16, tN=16, tK=16, target=target,
    )
    cost_32, _ = _matmul_tiled_cost(
        M=128, N=128, K=128, tM=32, tN=32, tK=32, target=target,
    )
    cost_128, _ = _matmul_tiled_cost(
        M=128, N=128, K=128, tM=128, tN=128, tK=128, target=target,
    )
    distinct = {round(cost_16, 6), round(cost_32, 6), round(cost_128, 6)}
    assert len(distinct) >= 2


# --------------------------------------------------------------------------- #
# Negative tests
# --------------------------------------------------------------------------- #


@pytest.fixture
def merlin_mlp_wide_run(tmp_path: Path) -> Path:
    _need_wide()
    src = WIDE / "merlin_mlp_wide"
    if not src.is_dir():
        pytest.skip(f"merlin_mlp_wide fixture missing: {src}")
    dst = tmp_path / "merlin_mlp_wide"
    shutil.copytree(src, dst)
    return dst


def test_missing_region_dossier_marks_unavailable(merlin_mlp_wide_run: Path) -> None:
    """Delete a region-dossier file. The corresponding candidates must
    record an ``unavailable_reason`` for the cost preview rather than
    silently fabricating a value."""
    ga = merlin_mlp_wide_run / "02_graph_analysis"
    v2 = _read(ga / "graph_dossier_v2.json")
    # Delete one of the per-region dossiers.
    target_region = next(iter(v2["region_dossiers"]))
    target_path = merlin_mlp_wide_run / v2["region_dossiers"][target_region]
    if target_path.exists():
        target_path.unlink()
    target_yaml = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"
    run_cost_preview_v2(merlin_mlp_wide_run, target_yaml_path=target_yaml)
    cp = _read(ga / "cost_preview_v2.json")
    affected = [
        p for p in cp["cost_previews"] if p["region_id"] == target_region
    ]
    if affected:
        assert any("unavailable_reason" in p for p in affected), (
            f"deleted dossier for {target_region!r} did not surface an "
            f"unavailable_reason"
        )


def test_constant_cost_across_tile_candidates_fails(
    merlin_mlp_wide_run: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force baseline AND tiled to return identical constants — every
    relative_cost becomes 1.0 — non-degeneracy on tile candidates must
    fail."""
    from compgen.graph_compilation import cost_preview_v2 as mod

    def constant_cost(*args, **kwargs):
        return 1.0, {"tier": "scratchpad", "bw_multiplier": 4.0,
                     "compute_time_us": 1.0, "memory_time_us": 1.0,
                     "bottleneck": "compute"}

    monkeypatch.setattr(mod, "_matmul_tiled_cost", constant_cost)
    monkeypatch.setattr(mod, "_matmul_baseline_cost", constant_cost)
    target_yaml = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"
    result = run_cost_preview_v2(
        merlin_mlp_wide_run, target_yaml_path=target_yaml,
    )
    assert result.overall == "fail"
    val = _read(result.validation_path)
    cpv2r002 = next(c for c in val["checks"] if c["id"].startswith("CPV2R002"))
    assert cpv2r002["status"] == "fail"


def test_opaque_confidence_lower_than_structured_linalg() -> None:
    _need_canonical()
    cp = _read(SUITE / "tiny_mlp" / "02_graph_analysis" / "cost_preview_v2.json")
    structured_unboosted = [
        p["confidence"] for p in cp["cost_previews"]
        if p["candidate_kind"] == "set_tile_params"
        and not p["features"].get("real_transform_verified")
        and p.get("legality_ok")
    ]
    opaque = [
        p["confidence"] for p in cp["cost_previews"]
        if p["candidate_kind"] in {
            "create_kernel_contract", "create_payload_lowering_extension",
            "keep_as_fallback",
        }
    ]
    if structured_unboosted and opaque:
        assert min(structured_unboosted) > max(opaque), (
            f"structured baseline {min(structured_unboosted)} not > "
            f"opaque max {max(opaque)}"
        )


def test_fake_real_transform_verified_without_m12_report_fails() -> None:
    """A cost preview claiming ``real_transform_verified=true`` without
    a corresponding report must fail validation."""
    fake_cp = {
        "schema_version": "candidate_cost_preview_v2",
        "candidate_id": "cand_fake",
        "candidate_kind": "set_tile_params",
        "region_id": "matmul_0",
        "legality_ok": True,
        "baseline_static_latency_us": 1.0,
        "candidate_static_latency_us": 0.5,
        "relative_cost": 0.5,
        "confidence": 0.9,
        "features": {"real_transform_verified": True},
        "known_limitations": [],
        "evidence": {},
    }
    from compgen.graph_compilation.cost_preview_v2 import _validate

    fake_cas = {
        "candidates": [
            {
                "candidate_id": "cand_fake",
                "kind": "set_tile_params",
                "legality": {"ok": True},
            }
        ]
    }
    val = _validate(
        cost_previews=[fake_cp],
        candidate_actions=fake_cas,
        real_diff_report=None,  # no report exists
        selected_candidate_id="cand_fake",
    )
    cpv2r004 = next(c for c in val["checks"] if c["id"].startswith("CPV2R004"))
    assert cpv2r004["status"] == "fail"
    assert any("no M-12 report" in d for d in cpv2r004["details"])


def test_no_legal_candidate_left_without_cost_preview_v2() -> None:
    """Sanity: every legal candidate across the whole canonical suite
    has a cost preview."""
    _need_canonical()
    for model in _CANONICAL:
        cas = _read(
            SUITE / model / "02_graph_analysis" / "candidate_actions.json"
        )
        cp = _read(
            SUITE / model / "02_graph_analysis" / "cost_preview_v2.json"
        )
        cp_ids = {p["candidate_id"] for p in cp["cost_previews"]}
        for c in cas["candidates"]:
            if (c.get("legality") or {}).get("ok") is True:
                assert c["candidate_id"] in cp_ids, (
                    f"{model}: legal candidate {c['candidate_id']!r} missing"
                )


def test_no_compiler_core_imports_in_module() -> None:
    src = (
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "cost_preview_v2.py"
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
        assert pat not in src, f"cost_preview_v2 must not import: {pat}"
