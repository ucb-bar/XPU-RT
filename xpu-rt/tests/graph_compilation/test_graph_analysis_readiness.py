"""Acceptance tests for M-17.1 Graph Analysis Readiness Lock.

Verifies:

- All 6 readiness artifacts + the top-level matrix + the summary
  markdown are emitted under ``02_graph_analysis/readiness/``.
- Five PNG figures land under ``readiness/figures/`` with valid magic
  bytes.
- The matrix overall is ``pass`` for the canonical/wide fixture suite.
- The 6 slide claims are each typed (``ready`` or ``ready_for_m18``).
- Each report's per-region invariants hold (precision budget, working
  set, reuse / lifetime, counterfactuals, agent view, hardware).
- The M-17 evidence pack ingests the readiness status (model matrix
  + aggregates + claim matrix entry).
- No compiler-core imports.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


_ALLOWED_DTYPE_STATUSES = {
    "safe", "risky", "exceeds_budget", "requires_reference",
}
_PRECISION_ORDER = ("fp32", "fast_math", "fp16_accum", "fp8_e4m3")
_DTYPE_KEYS = _PRECISION_ORDER


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _run(model: str, out_dir: Path, stop: str = "agent-decision-request") -> int:
    cmd = [
        sys.executable, "-m", "compgen.graph_compilation", "run",
        "--model", str(REPO_ROOT / f"configs/models/{model}.yaml"),
        "--target", str(REPO_ROOT / "configs/targets/host_cpu.yaml"),
        "--out", str(out_dir),
        "--stop-after", stop,
        "--selection-mode", "greedy",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return res.returncode


@pytest.fixture(scope="module")
def proxy_vla_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m171_proxy_vla") / "run"
    _run("proxy_vla", out)
    return out


@pytest.fixture(scope="module")
def merlin_mlp_wide_run(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("m171_merlin_mlp_wide") / "run"
    _run("merlin_mlp_wide", out)
    return out


# --------------------------------------------------------------------------- #
# All artifacts emit
# --------------------------------------------------------------------------- #


def test_all_readiness_artifacts_exist(proxy_vla_run: Path) -> None:
    base = proxy_vla_run / "02_graph_analysis" / "readiness"
    expected = [
        "graph_analysis_readiness_matrix.json",
        "graph_analysis_readiness_summary.md",
        "precision_budget_report.json",
        "working_set_fit_report.json",
        "reuse_lifetime_report.json",
        "candidate_counterfactual_report.json",
        "agent_view_completeness_report.json",
        "hardware_resource_report.json",
    ]
    for name in expected:
        assert (base / name).exists(), f"missing {name}"


def test_required_figures_exist_and_are_valid_pngs(proxy_vla_run: Path) -> None:
    figs_dir = proxy_vla_run / "02_graph_analysis" / "readiness" / "figures"
    expected = [
        "precision_budget_by_region.png",
        "working_set_fit_by_tile.png",
        "reuse_lifetime_histogram.png",
        "candidate_counterfactual_coverage.png",
        "bottleneck_by_region.png",
    ]
    for name in expected:
        path = figs_dir / name
        assert path.exists(), f"missing figure {name}"
        with path.open("rb") as f:
            head = f.read(8)
        assert head == b"\x89PNG\r\n\x1a\n", f"{name} not a valid PNG"


def test_readiness_matrix_overall_pass(proxy_vla_run: Path) -> None:
    m = _read(
        proxy_vla_run / "02_graph_analysis" / "readiness"
        / "graph_analysis_readiness_matrix.json"
    )
    assert m["overall"] == "pass"
    assert len(m["slide_rows"]) == 6
    statuses = [r["status"] for r in m["slide_rows"]]
    # Rows 1-5 must be "ready"; row 6 is "ready_for_m18".
    assert statuses[:5] == ["ready"] * 5
    assert statuses[5] == "ready_for_m18"


# --------------------------------------------------------------------------- #
# Precision budget
# --------------------------------------------------------------------------- #


def test_precision_report_covers_every_non_opaque_region(
    proxy_vla_run: Path,
) -> None:
    pb = _read(
        proxy_vla_run / "02_graph_analysis" / "readiness"
        / "precision_budget_report.json"
    )
    assert pb["status"] == "pass"
    for r in pb["regions"]:
        if r["is_opaque"]:
            continue
        for dt in _DTYPE_KEYS:
            ds = r["dtype_sensitivity"][dt]
            assert ds["status"] in _ALLOWED_DTYPE_STATUSES
            # eps_out, budget, budget_used_fraction are present.
            assert "eps_out" in ds
            assert "budget" in ds
            assert "budget_used_fraction" in ds


def test_precision_dtype_statuses_are_monotone(proxy_vla_run: Path) -> None:
    pb = _read(
        proxy_vla_run / "02_graph_analysis" / "readiness"
        / "precision_budget_report.json"
    )
    monotone_check = next(
        c for c in pb["checks"]
        if c["name"] == "monotone_precision_order"
    )
    assert monotone_check["status"] == "pass"


def test_opaque_regions_marked_requires_reference(proxy_vla_run: Path) -> None:
    pb = _read(
        proxy_vla_run / "02_graph_analysis" / "readiness"
        / "precision_budget_report.json"
    )
    opaque_regions = [r for r in pb["regions"] if r["is_opaque"]]
    if opaque_regions:
        for r in opaque_regions:
            for dt in _DTYPE_KEYS:
                assert r["dtype_sensitivity"][dt]["status"] == "requires_reference"


# --------------------------------------------------------------------------- #
# Working-set fit
# --------------------------------------------------------------------------- #


def test_working_set_report_covers_every_set_tile_candidate(
    merlin_mlp_wide_run: Path,
) -> None:
    """merlin_mlp_wide is the canonical SetTileParams discharge case;
    its working-set report must list every legal tile candidate."""
    ws = _read(
        merlin_mlp_wide_run / "02_graph_analysis" / "readiness"
        / "working_set_fit_report.json"
    )
    assert ws["status"] == "pass"
    cas = _read(
        merlin_mlp_wide_run / "02_graph_analysis" / "candidate_actions.json"
    )
    set_tile_legal_ids = {
        c["candidate_id"] for c in cas["candidates"]
        if c.get("kind") == "set_tile_params"
        and (c.get("legality") or {}).get("ok")
    }
    seen_ids: set[str] = set()
    for r in ws["regions"]:
        for t in r["candidate_tiles"]:
            if t["legality_ok"]:
                seen_ids.add(t["candidate_id"])
    missing = set_tile_legal_ids - seen_ids
    assert not missing, (
        f"working_set_fit_report missing {len(missing)} legal tile candidates"
    )


def test_working_set_has_scratchpad_fits_and_misses(
    merlin_mlp_wide_run: Path,
) -> None:
    ws = _read(
        merlin_mlp_wide_run / "02_graph_analysis" / "readiness"
        / "working_set_fit_report.json"
    )
    s = ws["summary"]
    assert s["any_tile_fits_scratchpad"]
    assert s["any_tile_misses_scratchpad"]


# --------------------------------------------------------------------------- #
# Reuse / lifetime
# --------------------------------------------------------------------------- #


def test_reuse_lifetime_covers_every_tensor(proxy_vla_run: Path) -> None:
    """Every tensor whose producer is a real region (i.e. NOT a
    pseudo-region like 'input'/'output' for graph boundaries) appears
    in the reuse_lifetime_report."""
    ru = _read(
        proxy_vla_run / "02_graph_analysis" / "readiness"
        / "reuse_lifetime_report.json"
    )
    assert ru["status"] == "pass"
    use_def = _read(
        proxy_vla_run / "02_graph_analysis" / "tensor_use_def_graph.json"
    )
    region_map = _read(
        proxy_vla_run / "02_graph_analysis" / "region_map.json"
    )
    region_ids = {r["region_id"] for r in region_map["regions"]}
    expected_count = len([
        t for t in use_def["tensors"]
        if t.get("producer_region") in region_ids
    ])
    seen = sum(
        1 for r in ru["regions"] for _ in r.get("outputs", []) or []
    )
    assert seen >= expected_count, (
        f"seen={seen} expected>={expected_count} "
        f"(only counting tensors whose producer is a real region, "
        f"not graph-input pseudo-regions)"
    )


def test_single_consumer_transients_identified(proxy_vla_run: Path) -> None:
    ru = _read(
        proxy_vla_run / "02_graph_analysis" / "readiness"
        / "reuse_lifetime_report.json"
    )
    assert ru["summary"]["single_consumer_transients_seen"]
    chk = next(
        c for c in ru["checks"]
        if c["name"] == "single_consumer_transients_identified"
    )
    assert chk["status"] == "pass"


def test_multi_consumer_values_not_marked_simple_fusible(
    proxy_vla_run: Path,
) -> None:
    ru = _read(
        proxy_vla_run / "02_graph_analysis" / "readiness"
        / "reuse_lifetime_report.json"
    )
    chk = next(
        c for c in ru["checks"]
        if c["name"] == "multi_consumer_values_not_marked_fusible"
    )
    assert chk["status"] == "pass"
    assert ru["summary"]["multi_consumer_marked_fusible_count"] == 0


# --------------------------------------------------------------------------- #
# Counterfactual
# --------------------------------------------------------------------------- #


def test_counterfactual_covers_every_candidate(proxy_vla_run: Path) -> None:
    cf = _read(
        proxy_vla_run / "02_graph_analysis" / "readiness"
        / "candidate_counterfactual_report.json"
    )
    assert cf["status"] == "pass"
    cas = _read(
        proxy_vla_run / "02_graph_analysis" / "candidate_actions.json"
    )
    assert cf["summary"]["candidate_count"] == len(cas["candidates"])
    assert cf["summary"]["with_recipe_delta"] == cf["summary"]["candidate_count"]
    assert cf["summary"]["with_legality"] == cf["summary"]["candidate_count"]


def test_every_legal_candidate_has_cost_preview(
    proxy_vla_run: Path,
) -> None:
    cf = _read(
        proxy_vla_run / "02_graph_analysis" / "readiness"
        / "candidate_counterfactual_report.json"
    )
    legal_with_cp = sum(
        1 for c in cf["candidates"]
        if (c.get("legality") or {}).get("ok") and "cost_preview_v2" in c
    )
    legal_total = sum(
        1 for c in cf["candidates"]
        if (c.get("legality") or {}).get("ok")
    )
    assert legal_with_cp == legal_total > 0


# --------------------------------------------------------------------------- #
# Agent view
# --------------------------------------------------------------------------- #


def test_llm_view_contains_zero_illegal_candidates(proxy_vla_run: Path) -> None:
    av = _read(
        proxy_vla_run / "02_graph_analysis" / "readiness"
        / "agent_view_completeness_report.json"
    )
    assert av["status"] == "pass"
    chk = next(
        c for c in av["checks"]
        if c["name"] == "view_contains_no_illegal_candidates"
    )
    assert chk["status"] == "pass"


def test_view_is_bounded(proxy_vla_run: Path) -> None:
    av = _read(
        proxy_vla_run / "02_graph_analysis" / "readiness"
        / "agent_view_completeness_report.json"
    )
    chk = next(c for c in av["checks"] if c["name"] == "view_is_bounded")
    assert chk["status"] == "pass"
    assert chk["max_regions"] >= 1
    assert chk["max_candidates_per_region"] >= 1


# --------------------------------------------------------------------------- #
# Hardware resource
# --------------------------------------------------------------------------- #


def test_hardware_report_calibration_is_explicit(proxy_vla_run: Path) -> None:
    hw = _read(
        proxy_vla_run / "02_graph_analysis" / "readiness"
        / "hardware_resource_report.json"
    )
    assert hw["calibration_status"] == "not_profiler_calibrated"
    chk = next(
        c for c in hw["checks"]
        if c["name"] == "calibration_status_explicitly_recorded"
    )
    assert chk["status"] == "pass"


def test_hardware_report_has_memory_bound_region(proxy_vla_run: Path) -> None:
    """proxy_vla's tiny matmuls/pointwise ops are all memory-bound under
    the deterministic roofline. ``any_memory_bound`` is a stable
    invariant across small models."""
    hw = _read(
        proxy_vla_run / "02_graph_analysis" / "readiness"
        / "hardware_resource_report.json"
    )
    assert hw["summary"]["any_memory_bound"]


def test_hardware_report_per_region_invariants(proxy_vla_run: Path) -> None:
    hw = _read(
        proxy_vla_run / "02_graph_analysis" / "readiness"
        / "hardware_resource_report.json"
    )
    for r in hw["regions"]:
        if r["is_opaque"]:
            continue
        assert r["estimated_latency_us"] is not None
        assert r["bottleneck_resource"] in ("compute", "memory", "balanced")
        assert "known_limitations" in r
        assert "not calibrated with profiler" in r["known_limitations"]


# --------------------------------------------------------------------------- #
# Evidence pack ingestion
# --------------------------------------------------------------------------- #


def test_evidence_pack_ingests_readiness(
    proxy_vla_run: Path, merlin_mlp_wide_run: Path, tmp_path: Path,
) -> None:
    """The M-17 evidence pack must surface readiness_overall in the
    model matrix + readiness counts in aggregates + a readiness claim
    in the claim matrix."""
    suite = tmp_path / "fixture_suite"
    canonical = suite / "canonical"
    wide = suite / "wide"
    canonical.mkdir(parents=True)
    wide.mkdir(parents=True)

    import shutil
    shutil.copytree(proxy_vla_run, canonical / "proxy_vla")
    shutil.copytree(merlin_mlp_wide_run, wide / "merlin_mlp_wide")

    pack_out = suite / "evidence_pack"

    from compgen.graph_compilation.evidence_pack import build_evidence_pack
    res = build_evidence_pack(
        canonical_suite_root=canonical, wide_suite_root=wide,
        out_dir=pack_out, skip_figures=True,
    )

    rows = list(csv.DictReader(
        (pack_out / "graph_section_model_matrix.csv").open(encoding="utf-8"),
    ))
    by_id = {r["model_id"]: r for r in rows}
    assert by_id["proxy_vla"]["readiness_overall"] == "pass"
    assert by_id["merlin_mlp_wide"]["readiness_overall"] == "pass"

    agg = _read(pack_out / "graph_section_evidence_tables.json")
    assert agg["readiness_pass_count"] >= 2
    assert agg["readiness_fail_count"] == 0

    cm = _read(res.claim_matrix)
    readiness_claim = next(
        c for c in cm["claims"]
        if "Graph analysis readiness lock" in c["claim"]
    )
    assert readiness_claim["status"] == "implemented"
    assert readiness_claim["observed_metric"]["readiness_pass_count"] >= 2


# --------------------------------------------------------------------------- #
# No compiler-core imports
# --------------------------------------------------------------------------- #


def test_readiness_module_does_not_import_compiler_core() -> None:
    forbidden = (
        "from compgen.ir",
        "import compgen.ir",
        "from compgen.capture",
        "import compgen.capture",
        "from compgen.pipeline",
        "import compgen.pipeline",
        "from compgen.runtime.bundle_emit",
    )
    for src_path in (
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "graph_analysis_readiness.py",
        REPO_ROOT / "python" / "compgen" / "graph_compilation"
        / "graph_analysis_readiness_figures.py",
    ):
        text = src_path.read_text(encoding="utf-8")
        for pat in forbidden:
            assert pat not in text, (
                f"{src_path.name} imports forbidden module: {pat}"
            )
