"""Acceptance tests for Region Dossier V2 MVP (Milestone 03).

Asserts that the canonical 6-model suite produces decision-quality
per-region dossiers — every region has cost / reuse / numerical /
working-set / placement / legality fields with realistic values, and
the suite-wide acceptance bars from the user spec are met:

- at least one compute-bound region
- at least one memory-bound region
- at least one FP8 numerical setting flagged as risky/exceeds_budget
- at least one tile fits scratchpad/L2 and at least one does not
- single-consumer transient outputs detected
- opaque regions marked requires_reference_or_extension
- graph_analysis.mlir exists and reflects every region/tensor
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compgen.graph_compilation.run import run_graph_compilation

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_CPU_TARGET = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"

DOSSIER_MODELS: tuple[str, ...] = (
    "tiny_mlp",
    "tiny_attention",
    "tiny_conv_block",
    "proxy_vlm",
    "proxy_vla",
    "custom_unsupported_op",
)


@pytest.fixture(scope="module")
def dossier_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("dossier_runs")
    out: dict[str, Path] = {}
    for model_id in DOSSIER_MODELS:
        cfg = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
        run_dir = base / model_id
        run_graph_compilation(
            model_config_path=cfg,
            target_config_path=HOST_CPU_TARGET,
            out_dir=run_dir,
            stop_after="graph-analysis",
            run_id=f"dossier_{model_id}",
        )
        out[model_id] = run_dir
    return out


# --------------------------------------------------------------------------- #
# Per-model existence / shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", DOSSIER_MODELS)
def test_required_artifacts_emitted(
    model_id: str, dossier_runs: dict[str, Path]
) -> None:
    ga = dossier_runs[model_id] / "02_graph_analysis"
    for name in (
        "graph_analysis.mlir",
        "graph_dossier_v2.json",
        "dossier_validation.json",
    ):
        p = ga / name
        assert p.exists(), f"{model_id}: missing {name}"
        assert p.stat().st_size > 0, f"{model_id}: empty {name}"
    region_dossier_dir = ga / "region_dossiers"
    assert region_dossier_dir.is_dir(), f"{model_id}: region_dossiers/ missing"
    assert any(region_dossier_dir.glob("*.json")), f"{model_id}: no region dossiers"


@pytest.mark.parametrize("model_id", DOSSIER_MODELS)
def test_dossier_validation_pass(
    model_id: str, dossier_runs: dict[str, Path]
) -> None:
    p = dossier_runs[model_id] / "02_graph_analysis" / "dossier_validation.json"
    obj = json.loads(p.read_text())
    failed = [c for c in obj["checks"] if c["status"] == "fail"]
    assert obj["overall"] == "pass", failed


@pytest.mark.parametrize("model_id", DOSSIER_MODELS)
def test_every_region_has_dossier(
    model_id: str, dossier_runs: dict[str, Path]
) -> None:
    run = dossier_runs[model_id]
    rm = json.loads((run / "02_graph_analysis" / "region_map.json").read_text())
    gd = json.loads((run / "02_graph_analysis" / "graph_dossier_v2.json").read_text())
    rm_ids = {r["region_id"] for r in rm["regions"]}
    dossier_ids = set(gd["region_dossiers"].keys())
    assert rm_ids == dossier_ids, (model_id, rm_ids ^ dossier_ids)
    # Every dossier file referenced exists.
    for rid, ref in gd["region_dossiers"].items():
        full = run / ref
        assert full.exists(), (model_id, rid, ref)


@pytest.mark.parametrize("model_id", DOSSIER_MODELS)
def test_per_region_dossier_has_required_fields(
    model_id: str, dossier_runs: dict[str, Path]
) -> None:
    run = dossier_runs[model_id]
    gd = json.loads((run / "02_graph_analysis" / "graph_dossier_v2.json").read_text())
    required_top = {
        "schema_version", "region_id", "module_id", "kind",
        "source", "cost", "reuse", "numerical_sensitivity",
        "working_set_curve", "placement_envelope", "legality_constraints",
    }
    required_source = {"fx_nodes", "fx_targets", "payload_ops", "source_classification"}
    required_cost = {"flops", "bytes", "arithmetic_intensity",
                     "estimated_latency_us", "bottleneck_resource"}
    required_numerics = {"fp32", "fp16_accum", "fp8_e4m3", "fast_math"}
    for ref in gd["region_dossiers"].values():
        d = json.loads((run / ref).read_text())
        assert d["schema_version"] == "region_dossier_v2", (model_id, ref)
        assert set(d.keys()) >= required_top, (model_id, set(d.keys()))
        assert set(d["source"].keys()) >= required_source, (model_id, d["source"])
        assert set(d["cost"].keys()) >= required_cost, (model_id, d["cost"])
        assert set(d["numerical_sensitivity"].keys()) == required_numerics, (
            model_id, d["numerical_sensitivity"]
        )
        # Reuse fields shape
        for port in d["reuse"]["inputs"] + d["reuse"]["outputs"]:
            assert {"tensor_id", "consumer_count", "reuse_horizon", "lifetime_class"} <= set(port.keys())


@pytest.mark.parametrize("model_id", DOSSIER_MODELS)
def test_every_payload_ref_resolves(
    model_id: str, dossier_runs: dict[str, Path]
) -> None:
    run = dossier_runs[model_id]
    gd = json.loads((run / "02_graph_analysis" / "graph_dossier_v2.json").read_text())
    for ref in gd["region_dossiers"].values():
        d = json.loads((run / ref).read_text())
        for po in d["source"]["payload_ops"]:
            full = run / po["payload_ref"]
            assert full.exists(), (model_id, po)


@pytest.mark.parametrize("model_id", DOSSIER_MODELS)
def test_matmul_like_regions_have_working_set_curve(
    model_id: str, dossier_runs: dict[str, Path]
) -> None:
    """Acceptance: matmul/conv regions must have a non-empty working_set_curve."""
    run = dossier_runs[model_id]
    gd = json.loads((run / "02_graph_analysis" / "graph_dossier_v2.json").read_text())
    for ref in gd["region_dossiers"].values():
        d = json.loads((run / ref).read_text())
        if d["kind"] in ("matmul", "conv"):
            assert d["working_set_curve"], (model_id, d["region_id"])


@pytest.mark.parametrize("model_id", DOSSIER_MODELS)
def test_opaque_regions_marked_requires_reference(
    model_id: str, dossier_runs: dict[str, Path]
) -> None:
    """Acceptance: opaque_<…> regions must carry the
    requires_reference_or_extension legality with ok=False, and have
    fp32/fp16/fp8/fast_math all status='requires_reference'."""
    run = dossier_runs[model_id]
    gd = json.loads((run / "02_graph_analysis" / "graph_dossier_v2.json").read_text())
    for ref in gd["region_dossiers"].values():
        d = json.loads((run / ref).read_text())
        if not d["kind"].startswith("opaque_"):
            continue
        constraints = {c["constraint"]: c for c in d["legality_constraints"]}
        assert "requires_reference_or_extension" in constraints, (model_id, d["region_id"])
        assert constraints["requires_reference_or_extension"]["ok"] is False
        for dt in ("fp32", "fp16_accum", "fp8_e4m3", "fast_math"):
            assert d["numerical_sensitivity"][dt]["status"] == "requires_reference", (
                model_id, d["region_id"], dt
            )


@pytest.mark.parametrize("model_id", DOSSIER_MODELS)
def test_graph_analysis_mlir_mentions_every_region_and_tensor(
    model_id: str, dossier_runs: dict[str, Path]
) -> None:
    run = dossier_runs[model_id]
    mlir = (run / "02_graph_analysis" / "graph_analysis.mlir").read_text()
    rm = json.loads((run / "02_graph_analysis" / "region_map.json").read_text())
    tg = json.loads((run / "02_graph_analysis" / "tensor_use_def_graph.json").read_text())
    for r in rm["regions"]:
        assert f'region_id = "{r["region_id"]}"' in mlir, (
            model_id, r["region_id"]
        )
    for t in tg["tensors"]:
        assert f'tensor_id = "{t["tensor_id"]}"' in mlir, (
            model_id, t["tensor_id"]
        )


# --------------------------------------------------------------------------- #
# Suite-level acceptance bars (across all 6 models)
# --------------------------------------------------------------------------- #


def test_suite_has_compute_and_memory_bound_regions(
    dossier_runs: dict[str, Path]
) -> None:
    compute = memory = 0
    for run in dossier_runs.values():
        v = json.loads(
            (run / "02_graph_analysis" / "dossier_validation.json").read_text()
        )
        compute += v["totals"]["bottleneck_compute_count"]
        memory += v["totals"]["bottleneck_memory_count"]
    assert compute >= 1, f"no compute-bound regions across suite (got {compute})"
    assert memory >= 1, f"no memory-bound regions across suite (got {memory})"


def test_suite_has_fp8_exceeds_budget_or_risky(
    dossier_runs: dict[str, Path]
) -> None:
    flagged = 0
    for run in dossier_runs.values():
        v = json.loads(
            (run / "02_graph_analysis" / "dossier_validation.json").read_text()
        )
        hist = v["totals"]["fp8_status_histogram"]
        flagged += hist.get("exceeds_budget", 0) + hist.get("risky", 0)
    assert flagged >= 1, f"no fp8 regions flagged (got {flagged})"


def test_suite_has_tile_that_fits_and_one_that_does_not(
    dossier_runs: dict[str, Path]
) -> None:
    fits_any = not_fits_any = False
    for run in dossier_runs.values():
        v = json.loads(
            (run / "02_graph_analysis" / "dossier_validation.json").read_text()
        )
        fits_any |= v["totals"]["fits_scratchpad_any"]
        not_fits_any |= v["totals"]["not_fits_scratchpad_any"]
    assert fits_any
    assert not_fits_any


def test_suite_detects_single_consumer_transient_output(
    dossier_runs: dict[str, Path]
) -> None:
    seen = False
    for run in dossier_runs.values():
        v = json.loads(
            (run / "02_graph_analysis" / "dossier_validation.json").read_text()
        )
        seen |= v["totals"]["single_consumer_transient_seen"]
    assert seen, "expected at least one single-consumer transient output across suite"


def test_numerical_sensitivity_varies_across_regions(
    dossier_runs: dict[str, Path]
) -> None:
    """Acceptance: dossiers must NOT have identical numerical_sensitivity
    across all regions. Reject hardcoded constant outputs."""
    eps_values: set[float] = set()
    for run in dossier_runs.values():
        gd = json.loads(
            (run / "02_graph_analysis" / "graph_dossier_v2.json").read_text()
        )
        for ref in gd["region_dossiers"].values():
            d = json.loads((run / ref).read_text())
            eps_values.add(d["numerical_sensitivity"]["fp16_accum"]["eps_out"])
    assert len(eps_values) >= 2, (
        f"fp16_accum eps_out is constant across all regions ({eps_values})"
    )


def test_graph_dossier_critical_path_resolves(
    dossier_runs: dict[str, Path]
) -> None:
    """Every region named in graph_dossier.critical_path must exist in
    region_dossiers (or be the empty list when the region graph is
    trivial)."""
    for run in dossier_runs.values():
        gd = json.loads(
            (run / "02_graph_analysis" / "graph_dossier_v2.json").read_text()
        )
        dossier_ids = set(gd["region_dossiers"].keys())
        for rid in gd["critical_path"]:
            assert rid in dossier_ids, (run, rid, dossier_ids)


def test_top_regions_link_to_existing_dossiers(
    dossier_runs: dict[str, Path]
) -> None:
    for run in dossier_runs.values():
        gd = json.loads(
            (run / "02_graph_analysis" / "graph_dossier_v2.json").read_text()
        )
        for top in gd["top_regions_by_estimated_latency"]:
            ref = top["dossier_ref"]
            full = run / ref
            assert full.exists(), (run, ref)


# --------------------------------------------------------------------------- #
# M-03.5 — Numerical Sensitivity Sanity Audit
# --------------------------------------------------------------------------- #


_STATUS_RANK = {"safe": 0, "risky": 1, "exceeds_budget": 2, "requires_reference": 3}


@pytest.mark.parametrize("model_id", DOSSIER_MODELS)
def test_numerical_sensitivity_audit_emitted(
    model_id: str, dossier_runs: dict[str, Path]
) -> None:
    p = (
        dossier_runs[model_id]
        / "02_graph_analysis"
        / "numerical_sensitivity_audit.json"
    )
    assert p.exists()
    obj = json.loads(p.read_text())
    assert obj["schema_version"] == "numerical_sensitivity_audit_v1"
    assert "checks" in obj and "violations" in obj


@pytest.mark.parametrize("model_id", DOSSIER_MODELS)
def test_numerical_sensitivity_audit_passes(
    model_id: str, dossier_runs: dict[str, Path]
) -> None:
    p = (
        dossier_runs[model_id]
        / "02_graph_analysis"
        / "numerical_sensitivity_audit.json"
    )
    obj = json.loads(p.read_text())
    assert obj["status"] == "pass", obj["violations"]


@pytest.mark.parametrize("model_id", DOSSIER_MODELS)
def test_no_fp32_marked_less_safe_than_fast_math(
    model_id: str, dossier_runs: dict[str, Path]
) -> None:
    """The exact bug the user flagged: fp32 must never be classified as
    less safe than fast_math for the same region."""
    run = dossier_runs[model_id]
    gd = json.loads((run / "02_graph_analysis" / "graph_dossier_v2.json").read_text())
    for ref in gd["region_dossiers"].values():
        d = json.loads((run / ref).read_text())
        sens = d["numerical_sensitivity"]
        if sens["fp32"]["status"] == "requires_reference":
            continue
        assert _STATUS_RANK[sens["fp32"]["status"]] <= _STATUS_RANK[
            sens["fast_math"]["status"]
        ], (model_id, d["region_id"], sens["fp32"], sens["fast_math"])


@pytest.mark.parametrize("model_id", DOSSIER_MODELS)
def test_precision_order_monotone_per_region(
    model_id: str, dossier_runs: dict[str, Path]
) -> None:
    """For every non-opaque region: fp32 ≤ fast_math ≤ fp16_accum ≤ fp8_e4m3
    in status rank (safer → less safe)."""
    run = dossier_runs[model_id]
    gd = json.loads((run / "02_graph_analysis" / "graph_dossier_v2.json").read_text())
    order = ("fp32", "fast_math", "fp16_accum", "fp8_e4m3")
    for ref in gd["region_dossiers"].values():
        d = json.loads((run / ref).read_text())
        if d["kind"].startswith("opaque_"):
            continue
        sens = d["numerical_sensitivity"]
        if sens["fp32"]["status"] == "requires_reference":
            continue
        ranks = [_STATUS_RANK[sens[k]["status"]] for k in order]
        for i in range(len(ranks) - 1):
            assert ranks[i] <= ranks[i + 1], (model_id, d["region_id"], dict(zip(order, ranks)))


@pytest.mark.parametrize("model_id", DOSSIER_MODELS)
def test_dossier_validation_includes_numerical_audit(
    model_id: str, dossier_runs: dict[str, Path]
) -> None:
    """dossier_validation.json must reflect the numerical_sensitivity audit
    as one of its top-level checks."""
    p = dossier_runs[model_id] / "02_graph_analysis" / "dossier_validation.json"
    obj = json.loads(p.read_text())
    names = [c["name"] for c in obj["checks"]]
    assert "numerical_sensitivity_audit" in names, names
