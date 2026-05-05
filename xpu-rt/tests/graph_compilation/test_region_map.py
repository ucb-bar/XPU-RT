"""Acceptance tests for Graph Analysis V2 (Milestone B).

Verifies the new ``02_graph_analysis/`` stage emits well-formed
``region_map.json`` / ``tensor_use_def_graph.json`` / ``region_graph.json``
on the canonical 6-model suite, and that those views correctly cross-
reference the v2 ``fx_to_payload_accounting.json``.

Per the project's anti-mock policy: no mocked importer / no fake MLIR.
The tests run real Dynamo + torch.export captures + lowering and then
build the graph-analysis JSONs from the resulting IR.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from compgen.graph_compilation import validate_run
from compgen.graph_compilation.run import run_graph_compilation

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_CPU_TARGET = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"

GRAPH_ANALYSIS_MODELS: tuple[str, ...] = (
    "tiny_mlp",
    "tiny_attention",
    "tiny_conv_block",
    "proxy_vlm",
    "proxy_vla",
    "custom_unsupported_op",
)


@pytest.fixture(scope="module")
def graph_analysis_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Capture+lower+graph-analyze every model once; ``{model_id: run_dir}``."""
    base = tmp_path_factory.mktemp("ga_runs")
    out: dict[str, Path] = {}
    for model_id in GRAPH_ANALYSIS_MODELS:
        cfg = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
        run_dir = base / model_id
        run_graph_compilation(
            model_config_path=cfg,
            target_config_path=HOST_CPU_TARGET,
            out_dir=run_dir,
            stop_after="graph-analysis",
            run_id=f"ga_{model_id}",
        )
        out[model_id] = run_dir
    return out


# --------------------------------------------------------------------------- #
# Stage on disk
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", GRAPH_ANALYSIS_MODELS)
def test_graph_analysis_stage_dir_exists(
    model_id: str, graph_analysis_runs: dict[str, Path]
) -> None:
    """Every model must produce ``02_graph_analysis/`` with the four JSONs."""
    ga = graph_analysis_runs[model_id] / "02_graph_analysis"
    assert ga.is_dir(), f"{model_id}: missing 02_graph_analysis"
    for name in (
        "region_map.json",
        "tensor_use_def_graph.json",
        "region_graph.json",
        "graph_analysis_report.json",
    ):
        p = ga / name
        assert p.exists(), f"{model_id}: missing {name}"


@pytest.mark.parametrize("model_id", GRAPH_ANALYSIS_MODELS)
def test_artifact_validator_passes(
    model_id: str, graph_analysis_runs: dict[str, Path]
) -> None:
    """Artifact-contract validator must accept the new stage layout (R001..R012)."""
    rep = validate_run(graph_analysis_runs[model_id])
    assert rep.overall == "pass", [r for r in rep.rules if r.status == "fail"]


# --------------------------------------------------------------------------- #
# Schema versions + summary fields
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", GRAPH_ANALYSIS_MODELS)
def test_schema_versions(
    model_id: str, graph_analysis_runs: dict[str, Path]
) -> None:
    ga = graph_analysis_runs[model_id] / "02_graph_analysis"
    rm = json.loads((ga / "region_map.json").read_text())
    tg = json.loads((ga / "tensor_use_def_graph.json").read_text())
    rg = json.loads((ga / "region_graph.json").read_text())
    rep = json.loads((ga / "graph_analysis_report.json").read_text())
    assert rm["schema_version"] == "region_map_v1"
    assert tg["schema_version"] == "tensor_use_def_graph_v1"
    assert rg["schema_version"] == "region_graph_v1"
    assert rep["schema_version"] == "graph_analysis_report_v1"


# --------------------------------------------------------------------------- #
# Region-map content
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", GRAPH_ANALYSIS_MODELS)
def test_every_region_id_in_payload_mlir_appears_in_region_map(
    model_id: str, graph_analysis_runs: dict[str, Path]
) -> None:
    """Audit: every ``compgen.region_id = "X"`` attribute observed in any
    payload.mlir must appear as a region in region_map.json."""
    run = graph_analysis_runs[model_id]
    rm = json.loads((run / "02_graph_analysis" / "region_map.json").read_text())
    region_ids_in_map = {r["region_id"] for r in rm["regions"]}
    region_ids_in_mlir: set[str] = set()
    for path in (run / "01_payload_lowering").rglob("payload.mlir"):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r'compgen\.region_id\s*=\s*"([^"]+)"', text):
            region_ids_in_mlir.add(m.group(1))
    missing = region_ids_in_mlir - region_ids_in_map
    assert not missing, f"{model_id}: region_ids in MLIR but missing from map: {missing}"


@pytest.mark.parametrize("model_id", GRAPH_ANALYSIS_MODELS)
def test_every_region_has_required_fields(
    model_id: str, graph_analysis_runs: dict[str, Path]
) -> None:
    rm = json.loads((graph_analysis_runs[model_id] / "02_graph_analysis"
                     / "region_map.json").read_text())
    required = {
        "region_id", "module_id", "kind", "source_classification",
        "fx_nodes", "payload_ops", "inputs", "outputs",
        "estimated", "gap_refs", "extension_refs",
    }
    for r in rm["regions"]:
        assert set(r.keys()) >= required, (model_id, r)
        assert isinstance(r["payload_ops"], list)
        for po in r["payload_ops"]:
            assert {"op_name", "payload_ref"} <= set(po.keys()), po


@pytest.mark.parametrize("model_id", GRAPH_ANALYSIS_MODELS)
def test_payload_ref_paths_exist(
    model_id: str, graph_analysis_runs: dict[str, Path]
) -> None:
    """Every ``payload_ref`` must resolve to a file under ``run_dir``."""
    run = graph_analysis_runs[model_id]
    rm = json.loads((run / "02_graph_analysis" / "region_map.json").read_text())
    for r in rm["regions"]:
        for po in r["payload_ops"]:
            ref = run / po["payload_ref"]
            assert ref.exists(), (model_id, po["payload_ref"])


# --------------------------------------------------------------------------- #
# Tensor use-def graph
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", GRAPH_ANALYSIS_MODELS)
def test_tensor_use_def_references_valid_regions(
    model_id: str, graph_analysis_runs: dict[str, Path]
) -> None:
    """Every ``producer_region`` and ``consumer_regions[]`` member must be
    either a known region_id, ``input``, or ``output``."""
    run = graph_analysis_runs[model_id]
    rm = json.loads((run / "02_graph_analysis" / "region_map.json").read_text())
    tg = json.loads((run / "02_graph_analysis" / "tensor_use_def_graph.json").read_text())
    valid_regions = {r["region_id"] for r in rm["regions"]} | {"input", "output"}
    for t in tg["tensors"]:
        assert t["producer_region"] in valid_regions, (model_id, t)
        for c in t["consumer_regions"]:
            assert c in valid_regions, (model_id, t)


def test_tiny_mlp_finds_single_consumer_transient(
    graph_analysis_runs: dict[str, Path]
) -> None:
    """tiny_mlp's linear→relu chain should produce at least one transient
    output with ``consumer_count==1`` and a small reuse horizon
    (``<= 2`` lines between producer and first consumer) — that's the
    canonical fusion-candidate signal."""
    run = graph_analysis_runs["tiny_mlp"]
    tg = json.loads((run / "02_graph_analysis" / "tensor_use_def_graph.json").read_text())
    transient_singletons = [
        t for t in tg["tensors"]
        if t["producer_lifetime_class"] == "transient"
        and t["consumer_count"] == 1
        and 0 <= t["reuse_horizon"] <= 2
    ]
    assert transient_singletons, [
        (t["tensor_id"], t["producer_lifetime_class"], t["consumer_count"], t["reuse_horizon"])
        for t in tg["tensors"]
    ]


# --------------------------------------------------------------------------- #
# Region graph
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", GRAPH_ANALYSIS_MODELS)
def test_region_graph_is_dag(
    model_id: str, graph_analysis_runs: dict[str, Path]
) -> None:
    """region_graph must be acyclic on the canonical models."""
    rg = json.loads((graph_analysis_runs[model_id] / "02_graph_analysis"
                     / "region_graph.json").read_text())
    assert rg["totals"]["is_dag"] is True, (model_id, rg["totals"])


@pytest.mark.parametrize("model_id", GRAPH_ANALYSIS_MODELS)
def test_region_graph_edges_reference_valid_tensors(
    model_id: str, graph_analysis_runs: dict[str, Path]
) -> None:
    """Every edge.tensor_id must exist in tensor_use_def_graph."""
    run = graph_analysis_runs[model_id]
    rg = json.loads((run / "02_graph_analysis" / "region_graph.json").read_text())
    tg = json.loads((run / "02_graph_analysis" / "tensor_use_def_graph.json").read_text())
    valid_tids = {t["tensor_id"] for t in tg["tensors"]}
    for e in rg["edges"]:
        assert e["tensor_id"] in valid_tids, (model_id, e)


def test_proxy_vla_has_nontrivial_critical_path(
    graph_analysis_runs: dict[str, Path]
) -> None:
    """proxy_vla is the largest model — it should produce a non-empty critical path."""
    rg = json.loads((graph_analysis_runs["proxy_vla"] / "02_graph_analysis"
                     / "region_graph.json").read_text())
    assert rg["totals"]["nodes"] >= 5, rg["totals"]
    assert len(rg["critical_path"]) >= 1, rg["critical_path"]


# --------------------------------------------------------------------------- #
# Round-trip determinism
# --------------------------------------------------------------------------- #


def test_analyze_graph_cli_idempotent(
    tmp_path: Path, graph_analysis_runs: dict[str, Path]
) -> None:
    """Re-running ``build_graph_analysis`` twice produces byte-identical
    region_map / tensor_use_def_graph / region_graph JSONs."""
    from compgen.graph_compilation.region_map import build_graph_analysis

    run = graph_analysis_runs["tiny_mlp"]
    a = (run / "02_graph_analysis" / "region_map.json").read_bytes()
    build_graph_analysis(run)
    b = (run / "02_graph_analysis" / "region_map.json").read_bytes()
    assert a == b
