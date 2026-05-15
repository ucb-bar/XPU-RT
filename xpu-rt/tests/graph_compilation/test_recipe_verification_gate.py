"""Acceptance tests for Recipe Verification Gate.

Asserts:
- Required gate artifacts emitted for all canonical models
- Every recipe op has a gate verdict, declared refinement, semantic obligation
- Family-specific checks fire for SetTileParams / FuseProducerConsumer /
  extension closure
- Negative paths reject stale, tampered, illegal recipes
- payload.mlir is not modified by the gate
- compiler core untouched
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from compgen.graph_compilation.recipe_gate import (
    run_recipe_gate,
)
from compgen.graph_compilation.run import run_graph_compilation

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_CPU_TARGET = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"

GATE_MODELS: tuple[str, ...] = (
    "tiny_mlp",
    "tiny_attention",
    "tiny_conv_block",
    "proxy_vlm",
    "proxy_vla",
    "custom_unsupported_op",
)


@pytest.fixture(scope="module")
def gate_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("gate_runs")
    out: dict[str, Path] = {}
    for model_id in GATE_MODELS:
        cfg = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
        run_dir = base / model_id
        run_graph_compilation(
            model_config_path=cfg,
            target_config_path=HOST_CPU_TARGET,
            out_dir=run_dir,
            stop_after="recipe-verification",
            run_id=f"gate_{model_id}",
            selection_mode="greedy",
        )
        out[model_id] = run_dir
    return out


# --------------------------------------------------------------------------- #
# Existence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", GATE_MODELS)
def test_gate_artifacts_emitted(
    model_id: str, gate_runs: dict[str, Path]
) -> None:
    rp = gate_runs[model_id] / "03_recipe_planning"
    for name in (
        "recipe_gate_verdict.json",
        "recipe_gate_trace.jsonl",
        "semantic_obligations.mlir",
        "semantic_obligations.json",
        "verified_recipe.mlir",
    ):
        p = rp / name
        assert p.exists() and p.stat().st_size > 0, f"{model_id}: missing/empty {name}"


@pytest.mark.parametrize("model_id", GATE_MODELS)
def test_gate_overall_pass(model_id: str, gate_runs: dict[str, Path]) -> None:
    obj = json.loads(
        (
            gate_runs[model_id]
            / "03_recipe_planning"
            / "recipe_gate_verdict.json"
        ).read_text()
    )
    assert obj["status"] == "pass", obj


@pytest.mark.parametrize("model_id", GATE_MODELS)
def test_artifact_validator_passes_with_gate(
    model_id: str, gate_runs: dict[str, Path]
) -> None:
    from compgen.graph_compilation import validate_run
    rep = validate_run(gate_runs[model_id])
    assert rep.overall == "pass", [r for r in rep.rules if r.status == "fail"]


# --------------------------------------------------------------------------- #
# Per-recipe-op shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", GATE_MODELS)
def test_every_recipe_op_has_verdict_refinement_and_obligation(
    model_id: str, gate_runs: dict[str, Path]
) -> None:
    obj = json.loads(
        (
            gate_runs[model_id]
            / "03_recipe_planning"
            / "recipe_gate_verdict.json"
        ).read_text()
    )
    assert obj["checked_recipe_ops"], model_id
    for op in obj["checked_recipe_ops"]:
        assert op["gate_status"] in ("pass", "fail"), op
        assert op["declared_refinement"], op
        assert op["semantic_obligation"], op
        assert isinstance(op["discharged_now"], list)
        assert isinstance(op["deferred_until_lowering"], list)


@pytest.mark.parametrize("model_id", GATE_MODELS)
def test_semantic_obligations_json_matches_mlir(
    model_id: str, gate_runs: dict[str, Path]
) -> None:
    rp = gate_runs[model_id] / "03_recipe_planning"
    obj = json.loads((rp / "semantic_obligations.json").read_text())
    mlir = (rp / "semantic_obligations.mlir").read_text()
    assert obj["schema_version"] == "semantic_obligations_v1"
    assert obj["obligations"], model_id
    for ob in obj["obligations"]:
        assert ob["id"] in mlir, (model_id, ob["id"])
        # Refinement and recipe_kind are recorded in both.
        assert ob["refinement"] in mlir
        assert ob["recipe_kind"] in mlir


@pytest.mark.parametrize("model_id", GATE_MODELS)
def test_verified_recipe_carries_gate_annotations(
    model_id: str, gate_runs: dict[str, Path]
) -> None:
    text = (
        gate_runs[model_id] / "03_recipe_planning" / "verified_recipe.mlir"
    ).read_text()
    assert 'recipe_gate_status = "pass"' in text, model_id
    assert "gate_status = " in text
    assert "declared_refinement = " in text
    assert "semantic_obligation = @obl_" in text


# --------------------------------------------------------------------------- #
# Family-specific checks fire on the canonical 6 models
# --------------------------------------------------------------------------- #


def test_set_tile_params_checks_working_set_membership(
    gate_runs: dict[str, Path]
) -> None:
    """Models that selected a SetTileParams candidate must record the
    working_set_curve membership discharge."""
    seen = False
    for run in gate_runs.values():
        obj = json.loads(
            (run / "03_recipe_planning" / "recipe_gate_verdict.json").read_text()
        )
        for op in obj["checked_recipe_ops"]:
            if op["op"] != "SetTileParams":
                continue
            assert "tile_exists_in_working_set_curve" in op["discharged_now"], op
            assert "working_set_fits_required_memory_tier" in op["discharged_now"], op
            seen = True
    assert seen, "no SetTileParams op selected across the suite"


def test_fuse_producer_consumer_checks_use_def_invariants(
    gate_runs: dict[str, Path]
) -> None:
    """Fusion picks must verify producer/consumer/edge/use-def discharge."""
    seen = False
    for run in gate_runs.values():
        obj = json.loads(
            (run / "03_recipe_planning" / "recipe_gate_verdict.json").read_text()
        )
        for op in obj["checked_recipe_ops"]:
            if op["op"] != "FuseProducerConsumer":
                continue
            d = set(op["discharged_now"])
            assert "producer_region_exists" in d
            assert "consumer_region_exists" in d
            assert "via_tensor_exists" in d
            assert "via_tensor_consumer_count_is_one" in d
            assert "via_tensor_is_transient" in d
            assert "region_graph_edge_exists" in d
            seen = True
    assert seen, "no FuseProducerConsumer op selected across the suite"


def test_extension_closure_declares_contract_obligation(
    gate_runs: dict[str, Path]
) -> None:
    """custom_unsupported_op selects a CreateKernelContract candidate.
    The gate must declare contract_obligation, not bit_equality."""
    obj = json.loads(
        (
            gate_runs["custom_unsupported_op"]
            / "03_recipe_planning"
            / "recipe_gate_verdict.json"
        ).read_text()
    )
    assert obj["checked_recipe_ops"]
    op = obj["checked_recipe_ops"][0]
    assert op["op"] in (
        "CreateKernelContract", "CreatePayloadLoweringExtension", "KeepAsFallback"
    )
    assert op["declared_refinement"] in (
        "contract_obligation", "extension_obligation", "fallback_obligation"
    )
    assert op["proof_stage"] != "post_lowering", (
        "extension closures should not declare post_lowering proof_stage"
    )


# --------------------------------------------------------------------------- #
# Trace + summary plumbing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", GATE_MODELS)
def test_recipe_validation_records_gate_overall(
    model_id: str, gate_runs: dict[str, Path]
) -> None:
    obj = json.loads(
        (
            gate_runs[model_id]
            / "03_recipe_planning"
            / "recipe_validation.json"
        ).read_text()
    )
    names = [c["name"] for c in obj["checks"]]
    assert "recipe_gate_overall" in names, names


@pytest.mark.parametrize("model_id", GATE_MODELS)
def test_recipe_summary_records_gate_status(
    model_id: str, gate_runs: dict[str, Path]
) -> None:
    obj = json.loads(
        (
            gate_runs[model_id]
            / "03_recipe_planning"
            / "recipe_summary.json"
        ).read_text()
    )
    assert obj["recipe_gate_status"] == "pass", model_id


@pytest.mark.parametrize("model_id", GATE_MODELS)
def test_trace_jsonl_well_formed(
    model_id: str, gate_runs: dict[str, Path]
) -> None:
    p = gate_runs[model_id] / "03_recipe_planning" / "recipe_gate_trace.jsonl"
    events: list[dict] = []
    with p.open() as f:
        for line in f:
            events.append(json.loads(line))
    assert events, model_id
    for ev in events:
        assert ev["schema_version"] == "recipe_gate_trace_event_v1"
        assert "recipe_op_id" in ev
        assert ev["status"] in ("pass", "fail")


# --------------------------------------------------------------------------- #
# Negative paths
# --------------------------------------------------------------------------- #


def _gate_run(work: Path) -> dict:
    """Helper: run gate, return verdict dict (or raise on internal error)."""
    run_recipe_gate(work)
    return json.loads(
        (work / "03_recipe_planning" / "recipe_gate_verdict.json").read_text()
    )


def test_tampered_recipe_source_candidate_fails(
    tmp_path: Path, gate_runs: dict[str, Path],
) -> None:
    """Replace source_candidate in recipe.mlir with a non-existent ID;
    gate must fail."""
    src = gate_runs["tiny_mlp"]
    work = tmp_path / "tampered_src"
    shutil.copytree(src, work)
    recipe_path = work / "03_recipe_planning" / "recipe.mlir"
    text = recipe_path.read_text()
    text = text.replace(
        'source_candidate = "cand_',
        'source_candidate = "cand_nonexistent_xxxx_',
    )
    recipe_path.write_text(text)
    verdict = _gate_run(work)
    assert verdict["status"] == "fail"
    assert any(
        "CandidateNotFoundError" in (r or "")
        for op in verdict["checked_recipe_ops"]
        for r in op["failure_reasons"]
    )


def test_tampered_recipe_tile_to_invented_size_fails(
    tmp_path: Path, gate_runs: dict[str, Path],
) -> None:
    """Mutate the tile to a size that's not in the working_set_curve."""
    src = gate_runs["tiny_mlp"]
    work = tmp_path / "tampered_tile"
    shutil.copytree(src, work)
    recipe_path = work / "03_recipe_planning" / "recipe.mlir"
    text = recipe_path.read_text()
    # Inject an invented tile dimension. Locate whatever ``M = N : i64``
    # value the planner picked (post-this varies — tiny_mlp's
    # shape-fit tile is M=4, while older runs used M=16) and bump it
    # to the invented 999.
    import re as _re
    m_match = _re.search(r"M\s*=\s*(\d+)\s*:\s*i64", text)
    assert m_match is not None, f"recipe.mlir has no M tile attr: {text}"
    actual_m = m_match.group(1)
    text = text.replace(f"M = {actual_m} : i64", "M = 999 : i64", 1)
    recipe_path.write_text(text)
    verdict = _gate_run(work)
    # Tampering source_candidate is unchanged so resolver still passes,
    # but the tile-membership check inside the family gate fails.
    if verdict["status"] == "pass":
        # If the tampered tile happened to round-trip somehow, surface for diagnostics.
        pytest.fail(f"expected failure, got: {verdict}")
    fr = verdict["checked_recipe_ops"][0]["failure_reasons"]
    assert any("not in working_set_curve" in (r or "") for r in fr), fr


def test_tampered_action_space_sha_fails(
    tmp_path: Path, gate_runs: dict[str, Path],
) -> None:
    """Tamper action_space_ir_sha256 in any projection; resolver detects."""
    src = gate_runs["tiny_mlp"]
    work = tmp_path / "tampered_sha"
    shutil.copytree(src, work)
    p = work / "02_graph_analysis" / "decision_sites.json"
    obj = json.loads(p.read_text())
    obj["source"]["action_space_ir_sha256"] = "sha256:" + "0" * 64
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))
    verdict = _gate_run(work)
    assert verdict["status"] == "fail"
    assert any(
        "HashMismatchError" in (r or "")
        for op in verdict["checked_recipe_ops"]
        for r in op["failure_reasons"]
    )


def test_tampered_fusion_via_tensor_fails(
    tmp_path: Path, gate_runs: dict[str, Path],
) -> None:
    """For models that selected fusion, tamper via_tensor in recipe.mlir."""
    src = gate_runs["proxy_vla"]  # proxy_vla selects FuseProducerConsumer
    work = tmp_path / "tampered_fuse"
    shutil.copytree(src, work)
    recipe_path = work / "03_recipe_planning" / "recipe.mlir"
    text = recipe_path.read_text()
    text = text.replace(
        'via_tensor = "export_program::',
        'via_tensor = "nonexistent_tensor::',
    )
    recipe_path.write_text(text)
    verdict = _gate_run(work)
    assert verdict["status"] == "fail"
    fr = verdict["checked_recipe_ops"][0]["failure_reasons"]
    # Either the tensor-not-found OR the edge-not-found check fires.
    assert any(
        "not in tensor_use_def_graph" in (r or "")
        or "no region_graph edge" in (r or "")
        for r in fr
    ), fr


def test_deleted_region_dossier_fails(
    tmp_path: Path, gate_runs: dict[str, Path],
) -> None:
    """Delete the dossier for the region the recipe targets."""
    src = gate_runs["tiny_mlp"]
    work = tmp_path / "deleted_dossier"
    shutil.copytree(src, work)
    sel = json.loads(
        (work / "03_recipe_planning" / "candidate_selection.json").read_text()
    )
    region_id = sel["region_id"]
    gd = json.loads((work / "02_graph_analysis" / "graph_dossier_v2.json").read_text())
    dossier_ref = gd["region_dossiers"][region_id]
    (work / dossier_ref).unlink()
    verdict = _gate_run(work)
    assert verdict["status"] == "fail"


# --------------------------------------------------------------------------- #
# Read-only against payload + compiler core
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", GATE_MODELS)
def test_gate_does_not_modify_payload_mlir(
    model_id: str, gate_runs: dict[str, Path], tmp_path: Path,
) -> None:
    """Re-running the gate must leave 01_payload_lowering/ byte-identical."""
    src = gate_runs[model_id]
    work = tmp_path / f"readonly_{model_id}"
    shutil.copytree(src, work)
    pl = work / "01_payload_lowering"
    before: dict[str, bytes] = {
        str(p.relative_to(pl)): p.read_bytes()
        for p in pl.rglob("*") if p.is_file()
    }
    run_recipe_gate(work)
    after: dict[str, bytes] = {
        str(p.relative_to(pl)): p.read_bytes()
        for p in pl.rglob("*") if p.is_file()
    }
    assert before == after, f"{model_id}: gate mutated 01_payload_lowering"


def test_compiler_core_not_modified_by_m06() -> None:
    import subprocess
    forbidden = [
        "python/compgen/ir/payload/import_fx.py",
        "python/compgen/capture/torch_export.py",
        "python/compgen/capture/torch_mlir_bridge.py",
        "python/compgen/pipeline/driver.py",
        "python/compgen/runtime/bundle_emit.py",
    ]
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD", "--"] + forbidden,
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        pytest.skip("git unavailable")
    changed = [line.strip() for line in diff.splitlines() if line.strip()]
    assert not changed, f"M-06 modified compiler core: {changed}"
