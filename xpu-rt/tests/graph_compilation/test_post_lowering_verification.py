"""Acceptance tests for M-08: Post-Lowering Verification.

Asserts:
- transformed_payload.mlir is emitted for transform-like recipes only
- contract_structural_validation.json is emitted for contract-only recipes
- source 01_payload_lowering/ tree is byte-identical pre/post
- transformed_payload.mlir lives ONLY under 03_recipe_planning/post_lowering/
- Family-specific metadata is injected on the right anchor op
- Semantic obligations are recorded as partially_discharged_structural /
  pending_kernel_contract_generation — no full discharge claimed
- Negative paths reject tampered transforms, opaque tile targets,
  source mutation, contract producing transformed_payload, etc.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from compgen.graph_compilation.post_lowering import (
    run_post_lowering_verification,
)
from compgen.graph_compilation.run import run_graph_compilation

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_CPU_TARGET = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"

POST_LOWERING_MODELS: tuple[str, ...] = (
    "tiny_mlp",
    "tiny_attention",
    "tiny_conv_block",
    "proxy_vlm",
    "proxy_vla",
    "custom_unsupported_op",
)


def _sha256_tree(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        h.update(str(p.relative_to(path)).encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\n")
    return h.hexdigest()


@pytest.fixture(scope="module")
def post_lowering_runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("post_lowering_runs")
    out: dict[str, Path] = {}
    for model_id in POST_LOWERING_MODELS:
        cfg = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
        run_dir = base / model_id
        run_graph_compilation(
            model_config_path=cfg,
            target_config_path=HOST_CPU_TARGET,
            out_dir=run_dir,
            stop_after="post-lowering-verification",
            run_id=f"plv_{model_id}",
            selection_mode="greedy",
        )
        out[model_id] = run_dir
    return out


# --------------------------------------------------------------------------- #
# Existence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", POST_LOWERING_MODELS)
def test_required_post_lowering_artifacts_emitted(
    model_id: str, post_lowering_runs: dict[str, Path]
) -> None:
    pl = post_lowering_runs[model_id] / "03_recipe_planning" / "post_lowering"
    assert pl.is_dir()
    for name in (
        "post_lowering_verification_report.json",
        "semantic_obligations_status.json",
    ):
        p = pl / name
        assert p.exists() and p.stat().st_size > 0, f"{model_id}: missing/empty {name}"


@pytest.mark.parametrize("model_id", POST_LOWERING_MODELS)
def test_post_lowering_overall_pass(
    model_id: str, post_lowering_runs: dict[str, Path]
) -> None:
    obj = json.loads(
        (
            post_lowering_runs[model_id] / "03_recipe_planning" / "post_lowering"
            / "post_lowering_verification_report.json"
        ).read_text()
    )
    failed = [c for c in obj["checks"] if c["status"] == "fail"]
    assert obj["status"] == "pass", failed


@pytest.mark.parametrize("model_id", POST_LOWERING_MODELS)
def test_artifact_validator_passes(
    model_id: str, post_lowering_runs: dict[str, Path]
) -> None:
    from compgen.graph_compilation import validate_run
    rep = validate_run(post_lowering_runs[model_id])
    assert rep.overall == "pass", [r for r in rep.rules if r.status == "fail"]


# --------------------------------------------------------------------------- #
# Family-specific routing
# --------------------------------------------------------------------------- #


def test_set_tile_params_emits_transformed_payload(
    post_lowering_runs: dict[str, Path],
) -> None:
    seen = False
    for run in post_lowering_runs.values():
        sel = json.loads(
            (run / "03_recipe_planning" / "candidate_selection.json").read_text()
        )
        if sel["candidate_kind"] != "set_tile_params":
            continue
        tp = run / "03_recipe_planning" / "post_lowering" / "transformed_payload.mlir"
        assert tp.exists(), run
        text = tp.read_text()
        assert "compgen.tile = [" in text
        assert "compgen.recipe_op = " in text
        assert "compgen.semantic_obligation = " in text
        # Anchored to a linalg.matmul op
        assert "linalg.matmul" in text
        seen = True
    assert seen


def test_fuse_producer_consumer_emits_metadata_on_consumer(
    post_lowering_runs: dict[str, Path],
) -> None:
    seen = False
    for run in post_lowering_runs.values():
        sel = json.loads(
            (run / "03_recipe_planning" / "candidate_selection.json").read_text()
        )
        if sel["candidate_kind"] != "fuse_producer_consumer":
            continue
        tp = run / "03_recipe_planning" / "post_lowering" / "transformed_payload.mlir"
        assert tp.exists(), run
        text = tp.read_text()
        assert "compgen.fuse_producer = " in text
        assert "compgen.fuse_consumer = " in text
        assert "compgen.fuse_via_tensor = " in text
        seen = True
    assert seen


def test_create_kernel_contract_does_not_emit_transformed_payload(
    post_lowering_runs: dict[str, Path],
) -> None:
    run = post_lowering_runs["custom_unsupported_op"]
    tp = run / "03_recipe_planning" / "post_lowering" / "transformed_payload.mlir"
    assert not tp.exists(), tp
    cv = run / "03_recipe_planning" / "post_lowering" / "contract_structural_validation.json"
    assert cv.exists()
    obj = json.loads(cv.read_text())
    assert obj["status"] == "pass"
    val = obj["validations"][0]
    assert val["recipe_kind"] == "CreateKernelContract"
    assert val["proof_stage"] == "kernel_contract_generation"
    for k in (
        "contract_draft_exists",
        "references_semantic_obligation",
        "references_opaque_region",
        "proof_stage_is_kernel_contract_generation",
    ):
        assert val["checks"][k] is True, val


# --------------------------------------------------------------------------- #
# Hard invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", POST_LOWERING_MODELS)
def test_source_payload_byte_identical(
    model_id: str, post_lowering_runs: dict[str, Path], tmp_path: Path,
) -> None:
    """Re-run M-08 on a copy and verify ``01_payload_lowering/`` is
    byte-identical afterwards."""
    src = post_lowering_runs[model_id]
    work = tmp_path / f"readonly_{model_id}"
    shutil.copytree(src, work)
    pl = work / "01_payload_lowering"
    before = _sha256_tree(pl)
    run_post_lowering_verification(work)
    after = _sha256_tree(pl)
    assert before == after, f"{model_id}: source payload tree changed"


@pytest.mark.parametrize("model_id", POST_LOWERING_MODELS)
def test_transformed_payload_not_under_01_payload_lowering(
    model_id: str, post_lowering_runs: dict[str, Path]
) -> None:
    pl = post_lowering_runs[model_id] / "01_payload_lowering"
    leaks = list(pl.rglob("transformed_payload*"))
    assert not leaks, leaks


@pytest.mark.parametrize("model_id", POST_LOWERING_MODELS)
def test_no_full_differential_discharge_claimed(
    model_id: str, post_lowering_runs: dict[str, Path]
) -> None:
    obj = json.loads(
        (
            post_lowering_runs[model_id] / "03_recipe_planning" / "post_lowering"
            / "semantic_obligations_status.json"
        ).read_text()
    )
    for s in obj["statuses"]:
        if s["declared_refinement"] in ("bit_equality", "tolerance_eps"):
            assert s["status"] == "partially_discharged_structural", s
            assert "differential_check" in s["remaining"], s
        elif s["declared_refinement"] == "contract_obligation":
            assert s["status"] == "pending_kernel_contract_generation", s


# --------------------------------------------------------------------------- #
# Manifest + diff coherence
# --------------------------------------------------------------------------- #


def test_applied_transform_manifest_records_exactly_one_recipe(
    post_lowering_runs: dict[str, Path],
) -> None:
    for model_id, run in post_lowering_runs.items():
        if model_id == "custom_unsupported_op":
            continue
        m = json.loads(
            (
                run / "03_recipe_planning" / "post_lowering"
                / "applied_transform_manifest.json"
            ).read_text()
        )
        assert len(m["applied"]) == 1, (model_id, m)
        applied = m["applied"][0]
        assert applied["status"] == "applied"
        assert applied["application_mode"] == "metadata_only_structural_mvp"
        assert applied["recipe_op_id"] == "recipe_0000"


def test_structural_diff_records_added_attributes(
    post_lowering_runs: dict[str, Path],
) -> None:
    for model_id, run in post_lowering_runs.items():
        if model_id == "custom_unsupported_op":
            continue
        d = json.loads(
            (
                run / "03_recipe_planning" / "post_lowering"
                / "structural_diff.json"
            ).read_text()
        )
        assert d["status"] == "pass"
        assert d["summary"]["semantic_change_claimed"] is False
        assert d["diffs"], (model_id, d)
        for entry in d["diffs"]:
            assert entry["kind"] == "annotation_added"
            assert entry["before"] is None
            assert entry["after"] is not None


def test_set_tile_params_diff_matches_verified_recipe(
    post_lowering_runs: dict[str, Path],
) -> None:
    """The tile in the structural diff must match what verified_recipe.mlir says."""
    for model_id, run in post_lowering_runs.items():
        sel = json.loads(
            (run / "03_recipe_planning" / "candidate_selection.json").read_text()
        )
        if sel["candidate_kind"] != "set_tile_params":
            continue
        verified = (run / "03_recipe_planning" / "verified_recipe.mlir").read_text()
        d = json.loads(
            (
                run / "03_recipe_planning" / "post_lowering"
                / "structural_diff.json"
            ).read_text()
        )
        tile = d["diffs"][0]["after"]
        assert isinstance(tile, list) and len(tile) == 3
        for v in tile:
            assert f"= {v} :" in verified, (model_id, tile, v)


# --------------------------------------------------------------------------- #
# Negative paths
# --------------------------------------------------------------------------- #


def test_deleting_transform_script_fails(
    tmp_path: Path, post_lowering_runs: dict[str, Path],
) -> None:
    src = post_lowering_runs["tiny_mlp"]
    work = tmp_path / "no_transform"
    shutil.copytree(src, work)
    # Wipe post_lowering outputs so we re-run cleanly.
    pl_out = work / "03_recipe_planning" / "post_lowering"
    if pl_out.exists():
        shutil.rmtree(pl_out)
    # Delete the transform script.
    (work / "03_recipe_planning" / "lowering_artifacts" / "transforms" / "recipe_0000.mlir").unlink()
    result = run_post_lowering_verification(work)
    assert result.overall == "fail"


def test_tampered_tile_in_transform_script_fails(
    tmp_path: Path, post_lowering_runs: dict[str, Path],
) -> None:
    src = post_lowering_runs["tiny_mlp"]
    work = tmp_path / "tampered_tile_script"
    shutil.copytree(src, work)
    pl_out = work / "03_recipe_planning" / "post_lowering"
    if pl_out.exists():
        shutil.rmtree(pl_out)
    p = work / "03_recipe_planning" / "lowering_artifacts" / "transforms" / "recipe_0000.mlir"
    text = p.read_text()
    # Mutate the M tile dim to 999 — disagrees with verified_recipe.mlir.
    # Post-M-37.11 tiny_mlp's shape-fit tile is M=4 (was M=16 pre-M-37.11);
    # match whichever value the planner picked.
    import re as _re
    m_match = _re.search(r"M\s*=\s*(\d+)\s*:\s*i64", text)
    assert m_match is not None, f"recipe_0000.mlir has no M tile attr: {text}"
    actual_m = m_match.group(1)
    p.write_text(text.replace(f"M = {actual_m} : i64", "M = 999 : i64", 1))
    result = run_post_lowering_verification(work)
    assert result.overall == "fail"
    rep = json.loads(result.verification_report_path.read_text())
    assert any(
        "disagrees with verified_recipe" in r for r in rep["failure_reasons"]
    ), rep["failure_reasons"]


def test_transform_script_referencing_missing_region_fails(
    tmp_path: Path, post_lowering_runs: dict[str, Path],
) -> None:
    src = post_lowering_runs["tiny_mlp"]
    work = tmp_path / "missing_region"
    shutil.copytree(src, work)
    pl_out = work / "03_recipe_planning" / "post_lowering"
    if pl_out.exists():
        shutil.rmtree(pl_out)
    p = work / "03_recipe_planning" / "lowering_artifacts" / "transforms" / "recipe_0000.mlir"
    text = p.read_text()
    p.write_text(text.replace("region: matmul_0", "region: nonexistent_region_xxx")
                     .replace('region = "matmul_0"', 'region = "nonexistent_region_xxx"'))
    result = run_post_lowering_verification(work)
    assert result.overall == "fail"


def test_fusion_with_missing_via_tensor_fails(
    tmp_path: Path, post_lowering_runs: dict[str, Path],
) -> None:
    src = post_lowering_runs["proxy_vla"]  # selected fusion
    work = tmp_path / "bad_via_tensor"
    shutil.copytree(src, work)
    pl_out = work / "03_recipe_planning" / "post_lowering"
    if pl_out.exists():
        shutil.rmtree(pl_out)
    p = work / "03_recipe_planning" / "lowering_artifacts" / "transforms" / "recipe_0000.mlir"
    text = p.read_text()
    p.write_text(
        text.replace(
            'via_tensor = "export_program::',
            'via_tensor = "nonexistent_tensor::',
        )
    )
    result = run_post_lowering_verification(work)
    assert result.overall == "fail"


def test_source_payload_mutation_detected(
    tmp_path: Path, post_lowering_runs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch ``sha256_tree`` so the post-snapshot of 01_payload_lowering
    differs from the pre-snapshot. The verification report must surface
    a ``source_payload_unchanged`` failure."""
    import compgen.graph_compilation.post_lowering as pl_mod

    src = post_lowering_runs["tiny_mlp"]
    work = tmp_path / "payload_mut"
    shutil.copytree(src, work)
    pl_out = work / "03_recipe_planning" / "post_lowering"
    if pl_out.exists():
        shutil.rmtree(pl_out)

    real = pl_mod.sha256_tree
    n = {"i": 0}

    def fake(path: Path) -> str:
        n["i"] += 1
        if n["i"] == 1:
            return real(path)
        return "deadbeef" * 8

    monkeypatch.setattr(pl_mod, "sha256_tree", fake)
    result = run_post_lowering_verification(work)
    assert result.overall == "fail"
    rep = json.loads(result.verification_report_path.read_text())
    src_check = next(c for c in rep["checks"] if c["name"] == "source_payload_unchanged")
    assert src_check["status"] == "fail"


def test_transformed_payload_under_01_payload_lowering_is_caught(
    tmp_path: Path, post_lowering_runs: dict[str, Path],
) -> None:
    """If a malicious M-08 (or buggy refactor) writes
    ``transformed_payload`` under 01_payload_lowering, the
    ``transformed_payload_not_under_01_payload_lowering`` check fails."""
    src = post_lowering_runs["tiny_mlp"]
    work = tmp_path / "leak"
    shutil.copytree(src, work)
    leak = work / "01_payload_lowering" / "transformed_payload.mlir"
    leak.write_text("// leaked", encoding="utf-8")
    pl_out = work / "03_recipe_planning" / "post_lowering"
    if pl_out.exists():
        shutil.rmtree(pl_out)
    result = run_post_lowering_verification(work)
    assert result.overall == "fail"
    rep = json.loads(result.verification_report_path.read_text())
    leak_check = next(
        c for c in rep["checks"]
        if c["name"] == "transformed_payload_not_under_01_payload_lowering"
    )
    assert leak_check["status"] == "fail"


def test_contract_only_recipe_does_not_emit_transformed_payload(
    post_lowering_runs: dict[str, Path],
) -> None:
    """Sanity: the routing rule is enforced for every contract-only run."""
    run = post_lowering_runs["custom_unsupported_op"]
    pl = run / "03_recipe_planning" / "post_lowering"
    assert (pl / "contract_structural_validation.json").exists()
    assert not (pl / "transformed_payload.mlir").exists()
    assert not (pl / "applied_transform_manifest.json").exists()
    assert not (pl / "structural_diff.json").exists()


def test_post_lowering_does_not_claim_full_semantic_discharge(
    post_lowering_runs: dict[str, Path],
) -> None:
    """The report must not claim differential checks have been discharged."""
    for run in post_lowering_runs.values():
        rep = json.loads(
            (
                run / "03_recipe_planning" / "post_lowering"
                / "post_lowering_verification_report.json"
            ).read_text()
        )
        check = next(
            c for c in rep["checks"]
            if c["name"] == "no_full_differential_discharge_claimed"
        )
        assert check["status"] == "pass"


# --------------------------------------------------------------------------- #
# Compiler core untouched
# --------------------------------------------------------------------------- #


def test_compiler_core_not_modified_by_m08() -> None:
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
    assert not changed, f"M-08 modified compiler core: {changed}"
