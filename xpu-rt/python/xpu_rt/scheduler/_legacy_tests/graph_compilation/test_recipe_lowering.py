"""Acceptance tests for Recipe Lowering to Lowering Artifacts.

Asserts:
- Required lowering artifacts are emitted for all canonical models
- Each verified recipe op has exactly one lowering artifact
- Family-specific routing: tile/fusion → transform_script;
  CreateKernelContract → kernel_contract_draft
- payload.mlir is byte-identical before/after lowering
- Negative paths reject missing/tampered/illegal inputs
- Compiler core untouched
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from compgen.graph_compilation.recipe_lowering import run_recipe_lowering
from compgen.graph_compilation.run import run_graph_compilation

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_CPU_TARGET = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"

LOWERING_MODELS: tuple[str, ...] = (
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
def lowering_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("lowering_runs")
    out: dict[str, Path] = {}
    for model_id in LOWERING_MODELS:
        cfg = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
        run_dir = base / model_id
        run_graph_compilation(
            model_config_path=cfg,
            target_config_path=HOST_CPU_TARGET,
            out_dir=run_dir,
            stop_after="recipe-lowering",
            run_id=f"lower_{model_id}",
            selection_mode="greedy",
        )
        out[model_id] = run_dir
    return out


# --------------------------------------------------------------------------- #
# Existence + shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", LOWERING_MODELS)
def test_required_lowering_artifacts_emitted(
    model_id: str, lowering_runs: dict[str, Path]
) -> None:
    rp = lowering_runs[model_id] / "03_recipe_planning"
    for name in (
        "lowering_artifact_manifest.json",
        "transform_lowering_report.json",
        "transform_validation.json",
        "lowering_artifacts/README.md",
    ):
        p = rp / name
        assert p.exists() and p.stat().st_size > 0, f"{model_id}: missing {name}"
    # At least one transform OR contract artifact
    artifacts = list((rp / "lowering_artifacts").rglob("*.mlir"))
    assert artifacts, f"{model_id}: no .mlir lowering artifacts"


@pytest.mark.parametrize("model_id", LOWERING_MODELS)
def test_transform_validation_overall_pass(
    model_id: str, lowering_runs: dict[str, Path]
) -> None:
    obj = json.loads(
        (
            lowering_runs[model_id] / "03_recipe_planning" / "transform_validation.json"
        ).read_text()
    )
    failed = [c for c in obj["checks"] if c["status"] == "fail"]
    assert obj["overall"] == "pass", failed


@pytest.mark.parametrize("model_id", LOWERING_MODELS)
def test_artifact_validator_passes(
    model_id: str, lowering_runs: dict[str, Path]
) -> None:
    from compgen.graph_compilation import validate_run
    rep = validate_run(lowering_runs[model_id])
    assert rep.overall == "pass", [r for r in rep.rules if r.status == "fail"]


# --------------------------------------------------------------------------- #
# Manifest invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", LOWERING_MODELS)
def test_one_artifact_per_verified_recipe_op(
    model_id: str, lowering_runs: dict[str, Path]
) -> None:
    run = lowering_runs[model_id]
    manifest = json.loads(
        (run / "03_recipe_planning" / "lowering_artifact_manifest.json").read_text()
    )
    s = manifest["summary"]
    assert s["artifacts_emitted"] == s["recipe_ops_total"], manifest


@pytest.mark.parametrize("model_id", LOWERING_MODELS)
def test_manifest_hashes_match_disk(
    model_id: str, lowering_runs: dict[str, Path]
) -> None:
    run = lowering_runs[model_id]
    manifest = json.loads(
        (run / "03_recipe_planning" / "lowering_artifact_manifest.json").read_text()
    )
    for a in manifest["artifacts"]:
        path = run / a["path"]
        actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert (
            actual.removeprefix("sha256:") == a["sha256"].removeprefix("sha256:")
            or actual == a["sha256"]
        ), (model_id, a["path"], actual, a["sha256"])


# --------------------------------------------------------------------------- #
# Family-specific routing
# --------------------------------------------------------------------------- #


def test_set_tile_params_emits_transform_script(
    lowering_runs: dict[str, Path],
) -> None:
    """Models that selected SetTileParams must produce a transform_script."""
    seen = False
    for model_id, run in lowering_runs.items():
        sel = json.loads(
            (run / "03_recipe_planning" / "candidate_selection.json").read_text()
        )
        if sel["candidate_kind"] != "set_tile_params":
            continue
        manifest = json.loads(
            (run / "03_recipe_planning" / "lowering_artifact_manifest.json").read_text()
        )
        artifact = manifest["artifacts"][0]
        assert artifact["artifact_kind"] == "transform_script", (model_id, artifact)
        assert "/transforms/" in artifact["path"], (model_id, artifact)
        text = (run / artifact["path"]).read_text()
        assert "SetTileParams" in text
        assert "set_tile_params" in text
        assert "M = " in text and "N = " in text and "K = " in text
        seen = True
    assert seen


def test_fuse_producer_consumer_emits_transform_script(
    lowering_runs: dict[str, Path],
) -> None:
    seen = False
    for model_id, run in lowering_runs.items():
        sel = json.loads(
            (run / "03_recipe_planning" / "candidate_selection.json").read_text()
        )
        if sel["candidate_kind"] != "fuse_producer_consumer":
            continue
        manifest = json.loads(
            (run / "03_recipe_planning" / "lowering_artifact_manifest.json").read_text()
        )
        artifact = manifest["artifacts"][0]
        assert artifact["artifact_kind"] == "transform_script", (model_id, artifact)
        text = (run / artifact["path"]).read_text()
        assert "FuseProducerConsumer" in text
        assert "fuse_producer_consumer" in text
        assert "producer = " in text and "consumer = " in text
        assert "via_tensor = " in text
        seen = True
    assert seen


def test_create_kernel_contract_emits_contract_draft(
    lowering_runs: dict[str, Path],
) -> None:
    """custom_unsupported_op must emit a kernel_contract_draft, not
    a transform_script."""
    run = lowering_runs["custom_unsupported_op"]
    manifest = json.loads(
        (run / "03_recipe_planning" / "lowering_artifact_manifest.json").read_text()
    )
    artifact = manifest["artifacts"][0]
    assert artifact["artifact_kind"] == "kernel_contract_draft"
    assert "/contracts/" in artifact["path"]
    text = (run / artifact["path"]).read_text()
    assert "CreateKernelContract" in text
    assert "contract.kernel" in text
    assert "kernel_contract_generation" in text
    # Must NOT claim bit-equality
    assert "bit_equality" not in text


# --------------------------------------------------------------------------- #
# Per-artifact content invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", LOWERING_MODELS)
def test_every_artifact_references_source_candidate_and_obligation(
    model_id: str, lowering_runs: dict[str, Path]
) -> None:
    run = lowering_runs[model_id]
    manifest = json.loads(
        (run / "03_recipe_planning" / "lowering_artifact_manifest.json").read_text()
    )
    sel = json.loads(
        (run / "03_recipe_planning" / "candidate_selection.json").read_text()
    )
    expected_cand = sel["selected_candidate_id"]
    for a in manifest["artifacts"]:
        text = (run / a["path"]).read_text()
        assert expected_cand in text, (model_id, a["path"])
        assert "semantic_obligation:" in text or "sem.obligation_ref" in text


@pytest.mark.parametrize("model_id", LOWERING_MODELS)
def test_payload_refs_in_transform_resolve(
    model_id: str, lowering_runs: dict[str, Path]
) -> None:
    run = lowering_runs[model_id]
    report = json.loads(
        (run / "03_recipe_planning" / "transform_lowering_report.json").read_text()
    )
    for op in report["lowered_ops"]:
        for pr in op["payload_refs"]:
            assert (run / pr).exists(), (model_id, pr)


# --------------------------------------------------------------------------- #
# Hard invariant: payload.mlir not mutated
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", LOWERING_MODELS)
def test_no_payload_mutation_check_passes(
    model_id: str, lowering_runs: dict[str, Path]
) -> None:
    obj = json.loads(
        (
            lowering_runs[model_id] / "03_recipe_planning" / "transform_validation.json"
        ).read_text()
    )
    npm = next(c for c in obj["checks"] if c["name"] == "no_payload_mutation")
    assert npm["status"] == "pass", npm


@pytest.mark.parametrize("model_id", LOWERING_MODELS)
def test_payload_unchanged_after_rerun(
    model_id: str, lowering_runs: dict[str, Path], tmp_path: Path,
) -> None:
    """Snapshot 01_payload_lowering/, re-run on a copy, and verify
    the directory is byte-identical afterwards. This is the hard
    guarantee must keep."""
    src = lowering_runs[model_id]
    work = tmp_path / f"readonly_{model_id}"
    shutil.copytree(src, work)
    pl = work / "01_payload_lowering"
    before = _sha256_tree(pl)
    run_recipe_lowering(work)
    after = _sha256_tree(pl)
    assert before == after, f"{model_id}: M-07 mutated 01_payload_lowering/"


# --------------------------------------------------------------------------- #
# Negative paths
# --------------------------------------------------------------------------- #


def test_deleting_verified_recipe_fails(
    tmp_path: Path, lowering_runs: dict[str, Path],
) -> None:
    src = lowering_runs["tiny_mlp"]
    work = tmp_path / "no_verified"
    shutil.copytree(src, work)
    (work / "03_recipe_planning" / "verified_recipe.mlir").unlink()
    with pytest.raises(FileNotFoundError):
        run_recipe_lowering(work)


def test_deleting_semantic_obligations_fails(
    tmp_path: Path, lowering_runs: dict[str, Path],
) -> None:
    src = lowering_runs["tiny_mlp"]
    work = tmp_path / "no_sem"
    shutil.copytree(src, work)
    (work / "03_recipe_planning" / "semantic_obligations.mlir").unlink()
    with pytest.raises(FileNotFoundError):
        run_recipe_lowering(work)


def test_tampered_gate_status_fails_lowering(
    tmp_path: Path, lowering_runs: dict[str, Path],
) -> None:
    src = lowering_runs["tiny_mlp"]
    work = tmp_path / "tamper_gate"
    shutil.copytree(src, work)
    p = work / "03_recipe_planning" / "verified_recipe.mlir"
    text = p.read_text()
    # Flip pass → fail on the recipe op so refuses to lower.
    text = text.replace('gate_status = "pass"', 'gate_status = "fail"', 2)
    p.write_text(text)
    result = run_recipe_lowering(work)
    assert result.overall == "fail", "expected lowering to fail when gate_status='fail'"


def test_tampered_semantic_obligation_id_fails(
    tmp_path: Path, lowering_runs: dict[str, Path],
) -> None:
    src = lowering_runs["tiny_mlp"]
    work = tmp_path / "tamper_obl"
    shutil.copytree(src, work)
    p = work / "03_recipe_planning" / "verified_recipe.mlir"
    text = p.read_text()
    text = text.replace("@obl_recipe_0000", "@obl_recipe_does_not_exist", 1)
    p.write_text(text)
    result = run_recipe_lowering(work)
    val = json.loads(result.validation_path.read_text())
    assert val["overall"] == "fail"
    assert any(
        c["name"] == "semantic_obligations_resolved" and c["status"] == "fail"
        for c in val["checks"]
    ), val["checks"]


def test_tampered_recipe_kind_to_unsupported_fails(
    tmp_path: Path, lowering_runs: dict[str, Path],
) -> None:
    src = lowering_runs["tiny_mlp"]
    work = tmp_path / "unknown_kind"
    shutil.copytree(src, work)
    p = work / "03_recipe_planning" / "verified_recipe.mlir"
    text = p.read_text()
    text = text.replace("recipe.set_tile_params", "recipe.do_something_unsupported", 1)
    p.write_text(text)
    result = run_recipe_lowering(work)
    assert result.overall == "fail"


def test_payload_mutation_during_lowering_is_detected(
    tmp_path: Path, lowering_runs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a payload mutation that happens during lowering by
    patching ``sha256_tree`` to return a different hash on the post-call.
    The ``no_payload_mutation`` gate must catch it."""
    src = lowering_runs["tiny_mlp"]
    work = tmp_path / "mutation_during"
    shutil.copytree(src, work)

    # Wipe lowering artifacts so we re-run cleanly.
    rp = work / "03_recipe_planning"
    for p in (
        rp / "lowering_artifact_manifest.json",
        rp / "transform_lowering_report.json",
        rp / "transform_validation.json",
    ):
        if p.exists():
            p.unlink()
    if (rp / "lowering_artifacts").exists():
        shutil.rmtree(rp / "lowering_artifacts")

    import compgen.graph_compilation.recipe_lowering as rl_mod

    real = rl_mod.sha256_tree
    call_count = {"n": 0}

    def fake(path: Path) -> str:
        call_count["n"] += 1
        # First call (pre-snapshot) → real hash; second call (post) → fake.
        if call_count["n"] == 1:
            return real(path)
        return "deadbeef" * 8  # 64 hex chars, intentionally != real

    monkeypatch.setattr(rl_mod, "sha256_tree", fake)
    result = rl_mod.run_recipe_lowering(work)
    assert result.overall == "fail"
    val = json.loads(result.validation_path.read_text())
    npm = next(c for c in val["checks"] if c["name"] == "no_payload_mutation")
    assert npm["status"] == "fail", val


def test_create_kernel_contract_does_not_emit_transform_script(
    lowering_runs: dict[str, Path],
) -> None:
    """The opposite of the routing test: custom_unsupported_op must NOT
    have a transform script under transforms/."""
    run = lowering_runs["custom_unsupported_op"]
    rp = run / "03_recipe_planning"
    assert not (rp / "lowering_artifacts" / "transforms" / "recipe_0000.mlir").exists()
    assert (
        rp / "lowering_artifacts" / "contracts" / "recipe_0000.kernel_contract.mlir"
    ).exists()


# --------------------------------------------------------------------------- #
# Compiler core untouched
# --------------------------------------------------------------------------- #


def test_compiler_core_not_modified_by_m07() -> None:
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
    assert not changed, f"M-07 modified compiler core: {changed}"


# --------------------------------------------------------------------------- #
# No transformed_payload.mlir, no benchmarks/profiler/codegen artifacts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", LOWERING_MODELS)
def test_no_transformed_payload_mlir_yet(
    model_id: str, lowering_runs: dict[str, Path]
) -> None:
    run = lowering_runs[model_id]
    assert not (run / "transformed_payload.mlir").exists()
    assert not list(
        (run / "01_payload_lowering").rglob("transformed_payload*")
    ), f"{model_id}: M-07 emitted transformed_payload.mlir prematurely"
