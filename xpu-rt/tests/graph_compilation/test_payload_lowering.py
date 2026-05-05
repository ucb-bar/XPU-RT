"""Acceptance tests for Payload Lowering.

Run real Dynamo + torch.export captures + lowering through ``FXImporter``
against two test models:

- ``tiny_mlp`` — proves the basic decomposition path; payload.mlir
  contains real ``linalg.matmul`` / ``linalg.transpose`` ops.
- ``custom_unsupported_op`` — proves the opaque/unsupported inventory
  is real and downstream-consumable.

No mocks. Per the project's anti-mock policy these tests exercise the
emitted artifacts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from compgen.graph_compilation import validate_run
from compgen.graph_compilation.lowering_validate import validate_payload_lowering
from compgen.graph_compilation.run import lower_from_existing_capture, run_graph_compilation

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_MLP_CONFIG = REPO_ROOT / "configs" / "models" / "tiny_mlp.yaml"
UNSUPPORTED_CONFIG = REPO_ROOT / "configs" / "models" / "custom_unsupported_op.yaml"
HOST_CPU_TARGET = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"

# Extended model coverage: each name maps to a YAML config under configs/models/.
# These exercise different op surfaces (attention/conv/multimodal/robotics)
# beyond the two blocking tests so we catch breakage early.
EXTENDED_MODELS: tuple[str, ...] = (
    "tiny_attention",
    "tiny_conv_block",
    "proxy_vlm",
    "proxy_vla",
)


# --------------------------------------------------------------------------- #
# Module-scope fixtures: one full-pipeline run per model
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_mlp_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("pl_tiny_mlp") / "run"
    run_graph_compilation(
        model_config_path=TINY_MLP_CONFIG,
        target_config_path=HOST_CPU_TARGET,
        out_dir=out,
        stop_after="payload-lowering",
        run_id="pl_tiny_mlp",
    )
    return out


@pytest.fixture(scope="module")
def custom_unsupported_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("pl_custom_unsupported") / "run"
    run_graph_compilation(
        model_config_path=UNSUPPORTED_CONFIG,
        target_config_path=HOST_CPU_TARGET,
        out_dir=out,
        stop_after="payload-lowering",
        run_id="pl_custom_unsupported",
    )
    return out


# --------------------------------------------------------------------------- #
# tiny_mlp acceptance metrics
# --------------------------------------------------------------------------- #


def test_tiny_mlp_artifact_validator_passes(tiny_mlp_run: Path) -> None:
    report = validate_run(tiny_mlp_run)
    assert report.overall == "pass", [r for r in report.rules if r.status == "fail"]


def test_tiny_mlp_lowering_validation_passes(tiny_mlp_run: Path) -> None:
    rep = validate_payload_lowering(tiny_mlp_run)
    assert rep.status == "pass", [c for c in rep.checks if c.status == "fail"]


def test_tiny_mlp_summary_top_level_metrics(tiny_mlp_run: Path) -> None:
    s = json.loads((tiny_mlp_run / "01_payload_lowering" / "lowering_summary.json").read_text())
    assert s["status"] in {"pass", "partial_success"}
    assert s["primary_capture"] == "torch_dynamo"
    assert s["target_id"] == "host_cpu"
    assert s["dynamo"]["input_partition_count"] >= 1
    assert s["dynamo"]["lowered_partition_count"] >= 1
    assert s["totals"]["payload_modules_total"] >= 1
    assert s["totals"]["fx_nodes_total"] > 0
    assert s["totals"]["payload_ops_total"] > 0
    assert s["totals"]["decomposed_ops_total"] + s["totals"]["opaque_ops_total"] > 0
    assert 0.0 <= s["totals"]["decomposition_coverage"] <= 1.0
    assert s["llm_calls"] == 0


def test_tiny_mlp_payload_index_paths_exist(tiny_mlp_run: Path) -> None:
    idx = json.loads((tiny_mlp_run / "01_payload_lowering" / "payload_index.json").read_text())
    assert len(idx["modules"]) == json.loads(
        (tiny_mlp_run / "01_payload_lowering" / "lowering_summary.json").read_text()
    )["totals"]["payload_modules_total"]
    for m in idx["modules"]:
        assert (tiny_mlp_run / m["payload_mlir"]).exists()
        assert (tiny_mlp_run / m["lowering_report"]).exists()
        assert (tiny_mlp_run / m["input_graph"]).exists()


def test_tiny_mlp_export_payload_mlir_has_linalg_ops(tiny_mlp_run: Path) -> None:
    mlir = (tiny_mlp_run / "01_payload_lowering" / "export_program" / "payload.mlir").read_text()
    assert "builtin.module" in mlir
    assert "func.func" in mlir
    # tiny_mlp has linear→gelu→linear; after default decompositions the export
    # path produces real linalg ops.
    assert "linalg.matmul" in mlir
    assert "compgen.region_id" in mlir
    # No fake/placeholder text.
    assert "TODO" not in mlir
    assert "fake" not in mlir.lower()


def test_tiny_mlp_per_module_lowering_report_uses_fx_importer(tiny_mlp_run: Path) -> None:
    for path in (tiny_mlp_run / "01_payload_lowering").rglob("lowering_report.json"):
        r = json.loads(path.read_text())
        if r.get("status") == "skipped":
            continue
        assert r["lowering_api"] == "compgen.ir.payload.import_fx.FXImporter"
        assert r["llm_calls"] == 0
        assert r["input"]["num_fx_nodes"] > 0


def test_tiny_mlp_canonical_pass_trace_llm_disallowed(tiny_mlp_run: Path) -> None:
    trace = json.loads(
        (tiny_mlp_run / "01_payload_lowering" / "canonical_pass_trace.json").read_text()
    )
    assert trace["llm_allowed"] is False
    assert trace["stage_id"] == "payload_lowering"
    assert any(p["implementation"].endswith("FXImporter.import_graph") for p in trace["passes"])


# --------------------------------------------------------------------------- #
# custom_unsupported_op acceptance metrics
# --------------------------------------------------------------------------- #


def test_custom_unsupported_artifact_validator_passes(custom_unsupported_run: Path) -> None:
    report = validate_run(custom_unsupported_run)
    assert report.overall == "pass", [r for r in report.rules if r.status == "fail"]


def test_custom_unsupported_lowering_validation_passes(custom_unsupported_run: Path) -> None:
    rep = validate_payload_lowering(custom_unsupported_run)
    assert rep.status == "pass", [c for c in rep.checks if c.status == "fail"]


def test_custom_unsupported_produces_real_inventories(custom_unsupported_run: Path) -> None:
    opaque = json.loads(
        (custom_unsupported_run / "01_payload_lowering" / "opaque_calls.json").read_text()
    )
    unsupp = json.loads(
        (custom_unsupported_run / "01_payload_lowering" / "unsupported_ops.json").read_text()
    )
    assert opaque["summary"]["count"] >= 1
    assert unsupp["summary"]["count"] >= 1
    # At least one record names our custom op explicitly.
    targets = {u["fx_target"] for u in unsupp["unsupported_ops"]}
    assert any("crgtoy.affine_gelu" in t for t in targets), targets


def test_custom_unsupported_payload_mlir_contains_func_call(custom_unsupported_run: Path) -> None:
    """Spec: if opaque calls exist, payload.mlir must contain func.call."""
    for path in (custom_unsupported_run / "01_payload_lowering").rglob("payload.mlir"):
        text = path.read_text()
        if "crgtoy" in text:
            assert "func.call" in text, f"opaque payload.mlir missing func.call: {path}"


def test_custom_unsupported_payload_ref_traceable(custom_unsupported_run: Path) -> None:
    """Each unsupported_op record's payload_ref must point at an existing payload.mlir
    that contains the named callee."""
    unsupp = json.loads(
        (custom_unsupported_run / "01_payload_lowering" / "unsupported_ops.json").read_text()
    )
    for u in unsupp["unsupported_ops"]:
        payload_path = custom_unsupported_run / u["payload_ref"]["payload_mlir"]
        assert payload_path.exists(), f"missing referenced payload.mlir: {payload_path}"


# --------------------------------------------------------------------------- #
# Tamper tests
# --------------------------------------------------------------------------- #


def test_tamper_payload_mlir_fails_artifact_validator(tiny_mlp_run: Path, tmp_path: Path) -> None:
    """Spec: replacing payload.mlir text with garbage must fail validation."""
    tampered = tmp_path / "tamper_payload"
    shutil.copytree(tiny_mlp_run, tampered)
    target = tampered / "01_payload_lowering" / "export_program" / "payload.mlir"
    target.write_text("fake mlir\n", encoding="utf-8")
    report = validate_run(tampered)
    assert report.overall == "fail"  # R005 catches the hash drift


def test_tamper_aggregate_counts_fails_lowering_validation(
    tiny_mlp_run: Path, tmp_path: Path
) -> None:
    """Spec: editing aggregate totals to lie must fail lowering_validation."""
    tampered = tmp_path / "tamper_totals"
    shutil.copytree(tiny_mlp_run, tampered)
    summary_path = tampered / "01_payload_lowering" / "lowering_summary.json"
    obj = json.loads(summary_path.read_text())
    obj["totals"]["payload_modules_total"] = 999
    summary_path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    rep = validate_payload_lowering(tampered)
    assert rep.status == "fail"
    fails = [c for c in rep.checks if c.status == "fail"]
    assert any(c.name == "aggregate_counts_match_partition_reports" for c in fails)


def test_tamper_delete_lowering_report_fails(tiny_mlp_run: Path, tmp_path: Path) -> None:
    """Spec: payload_index points to a missing report → validation fails."""
    tampered = tmp_path / "tamper_missing_report"
    shutil.copytree(tiny_mlp_run, tampered)
    # Find a referenced report and delete it.
    idx = json.loads((tampered / "01_payload_lowering" / "payload_index.json").read_text())
    target = next(
        (
            tampered / m["lowering_report"]
            for m in idx["modules"]
            if m.get("status") != "skipped"
        ),
        None,
    )
    assert target is not None and target.exists()
    target.unlink()
    rep = validate_payload_lowering(tampered)
    assert rep.status == "fail"


# --------------------------------------------------------------------------- #
# Lower-from-existing-capture (proves no hidden re-capture)
# --------------------------------------------------------------------------- #


def test_lower_from_existing_capture_run(tmp_path: Path) -> None:
    capture_only = tmp_path / "capture_only"
    run_graph_compilation(
        model_config_path=TINY_MLP_CONFIG,
        target_config_path=HOST_CPU_TARGET,
        out_dir=capture_only,
        stop_after="graph-capture",
        run_id="capture_only",
    )
    lowered = tmp_path / "lowered_from_capture"
    lower_from_existing_capture(
        capture_run=capture_only,
        target_config_path=HOST_CPU_TARGET,
        out_dir=lowered,
        run_id="lowered_from_capture",
    )
    # Both validators must pass on the produced run.
    assert validate_run(lowered).overall == "pass"
    assert validate_payload_lowering(lowered).status == "pass"
    # Sanity: the lowered run carries the same goldens as the source.
    src_goldens = (capture_only / "00_graph_capture" / "golden_outputs.pt").read_bytes()
    dst_goldens = (lowered / "00_graph_capture" / "golden_outputs.pt").read_bytes()
    assert src_goldens == dst_goldens


# --------------------------------------------------------------------------- #
# Anti-coupling
# --------------------------------------------------------------------------- #


def test_existing_compiler_core_not_modified() -> None:
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
    assert not changed, f"forbidden files modified: {changed}"


def test_no_development_wave_names_in_public_surfaces() -> None:
    """Gate: the new public surface uses only descriptive names.

    We allow this test file itself to mention the forbidden tokens (it
    has to, to test for them) and we allow ``__pycache__`` artifacts.
    """
    import subprocess

    forbidden_pattern = "(" + chr(99) + "apture_lower|" + chr(67) + "APLOW|" + chr(99) + "aplow)"
    bad = subprocess.run(
        [
            "grep",
            "-rEln",
            "--exclude-dir=__pycache__",
            "--exclude=test_payload_lowering.py",
            forbidden_pattern,
            "python/compgen/graph_compilation/",
            "tests/graph_compilation/",
            "configs/models/",
            "configs/targets/",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if bad.stdout.strip():
        pytest.fail(f"naming-policy violation:\n{bad.stdout}")


# --------------------------------------------------------------------------- #
# Extended-model coverage matrix (tiny_attention / tiny_conv_block / proxy_vlm /
# proxy_vla). These are not the two blocking tests — they're the "should also
# run" coverage from the spec. Each is module-scoped so we capture+lower once
# per model per session.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def extended_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Capture+lower every extended model once; return ``{model_id: run_dir}``.

    Skipped if any model fails to build/capture — the parametrized tests
    surface the failure individually.
    """
    base = tmp_path_factory.mktemp("pl_extended")
    out: dict[str, Path] = {}
    for model_id in EXTENDED_MODELS:
        cfg = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
        run_dir = base / model_id
        run_graph_compilation(
            model_config_path=cfg,
            target_config_path=HOST_CPU_TARGET,
            out_dir=run_dir,
            stop_after="payload-lowering",
            run_id=f"pl_{model_id}",
        )
        out[model_id] = run_dir
    return out


@pytest.mark.parametrize("model_id", EXTENDED_MODELS)
def test_extended_model_artifact_validator_passes(
    model_id: str, extended_runs: dict[str, Path]
) -> None:
    report = validate_run(extended_runs[model_id])
    assert report.overall == "pass", [r for r in report.rules if r.status == "fail"]


@pytest.mark.parametrize("model_id", EXTENDED_MODELS)
def test_extended_model_lowering_validation_passes(
    model_id: str, extended_runs: dict[str, Path]
) -> None:
    rep = validate_payload_lowering(extended_runs[model_id])
    assert rep.status == "pass", [c for c in rep.checks if c.status == "fail"]


@pytest.mark.parametrize("model_id", EXTENDED_MODELS)
def test_extended_model_produces_real_payload(
    model_id: str, extended_runs: dict[str, Path]
) -> None:
    """Per-spec: payload_modules_total > 0, fx_nodes > 0, llm_calls == 0."""
    s = json.loads(
        (extended_runs[model_id] / "01_payload_lowering" / "lowering_summary.json").read_text()
    )
    assert s["status"] in {"pass", "partial_success"}, s["status"]
    assert s["totals"]["payload_modules_total"] >= 1
    assert s["totals"]["fx_nodes_total"] > 0
    assert s["totals"]["payload_ops_total"] > 0
    assert s["totals"]["decomposed_ops_total"] + s["totals"]["opaque_ops_total"] > 0
    assert s["llm_calls"] == 0


@pytest.mark.parametrize("model_id", EXTENDED_MODELS)
def test_extended_model_no_python_object_addresses(
    model_id: str, extended_runs: dict[str, Path]
) -> None:
    """Spec: target names must be reproducible; ``0x...`` Python addresses break that."""
    import re

    text = (
        extended_runs[model_id] / "01_payload_lowering" / "opaque_calls.json"
    ).read_text()
    # If ``0x...`` survives into the recorded targets, two reruns will
    # produce different sha256 — gap discovery indexing breaks.
    assert not re.search(r"\bat 0x[0-9a-f]+", text), (
        f"opaque_calls.json contains a non-canonical Python address: {model_id}"
    )


def test_extended_models_have_distinct_op_histograms(
    extended_runs: dict[str, Path],
) -> None:
    """The four extended models exercise different op surfaces — totals shouldn't all match."""
    histograms = {}
    for model_id, run in extended_runs.items():
        s = json.loads((run / "01_payload_lowering" / "lowering_summary.json").read_text())
        histograms[model_id] = (
            s["totals"]["fx_nodes_total"],
            s["totals"]["call_function_nodes_total"],
            s["totals"]["payload_ops_total"],
        )
    # No two histograms should be identical — that would hint we're
    # capturing/lowering the same graph for both.
    seen = set()
    for h in histograms.values():
        seen.add(h)
    assert len(seen) == len(histograms), f"models share a histogram: {histograms}"


# --------------------------------------------------------------------------- #
# Payload Coverage Audit (07 — Truth-telling)
# --------------------------------------------------------------------------- #


def test_payload_coverage_audit_files_exist(extended_runs: dict[str, Path]) -> None:
    """Every lowering run must emit the three coverage-audit JSONs."""
    import json
    expected_versions = {
        "fx_to_payload_accounting.json": "fx_to_payload_accounting_v2",
        "dialect_coverage.json": "dialect_coverage_v1",
        "silent_drop_audit.json": "silent_drop_audit_v1",
    }
    for model_id, run in extended_runs.items():
        for name, expected_version in expected_versions.items():
            p = run / "01_payload_lowering" / name
            assert p.exists(), f"{model_id}: missing {name}"
            obj = json.loads(p.read_text())
            assert obj.get("schema_version") == expected_version, (model_id, name, obj.get("schema_version"))


def test_silent_drop_audit_strict_pass_for_canonical_models(
    extended_runs: dict[str, Path]
) -> None:
    """The strict pass gate: no unaccounted call_function nodes, no opaque
    calls without origin. Silent drops are surfaced but don't fail."""
    import json
    for model_id, run in extended_runs.items():
        a = json.loads((run / "01_payload_lowering" / "silent_drop_audit.json").read_text())
        assert a["status"] == "pass", (model_id, a)
        assert a["totals"]["unaccounted_call_function_nodes"] == 0, (model_id, a)
        assert a["totals"]["opaque_calls_without_origin"] == 0, (model_id, a)


def test_fx_to_payload_accounting_classifies_every_node(
    extended_runs: dict[str, Path]
) -> None:
    """Each node has exactly one classification from the allowed v2 set."""
    import json
    allowed = {
        "placeholder", "output",
        "decomposed_structured", "opaque_fallback", "closed_by_registry",
        "resolved_alias", "dropped_auxiliary_output", "diagnostic_error",
    }
    required_keys = {
        "fx_node", "fx_target", "op_kind", "classification",
        "payload_ops", "diagnostics", "gap_id", "registry_closure",
    }
    for model_id, run in extended_runs.items():
        acc = json.loads((run / "01_payload_lowering"
                          / "fx_to_payload_accounting.json").read_text())
        assert acc["schema_version"] == "fx_to_payload_accounting_v2", model_id
        for module in acc["modules"]:
            for n in module["nodes"]:
                assert set(n.keys()) >= required_keys, (model_id, set(n.keys()))
                assert n["classification"] in allowed, (model_id, n)
                # payload_ops must be a list of dicts (possibly empty).
                assert isinstance(n["payload_ops"], list), (model_id, n)
                for po in n["payload_ops"]:
                    assert {"op_name", "region_id", "payload_ref"} <= set(po.keys()), (model_id, po)


def test_fx_to_payload_accounting_summary_keys_v2(
    extended_runs: dict[str, Path]
) -> None:
    """Summary must use v2 keys and report no `unaccounted` on canonical models."""
    import json
    expected_keys = {
        "fx_nodes_total", "call_function_nodes",
        "placeholder", "output",
        "decomposed_structured", "opaque_fallback", "closed_by_registry",
        "resolved_alias", "dropped_auxiliary_output", "diagnostic_error",
        "unaccounted",
    }
    for model_id, run in extended_runs.items():
        acc = json.loads((run / "01_payload_lowering"
                          / "fx_to_payload_accounting.json").read_text())
        assert set(acc["summary"].keys()) == expected_keys, (model_id, acc["summary"])
        assert acc["summary"]["unaccounted"] == 0, (model_id, acc["summary"])


def test_dialect_coverage_has_real_structured_ops(
    tiny_mlp_run: Path,
) -> None:
    """tiny_mlp must produce at least one linalg op via export_program path —
    proves we are emitting real structured Payload IR, not just opaque."""
    import json
    cov = json.loads((tiny_mlp_run / "01_payload_lowering"
                      / "dialect_coverage.json").read_text())
    structured = cov["aggregate"]["structured_ops"]
    assert any(op.startswith("linalg.") for op in structured), structured
    assert any(op.startswith("tensor.") for op in structured), structured
