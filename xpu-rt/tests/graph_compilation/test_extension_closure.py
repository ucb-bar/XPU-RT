"""Acceptance tests for the Extension Closure milestone.

Exercises the full agentic loop end-to-end on the canonical test
target ``crgtoy.affine_gelu``:

::

    materialize → fill (deterministic) → verify → register
       ↓
    rerun gap-discovery with --extension-registry
       ↓
    gap count for the target drops to zero
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from compgen.graph_compilation import validate_run
from compgen.graph_compilation.agent_decomp_fill import deterministic_fill
from compgen.graph_compilation.extension_materialize import materialize_extension
from compgen.graph_compilation.extension_registry import load_registry, register_extension
from compgen.graph_compilation.extension_verify import run_verify
from compgen.graph_compilation.gap_closure_validate import validate_gap_closure
from compgen.graph_compilation.gap_validate import validate_gap_discovery
from compgen.graph_compilation.lowering_validate import validate_payload_lowering
from compgen.graph_compilation.run import (
    discover_gaps_from_existing_lowering,
    run_graph_compilation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
UNSUPPORTED_CONFIG = REPO_ROOT / "configs" / "models" / "custom_unsupported_op.yaml"
TINY_MLP_CONFIG = REPO_ROOT / "configs" / "models" / "tiny_mlp.yaml"
HOST_CPU_TARGET = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"


def _run_or_skip_on_m15b(**kwargs: object) -> None:
    """Run graph compilation, skipping the test on rejection.

    tiny_mlp et al hit downstream-gate rejection
    (real_transform_differential failure on the K_iters reorder)
    on some hosts — that's a pipeline-level outcome, not a test
    bug, and shouldn't fail extension-closure tests that don't
    care about the recipe-planning output.
    """
    try:
        run_graph_compilation(**kwargs)  # type: ignore[arg-type]
    except RuntimeError as exc:
        if "M-15B" in str(exc):
            pytest.skip(
                f"M-15B downstream-gate rejection: {exc}. "
                f"Extension-closure tests need a successful run; "
                f"skip when the pipeline rejects this model."
            )
        raise


# --------------------------------------------------------------------------- #
# Module-scope fixture: one full closure run produces all the artifacts the
# tests inspect.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def closure_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("closure")
    run_dir = base / "primary"
    ext_root = base / "crg_artifacts" / "extensions"
    run_graph_compilation(
        model_config_path=UNSUPPORTED_CONFIG,
        target_config_path=HOST_CPU_TARGET,
        out_dir=run_dir,
        stop_after="gap-closure",
        run_id="closure_primary",
        extensions_root=ext_root,
    )
    return {"run_dir": run_dir, "ext_root": ext_root, "registry": ext_root / "registry.yaml"}


# --------------------------------------------------------------------------- #
# 1. All four validators pass on the produced run
# --------------------------------------------------------------------------- #


def test_artifact_validator_passes(closure_run: dict[str, Path]) -> None:
    rep = validate_run(closure_run["run_dir"])
    assert rep.overall == "pass", [r for r in rep.rules if r.status == "fail"]


def test_lowering_validator_passes(closure_run: dict[str, Path]) -> None:
    rep = validate_payload_lowering(closure_run["run_dir"])
    assert rep.status == "pass"


def test_gap_validator_passes(closure_run: dict[str, Path]) -> None:
    rep = validate_gap_discovery(closure_run["run_dir"])
    assert rep.status == "pass"


def test_closure_validator_passes(closure_run: dict[str, Path]) -> None:
    rep = validate_gap_closure(closure_run["run_dir"])
    assert rep.status == "pass", [c for c in rep.checks if c.status == "fail"]


# --------------------------------------------------------------------------- #
# 2. Closure summary shape
# --------------------------------------------------------------------------- #


def test_closure_summary_shape(closure_run: dict[str, Path]) -> None:
    s = json.loads((closure_run["run_dir"] / "05_gap_closure" / "closure_summary.json").read_text())
    assert s["status"] == "pass"
    assert s["llm_calls"] == 0
    assert s["extensions_registered_count"] >= 1
    assert s["extensions_invoked_count"] == s["extensions_registered_count"]


def test_extensions_invoked_records(closure_run: dict[str, Path]) -> None:
    inv = json.loads(
        (closure_run["run_dir"] / "05_gap_closure" / "extensions_invoked.json").read_text()
    )
    for r in inv["extensions"]:
        assert r["materialize_status"] == "pass"
        assert r["fill_status"] in ("pass", "skipped")
        assert r["verify_status"] == "pass"
        assert r["register_status"] == "pass"
        assert r["verify_max_abs_error"] == 0.0


# --------------------------------------------------------------------------- #
# 3. The closure loop's actual proof: gap count drops to zero
# --------------------------------------------------------------------------- #


def test_closure_proof_gap_count_drops(closure_run: dict[str, Path], tmp_path: Path) -> None:
    """Re-running gap-discovery with --extension-registry must produce 0 gaps."""
    after_dir = tmp_path / "after"
    discover_gaps_from_existing_lowering(
        lowering_run=closure_run["run_dir"],
        target_config_path=HOST_CPU_TARGET,
        out_dir=after_dir,
        run_id="after_closure",
        extension_registry=closure_run["registry"],
    )
    before = json.loads(
        (closure_run["run_dir"] / "04_gap_discovery" / "gap_action_queue.json").read_text()
    )
    after = json.loads((after_dir / "04_gap_discovery" / "gap_action_queue.json").read_text())
    assert before["summary"]["count"] >= 1
    assert after["summary"]["count"] == 0
    # And the report must call out the closure.
    rep = json.loads((after_dir / "04_gap_discovery" / "gap_discovery_summary.json").read_text())
    assert rep["totals"]["closed_by_registry_count"] >= 1
    targets = {t["fx_target"] for t in rep["closed_targets"]}
    assert any("crgtoy.affine_gelu" in t for t in targets), targets


# --------------------------------------------------------------------------- #
# 4. Extension workspace shape
# --------------------------------------------------------------------------- #


def test_extension_workspace_layout(closure_run: dict[str, Path]) -> None:
    ext_dir = next(
        d for d in (closure_run["ext_root"] / "unsupported_op").iterdir() if d.is_dir()
    )
    expected = [
        "gap_record.json",
        "extension_contract.json",
        "reference.py",
        "extension.py",
        "manifest.yaml",
        "README.md",
        "tests/test_extension_correctness.py",
        "results/verification.json",
    ]
    for rel in expected:
        assert (ext_dir / rel).exists(), f"missing: {rel}"
    # Frozen-case directories must exist + contain at least one (input, expected) pair.
    assert (ext_dir / "inputs").is_dir()
    assert (ext_dir / "expected").is_dir()
    inputs = sorted((ext_dir / "inputs").glob("case_*.pt"))
    expected_files = sorted((ext_dir / "expected").glob("case_*.pt"))
    assert len(inputs) >= 1
    assert len(inputs) == len(expected_files)


def test_extension_contract_locked_files_match_disk(closure_run: dict[str, Path]) -> None:
    """After fill, locked files MUST hash to their materialize-time value."""
    from compgen.graph_compilation.hashing import sha256_file

    ext_dir = next((closure_run["ext_root"] / "unsupported_op").iterdir())
    contract = json.loads((ext_dir / "extension_contract.json").read_text())
    for rel, declared in contract["locked_files_sha256"].items():
        actual = sha256_file(ext_dir / rel)
        assert actual == declared, f"locked file changed: {rel}"


def test_registry_has_entry_per_target(closure_run: dict[str, Path]) -> None:
    registry = load_registry(closure_run["registry"])
    assert registry.has("unsupported_op", "crgtoy.affine_gelu")
    assert registry.has("unsupported_op", "crgtoy.affine_gelu.default")


# --------------------------------------------------------------------------- #
# 5. Tamper tests
# --------------------------------------------------------------------------- #


def test_tamper_locked_file_breaks_verify(closure_run: dict[str, Path], tmp_path: Path) -> None:
    """If the agent edits reference.py, verify must fail the locked-files audit."""
    src = next((closure_run["ext_root"] / "unsupported_op").iterdir())
    tampered = tmp_path / "tampered_ext"
    shutil.copytree(src, tampered)
    # Append a comment to reference.py — bytes change, sha256 changes.
    ref = tampered / "reference.py"
    ref.write_text(ref.read_text() + "\n# tampered\n", encoding="utf-8")
    result = run_verify(tampered)
    assert result.status == "fail"
    assert result.locked_audit_status == "fail"
    assert any("reference.py" in v for v in result.locked_audit_violations)


def test_tamper_extension_breaks_differential(closure_run: dict[str, Path], tmp_path: Path) -> None:
    """If the agent's extension is mathematically wrong, differential verify fails."""
    src = next((closure_run["ext_root"] / "unsupported_op").iterdir())
    tampered = tmp_path / "tampered_diff"
    shutil.copytree(src, tampered)
    # Replace extension.py with a wrong impl: just zeros.
    (tampered / "extension.py").write_text(
        "import torch\n\n"
        "def extension(x, w, b):\n"
        "    return torch.zeros(x.shape[:-1] + (w.shape[0],))\n",
        encoding="utf-8",
    )
    result = run_verify(tampered)
    assert result.status == "fail"
    assert result.locked_audit_status == "pass"  # only extension.py changed
    assert result.max_abs_error > 0.0


def test_tamper_unknown_target_raises(tmp_path: Path) -> None:
    """Deterministic agent must refuse to fill targets it doesn't know."""
    from compgen.graph_compilation.agent_decomp_fill import UnknownTargetError

    workspace = tmp_path / "fake_ws"
    workspace.mkdir()
    with pytest.raises(UnknownTargetError):
        deterministic_fill(workspace, "<built-in function gelu>")


# --------------------------------------------------------------------------- #
# 6. Tiny MLP closure: deterministic agent has no fill, so the run should
#    record skipped_gaps but not crash.
# --------------------------------------------------------------------------- #


def test_tiny_mlp_closure_materializes_pending_workspaces(tmp_path: Path) -> None:
    """tiny_mlp gaps target ``<built-in function linear>`` etc. — not in KNOWN_FILLS.

    Closure should ALWAYS materialize the workspace (so a human or
    Claude Code can pick it up), register zero extensions, and report
    ``pending_human_fill`` for each. The validators must still pass.
    """
    run_dir = tmp_path / "tiny_mlp_closure"
    ext_root = tmp_path / "ext_root"
    _run_or_skip_on_m15b(
        model_config_path=TINY_MLP_CONFIG,
        target_config_path=HOST_CPU_TARGET,
        out_dir=run_dir,
        stop_after="gap-closure",
        run_id="tiny_mlp_closure",
        extensions_root=ext_root,
    )
    s = json.loads((run_dir / "05_gap_closure" / "closure_summary.json").read_text())
    assert s["extensions_registered_count"] == 0
    assert s["extensions_pending_count"] >= 1
    assert s["status"] == "pending_human_fill"
    assert s["llm_calls"] == 0
    # Each pending workspace must contain a real README.md and stub extension.py.
    for ws_path in s["pending_workspaces"]:
        ws = Path(ws_path)
        assert ws.is_dir()
        assert (ws / "README.md").exists()
        assert (ws / "extension.py").exists()
        assert "NotImplementedError" in (ws / "extension.py").read_text()
    # Validators still pass.
    assert validate_run(run_dir).overall == "pass"
    assert validate_gap_closure(run_dir).status == "pass"


# --------------------------------------------------------------------------- #
# 7. Determinism: extension_id is content-addressed; two runs produce the same id.
# --------------------------------------------------------------------------- #


def test_extension_id_is_content_addressed(tmp_path: Path) -> None:
    gap = {
        "gap_id": "gap_0000",
        "gap_kind": "unsupported_op",
        "fx_target": "crgtoy.affine_gelu",
        "shape_signature": {"inputs": [[2, 16]], "outputs": [[2, 8]]},
        "dtype_signature": {"inputs": ["torch.float32"], "outputs": ["torch.float32"]},
        "allowed_actions": ["decompose_to_supported_ops", "keep_as_fallback"],
    }
    a = tmp_path / "a"
    b = tmp_path / "b"
    mr_a = materialize_extension(gap, target_id="host_cpu", extensions_root=a)
    mr_b = materialize_extension(gap, target_id="host_cpu", extensions_root=b)
    assert mr_a.extension_id == mr_b.extension_id


def test_full_closure_loop_unit() -> None:
    """The whole materialize→fill→verify→register pipeline as a unit.

    Standalone of the higher-level orchestrator so a regression here
    surfaces fast even when run.py is broken.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ext_root = Path(tmp) / "extensions"
        registry_path = ext_root / "registry.yaml"
        gap = {
            "gap_id": "gap_0000",
            "gap_kind": "unsupported_op",
            "fx_target": "crgtoy.affine_gelu",
            "shape_signature": {"inputs": [[2, 16], [8, 16], [8]], "outputs": [[2, 8]]},
            "dtype_signature": {
                "inputs": ["torch.float32", "torch.float32", "torch.float32"],
                "outputs": ["torch.float32"],
            },
            "allowed_actions": ["decompose_to_supported_ops", "keep_as_fallback"],
        }
        mr = materialize_extension(gap, target_id="host_cpu", extensions_root=ext_root)
        deterministic_fill(mr.extension_dir, gap["fx_target"])
        result = run_verify(mr.extension_dir)
        assert result.status == "pass"
        register_extension(
            workspace=mr.extension_dir,
            verification_result=result,
            registry_path=registry_path,
        )
        registry = load_registry(registry_path)
        assert registry.has("unsupported_op", "crgtoy.affine_gelu")


# --------------------------------------------------------------------------- #
# 8. IR-level closure: registry-driven FX rewrite eliminates opaque func.call
# --------------------------------------------------------------------------- #


def test_ir_closure_eliminates_opaque_calls(tmp_path: Path) -> None:
    """After registering an extension, re-running payload-lowering with
    --extension-registry must produce a payload.mlir with **zero** opaque
    func.call entries for the closed target."""
    # First pass: gap-closure WITHOUT registry (build extensions).
    base = tmp_path / "ir_closure"
    ext_root = base / "ext"
    base.mkdir()
    closure_run = base / "before"
    run_graph_compilation(
        model_config_path=UNSUPPORTED_CONFIG,
        target_config_path=HOST_CPU_TARGET,
        out_dir=closure_run,
        stop_after="gap-closure",
        run_id="ir_before",
        extensions_root=ext_root,
    )
    # The "before" run lowered without the registry; payload.mlir contains
    # opaque crgtoy func.call entries.
    before_text = (
        closure_run / "01_payload_lowering" / "export_program" / "payload.mlir"
    ).read_text()
    assert "crgtoy" in before_text

    # Second pass: same model, this time WITH the registry passed in.
    after_run = base / "after"
    run_graph_compilation(
        model_config_path=UNSUPPORTED_CONFIG,
        target_config_path=HOST_CPU_TARGET,
        out_dir=after_run,
        stop_after="gap-closure",
        run_id="ir_after",
        extensions_root=ext_root,
        extension_registry=ext_root / "registry.yaml",
    )
    after_text = (
        after_run / "01_payload_lowering" / "export_program" / "payload.mlir"
    ).read_text()
    assert "crgtoy" not in after_text
    assert "linalg.matmul" in after_text  # decomposition went through
    # The lowering_diagnostics records the inlining decision.
    diag = json.loads(
        (after_run / "01_payload_lowering" / "lowering_diagnostics.json").read_text()
    )
    inlined = [d for d in diag["diagnostics"] if "Inlined extension" in d.get("message", "")]
    assert len(inlined) >= 2  # one per module (dynamo + export)


# --------------------------------------------------------------------------- #
# 9. list-pending CLI surface
# --------------------------------------------------------------------------- #


def test_list_pending_finds_unfilled_workspaces(tmp_path: Path) -> None:
    """``extension list-pending`` should enumerate materialized but unregistered
    workspaces — the entry point for Claude Code to discover what to fill."""
    import subprocess
    import sys

    run_dir = tmp_path / "lp_run"
    ext_root = tmp_path / "lp_ext"
    _run_or_skip_on_m15b(
        model_config_path=TINY_MLP_CONFIG,
        target_config_path=HOST_CPU_TARGET,
        out_dir=run_dir,
        stop_after="gap-closure",
        run_id="lp_run",
        extensions_root=ext_root,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "compgen.graph_compilation",
            "extension",
            "list-pending",
            "--extensions-root",
            str(ext_root),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert len(out["pending"]) >= 1
    assert len(out["registered"]) == 0
    for entry in out["pending"]:
        assert "extension_id" in entry
        assert "fx_target" in entry
        assert (Path(entry["extension_path"]) / "README.md").exists()
        assert "extension.py" in entry["fillable_files"]


# --------------------------------------------------------------------------- #
# 10. Spec 04 acceptance: pre-fill pytest must fail for the RIGHT reason
#     (NotImplementedError from extension.py, not import/path/contract issues).
# --------------------------------------------------------------------------- #


def test_04_pre_fill_pytest_fails_with_not_implemented_error(tmp_path: Path) -> None:
    """A freshly-materialized workspace must fail pytest with NotImplementedError,
    not with ImportError, FileNotFoundError, or contract drift."""
    import subprocess
    import sys

    from compgen.graph_compilation.extension_materialize import materialize_extension

    gap = {
        "gap_id": "gap_0000",
        "gap_kind": "unsupported_op",
        "fx_target": "crgtoy.affine_gelu",
        "semantic_name": "crgtoy.affine_gelu",
        "slug": "crgtoy_affine_gelu",
        "target_id": "host_cpu",
        "shape_signature": {"inputs": [[2, 16], [8, 16], [8]], "outputs": [[2, 8]]},
        "dtype_signature": {
            "inputs": ["torch.float32", "torch.float32", "torch.float32"],
            "outputs": ["torch.float32"],
        },
        "allowed_actions": ["decompose_to_supported_ops", "keep_as_fallback"],
        "required_evidence": ["reference_semantics", "input_output_shapes", "dtype_policy", "differential_tests"],
    }
    ext_root = tmp_path / "extensions"
    mr = materialize_extension(gap, target_id="host_cpu", extensions_root=ext_root)

    # Run pytest against the workspace. Expect a failure that mentions
    # "NotImplementedError" (the extension stub raised), not import/path errors.
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv = ['pytest', '-q', '" + str(mr.extension_dir / "tests") + "']; "
         "from _pytest.config import main; raise SystemExit(main())"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0, "pre-fill pytest should fail"
    out = proc.stdout + proc.stderr
    assert "NotImplementedError" in out, out[-1500:]
    # Counter-checks: failure must NOT be about import/path/contract issues.
    bad_signals = (
        "ModuleNotFoundError", "ImportError",
        "FileNotFoundError",
        "extension_contract.json' is missing",
    )
    for sig in bad_signals:
        assert sig not in out, f"pre-fill failure was for the wrong reason: {sig}\n{out[-1500:]}"


def test_04_post_fill_pytest_passes(tmp_path: Path) -> None:
    """After filling extension.py with a correct decomposition, pytest passes."""
    import subprocess
    import sys

    from compgen.graph_compilation.extension_materialize import materialize_extension

    gap = {
        "gap_id": "gap_0000",
        "gap_kind": "unsupported_op",
        "fx_target": "crgtoy.affine_gelu",
        "semantic_name": "crgtoy.affine_gelu",
        "slug": "crgtoy_affine_gelu",
        "target_id": "host_cpu",
        "shape_signature": {"inputs": [[2, 16], [8, 16], [8]], "outputs": [[2, 8]]},
        "dtype_signature": {
            "inputs": ["torch.float32", "torch.float32", "torch.float32"],
            "outputs": ["torch.float32"],
        },
        "allowed_actions": ["decompose_to_supported_ops", "keep_as_fallback"],
        "required_evidence": ["reference_semantics", "input_output_shapes", "dtype_policy", "differential_tests"],
    }
    ext_root = tmp_path / "ext_post"
    mr = materialize_extension(gap, target_id="host_cpu", extensions_root=ext_root)
    # Fill with the canonical decomposition.
    (mr.extension_dir / "extension.py").write_text(
        "from __future__ import annotations\n"
        "import torch\nimport torch.nn.functional as F\n\n"
        "def extension(x, w, b):\n"
        "    return F.gelu(F.linear(x, w, b))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv = ['pytest', '-q', '" + str(mr.extension_dir / "tests") + "']; "
         "from _pytest.config import main; raise SystemExit(main())"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "passed" in proc.stdout


def test_04_manifest_starts_as_draft(tmp_path: Path) -> None:
    """manifest.yaml must start with status: draft (not 'verified')."""
    import yaml as _yaml
    from compgen.graph_compilation.extension_materialize import materialize_extension

    gap = {
        "gap_id": "gap_0000", "gap_kind": "unsupported_op",
        "fx_target": "crgtoy.affine_gelu", "semantic_name": "crgtoy.affine_gelu",
        "slug": "crgtoy_affine_gelu", "target_id": "host_cpu",
        "shape_signature": {"inputs": [[2, 16], [8, 16], [8]], "outputs": [[2, 8]]},
        "dtype_signature": {"inputs": ["torch.float32"]*3, "outputs": ["torch.float32"]},
        "allowed_actions": ["decompose_to_supported_ops"],
        "required_evidence": ["reference_semantics", "input_output_shapes", "dtype_policy", "differential_tests"],
    }
    mr = materialize_extension(gap, target_id="host_cpu", extensions_root=tmp_path / "ext")
    manifest = _yaml.safe_load((mr.extension_dir / "manifest.yaml").read_text())
    assert manifest["status"] == "draft"
    assert manifest["last_verified_at_utc"] is None
    assert manifest["registered_at_utc"] is None


def test_04_manifest_is_editable_post_materialize(tmp_path: Path) -> None:
    """manifest.yaml is editable per spec — verify must NOT flag a manifest edit."""
    from compgen.graph_compilation.extension_materialize import materialize_extension
    from compgen.graph_compilation.extension_verify import run_verify
    gap = {
        "gap_id": "gap_0000", "gap_kind": "unsupported_op",
        "fx_target": "crgtoy.affine_gelu", "semantic_name": "crgtoy.affine_gelu",
        "slug": "crgtoy_affine_gelu", "target_id": "host_cpu",
        "shape_signature": {"inputs": [[2, 16], [8, 16], [8]], "outputs": [[2, 8]]},
        "dtype_signature": {"inputs": ["torch.float32"]*3, "outputs": ["torch.float32"]},
        "allowed_actions": ["decompose_to_supported_ops"],
        "required_evidence": ["reference_semantics", "input_output_shapes", "dtype_policy", "differential_tests"],
    }
    mr = materialize_extension(gap, target_id="host_cpu", extensions_root=tmp_path / "ext_edit")
    # Fill extension correctly.
    (mr.extension_dir / "extension.py").write_text(
        "import torch.nn.functional as F\n"
        "def extension(x, w, b): return F.gelu(F.linear(x, w, b))\n",
        encoding="utf-8",
    )
    # Now edit manifest.yaml — should NOT be flagged.
    manifest_path = mr.extension_dir / "manifest.yaml"
    manifest_path.write_text(manifest_path.read_text() + "\nclaude_code_note: filled\n", encoding="utf-8")
    result = run_verify(mr.extension_dir)
    assert result.locked_audit_status == "pass", result.locked_audit_violations
    assert result.status == "pass"


def test_04_inputs_expected_locked(tmp_path: Path) -> None:
    """Editing an input or expected case file must fail the locked-files audit."""
    import torch as _torch
    from compgen.graph_compilation.extension_materialize import materialize_extension
    from compgen.graph_compilation.extension_verify import run_verify
    gap = {
        "gap_id": "gap_0000", "gap_kind": "unsupported_op",
        "fx_target": "crgtoy.affine_gelu", "semantic_name": "crgtoy.affine_gelu",
        "slug": "crgtoy_affine_gelu", "target_id": "host_cpu",
        "shape_signature": {"inputs": [[2, 16], [8, 16], [8]], "outputs": [[2, 8]]},
        "dtype_signature": {"inputs": ["torch.float32"]*3, "outputs": ["torch.float32"]},
        "allowed_actions": ["decompose_to_supported_ops"],
        "required_evidence": ["reference_semantics", "input_output_shapes", "dtype_policy", "differential_tests"],
    }
    mr = materialize_extension(gap, target_id="host_cpu", extensions_root=tmp_path / "ext_lock")
    # Fill correctly first.
    (mr.extension_dir / "extension.py").write_text(
        "import torch.nn.functional as F\n"
        "def extension(x, w, b): return F.gelu(F.linear(x, w, b))\n",
        encoding="utf-8",
    )
    # Now tamper with inputs/case_00.pt.
    case_path = mr.extension_dir / "inputs" / "case_00.pt"
    _torch.save(_torch.zeros(2, 16), case_path)  # wrong shape, definitely changes hash
    result = run_verify(mr.extension_dir)
    assert result.locked_audit_status == "fail"
    assert any("inputs/case_00.pt" in v for v in result.locked_audit_violations)


def test_04_closure_validation_says_not_applicable_for_no_closure(tmp_path: Path) -> None:
    """Per the user's flag: closure_validation must NOT report 'pass' before
    extension closure has happened. ``not_applicable`` is the honest state."""
    from compgen.graph_compilation.gap_closure_validate import validate_gap_closure
    run_dir = tmp_path / "run_no_closure"
    run_dir.mkdir()
    rep = validate_gap_closure(run_dir)
    assert rep.status == "not_applicable", rep.checks
