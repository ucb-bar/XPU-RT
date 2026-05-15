"""graph_capture stage: Stage 0 capture acceptance tests.

These tests run real torch.export + Dynamo against the tiny MLP. They
are not mocked. Per the project's anti-mock policy, the implementation
is exercised end-to-end against a real workload.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from compgen.graph_compilation import validate_run
from compgen.graph_compilation.replay import replay_goldens
from compgen.graph_compilation.run import run_graph_compilation

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_MLP_CONFIG = REPO_ROOT / "configs" / "models" / "tiny_mlp.yaml"
HOST_CPU_TARGET = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"


# --------------------------------------------------------------------------- #
# Fixture: one fully-built graph_capture stage run, reused across tests where safe.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def graph_compilation_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("graph_compilation_") / "run"
    run_graph_compilation(
        model_config_path=TINY_MLP_CONFIG,
        target_config_path=HOST_CPU_TARGET,
        out_dir=out,
        stop_after="graph-capture",
        run_id="graph_compilation_module_run",
    )
    return out


# --------------------------------------------------------------------------- #
# 1. Clean capture: every required artifact is produced
# --------------------------------------------------------------------------- #


def test_required_artifacts_present(graph_compilation_run: Path) -> None:
    capture_dir = graph_compilation_run / "00_graph_capture"
    expected = [
        "golden_inputs.pt",
        "golden_outputs.pt",
        "dynamo_summary.json",
        "exported_program.pt2",
        "export_graph.json",
        "export_graph_readable.py",
        "graph_breaks.json",
        "compile_baseline.json",
        "capture_report.json",
    ]
    for rel in expected:
        assert (capture_dir / rel).exists(), f"missing artifact: {rel}"
    # at least one Dynamo partition
    parts = sorted((capture_dir / "dynamo_partitions").glob("partition_*_graph.json"))
    assert parts, "no Dynamo partitions captured"


def test_artifact_contract_validator_accepts_run(graph_compilation_run: Path) -> None:
    report = validate_run(graph_compilation_run)
    assert report.overall == "pass", [r for r in report.rules if r.status == "fail"]


def test_replay_goldens_pass_on_clean_run(graph_compilation_run: Path) -> None:
    result = replay_goldens(graph_compilation_run)
    assert result.status == "pass"
    assert result.mode == "exported_program"
    assert result.max_abs_error == 0.0
    assert result.max_rel_error == 0.0


# --------------------------------------------------------------------------- #
# 2. Real-content tests: report is not hardcoded
# --------------------------------------------------------------------------- #


def test_capture_report_has_real_content(graph_compilation_run: Path) -> None:
    report = json.loads((graph_compilation_run / "00_graph_capture" / "capture_report.json").read_text())
    assert report["schema_version"] == "capture_report_v1"
    assert report["model_id"] == "tiny_mlp"
    assert report["target_id"] == "host_cpu"
    assert report["primary_capture"] == "torch_dynamo"
    assert report["llm_calls"] == 0
    assert report["torch_dynamo"]["partition_count"] >= 1
    assert report["torch_export"]["status"] == "pass"
    # The TinyMLP exported graph has multiple ops; reject the
    # "0 nodes / hardcoded zeros" failure mode.
    assert report["torch_export"]["num_ops"] >= 3


def test_export_graph_has_real_fx_nodes(graph_compilation_run: Path) -> None:
    eg = json.loads((graph_compilation_run / "00_graph_capture" / "export_graph.json").read_text())
    assert eg["schema_version"] == "fx_graph_v1"
    summary = eg["summary"]
    # TinyMLP: 2 Linear + GELU = at least 3 call_function nodes after default decomp.
    assert summary["num_nodes"] >= 5
    assert summary["num_call_function"] >= 1
    assert summary["num_placeholders"] >= 1
    assert summary["num_outputs"] >= 1
    assert eg["graph_hash"].startswith("sha256:")


def test_dynamo_summary_partitions_match_disk(graph_compilation_run: Path) -> None:
    summary = json.loads((graph_compilation_run / "00_graph_capture" / "dynamo_summary.json").read_text())
    on_disk = sorted((graph_compilation_run / "00_graph_capture" / "dynamo_partitions").glob("partition_*_graph.json"))
    assert summary["partition_count"] == len(on_disk) >= 1
    for partition_meta, jpath in zip(summary["partitions"], on_disk):
        body = json.loads(jpath.read_text())
        assert partition_meta["graph_hash"] == body["graph_hash"]
        assert partition_meta["num_nodes"] == body["summary"]["num_nodes"]


def test_graph_breaks_report_present_and_zero(graph_compilation_run: Path) -> None:
    """Tiny MLP has no Python control flow; no graph breaks expected."""
    obj = json.loads((graph_compilation_run / "00_graph_capture" / "graph_breaks.json").read_text())
    assert obj["schema_version"] == "graph_breaks_v1"
    assert obj["graph_break_count"] == 0
    assert obj["graph_breaks"] == []


def test_compile_baseline_is_recorded_explicitly(graph_compilation_run: Path) -> None:
    """Even when torch.compile passes, the report must declare a status field."""
    obj = json.loads((graph_compilation_run / "00_graph_capture" / "compile_baseline.json").read_text())
    assert obj["schema_version"] == "compile_baseline_v1"
    assert obj["attempted"] is True
    assert obj["status"] in {"pass", "fail", "skipped"}
    if obj["status"] != "pass":
        assert obj.get("error"), "non-pass compile_baseline must carry an error message"


# --------------------------------------------------------------------------- #
# 3. Tamper / negative tests
# --------------------------------------------------------------------------- #


def test_deleted_exported_program_fails_validator(graph_compilation_run: Path, tmp_path: Path) -> None:
    tampered = tmp_path / "tampered_no_export"
    shutil.copytree(graph_compilation_run, tampered)
    (tampered / "00_graph_capture" / "exported_program.pt2").unlink()
    report = validate_run(tampered)
    assert report.overall == "fail"


def test_corrupted_goldens_fail_replay(graph_compilation_run: Path, tmp_path: Path) -> None:
    tampered = tmp_path / "tampered_goldens"
    shutil.copytree(graph_compilation_run, tampered)
    target = tampered / "00_graph_capture" / "golden_outputs.pt"
    expected = torch.load(target, weights_only=False)
    if isinstance(expected, (list, tuple)):
        expected = tuple(expected)
    else:
        expected = (expected,)
    # Corrupt: add a perturbation.
    perturbed = tuple(t + 1.0 if isinstance(t, torch.Tensor) else t for t in expected)
    torch.save(perturbed, target)

    result = replay_goldens(tampered)
    assert result.status == "fail"
    assert result.max_abs_error > 0.0


# --------------------------------------------------------------------------- #
# 4. Anti-coupling tests
# --------------------------------------------------------------------------- #


def test_existing_capture_files_not_modified() -> None:
    """graph_capture stage must not modify files under compgen.capture / pipeline / runtime."""
    forbidden = [
        "python/compgen/capture/torch_export.py",
        "python/compgen/capture/torch_mlir_bridge.py",
        "python/compgen/capture/dynamo_baseline.py",
        "python/compgen/ir/payload/import_fx.py",
        "python/compgen/pipeline/driver.py",
        "python/compgen/runtime/bundle_emit.py",
    ]
    # Use git to check mtime/working-copy delta against HEAD.
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD", "--"] + forbidden,
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        pytest.skip("git unavailable in this environment")
    changed = [line.strip() for line in diff.splitlines() if line.strip()]
    assert not changed, f"forbidden files modified: {changed}"


def test_llm_calls_zero_in_manifest_and_report(graph_compilation_run: Path) -> None:
    manifest = json.loads((graph_compilation_run / "run_manifest.json").read_text())
    for stage in manifest["stages"]:
        assert stage["llm_calls"] == 0, f"stage {stage['stage_id']} has llm_calls={stage['llm_calls']}"
    report = json.loads((graph_compilation_run / "00_graph_capture" / "capture_report.json").read_text())
    assert report["llm_calls"] == 0


# --------------------------------------------------------------------------- #
# 5. Determinism: two runs produce the same stable hashes
# --------------------------------------------------------------------------- #


def test_two_runs_have_matching_graph_hashes(tmp_path: Path) -> None:
    run_a = tmp_path / "repro_a"
    run_b = tmp_path / "repro_b"
    run_graph_compilation(
        model_config_path=TINY_MLP_CONFIG,
        target_config_path=HOST_CPU_TARGET,
        out_dir=run_a,
        stop_after="graph-capture",
        run_id="repro_a",
    )
    run_graph_compilation(
        model_config_path=TINY_MLP_CONFIG,
        target_config_path=HOST_CPU_TARGET,
        out_dir=run_b,
        stop_after="graph-capture",
        run_id="repro_b",
    )

    rep_a = json.loads((run_a / "00_graph_capture" / "capture_report.json").read_text())
    rep_b = json.loads((run_b / "00_graph_capture" / "capture_report.json").read_text())

    # Stable fields.
    assert rep_a["torch_dynamo"]["partition_count"] == rep_b["torch_dynamo"]["partition_count"]
    assert rep_a["torch_dynamo"]["graph_break_count"] == rep_b["torch_dynamo"]["graph_break_count"]
    assert rep_a["torch_export"]["graph_hash"] == rep_b["torch_export"]["graph_hash"]
    assert rep_a["torch_export"]["num_ops"] == rep_b["torch_export"]["num_ops"]
    assert rep_a["model_id"] == rep_b["model_id"]
    assert rep_a["seed"] == rep_b["seed"]


def test_compare_command_reports_pass(tmp_path: Path) -> None:
    run_a = tmp_path / "cmp_a"
    run_b = tmp_path / "cmp_b"
    for out in (run_a, run_b):
        run_graph_compilation(
            model_config_path=TINY_MLP_CONFIG,
            target_config_path=HOST_CPU_TARGET,
            out_dir=out,
            stop_after="graph-capture",
            run_id=out.name,
        )
    from compgen.graph_compilation.compare import compare_runs

    rep = compare_runs(run_a, run_b)
    assert rep.overall == "pass", rep.mismatches


# --------------------------------------------------------------------------- #
# 6. CLI smoke
# --------------------------------------------------------------------------- #


def test_cli_run_then_validate_then_replay(tmp_path: Path) -> None:
    out = tmp_path / "cli_run"
    rc1 = subprocess.run(
        [
            sys.executable,
            "-m",
            "compgen.graph_compilation",
            "run",
            "--model",
            str(TINY_MLP_CONFIG),
            "--target",
            str(HOST_CPU_TARGET),
            "--out",
            str(out),
            "--stop-after",
            "graph-capture",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert rc1.returncode == 0, rc1.stderr

    rc2 = subprocess.run(
        [sys.executable, "-m", "compgen.graph_compilation", "validate", "--run", str(out)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert rc2.returncode == 0, rc2.stderr

    rc3 = subprocess.run(
        [sys.executable, "-m", "compgen.graph_compilation", "replay-goldens", "--run", str(out)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert rc3.returncode == 0, rc3.stderr
