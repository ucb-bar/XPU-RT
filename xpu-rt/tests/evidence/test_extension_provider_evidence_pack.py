"""extension/provider evidence pack tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_builder(probe_dir: Path, out_dir: Path, snapshots_dir: Path | None = None) -> int:
    """Invoke the evidence-pack CLI in-process style."""
    cmd = [
        sys.executable,
        "scripts/dev/build_extension_provider_evidence_pack.py",
        "--probe-dir", str(probe_dir),
        "--out", str(out_dir),
    ]
    if snapshots_dir is not None:
        cmd += ["--snapshots-dir", str(snapshots_dir)]
    return subprocess.run(cmd, check=False).returncode


def _build_probe_dir(probe_dir: Path) -> None:
    """Bootstrap a probe report set in ``probe_dir``."""
    from compgen.providers.provider_reports import write_probe_reports
    write_probe_reports(probe_dir)


def test_evidence_pack_builds(tmp_path: Path):
    probe = tmp_path / "probe"
    out = tmp_path / "evidence_pack"
    _build_probe_dir(probe)
    rc = _run_builder(probe, out)
    assert rc == 0
    assert (out / "claim_matrix.json").is_file()
    assert (out / "architecture_audit.json").is_file()
    assert (out / "extension_summary.md").is_file()
    assert (out / "provider_status.json").is_file()
    assert (out / "dialect_provider_registry.json").is_file()
    assert (out / "pass_tool_registry.json").is_file()
    assert (out / "figures").is_dir()


def test_claim_matrix_has_twelve_rows(tmp_path: Path):
    probe = tmp_path / "probe"
    out = tmp_path / "evidence_pack"
    _build_probe_dir(probe)
    _run_builder(probe, out)
    body = json.loads((out / "claim_matrix.json").read_text())
    assert body["schema_version"] == "claim_matrix_v1"
    assert len(body["rows"]) == 12
    claims = {r["claim"] for r in body["rows"]}
    assert claims == {
        "extension_manifest_validated",
        "extension_sandbox_enforced",
        "provider_registry_present",
        "target_registry_present",
        "dialect_provider_registry_present",
        "pass_tool_registry_present",
        "provider_probe_typed",
        "provider_routing_deterministic",
        "unsupported_op_task_emitted",
        "extension_registered_after_verification",
        "multi_level_analysis_snapshots_present",
        "certified_artifacts_only_executed",
    }


def test_claim_status_is_typed_enum(tmp_path: Path):
    probe = tmp_path / "probe"
    out = tmp_path / "evidence_pack"
    _build_probe_dir(probe)
    _run_builder(probe, out)
    body = json.loads((out / "claim_matrix.json").read_text())
    for row in body["rows"]:
        assert row["status"] in ("implemented", "partial_scope", "blocked", "not_run"), row


def test_snapshot_dir_flips_corresponding_claim_to_implemented(tmp_path: Path):
    """When --snapshots-dir is supplied, the
    multi_level_analysis_snapshots_present row should flip to
    `implemented`."""
    from compgen.analysis.ir_snapshots import (
        RegionSummary, make_available, write_snapshots,
    )
    probe = tmp_path / "probe"
    out = tmp_path / "evidence_pack"
    snapshots = tmp_path / "snapshots"
    _build_probe_dir(probe)
    fx = make_available(
        level="fx_graph", source_artifact="x",
        regions=[RegionSummary(region_id="r0", ops=("a",))],
    )
    write_snapshots({"fx_graph": fx}, snapshots)
    _run_builder(probe, out, snapshots_dir=snapshots)
    body = json.loads((out / "claim_matrix.json").read_text())
    row = next(r for r in body["rows"] if r["claim"] == "multi_level_analysis_snapshots_present")
    assert row["status"] == "implemented"


def test_architecture_audit_artifact_present_and_passes(tmp_path: Path):
    probe = tmp_path / "probe"
    out = tmp_path / "evidence_pack"
    _build_probe_dir(probe)
    _run_builder(probe, out)
    body = json.loads((out / "architecture_audit.json").read_text())
    assert body["passed"] is True
    assert body["violation_count"] == 0


def test_evidence_summary_md_has_real_content(tmp_path: Path):
    probe = tmp_path / "probe"
    out = tmp_path / "evidence_pack"
    _build_probe_dir(probe)
    _run_builder(probe, out)
    text = (out / "extension_summary.md").read_text()
    assert "Phase F evidence pack" in text
    assert "Claim matrix" in text


def test_implemented_row_requires_real_evidence_path(tmp_path: Path):
    """A row marked `implemented` must reference at least one
    existing file in the pack."""
    probe = tmp_path / "probe"
    out = tmp_path / "evidence_pack"
    _build_probe_dir(probe)
    _run_builder(probe, out)
    body = json.loads((out / "claim_matrix.json").read_text())
    for row in body["rows"]:
        if row["status"] != "implemented":
            continue
        for evi in row["evidence"]:
            p = out / evi
            assert p.exists(), f"{row['claim']!r} marks implemented but {evi!r} is missing"
