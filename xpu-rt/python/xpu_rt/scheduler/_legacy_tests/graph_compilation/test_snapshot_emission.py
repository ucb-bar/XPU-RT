"""emit IR snapshots from a real run directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from compgen.analysis.ir_snapshots import (
    IR_LEVELS,
    SNAPSHOT_FILENAMES,
    load_snapshot,
)
from compgen.graph_compilation.snapshot_emitter import (
    EmitResult,
    emit_snapshots_for_run,
)


# ---------------------------------------------------------------------------
# A synthetic run-dir with REAL artifacts, just minimal ones, lets us
# exercise every per-level builder without needing a 30-minute pipeline.
# ---------------------------------------------------------------------------


def _make_minimal_run_dir(root: Path) -> Path:
    """Build a run-dir with the minimum-viable artifacts the emitter
    consumes for each level."""

    root.mkdir(parents=True, exist_ok=True)
    # fx_graph
    (root / "exported_program.pt2").write_bytes(b"\x00\x01")
    (root / "graph_breaks.json").write_text(
        json.dumps(
            {
                "regions": [
                    {"region_id": "r0", "ops": ["aten.matmul"]},
                    {"region_id": "r1", "ops": ["aten.relu"]},
                ]
            }
        )
    )
    # payload_ir
    (root / "payload.mlir").write_text(
        """\
module {
  func.func @main() {
    %0 = linalg.matmul ins(%arg0, %arg1)
    %1 = arith.addf(%0, %bias)
    %2 = linalg.relu(%1)
    return
  }
}
"""
    )
    # recipe_ir
    (root / "recipe_main.yaml").write_text("recipe: main\n")
    (root / "gap_analysis.json").write_text(
        json.dumps(
            {
                "decisions": [
                    {"region_id": "r0", "ops": ["FuseElementwise"], "rationale": "x"},
                ]
            }
        )
    )
    # tile_ir
    (root / "transforms").mkdir()
    (root / "transforms" / "tile_matmul.mlir").write_text(
        "transform.sequence { transform.tile %0 by [32, 32] }\n"
    )
    # dialect_ir + kernel_artifact
    gen = root / "generated_kernels"
    gen.mkdir()
    triton_dir = gen / "triton"
    triton_dir.mkdir()
    (triton_dir / "kernel.py").write_text("import triton\n")
    cffi_dir = gen / "cffi_c"
    cffi_dir.mkdir()
    (cffi_dir / "kernel.c").write_text("/* matmul kernel */\nint main(){}\n")
    # execution_plan
    (root / "execution_plan.yaml").write_text(
        yaml.safe_dump(
            {
                "operations": [
                    {"region_id": "r0", "kind": "matmul", "device": "host_cpu"},
                ]
            }
        )
    )
    (root / "memory_plan.yaml").write_text(
        yaml.safe_dump({"buffers": [{"id": "b0", "bytes": 1024}]})
    )
    # runtime_profile
    (root / "verification_report.json").write_text(
        json.dumps({"status": "ok", "levels_run": ["structural", "differential"]})
    )
    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_emit_snapshots_for_run_writes_all_eight_levels(tmp_path: Path):
    rd = _make_minimal_run_dir(tmp_path / "run_a")
    results = emit_snapshots_for_run(rd)
    assert set(results.keys()) == set(IR_LEVELS)
    for level in IR_LEVELS:
        snap_path = rd / "02_graph_analysis" / "analysis_snapshots" / SNAPSHOT_FILENAMES[level]
        assert snap_path.is_file()


def test_every_level_populated_from_real_artifacts(tmp_path: Path):
    """When every artifact is present, every level emits ``available``."""

    rd = _make_minimal_run_dir(tmp_path / "run_b")
    results = emit_snapshots_for_run(rd)
    for level, r in results.items():
        assert r.status == "available", (
            f"{level} did not flip available: reason={r.not_available_reason}"
        )
        assert r.region_count >= 1


def test_missing_artifacts_produce_typed_not_available(tmp_path: Path):
    """Empty run-dir → 8 typed not_available snapshots, never a crash."""

    rd = tmp_path / "empty_run"
    rd.mkdir()
    results = emit_snapshots_for_run(rd)
    for level, r in results.items():
        assert r.status == "not_available"
        assert r.not_available_reason in (
            "artifact_missing",
            "stage_not_run",
        )


def test_payload_ir_parses_real_ops(tmp_path: Path):
    rd = _make_minimal_run_dir(tmp_path / "run_payload")
    emit_snapshots_for_run(rd)
    snap = load_snapshot(
        rd / "02_graph_analysis" / "analysis_snapshots" / "payload_ir_analysis.json"
    )
    assert snap.status == "available"
    region = snap.regions[0]
    # The payload.mlir we wrote has linalg.matmul, arith.addf, linalg.relu.
    op_kinds = set(region.ops)
    assert any("linalg" in op for op in op_kinds)
    assert any("arith" in op for op in op_kinds)


def test_dialect_ir_picks_up_per_provider_directories(tmp_path: Path):
    rd = _make_minimal_run_dir(tmp_path / "run_dialect")
    emit_snapshots_for_run(rd)
    snap = load_snapshot(
        rd / "02_graph_analysis" / "analysis_snapshots" / "dialect_ir_analysis.json"
    )
    assert snap.status == "available"
    region_ids = {r.region_id for r in snap.regions}
    assert "dialect:triton" in region_ids
    assert "dialect:cffi_c" in region_ids


def test_runtime_profile_pulls_verification_levels(tmp_path: Path):
    rd = _make_minimal_run_dir(tmp_path / "run_profile")
    emit_snapshots_for_run(rd)
    snap = load_snapshot(
        rd / "02_graph_analysis" / "analysis_snapshots" / "runtime_profile_analysis.json"
    )
    assert snap.status == "available"
    region = snap.regions[0]
    assert "structural" in region.ops
    assert "differential" in region.ops


def test_emit_is_idempotent(tmp_path: Path):
    rd = _make_minimal_run_dir(tmp_path / "run_idem")
    a = emit_snapshots_for_run(rd)
    b = emit_snapshots_for_run(rd)
    # Same set of paths returned, same statuses.
    assert {(level, r.status) for level, r in a.items()} == {
        (level, r.status) for level, r in b.items()
    }


def test_nonexistent_run_dir_raises():
    with pytest.raises(FileNotFoundError):
        emit_snapshots_for_run("/tmp/does_not_exist_compgen_run_xyz")


def test_emit_result_carries_path_and_region_count(tmp_path: Path):
    rd = _make_minimal_run_dir(tmp_path / "run_meta")
    results = emit_snapshots_for_run(rd)
    fx = results["fx_graph"]
    assert isinstance(fx, EmitResult)
    assert fx.region_count >= 1
    assert fx.path.is_file()
    assert fx.path.name == "fx_graph_analysis.json"


def test_partial_run_mixes_available_and_not_available(tmp_path: Path):
    """A run-dir with only payload.mlir produces 1 available + 7 not_available."""

    rd = tmp_path / "run_partial"
    rd.mkdir()
    (rd / "payload.mlir").write_text("module { }\n")
    results = emit_snapshots_for_run(rd)
    assert results["payload_ir"].status == "available"
    for level in IR_LEVELS:
        if level == "payload_ir":
            continue
        assert results[level].status == "not_available", (
            f"{level} unexpectedly available"
        )
