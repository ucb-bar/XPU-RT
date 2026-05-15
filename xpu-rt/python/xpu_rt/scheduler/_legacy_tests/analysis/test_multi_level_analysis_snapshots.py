"""multi-level IR analysis snapshots schema + writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.analysis.ir_snapshots import (
    IR_LEVELS,
    NOT_AVAILABLE_REASONS,
    SCHEMA_VERSION,
    SNAPSHOT_FILENAMES,
    IRAnalysisSnapshot,
    IRSnapshotError,
    RegionSummary,
    UnsupportedProvider,
    discover_snapshots,
    load_snapshot,
    make_available,
    make_not_available,
    write_snapshots,
)


# ---------------------------------------------------------------------------
# Schema discipline
# ---------------------------------------------------------------------------


def test_eight_documented_levels():
    assert IR_LEVELS == (
        "fx_graph",
        "payload_ir",
        "recipe_ir",
        "tile_ir",
        "dialect_ir",
        "kernel_artifact",
        "execution_plan",
        "runtime_profile",
    )


def test_unknown_level_rejected():
    with pytest.raises(IRSnapshotError, match="unknown IR level"):
        IRAnalysisSnapshot(
            schema_version=SCHEMA_VERSION,
            level="totally_made_up",
            status="not_available",
            not_available_reason="stage_not_run",
        )


def test_unknown_status_rejected():
    with pytest.raises(IRSnapshotError, match="status"):
        IRAnalysisSnapshot(
            schema_version=SCHEMA_VERSION,
            level="fx_graph",
            status="wave_hands",
        )


def test_not_available_requires_typed_reason():
    with pytest.raises(IRSnapshotError, match="typed reason"):
        IRAnalysisSnapshot(
            schema_version=SCHEMA_VERSION,
            level="fx_graph",
            status="not_available",
        )


def test_not_available_untyped_reason_rejected():
    with pytest.raises(IRSnapshotError, match="not_available_reason"):
        IRAnalysisSnapshot(
            schema_version=SCHEMA_VERSION,
            level="fx_graph",
            status="not_available",
            not_available_reason="wave_hands",
        )


def test_not_available_must_not_carry_regions():
    with pytest.raises(IRSnapshotError, match="must not carry regions"):
        IRAnalysisSnapshot(
            schema_version=SCHEMA_VERSION,
            level="fx_graph",
            status="not_available",
            not_available_reason="stage_not_run",
            regions=(RegionSummary(region_id="x", ops=()),),
        )


def test_available_must_not_carry_not_available_reason():
    with pytest.raises(IRSnapshotError, match="not_available_reason"):
        IRAnalysisSnapshot(
            schema_version=SCHEMA_VERSION,
            level="fx_graph",
            status="available",
            not_available_reason="stage_not_run",
        )


def test_not_available_reasons_are_typed_enum():
    for reason in NOT_AVAILABLE_REASONS:
        snap = make_not_available(level="fx_graph", reason=reason)
        assert snap.not_available_reason == reason


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def _example_snapshot() -> IRAnalysisSnapshot:
    return make_available(
        level="payload_ir",
        source_artifact="payload.mlir",
        regions=[
            RegionSummary(
                region_id="region_017",
                ops=("linalg.matmul", "arith.addf"),
                supported_providers=("triton", "cffi_c"),
                unsupported_providers=(
                    UnsupportedProvider(
                        provider_id="cuda_tile_ir",
                        reason="toolchain_missing",
                    ),
                ),
                fusion_candidates=("matmul_bias_relu",),
                lowering_gaps=(),
            )
        ],
    )


def test_round_trip_through_dict():
    snap = _example_snapshot()
    body = snap.to_dict()
    restored = IRAnalysisSnapshot.from_dict(body)
    assert restored == snap


def test_round_trip_through_json(tmp_path: Path):
    snap = _example_snapshot()
    path = snap.write(tmp_path / "payload_ir_analysis.json")
    restored = load_snapshot(path)
    assert restored.level == snap.level
    assert restored.regions[0].region_id == "region_017"
    assert restored.regions[0].unsupported_providers[0].provider_id == "cuda_tile_ir"


# ---------------------------------------------------------------------------
# Bulk writer
# ---------------------------------------------------------------------------


def test_bulk_write_fills_missing_levels_with_not_available(tmp_path: Path):
    """The agent surface must be complete — every level gets a file
    even if no snapshot was supplied."""

    paths = write_snapshots({"fx_graph": _example_snapshot().to_dict() if False else make_available(
        level="fx_graph", source_artifact="exported_program.pt2",
        regions=[RegionSummary(region_id="r0", ops=("aten.matmul",))],
    )}, tmp_path)
    assert set(paths.keys()) == set(IR_LEVELS)
    for level, p in paths.items():
        assert p.is_file()
    # Levels we didn't supply are not_available with stage_not_run.
    runtime = load_snapshot(tmp_path / SNAPSHOT_FILENAMES["runtime_profile"])
    assert runtime.status == "not_available"
    assert runtime.not_available_reason == "stage_not_run"


def test_bulk_write_rejects_mislabeled_snapshot(tmp_path: Path):
    fx = make_available(
        level="fx_graph",
        source_artifact="x",
        regions=[RegionSummary(region_id="r0", ops=("a",))],
    )
    with pytest.raises(IRSnapshotError, match="mislabeled"):
        write_snapshots({"payload_ir": fx}, tmp_path)


def test_discover_snapshots_round_trips(tmp_path: Path):
    fx = make_available(
        level="fx_graph",
        source_artifact="exported_program.pt2",
        regions=[RegionSummary(region_id="r0", ops=("aten.matmul",))],
    )
    payload = make_available(
        level="payload_ir",
        source_artifact="payload.mlir",
        regions=[RegionSummary(region_id="r0", ops=("linalg.matmul",))],
    )
    write_snapshots({"fx_graph": fx, "payload_ir": payload}, tmp_path)
    found = discover_snapshots(tmp_path)
    assert set(found.keys()) == set(IR_LEVELS)
    assert found["fx_graph"].status == "available"
    assert found["payload_ir"].status == "available"
    assert found["runtime_profile"].status == "not_available"
    assert found["runtime_profile"].not_available_reason == "stage_not_run"


def test_no_silent_omission_of_levels(tmp_path: Path):
    """Hard rule: even with zero supplied snapshots, all 8 levels emit
    a typed not_available file. No silent missing levels."""

    paths = write_snapshots({}, tmp_path)
    assert len(paths) == 8
    for level, p in paths.items():
        body = json.loads(p.read_text())
        assert body["level"] == level
        assert body["status"] == "not_available"
        assert body["not_available_reason"] in NOT_AVAILABLE_REASONS


def test_filenames_are_documented():
    """Snapshot files follow the ``<level>_analysis.json`` convention."""

    for level in IR_LEVELS:
        assert SNAPSHOT_FILENAMES[level] == f"{level}_analysis.json"


# ---------------------------------------------------------------------------
# RegionSummary discipline
# ---------------------------------------------------------------------------


def test_region_summary_round_trip():
    region = RegionSummary(
        region_id="r0",
        ops=("a", "b"),
        supported_providers=("triton",),
        unsupported_providers=(
            UnsupportedProvider(provider_id="cuda_tile_ir", reason="toolchain_missing"),
        ),
        fusion_candidates=("af",),
        lowering_gaps=("g1",),
        extras={"flops": 8.4e9},
    )
    restored = RegionSummary.from_dict(region.to_dict())
    assert restored == region


def test_region_with_no_extras_omits_extras_key():
    region = RegionSummary(region_id="r0", ops=("a",))
    body = region.to_dict()
    assert "extras" not in body
