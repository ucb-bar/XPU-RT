"""Tests for :mod:`xpu_rt.benchmarks.canonical_metrics`."""

from __future__ import annotations

from pathlib import Path

import pytest

from xpu_rt.benchmarks.canonical_metrics import (
    BACKENDS,
    CYCLE_SOURCES,
    CanonicalCellRow,
    TARGETS,
    WORKLOADS,
    merge_jsonl,
    parse_matmul_shape_id,
    read_jsonl,
    shape_id_for_matmul,
    write_jsonl,
)


def _ok_row(**overrides) -> CanonicalCellRow:
    base = dict(
        backend="kb-v2",
        target="gemmini_mx",
        workload="smolvla_matmuls",
        shape_id="[64, 720]×[720, 320]",
        repeat=0,
        correctness=True,
        cycles=12251,
        rounds_used=1,
        tokens_in=1000,
        tokens_out=500,
        cost_usd=0.01,
        wall_s=12.3,
        cycle_source="MAIN_LD_ST_EX_CYCLES",
        notes="",
    )
    base.update(overrides)
    return CanonicalCellRow(**base)


def test_canonical_row_validates_enums() -> None:
    """Enum validation must reject unknown backend / target /
    workload / cycle_source strings — silent typos would corrupt the
    matrix."""
    with pytest.raises(ValueError, match="unknown backend"):
        _ok_row(backend="kbv3")
    with pytest.raises(ValueError, match="unknown target"):
        _ok_row(target="cuda_a100")
    with pytest.raises(ValueError, match="unknown workload"):
        _ok_row(workload="conv2d")
    with pytest.raises(ValueError, match="unknown cycle_source"):
        _ok_row(cycle_source="rdtsc")


def test_canonical_row_rejects_negative_repeat() -> None:
    with pytest.raises(ValueError, match="repeat must be >= 0"):
        _ok_row(repeat=-1)


def test_canonical_row_roundtrips_through_jsonl(tmp_path: Path) -> None:
    rows = [
        _ok_row(repeat=0),
        _ok_row(repeat=1, correctness=False, cycles=None, cycle_source="none"),
        _ok_row(backend="autocomp", cycle_source="Generated implementation latency"),
    ]
    path = tmp_path / "rows.jsonl"
    n = write_jsonl(rows, path)
    assert n == 3
    loaded = read_jsonl(path)
    assert len(loaded) == 3
    for a, b in zip(rows, loaded):
        assert a == b


def test_read_jsonl_missing_file_returns_empty() -> None:
    assert read_jsonl(Path("/tmp/__never_exists__.jsonl")) == []


def test_merge_jsonl_concatenates_in_order(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    write_jsonl([_ok_row(repeat=0)], a)
    write_jsonl([_ok_row(repeat=1), _ok_row(repeat=2)], b)
    rows = merge_jsonl(a, b)
    assert [r.repeat for r in rows] == [0, 1, 2]


def test_shape_id_helpers_round_trip() -> None:
    assert shape_id_for_matmul(64, 720, 320) == "[64, 720]×[720, 320]"
    assert parse_matmul_shape_id("[64, 720]×[720, 320]") == (64, 720, 320)
    assert parse_matmul_shape_id("action_expert.layer0.mlp") is None


def test_enum_lists_match_documented_set() -> None:
    """Regression guard — the report aggregator + loaders pivot on
    these literals; renaming silently would break the join."""
    assert set(BACKENDS) == {"kb-vanilla", "kb-v2", "autocomp"}
    assert set(TARGETS) == {"gemmini", "gemmini_mx", "saturn_opu_v128"}
    assert set(WORKLOADS) == {"smolvla_matmuls", "smolvla_mlp_block"}
    assert set(CYCLE_SOURCES) == {
        "MAIN_LD_ST_EX_CYCLES",
        "Generated implementation latency",
        "mcycle",
        "cached",
        "none",
    }
