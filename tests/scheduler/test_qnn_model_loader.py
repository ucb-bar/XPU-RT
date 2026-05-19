"""Tests for ``xpu_rt.scheduler.qnn_model_loader``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from xpu_rt.scheduler.qnn_model_loader import (
    ExtractionStats,
    _dtype_bytes,
    extract_buffer_specs,
)

DRONET_PATH = Path("/tmp/qnn_build/dronet_net.json")
YOLOV8N_Q_PATH = Path("/tmp/yqnn/yolov8n_q_net.json")


pytestmark = pytest.mark.skipif(
    not DRONET_PATH.exists(),
    reason="QNN converter artifacts not present (need /tmp/qnn_build/dronet_net.json)",
)


def test_dtype_table_known_codes() -> None:
    """Bit widths for the canonical QNN data types."""

    assert _dtype_bytes(562) == 4  # FLOAT_32
    assert _dtype_bytes(534) == 2  # FLOAT_16
    assert _dtype_bytes(1032) == 1  # UFIXED_POINT_8
    assert _dtype_bytes(50) == 4  # INT_32
    assert _dtype_bytes(999_999) is None  # unknown enum -> None


def test_extract_dronet_basic_shape() -> None:
    """Loader pulls activations and aliases for the dronet model."""

    plan, stats = extract_buffer_specs(DRONET_PATH)

    assert isinstance(stats, ExtractionStats)
    assert stats.parser == "qnn_net_json_v1"
    assert stats.num_ops == 29  # known from the file
    assert stats.num_activations >= 25
    assert stats.alias_candidates_proposed > 0
    assert stats.skipped_unknown_dtype == 0
    # dronet is float32; total activation bytes should be in the
    # 0.5 - 5 MB range (sanity check, not exact).
    assert 500_000 < stats.total_activation_bytes < 10_000_000

    # Every buffer must have a non-empty allowed_tiers tuple matching
    # one of the declared tier ids.
    declared_tiers = {t.tier_id for t in plan.tier_capacities}
    for b in plan.buffers:
        assert b.allowed_tiers, f"empty allowed_tiers on {b.buffer_id}"
        assert all(t in declared_tiers for t in b.allowed_tiers)
        assert b.lifetime_end >= b.lifetime_start
        assert b.size_bytes > 0


def test_extract_alias_candidates_are_disjoint_lifetimes() -> None:
    """Every proposed alias pair has lifetimes that don't overlap.

    This is the precondition that makes the alias decision *useful* to
    the MILP planner.
    """

    plan, _stats = extract_buffer_specs(DRONET_PATH)
    by_id = {b.buffer_id: b for b in plan.buffers}
    assert plan.alias_candidates, "expected at least one alias candidate for dronet"
    for cand in plan.alias_candidates:
        a = by_id.get(cand.buffer_a)
        b = by_id.get(cand.buffer_b)
        assert a is not None and b is not None, (
            f"alias references unknown buffer: {cand}"
        )
        overlap = not (a.lifetime_end < b.lifetime_start or b.lifetime_end < a.lifetime_start)
        assert not overlap, (
            f"alias {cand.buffer_a} <-> {cand.buffer_b} has overlapping "
            f"lifetimes ({a.lifetime_start},{a.lifetime_end}) "
            f"vs ({b.lifetime_start},{b.lifetime_end})"
        )


@pytest.mark.skipif(
    not YOLOV8N_Q_PATH.exists(),
    reason="yolov8n_q_net.json not present",
)
def test_extract_yolov8n_scales() -> None:
    """Loader scales to yolov8n (241 ops, 250 activations)."""

    plan, stats = extract_buffer_specs(YOLOV8N_Q_PATH)
    assert stats.num_ops >= 200
    assert stats.num_activations >= 200
    # Quantized model — activations are uint8.
    assert stats.total_activation_bytes < 50_000_000
    # Buffers should reference both scratch and dram as allowed tiers
    # (the loader's default), so the MILP can choose.
    tier_choices = {tuple(b.allowed_tiers) for b in plan.buffers}
    assert ("scratch", "dram") in tier_choices


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        extract_buffer_specs("/tmp/does_not_exist_qnn_net.json")


def test_malformed_input_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad_net.json"
    bad.write_text(json.dumps({"not_a_graph": True}))
    with pytest.raises(ValueError, match="graph"):
        extract_buffer_specs(bad)
