"""Tests for compgen.llm.target_coverage."""

from __future__ import annotations

from compgen.llm.target_coverage import (
    INDUCTOR_COVERAGE,
    cost_weight_for,
    coverage_notes_for_llm,
    get_coverage,
    update_measurement,
)


def test_table_seeded() -> None:
    assert len(INDUCTOR_COVERAGE) >= 28


def test_cost_weight_penalize_on_cuda_for_decompose_concat() -> None:
    # Plan matrix: decompose_concat is penalize on CUDA (inductor does it).
    assert cost_weight_for("decompose_concat", "cuda") > 1.0


def test_cost_weight_prefer_on_cuda_for_raise_special_ops() -> None:
    assert cost_weight_for("raise_special_ops", "cuda") < 1.0


def test_missing_row_defaults_to_prefer_on_non_inductor_target() -> None:
    # A pass that has no entry for rvv_cpu should default to 'prefer'
    # (CompGen expected to help where inductor is absent).
    weight = cost_weight_for("some_unseeded_pass", "rvv_cpu")
    assert weight < 1.0


def test_missing_row_defaults_to_neutral_on_cuda() -> None:
    weight = cost_weight_for("some_unseeded_pass", "cuda")
    assert weight == 1.0


def test_notes_include_rationale_on_seeded_rows() -> None:
    note = coverage_notes_for_llm("decompose_concat", "cuda")
    assert "cuda" in note
    assert "penalize" in note


def test_update_measurement_overwrites_seed() -> None:
    update_measurement(
        "decompose_concat",
        "cuda",
        coverage="partial",
        cost_weight_bias="neutral",
        measured_shapes_where_compgen_wins=("B=1,N=7",),
        notes="measurement override in test",
    )
    row = get_coverage("decompose_concat", "cuda")
    assert row is not None
    assert row.coverage == "partial"
    assert row.cost_weight_bias == "neutral"
    assert row.basis == "measured"
    assert "B=1,N=7" in row.measured_shapes_where_compgen_wins


def test_update_measurement_creates_new_row() -> None:
    update_measurement(
        "brand_new_pass",
        "arm_cpu",
        coverage="none",
        cost_weight_bias="prefer",
        notes="new row",
    )
    row = get_coverage("brand_new_pass", "arm_cpu")
    assert row is not None and row.basis == "measured"
