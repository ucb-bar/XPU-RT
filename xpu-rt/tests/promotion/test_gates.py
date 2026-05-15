"""tests for the promotion-gate ladder (Section 19)."""

from __future__ import annotations

import json
from pathlib import Path

from compgen.promotion.gates import (
    GateEvaluation,
    PromotionLevel,
    evaluate_gate,
)


def _write(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")


def _build_run_dir(
    tmp_path: Path,
    *,
    selected_candidate: bool = True,
    fx_pass: bool = True,
    kernel_pass: bool = True,
    analytical_present: bool = True,
    measured_present: bool = True,
    fx_readiness: str = "pass",
    kernel_readiness: str = "pass",
    promotion_certs: bool = True,
) -> Path:
    """Build a Phase B run dir with selectively present evidence."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    # candidate_selection — ``observed`` floor.
    rp = run_dir / "03_recipe_planning"
    if selected_candidate:
        _write(
            rp / "candidate_selection.json",
            {"selected_candidate_id": "cand_0001", "candidate_kind": "set_tile_params"},
        )
    else:
        _write(
            rp / "candidate_selection.json",
            {"selected_candidate_id": None},
        )

    # FX-level differential — ``verified_fx``.
    if fx_pass:
        _write(
            rp / "real_transform_differential_report.json",
            {"status": "pass"},
        )

    # Compiled-kernel differential — ``verified_kernel``.
    ga = run_dir / "02_graph_analysis"
    if kernel_pass:
        _write(
            ga / "kernel_execution" / "kernel_execution_report.json",
            {"status": "pass"},
        )

    # analytical + measured — ``characterized``.
    if analytical_present:
        _write(
            ga / "analytical_cost" / "per_candidate_analytical_cost.json",
            {"candidates_modeled": 5, "candidate_count": 5},
        )
    if measured_present:
        _write(
            ga / "compiled_bottleneck" / "compiled_bottleneck_report.json",
            {"region_count_with_evidence": 3, "region_count_total": 3},
        )

    # Readiness matrices + certs — ``promoted``.
    _write(
        ga / "readiness" / "graph_analysis_readiness_matrix.json",
        {"overall": fx_readiness},
    )
    _write(
        ga / "kernel_section_readiness" / "kernel_section_readiness_matrix.json",
        {"overall": kernel_readiness},
    )
    if promotion_certs:
        _write(
            run_dir / "04_promotion" / "verification_report.json",
            {"passed": True, "levels_passed": ["structural", "differential"]},
        )

    return run_dir


# -- Level enum --------------------------------------------------------------


def test_promotion_level_ordering() -> None:
    """The numeric values establish a strict total order low → high."""
    assert PromotionLevel.OBSERVED.value < PromotionLevel.VERIFIED_FX.value
    assert PromotionLevel.VERIFIED_FX.value < PromotionLevel.VERIFIED_KERNEL.value
    assert PromotionLevel.VERIFIED_KERNEL.value < PromotionLevel.CHARACTERIZED.value
    assert PromotionLevel.CHARACTERIZED.value < PromotionLevel.PROMOTED.value
    assert PromotionLevel.PROMOTED.value < PromotionLevel.PORTABLE.value


def test_promotion_level_string_round_trip() -> None:
    """str(level) → from_string(...) is the identity."""
    for level in PromotionLevel:
        assert PromotionLevel.from_string(str(level)) is level


def test_unknown_level_string_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        PromotionLevel.from_string("nonexistent_level")


# -- evaluate_gate by level --------------------------------------------------


def test_observed_floor_when_no_candidate(tmp_path: Path) -> None:
    """No candidate selected → caps at observed (without ok)."""
    run_dir = _build_run_dir(tmp_path, selected_candidate=False)
    result = evaluate_gate(run_dir)
    assert result.level == PromotionLevel.OBSERVED
    assert "no candidate" in result.reasons["observed"].lower()


def test_observed_when_only_candidate_selected(tmp_path: Path) -> None:
    """Candidate selected but no FX evidence → observed."""
    run_dir = _build_run_dir(
        tmp_path,
        fx_pass=False,
        kernel_pass=False,
        analytical_present=False,
        measured_present=False,
        promotion_certs=False,
    )
    result = evaluate_gate(run_dir)
    assert result.level == PromotionLevel.OBSERVED


def test_verified_fx(tmp_path: Path) -> None:
    """FX differential passes but no kernel evidence → verified_fx."""
    run_dir = _build_run_dir(
        tmp_path,
        kernel_pass=False,
        analytical_present=False,
        measured_present=False,
        promotion_certs=False,
    )
    result = evaluate_gate(run_dir)
    assert result.level == PromotionLevel.VERIFIED_FX


def test_verified_kernel(tmp_path: Path) -> None:
    """Kernel differential passes but no characterization → verified_kernel."""
    run_dir = _build_run_dir(
        tmp_path,
        analytical_present=False,
        measured_present=False,
        promotion_certs=False,
    )
    result = evaluate_gate(run_dir)
    assert result.level == PromotionLevel.VERIFIED_KERNEL


def test_characterized(tmp_path: Path) -> None:
    """Analytical + measured cost present → characterized."""
    run_dir = _build_run_dir(tmp_path, promotion_certs=False)
    result = evaluate_gate(run_dir)
    assert result.level == PromotionLevel.CHARACTERIZED


def test_promoted_when_full_pack(tmp_path: Path) -> None:
    """Full readiness pack passes → promoted."""
    run_dir = _build_run_dir(tmp_path)
    result = evaluate_gate(run_dir)
    assert result.level == PromotionLevel.PROMOTED


def test_promoted_blocked_when_fx_readiness_not_pass(tmp_path: Path) -> None:
    """FX readiness != pass demotes from promoted to characterized."""
    run_dir = _build_run_dir(tmp_path, fx_readiness="ready_for_m18")
    result = evaluate_gate(run_dir)
    assert result.level == PromotionLevel.CHARACTERIZED


def test_promoted_blocked_when_no_certificates(tmp_path: Path) -> None:
    """Missing 04_promotion/verification_report.json blocks promoted."""
    run_dir = _build_run_dir(tmp_path, promotion_certs=False)
    result = evaluate_gate(run_dir)
    assert result.level == PromotionLevel.CHARACTERIZED


# -- portable level ---------------------------------------------------------


def test_portable_skipped_without_signature(tmp_path: Path) -> None:
    """portable check is skipped when region_signature is empty."""
    run_dir = _build_run_dir(tmp_path)
    result = evaluate_gate(run_dir, region_signature="", target_class="host_cpu")
    assert result.level == PromotionLevel.PROMOTED
    assert "skipped" in result.reasons["portable"].lower()


def test_portable_blocked_with_only_one_target_class(tmp_path: Path) -> None:
    """One target_class = not portable; level stays at promoted."""
    run_dir = _build_run_dir(tmp_path)
    library = tmp_path / "library"
    library.mkdir()
    # Drop a sidecar with a different region_signature so it doesn't count.
    (library / "x_y_z_v1").mkdir()
    (library / "x_y_z_v1" / "promoted_recipe.json").write_text(
        json.dumps({
            "key": {"region_signature": "other_sig"},
            "recipe": {"validity": {"target_class": "cuda_sm75"}},
        })
    )
    result = evaluate_gate(
        run_dir,
        region_signature="abc123",
        target_class="host_cpu",
        library_path=library,
    )
    assert result.level == PromotionLevel.PROMOTED  # still promoted, not portable


def test_portable_when_two_target_classes(tmp_path: Path) -> None:
    """≥2 distinct target_class for the same region_signature → portable."""
    run_dir = _build_run_dir(tmp_path)
    library = tmp_path / "library"
    library.mkdir()
    (library / "x_y_z_v1").mkdir()
    (library / "x_y_z_v1" / "promoted_recipe.json").write_text(
        json.dumps({
            "key": {"region_signature": "abc123"},
            "recipe": {"validity": {"target_class": "cuda_sm75"}},
        })
    )
    result = evaluate_gate(
        run_dir,
        region_signature="abc123",
        target_class="host_cpu",  # this run's target — combined with cuda_sm75 = 2.
        library_path=library,
    )
    assert result.level == PromotionLevel.PORTABLE


# -- monotonicity ------------------------------------------------------------


def test_stripping_evidence_demotes_monotonically(tmp_path: Path) -> None:
    """Removing evidence can only lower the level, never raise it."""
    full = _build_run_dir(tmp_path / "full")
    no_kernel = _build_run_dir(tmp_path / "no_kernel", kernel_pass=False,
                               analytical_present=False, measured_present=False,
                               promotion_certs=False)
    no_fx = _build_run_dir(tmp_path / "no_fx", fx_pass=False, kernel_pass=False,
                           analytical_present=False, measured_present=False,
                           promotion_certs=False)
    no_candidate = _build_run_dir(tmp_path / "no_cand", selected_candidate=False)

    full_level = evaluate_gate(full).level
    no_kernel_level = evaluate_gate(no_kernel).level
    no_fx_level = evaluate_gate(no_fx).level
    no_cand_level = evaluate_gate(no_candidate).level

    assert full_level.value >= no_kernel_level.value
    assert no_kernel_level.value >= no_fx_level.value
    assert no_fx_level.value >= no_cand_level.value


# -- GateEvaluation dataclass -----------------------------------------------


def test_gate_evaluation_to_dict_round_trip(tmp_path: Path) -> None:
    """GateEvaluation.to_dict() carries the level, reasons, evidence."""
    run_dir = _build_run_dir(tmp_path)
    result = evaluate_gate(run_dir)
    body = result.to_dict()
    assert body["level"] == "promoted"
    assert isinstance(body["reasons"], dict)
    assert "verified_fx" in body["reasons"]
    assert isinstance(body["evidence_summary"], dict)
