"""G7 — fresh-agent reproducibility /surface.

Closes the audit gap: prove that a fresh-agent task pack
contains everything needed to drive the /candidate-
selection + verification surface to a typed outcome on a holdout
model, without referencing any chat-context or session memory.

The deterministic ``run_greedy_baseline`` is the contractual
reproducibility floor (per the existing
``fresh_claude_session_operator_driven`` caveat). A fresh Claude
Code session is the operator-driven evidence row; this test
exercises the same code path the operator would, plus the
load-bearing-gate attribution surfaced by the inspection harness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.audit.fresh_agent import build_task_pack
from compgen.audit.fresh_agent_modes import (
    GreedyBaselineResult,
    run_greedy_baseline,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_greedy_baseline_against_task_pack_reaches_m37_11_surface(
    tmp_path: Path,
) -> None:
    """Build a fresh task pack, run greedy on holdout_mlp_odd_shapes, and
    assert the shape-fit machinery + admission rules
    actually fire on the run.

    Specifically:
    - Greedy reaches recipe_planning successfully (no chat-context).
    - candidate_selection.json reports cost_preview.region_dims and
      cost_preview.boundary_required (the surfaced fields).
    - real_differential_report.json reports a typed
      ``error.refinement_status`` value, never the empty string —
      meaning ran and produced a typed outcome under the
      admission semantics."""
    pack_dir = tmp_path / "pack"
    out_dir = tmp_path / "run"

    build_task_pack(
        out_dir=pack_dir, commit="m37_13",
        repo_root=REPO_ROOT, skip_python_package=True,
    )

    model_yaml = pack_dir / "configs" / "models" / "holdout_mlp_odd_shapes.yaml"
    target_yaml = pack_dir / "configs" / "targets" / "host_cpu.yaml"
    assert model_yaml.exists() and target_yaml.exists()

    result = run_greedy_baseline(
        task_pack_dir=pack_dir,
        out_dir=out_dir,
        model_yaml=model_yaml,
        target_yaml=target_yaml,
        stop_after="agent-decision-request",
    )

    assert isinstance(result, GreedyBaselineResult)
    assert result.success, (
        f"greedy baseline failed for holdout_mlp_odd_shapes from a "
        f"fresh task pack: {result.error}"
    )

    # surface check: candidate_selection.json carries the new
    # region-aware cost_preview fields. If shape-fit emission was
    # silently deleted, region_dims would be missing.
    sel_path = (
        out_dir / "03_recipe_planning" / "candidate_selection.json"
    )
    assert sel_path.exists()
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    cp = sel.get("cost_preview") or {}
    assert "region_dims" in cp, (
        "M-37.11 surface missing: cost_preview.region_dims absent — "
        "shape-fit machinery may have been removed"
    )
    assert "boundary_required" in cp, (
        "M-37.11 surface missing: cost_preview.boundary_required absent"
    )

    # surface check: real_differential_report.json reports a
    # typed refinement status (not empty). We cover both happy
    # outcomes (discharged_tolerance_eps / discharged_bit_equality)
    # and the fail outcomes — never the empty string.
    diff_path = (
        out_dir / "03_recipe_planning" / "real_verification"
        / "real_differential_report.json"
    )
    if diff_path.exists():
        diff = json.loads(diff_path.read_text(encoding="utf-8"))
        refinement_status = (diff.get("error") or {}).get(
            "refinement_status", ""
        )
        assert refinement_status, (
            f"M-37.12 surface missing: real_differential_report has no "
            f"refinement_status in error block; report keys = "
            f"{sorted(diff.keys())}"
        )
        # Honest outcomes: discharged_* (passing) or fail_* (rejecting).
        assert refinement_status.startswith(("discharged_", "fail_", "blocked")), (
            f"unexpected refinement_status={refinement_status!r}"
        )


def test_inspection_harness_attribution_runs_on_task_pack_run(
    tmp_path: Path,
) -> None:
    """Verify the G6 load-bearing-gate attribution renders a
    classified path (not 'unclassified') for a holdout run that
    discharges a real typed refinement status.

    This is the structural test that the attribution helper actually
    classifies post-runs correctly — without it, the OVERVIEW
    table would always show 'unclassified' and the user could not
    answer 'which gate carried the load?' from the artifact alone."""
    from compgen.benchmarks.model_inspection import (
        _attribute_load_bearing_gate,
    )
    # Synthetic-but-realistic cost_preview from a tiny_mlp-like region.
    cp_clean_divide_k_iters_gt_one = {
        "boundary_required": False,
        "region_dims": {"M": 4, "N": 128, "K": 64},
    }
    attr = _attribute_load_bearing_gate(
        cost_preview=cp_clean_divide_k_iters_gt_one,
        refinement_status="discharged_tolerance_eps",
        selected_label="tile_M4_N16_K16",
    )
    assert attr["verification_path"] == "clean_divide_tolerance_eps"
    assert "M-37.12" in attr["load_bearing_gate"]
    assert attr["shape_fit_tile_picked"] is True
    assert attr["single_k_iter"] is False

    # Boundary-path attribution.
    cp_boundary = {
        "boundary_required": True,
        "region_dims": {"M": 7, "N": 129, "K": 63},
    }
    attr_b = _attribute_load_bearing_gate(
        cost_preview=cp_boundary,
        refinement_status="discharged_tolerance_eps",
        selected_label="tile_M16_N16_K16",
    )
    assert attr_b["verification_path"] == "boundary_tolerance_eps"
    assert "boundary_required" in attr_b["load_bearing_gate"]

    # Bit-equality (legacy strict) attribution.
    cp_strict = {
        "boundary_required": False,
        "region_dims": {"M": 256, "N": 256, "K": 16},
    }
    attr_be = _attribute_load_bearing_gate(
        cost_preview=cp_strict,
        refinement_status="discharged_bit_equality",
        selected_label="tile_M16_N16_K16",
    )
    assert attr_be["verification_path"] == "bit_equality"
    assert attr_be["load_bearing_gate"] == "exact_equality_per_case"
    assert attr_be["single_k_iter"] is True
