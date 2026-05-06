"""Tests for compgen.benchmarks.pass_pool_ablation (M-36.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.benchmarks.pass_pool_ablation import (
    AblationCellSpec,
    AblationPack,
    AblationResult,
    emit_pack,
    run_one_cell,
    run_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# AblationResult / AblationPack data model
# --------------------------------------------------------------------------- #


def test_ablation_result_round_trip() -> None:
    r = AblationResult(
        model_id="merlin_mlp_wide",
        target_id="host_cpu",
        mode="greedy",
        selected_candidate_id="tile_M16_N16_K16",
        candidate_kind="set_tile_params",
        pass_id="set_tile_params",
        validation_overall="pass",
        validation_failures=(),
        decision_seconds=0.5,
        typed_outcome="verified",
    )
    raw = r.to_dict()
    assert raw["model_id"] == "merlin_mlp_wide"
    assert raw["mode"] == "greedy"
    assert raw["validation_failures"] == []


def test_ablation_pack_summary_per_mode() -> None:
    pack = AblationPack(commit="abc")
    pack.cells.append(AblationResult(
        model_id="m1", target_id="t1", mode="greedy",
        selected_candidate_id="c1", candidate_kind="set_tile_params",
        pass_id="set_tile_params", validation_overall="pass",
        validation_failures=(), decision_seconds=0.5,
        typed_outcome="verified",
    ))
    pack.cells.append(AblationResult(
        model_id="m2", target_id="t1", mode="greedy",
        selected_candidate_id="", candidate_kind="",
        pass_id="", validation_overall="unknown",
        validation_failures=(), decision_seconds=0.1,
        typed_outcome="typed_blocked",
    ))
    summary = pack.summary()
    assert summary["modes"] == ["greedy"]
    assert summary["cell_count"] == 2
    assert summary["per_mode"]["greedy"]["verified"] == 1
    assert summary["per_mode"]["greedy"]["typed_blocked"] == 1


def test_ablation_pack_divergence_detection() -> None:
    pack = AblationPack()
    pack.cells.append(AblationResult(
        model_id="m1", target_id="t1", mode="greedy",
        selected_candidate_id="cand_a", candidate_kind="set_tile_params",
        pass_id="set_tile_params", validation_overall="pass",
        validation_failures=(), decision_seconds=0.0,
        typed_outcome="verified",
    ))
    pack.cells.append(AblationResult(
        model_id="m1", target_id="t1", mode="agent-file",
        selected_candidate_id="cand_b", candidate_kind="set_tile_params",
        pass_id="set_tile_params", validation_overall="pass",
        validation_failures=(), decision_seconds=0.0,
        typed_outcome="verified",
    ))
    div = pack.divergences()
    assert len(div) == 1
    assert div[0]["model_id"] == "m1"
    assert div[0]["picks_by_mode"] == {
        "greedy": "cand_a", "agent-file": "cand_b",
    }


def test_ablation_pack_no_divergence_when_modes_agree() -> None:
    pack = AblationPack()
    pack.cells.append(AblationResult(
        model_id="m1", target_id="t1", mode="greedy",
        selected_candidate_id="cand_a", candidate_kind="x",
        pass_id="x", validation_overall="pass",
        validation_failures=(), decision_seconds=0.0,
        typed_outcome="verified",
    ))
    pack.cells.append(AblationResult(
        model_id="m1", target_id="t1", mode="agent-file",
        selected_candidate_id="cand_a", candidate_kind="x",
        pass_id="x", validation_overall="pass",
        validation_failures=(), decision_seconds=0.0,
        typed_outcome="verified",
    ))
    assert pack.divergences() == []


# --------------------------------------------------------------------------- #
# run_one_cell + run_suite (real workload, single model)
# --------------------------------------------------------------------------- #


def test_run_one_cell_greedy_on_holdout(tmp_path: Path) -> None:
    """End-to-end: greedy mode on a holdout model returns a typed cell."""
    model_yaml = REPO_ROOT / "configs" / "models" / "holdout_mlp_odd_shapes.yaml"
    target_yaml = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"
    out = tmp_path / "cell"

    result = run_one_cell(
        model_yaml=model_yaml,
        target_yaml=target_yaml,
        out_dir=out,
        mode="greedy",
    )
    assert result.model_id == "holdout_mlp_odd_shapes"
    assert result.target_id == "host_cpu"
    assert result.mode == "greedy"
    # Honest outcomes only
    assert result.typed_outcome in ("verified", "typed_blocked", "error")
    # Greedy should never error on a holdout model
    assert result.typed_outcome != "error", f"unexpected error: {result.error}"
    assert result.decision_seconds > 0


def test_run_suite_two_models(tmp_path: Path) -> None:
    """Run a 2-cell ablation; aggregator captures both cells."""
    cells = [
        AblationCellSpec(
            model_yaml=REPO_ROOT / "configs" / "models" / "holdout_mlp_odd_shapes.yaml",
            target_yaml=REPO_ROOT / "configs" / "targets" / "host_cpu.yaml",
            mode="greedy",
        ),
        AblationCellSpec(
            model_yaml=REPO_ROOT / "configs" / "models" / "holdout_pointwise_chain_renamed.yaml",
            target_yaml=REPO_ROOT / "configs" / "targets" / "host_cpu.yaml",
            mode="greedy",
        ),
    ]
    pack = run_suite(cells, out_root=tmp_path, commit="test")
    assert len(pack.cells) == 2
    summary = pack.summary()
    assert summary["cell_count"] == 2
    assert "greedy" in summary["modes"]


def test_emit_pack_writes_json(tmp_path: Path) -> None:
    pack = AblationPack(commit="abc")
    pack.cells.append(AblationResult(
        model_id="m1", target_id="t1", mode="greedy",
        selected_candidate_id="c1", candidate_kind="set_tile_params",
        pass_id="set_tile_params", validation_overall="pass",
        validation_failures=(), decision_seconds=0.5,
        typed_outcome="verified",
    ))
    out = tmp_path / "ablation_pack.json"
    emit_pack(pack, out_path=out)
    assert out.exists()
    raw = json.loads(out.read_text())
    assert raw["schema_version"] == "ablation_pack_v1"
    assert raw["commit"] == "abc"
    assert len(raw["cells"]) == 1
