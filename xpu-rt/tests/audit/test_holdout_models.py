"""Holdout-model honesty tests (M-31A.4).

Each holdout model must reach a verified outcome OR a typed-blocked
outcome — never a silent partial pass. We exercise capture + payload
lowering for each, then assert the run dir contains either:

- A successful manifest reaching the requested ``stop_after``, OR
- A typed-blocked surface (M-15B downstream_retry_request, or a typed
  exception name in the stage_ledger).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.graph_compilation.evidence_pack import is_holdout_model
from compgen.graph_compilation.run import run_graph_compilation

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs" / "models"
TARGET_YAML = REPO_ROOT / "configs" / "targets" / "host_cpu.yaml"

HOLDOUT_MODELS = (
    "holdout_mlp_odd_shapes",
    "holdout_mlp_large_k",
    "holdout_pointwise_chain_renamed",
    "holdout_two_matmuls_shared_input",
    "holdout_unsupported_attention",
)


@pytest.mark.parametrize("model_id", HOLDOUT_MODELS)
def test_holdout_yaml_carries_holdout_flag(model_id: str) -> None:
    yaml_path = CONFIGS_DIR / f"{model_id}.yaml"
    assert yaml_path.exists(), f"{yaml_path} missing"
    assert is_holdout_model(yaml_path), (
        f"{model_id}.yaml must declare 'holdout: true' so the evidence "
        f"pack excludes it from canonical-22"
    )


def test_non_holdout_yaml_does_not_carry_holdout_flag() -> None:
    yaml_path = CONFIGS_DIR / "merlin_mlp_wide.yaml"
    assert is_holdout_model(yaml_path) is False


@pytest.mark.parametrize("model_id", HOLDOUT_MODELS)
def test_holdout_run_reaches_honest_outcome(
    model_id: str, tmp_path: Path,
) -> None:
    """Each holdout must reach verified OR typed-blocked, never silent partial.

    We stop at ``payload-lowering`` to keep the test cheap; the
    capture + lowering stages are where most "silent partial" bugs
    would land.
    """
    out = tmp_path / model_id
    model_yaml = CONFIGS_DIR / f"{model_id}.yaml"
    assert model_yaml.exists()

    typed_blocked = False
    error_text = ""
    try:
        run_graph_compilation(
            model_yaml,
            TARGET_YAML,
            out,
            stop_after="payload-lowering",
            selection_mode="greedy",
        )
    except Exception as exc:  # noqa: BLE001 - classify
        type_name = type(exc).__name__
        msg = str(exc)
        # Typed-blocked outcomes: M-15B downstream-rejection, typed
        # runtime errors, capture-stage typed errors. Any of these
        # is honest. Generic AssertionError / KeyError / etc. is not.
        honest_markers = (
            "M-15B",
            "downstream",
            "Unsupported",
            "BundleEmissionError",
            "SymbolicShape",
            "unsupported_op",
        )
        if any(m in msg for m in honest_markers) or any(
            m in type_name for m in honest_markers
        ):
            typed_blocked = True
            error_text = f"{type_name}: {msg[:200]}"
        else:
            pytest.fail(
                f"holdout {model_id} raised unexpected {type_name}: {msg}"
            )

    if typed_blocked:
        # Honest typed-blocked. Done.
        return

    # Pipeline ran without raising; verify the run produced sensible
    # artifacts.
    assert (out / "run_manifest.json").exists() or typed_blocked, (
        f"holdout {model_id}: no manifest and no typed-blocked outcome"
    )
    if (out / "stage_ledger.jsonl").exists():
        ledger = (out / "stage_ledger.jsonl").read_text()
        # If any stage was skipped without a typed reason, that's
        # silent-partial.
        assert "silent_partial" not in ledger, (
            f"holdout {model_id} stage_ledger contains 'silent_partial'"
        )
