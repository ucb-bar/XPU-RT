"""Greedy-baseline reproducibility test.

The contract: a fresh task pack on a holdout model can complete a
greedy/no-LLM run. If the public doc surface is sufficient for the
deterministic resolver, it is sufficient for a fresh agent.

This is the CI-runnable reproducibility floor. The actual fresh-Claude
run is operator-driven and recorded in the caveat ledger.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from compgen.audit.fresh_agent import build_task_pack
from compgen.audit.fresh_agent_modes import (
    GreedyBaselineResult,
    record_manual_session_result,
    run_greedy_baseline,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_greedy_baseline_runs_on_holdout(tmp_path: Path) -> None:
    """Build a task pack; run greedy on holdout_mlp_odd_shapes from inside it.

    Note: we don't actually CD into the task pack to invoke the
    pipeline (the real pipeline lives in the parent repo's installed
    package, not the pack's checkout). But we use the pack's *configs*
    and the pack's *task prompt* as inputs, ensuring those are
    sufficient.
    """
    pack_dir = tmp_path / "pack"
    out_dir = tmp_path / "run"

    build_task_pack(
        out_dir=pack_dir, commit="testcommit",
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
        stop_after="payload-lowering",
    )

    assert isinstance(result, GreedyBaselineResult)
    # Honest outcomes only: verified or typed-blocked.
    assert result.success, (
        f"greedy baseline failed for holdout_mlp_odd_shapes: {result.error}"
    )
    assert result.typed_outcome in ("verified", "typed_blocked"), (
        f"unexpected outcome {result.typed_outcome!r}"
    )


def test_record_manual_session_result_appends_to_ledger(tmp_path: Path) -> None:
    """Operator-recorded fresh/current Claude outcomes land in the ledger."""
    ledger_path = tmp_path / "ledger.json"

    caveat = record_manual_session_result(
        ledger_path=ledger_path,
        mode="fresh_claude",
        success=True,
        evidence_paths=["/tmp/run/run_manifest.json"],
        notes="fresh session reached typed_blocked on holdout",
    )
    assert caveat.id.startswith("manual_fresh_claude_")
    assert caveat.status == "resolved"

    # Adding a second one with mode=current_claude works
    caveat2 = record_manual_session_result(
        ledger_path=ledger_path,
        mode="current_claude",
        success=True,
        evidence_paths=["/tmp/run2/run_manifest.json"],
    )
    assert caveat2.id.startswith("manual_current_claude_")


def test_record_rejects_invalid_mode(tmp_path: Path) -> None:
    from compgen.audit.errors import AuditError

    with pytest.raises(AuditError, match="mode"):
        record_manual_session_result(
            ledger_path=tmp_path / "l.json",
            mode="bogus",
            success=True,
            evidence_paths=["x"],
        )


def test_record_rejects_greedy_baseline_mode(tmp_path: Path) -> None:
    from compgen.audit.errors import AuditError

    with pytest.raises(AuditError, match="greedy_baseline"):
        record_manual_session_result(
            ledger_path=tmp_path / "l.json",
            mode="greedy_baseline",
            success=True,
            evidence_paths=["x"],
        )


def test_record_rejects_empty_evidence(tmp_path: Path) -> None:
    from compgen.audit.errors import AuditError

    with pytest.raises(AuditError, match="evidence_paths"):
        record_manual_session_result(
            ledger_path=tmp_path / "l.json",
            mode="fresh_claude",
            success=True,
            evidence_paths=[],
        )
