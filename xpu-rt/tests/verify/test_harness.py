"""Tests for the verification harness."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from compgen.verify.harness import VerificationRun, verify_callable_against_reference

# -- identity pass ------------------------------------------------------------


def test_identity_passes(tmp_path: Path) -> None:
    """An identity candidate should pass verification."""
    t = torch.randn(4, 4)
    result = verify_callable_against_reference(
        name="identity",
        ref_fn=lambda: t,
        got_fn=lambda: t.clone(),
        out_dir=tmp_path,
    )
    assert result.passed
    assert result.latency_ref_ms >= 0.0
    assert result.latency_got_ms >= 0.0
    assert len(result.comparisons) == 1


# -- intentional fail ---------------------------------------------------------


def test_wrong_candidate_fails(tmp_path: Path) -> None:
    """A candidate returning a different tensor should fail."""
    ref = torch.zeros(10)
    got = torch.ones(10)
    result = verify_callable_against_reference(
        name="wrong",
        ref_fn=lambda: ref,
        got_fn=lambda: got,
        out_dir=tmp_path,
    )
    assert not result.passed
    assert result.comparisons[0].num_mismatched == 10


# -- exception handling -------------------------------------------------------


def test_exception_in_candidate_caught(tmp_path: Path) -> None:
    """If the candidate raises, the run should be marked as failed."""

    def _boom() -> torch.Tensor:
        msg = "intentional failure"
        raise RuntimeError(msg)

    result = verify_callable_against_reference(
        name="boom",
        ref_fn=lambda: torch.ones(3),
        got_fn=_boom,
        out_dir=tmp_path,
    )
    assert not result.passed
    assert result.latency_got_ms == 0.0
    assert result.comparisons[0].num_mismatched == -1


# -- verification.json --------------------------------------------------------


def test_verification_json_written(tmp_path: Path) -> None:
    """verify_callable_against_reference should write verification.json."""
    t = torch.randn(2, 2)
    verify_callable_against_reference(
        name="json-check",
        ref_fn=lambda: t,
        got_fn=lambda: t.clone(),
        out_dir=tmp_path,
    )
    report_path = tmp_path / "verification.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["name"] == "json-check"
    assert data["passed"] is True
    assert "comparisons" in data
    assert len(data["comparisons"]) == 1
    assert data["comparisons"][0]["passed"] is True


def test_verification_json_on_failure(tmp_path: Path) -> None:
    """verification.json should reflect failure correctly."""
    verify_callable_against_reference(
        name="fail-report",
        ref_fn=lambda: torch.zeros(5),
        got_fn=lambda: torch.ones(5),
        out_dir=tmp_path,
    )
    data = json.loads((tmp_path / "verification.json").read_text(encoding="utf-8"))
    assert data["passed"] is False
    assert data["comparisons"][0]["num_mismatched"] == 5


# -- tuple output handling ----------------------------------------------------


def test_tuple_output_all_pass(tmp_path: Path) -> None:
    """Verification should handle tuple outputs, comparing each tensor."""
    a = torch.randn(3)
    b = torch.randn(4)
    result = verify_callable_against_reference(
        name="tuple-pass",
        ref_fn=lambda: (a, b),
        got_fn=lambda: (a.clone(), b.clone()),
        out_dir=tmp_path,
    )
    assert result.passed
    assert len(result.comparisons) == 2


def test_tuple_output_partial_fail(tmp_path: Path) -> None:
    """If one tuple element mismatches, the whole run should fail."""
    a = torch.ones(3)
    b = torch.ones(4)
    result = verify_callable_against_reference(
        name="tuple-fail",
        ref_fn=lambda: (a, b),
        got_fn=lambda: (a.clone(), b + 1.0),
        out_dir=tmp_path,
    )
    assert not result.passed
    assert result.comparisons[0].passed
    assert not result.comparisons[1].passed


def test_list_output_supported(tmp_path: Path) -> None:
    """List outputs should be handled like tuples."""
    t = torch.randn(2)
    result = verify_callable_against_reference(
        name="list-out",
        ref_fn=lambda: [t],
        got_fn=lambda: [t.clone()],
        out_dir=tmp_path,
    )
    assert result.passed


# -- out_dir creation ---------------------------------------------------------


def test_nested_out_dir_created(tmp_path: Path) -> None:
    """out_dir should be created if it does not exist."""
    deep = tmp_path / "a" / "b" / "c"
    t = torch.randn(2)
    verify_callable_against_reference(
        name="nested",
        ref_fn=lambda: t,
        got_fn=lambda: t.clone(),
        out_dir=deep,
    )
    assert (deep / "verification.json").exists()


# -- dataclass frozen ---------------------------------------------------------


def test_verification_run_frozen() -> None:
    """VerificationRun should be immutable."""
    from compgen.verify.compare import NumericComparison

    cmp = NumericComparison(passed=True, max_abs_error=0.0, max_rel_error=0.0, atol=1e-5, rtol=1e-5, num_mismatched=0)
    run = VerificationRun(name="x", passed=True, latency_ref_ms=1.0, latency_got_ms=1.0, comparisons=(cmp,))
    try:
        run.passed = False  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass
