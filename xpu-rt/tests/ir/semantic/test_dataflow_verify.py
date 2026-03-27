"""Tests for dataflow analysis verification."""

from __future__ import annotations

import pytest
from compgen.ir.semantic.dataflow_verify import AnalysisVerificationResult


def test_analysis_verification_result_sound() -> None:
    result = AnalysisVerificationResult(sound=True, status="sound", solver_time_ms=10.0)
    assert result.sound is True
    assert result.status == "sound"
    assert result.counterexample is None
    assert result.solver_time_ms == 10.0


def test_analysis_verification_result_unsound() -> None:
    ce = {"input": [1, 2, 3], "expected_range": "(0, 5)", "actual": 7}
    result = AnalysisVerificationResult(sound=False, status="unsound", counterexample=ce)
    assert result.sound is False
    assert result.status == "unsound"
    assert result.counterexample is not None


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_verify_analysis_soundness() -> None:
    """verify_analysis should check that analysis is a sound over-approximation."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_verify_analysis_timeout() -> None:
    """verify_analysis should return 'timeout' status on solver timeout."""
