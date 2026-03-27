"""Tests for peephole rewrite verification."""

from __future__ import annotations

import pytest
from compgen.ir.semantic.peephole_verify import RewriteVerificationResult


def test_rewrite_verification_result_valid() -> None:
    result = RewriteVerificationResult(valid=True, status="valid", solver_time_ms=42.5)
    assert result.valid is True
    assert result.status == "valid"
    assert result.counterexample is None
    assert result.solver_time_ms == 42.5
    assert result.cached is False


def test_rewrite_verification_result_invalid() -> None:
    ce = {"x": 1, "y": -1}
    result = RewriteVerificationResult(valid=False, status="invalid", counterexample=ce)
    assert result.valid is False
    assert result.status == "invalid"
    assert result.counterexample == ce


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_verify_rewrite_equivalence() -> None:
    """verify_rewrite should prove equivalence of pattern and replacement."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_verify_rewrite_caching() -> None:
    """Verified rewrites should be cacheable and reusable."""
