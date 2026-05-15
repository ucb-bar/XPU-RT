"""Tests for the VerificationExecutor."""

from __future__ import annotations

from compgen.semantic.executor import VerificationExecutor
from xdsl.dialects.builtin import ModuleOp


class TestVerificationExecutor:
    """Test the executor dispatches obligations correctly."""

    def test_empty_obligations(self) -> None:
        """No obligations → empty results."""
        executor = VerificationExecutor()
        results = executor.execute_obligations([])
        assert results == []

    def test_differential_obligation(self) -> None:
        """Differential obligation runs structural comparison."""
        executor = VerificationExecutor()
        before = ModuleOp([])
        after = ModuleOp([])

        result = executor.execute_single(
            {"type": "differential", "region_id": "matmul_0"},
            payload_before=before,
            payload_after=after,
        )
        assert result.obligation_type == "differential"
        assert result.region_id == "matmul_0"
        # Empty modules should pass structural diff
        assert result.passed

    def test_profile_budget_skipped(self) -> None:
        """Profile budget is deferred to runtime."""
        executor = VerificationExecutor()
        result = executor.execute_single(
            {"type": "profile_budget", "region_id": "r0", "max_latency_us": 100},
        )
        assert result.status == "skipped"
        assert result.passed

    def test_tv_disabled_skips(self) -> None:
        """TV obligation is skipped when disabled."""
        executor = VerificationExecutor(enable_tv=False)
        result = executor.execute_single(
            {"type": "translation_validation", "region_id": "r0"},
            payload_before=ModuleOp([]),
            payload_after=ModuleOp([]),
        )
        assert result.status == "skipped"
        assert result.passed

    def test_tv_missing_modules(self) -> None:
        """TV without before/after modules returns unknown."""
        executor = VerificationExecutor()
        result = executor.execute_single(
            {"type": "translation_validation", "region_id": "r0"},
        )
        assert not result.passed
        assert result.status == "unknown"

    def test_unknown_type(self) -> None:
        """Unknown obligation type returns unknown status."""
        executor = VerificationExecutor()
        result = executor.execute_single(
            {"type": "magic_check", "region_id": "r0"},
        )
        assert not result.passed
        assert result.status == "unknown"

    def test_mixed_obligations(self) -> None:
        """Multiple obligations of different types."""
        executor = VerificationExecutor(enable_tv=False)
        obligations = [
            {"type": "differential", "region_id": "r0"},
            {"type": "translation_validation", "region_id": "r1"},
            {"type": "profile_budget", "region_id": "r2", "max_latency_us": 100},
        ]
        results = executor.execute_obligations(
            obligations,
            payload_before=ModuleOp([]),
            payload_after=ModuleOp([]),
        )
        assert len(results) == 3
        assert all(r.passed for r in results)
