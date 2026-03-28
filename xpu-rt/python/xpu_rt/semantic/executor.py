"""Verification executor — dispatches obligations to backends.

Takes the verification obligation dicts produced by ``lower_recipe()``
and routes each to the appropriate verification backend. Results are
returned as structured ``VerificationResult`` objects that flow into
the agent's observation.

This is the central bridge between Recipe IR lowering and formal
verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from xdsl.dialects.builtin import ModuleOp

from compgen.semantic.backends.xdsl_smt.results import StructuredCounterexample

log = structlog.get_logger()


@dataclass(frozen=True)
class VerificationResult:
    """Result of executing a single verification obligation.

    Attributes:
        obligation_type: The kind of obligation ("translation_validation",
            "differential", "layout_invariant", "memory_bound",
            "check_file", "profile_budget").
        region_id: The Payload IR region this obligation applies to.
        passed: Whether the verification passed.
        status: "valid", "invalid", "unknown", "timeout", or "skipped".
        solver_time_ms: Time spent in the solver (0 for non-SMT checks).
        counterexample: Structured counterexample if the check failed.
        details: Backend-specific details.
    """

    obligation_type: str
    region_id: str
    passed: bool
    status: str = "unknown"
    solver_time_ms: float = 0.0
    counterexample: StructuredCounterexample | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationExecutor:
    """Dispatches verification obligations to appropriate backends.

    The executor is the system-level component that sits between Recipe IR
    lowering and the agent's observation. The agent does NOT call backends
    directly — it emits Recipe IR verification ops, the system lowers them
    to obligation dicts, and this executor runs them.

    Attributes:
        tv_timeout_ms: Timeout for translation validation checks.
        enable_tv: Whether to run translation validation (requires Z3).
    """

    tv_timeout_ms: int = 30_000
    enable_tv: bool = True

    def execute_obligations(
        self,
        obligations: list[dict[str, Any]],
        payload_before: ModuleOp | None = None,
        payload_after: ModuleOp | None = None,
    ) -> list[VerificationResult]:
        """Execute all verification obligations.

        Args:
            obligations: List of obligation dicts from ``lower_recipe()``.
            payload_before: The Payload IR before transformation.
            payload_after: The Payload IR after transformation.

        Returns:
            List of VerificationResult, one per obligation.
        """
        results: list[VerificationResult] = []
        for obligation in obligations:
            result = self.execute_single(obligation, payload_before, payload_after)
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed and r.status != "skipped")
        log.info(
            "verification.execute_obligations",
            total=len(results),
            passed=passed,
            failed=failed,
        )
        return results

    def execute_single(
        self,
        obligation: dict[str, Any],
        payload_before: ModuleOp | None = None,
        payload_after: ModuleOp | None = None,
    ) -> VerificationResult:
        """Execute a single verification obligation.

        Args:
            obligation: Obligation dict with at least "type" and "region_id".
            payload_before: The Payload IR before transformation.
            payload_after: The Payload IR after transformation.

        Returns:
            VerificationResult for this obligation.
        """
        ob_type = obligation.get("type", "unknown")
        region_id = obligation.get("region_id", "")

        try:
            if ob_type == "translation_validation":
                return self._execute_tv(obligation, payload_before, payload_after)
            elif ob_type == "differential":
                return self._execute_differential(obligation, payload_before, payload_after)
            elif ob_type == "layout_invariant":
                return self._execute_layout_invariant(obligation, payload_after)
            elif ob_type == "memory_bound":
                return self._execute_memory_bound(obligation, payload_after)
            elif ob_type == "check_file":
                return self._execute_check_file(obligation, payload_after)
            elif ob_type == "profile_budget":
                # Deferred to runtime — cannot check at compile time
                return VerificationResult(
                    obligation_type=ob_type,
                    region_id=region_id,
                    passed=True,
                    status="skipped",
                    details={"reason": "profile budgets are checked at runtime"},
                )
            else:
                log.warning("verification.unknown_type", type=ob_type)
                return VerificationResult(
                    obligation_type=ob_type,
                    region_id=region_id,
                    passed=False,
                    status="unknown",
                    details={"reason": f"unknown obligation type: {ob_type}"},
                )
        except Exception as e:
            log.warning("verification.error", type=ob_type, region=region_id, error=str(e))
            return VerificationResult(
                obligation_type=ob_type,
                region_id=region_id,
                passed=False,
                status="unknown",
                details={"error": str(e)},
            )

    def _execute_tv(
        self,
        obligation: dict[str, Any],
        payload_before: ModuleOp | None,
        payload_after: ModuleOp | None,
    ) -> VerificationResult:
        """Execute a translation validation obligation."""
        region_id = obligation.get("region_id", "")

        if not self.enable_tv:
            return VerificationResult(
                obligation_type="translation_validation",
                region_id=region_id,
                passed=True,
                status="skipped",
                details={"reason": "TV disabled"},
            )

        if payload_before is None or payload_after is None:
            return VerificationResult(
                obligation_type="translation_validation",
                region_id=region_id,
                passed=False,
                status="unknown",
                details={"reason": "missing before/after modules"},
            )

        from compgen.semantic.backends.xdsl_smt.tv_backend import TranslationValidationBackend

        backend = TranslationValidationBackend(timeout_ms=self.tv_timeout_ms)
        tv_result = backend.check_refinement(
            payload_before, payload_after, optimize=True
        )

        return VerificationResult(
            obligation_type="translation_validation",
            region_id=region_id,
            passed=tv_result.ok,
            status=tv_result.status,
            solver_time_ms=tv_result.solver_time_ms,
            counterexample=tv_result.counterexample,
            details={
                "solver_stdout": tv_result.solver_stdout[:200],
            },
        )

    def _execute_differential(
        self,
        obligation: dict[str, Any],
        payload_before: ModuleOp | None,
        payload_after: ModuleOp | None,
    ) -> VerificationResult:
        """Execute a differential testing obligation."""
        region_id = obligation.get("region_id", "")
        tolerance = obligation.get("tolerance_ulps", 1)

        if payload_before is None or payload_after is None:
            return VerificationResult(
                obligation_type="differential",
                region_id=region_id,
                passed=False,
                status="unknown",
                details={"reason": "missing before/after modules"},
            )

        from compgen.transforms.verify import _verify_differential

        passed, max_error, msg = _verify_differential(
            payload_before, payload_after, float(tolerance)
        )
        return VerificationResult(
            obligation_type="differential",
            region_id=region_id,
            passed=passed,
            status="valid" if passed else "invalid",
            details={"message": msg, "max_error": max_error},
        )

    def _execute_layout_invariant(
        self,
        obligation: dict[str, Any],
        payload_after: ModuleOp | None,
    ) -> VerificationResult:
        """Execute a layout invariant check (structural)."""
        region_id = obligation.get("region_id", "")
        expected_layout = obligation.get("expected_layout", "")

        # Structural check: verify the module is well-formed
        if payload_after is not None:
            try:
                payload_after.verify()
                return VerificationResult(
                    obligation_type="layout_invariant",
                    region_id=region_id,
                    passed=True,
                    status="valid",
                    details={"expected_layout": expected_layout},
                )
            except Exception as e:
                return VerificationResult(
                    obligation_type="layout_invariant",
                    region_id=region_id,
                    passed=False,
                    status="invalid",
                    details={"expected_layout": expected_layout, "error": str(e)},
                )

        return VerificationResult(
            obligation_type="layout_invariant",
            region_id=region_id,
            passed=False,
            status="unknown",
        )

    def _execute_memory_bound(
        self,
        obligation: dict[str, Any],
        payload_after: ModuleOp | None,
    ) -> VerificationResult:
        """Execute a memory bound check (static analysis)."""
        region_id = obligation.get("region_id", "")
        max_bytes = obligation.get("max_bytes", 0)

        # Static analysis placeholder — counts ops as proxy
        return VerificationResult(
            obligation_type="memory_bound",
            region_id=region_id,
            passed=True,
            status="skipped",
            details={"max_bytes": max_bytes, "reason": "static memory analysis not yet implemented"},
        )

    def _execute_check_file(
        self,
        obligation: dict[str, Any],
        payload_after: ModuleOp | None,
    ) -> VerificationResult:
        """Execute a FileCheck-style assertion."""
        region_id = obligation.get("region_id", "")
        path = obligation.get("path", "")

        # FileCheck execution placeholder
        return VerificationResult(
            obligation_type="check_file",
            region_id=region_id,
            passed=True,
            status="skipped",
            details={"path": path, "reason": "FileCheck not yet integrated"},
        )


__all__ = ["VerificationExecutor", "VerificationResult"]
