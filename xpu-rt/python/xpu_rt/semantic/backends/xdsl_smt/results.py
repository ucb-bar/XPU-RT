"""Unified result types and counterexample structuring for SMT-based verification.

All verification backends return structured results that can be consumed
by the agent through the Observation. Counterexamples from Z3 are parsed
into agent-readable ``StructuredCounterexample`` objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StructuredCounterexample:
    """Agent-readable counterexample extracted from Z3 output.

    Attributes:
        inputs: Variable name -> concrete value mapping.
        expected: Expected output values (from the source program).
        actual: Actual output values (from the target program).
        summary: One-line human-readable description.
    """

    inputs: dict[str, str] = field(default_factory=dict)
    expected: dict[str, str] = field(default_factory=dict)
    actual: dict[str, str] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True)
class TVResult:
    """Result of translation validation (refinement check).

    Attributes:
        ok: True if the transformation preserves semantics (unsat).
        status: "valid", "invalid", "unknown", or "timeout".
        smtlib: The SMT-LIB query that was sent to the solver.
        solver_stdout: Raw stdout from Z3.
        solver_stderr: Raw stderr from Z3.
        solver_time_ms: Wall-clock time in the solver.
        counterexample: Structured counterexample if invalid.
    """

    ok: bool
    status: str = "unknown"
    smtlib: str = ""
    solver_stdout: str = ""
    solver_stderr: str = ""
    solver_time_ms: float = 0.0
    counterexample: StructuredCounterexample | None = None


@dataclass(frozen=True)
class PDLResult:
    """Result of verifying a PDL rewrite pattern across bitwidths.

    Attributes:
        sound: True if the pattern is sound at all checked bitwidths.
        status: "sound", "unsound", "unknown", or "timeout".
        bitwidths_checked: List of bitwidths that were verified.
        unsound_bitwidths: Bitwidths where the pattern was unsound.
        solver_time_ms: Total solver time across all bitwidths.
    """

    sound: bool
    status: str = "unknown"
    bitwidths_checked: list[int] = field(default_factory=list)
    unsound_bitwidths: list[int] = field(default_factory=list)
    solver_time_ms: float = 0.0


@dataclass(frozen=True)
class TransferResult:
    """Result of verifying a transfer function for soundness.

    Attributes:
        sound: True if the transfer function is a sound over-approximation.
        status: "sound", "unsound", "unknown", or "timeout".
        solver_time_ms: Solver time.
        counterexample: Structured counterexample if unsound.
    """

    sound: bool
    status: str = "unknown"
    solver_time_ms: float = 0.0
    counterexample: StructuredCounterexample | None = None


def parse_z3_counterexample(stdout: str, stderr: str) -> StructuredCounterexample | None:
    """Parse a Z3 model output into a structured counterexample.

    Z3 outputs models in the form::

        sat
        (model
          (define-fun x () (_ BitVec 32) #x00000003)
          (define-fun y () (_ BitVec 32) #xffffffff)
        )

    Args:
        stdout: Z3 stdout containing the model.
        stderr: Z3 stderr (for diagnostics).

    Returns:
        StructuredCounterexample if a model was found, None otherwise.
    """
    if "sat" not in stdout or "unsat" in stdout:
        return None

    inputs: dict[str, str] = {}

    # Match define-fun lines: (define-fun name () type value)
    pattern = re.compile(r"\(define-fun\s+(\S+)\s+\(\)\s+\S+\s+(\S+)\)")
    for match in pattern.finditer(stdout):
        name, value = match.group(1), match.group(2)
        inputs[name] = value

    if not inputs:
        return None

    summary_parts = [f"{k}={v}" for k, v in sorted(inputs.items())[:5]]
    summary = "counterexample: " + ", ".join(summary_parts)
    if len(inputs) > 5:
        summary += f" (+{len(inputs) - 5} more)"

    return StructuredCounterexample(
        inputs=inputs,
        summary=summary,
    )


def interpret_z3_result(stdout: str, stderr: str, elapsed_ms: float) -> tuple[str, bool]:
    """Interpret Z3 stdout into a (status, ok) pair.

    Returns:
        (status, ok) where status is "valid"/"invalid"/"unknown"/"timeout"
        and ok is True only when status == "valid" (i.e., unsat).
    """
    stdout_stripped = stdout.strip()
    if "unsat" in stdout_stripped:
        return "valid", True
    if "sat" in stdout_stripped and "unsat" not in stdout_stripped:
        return "invalid", False
    if "timeout" in stdout_stripped or "timeout" in stderr:
        return "timeout", False
    return "unknown", False


__all__ = [
    "PDLResult",
    "StructuredCounterexample",
    "TVResult",
    "TransferResult",
    "interpret_z3_result",
    "parse_z3_counterexample",
]
