"""Typed counterexamples + delta-debug minimiser (P2.3).

Today a failed differential gate returns a single scalar
``max_abs_error`` — the LLM has no idea where to look. The plan calls
for a typed payload the agent can navigate:

* an :class:`InputSlice` naming the input tensor + indices,
* an :class:`OutputSlice` with the actual + reference values + abs
  error,
* an :class:`IRSlice` pointing at the offending op + the *tactic*
  that produced it,
* a free-text ``likely_cause``,
* a structured :class:`RemediationHint` whose ``suggest`` field
  references a candidate id (or ``None``) from the precomputed
  candidate set.

The :func:`delta_debug_input` helper bisects a failing tensor input
down to the smallest slice that still fails, so the LLM learns from
"this single batch index" rather than "your error was big."

This module is pure-function and standalone. Wire-in to the existing
Z3 verifier + remediation library lives in a follow-up.

Hard rules:

* The LLM is *never* the verifier — the verifier produces the
  counterexample; the LLM produces the remediation hint.
* ``RemediationHint.suggest`` is a candidate id (or ``None``); the
  LLM cannot fabricate an edit out of thin air.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, TypeVar

REMEDIATION_KINDS: Final[tuple[str, ...]] = (
    "tactic_change",
    "param_change",
    "abandon_tactic",
)

REJECTION_CLASSES: Final[tuple[str, ...]] = (
    "tactic_fatal",
    "tactic_recoverable",
    "surprising",
)


class CounterexampleError(ValueError):
    """A counterexample dataclass got a closed-enum violation."""


@dataclass(frozen=True)
class InputSlice:
    """The (possibly minimised) input that triggers the failure."""

    name: str
    indices: dict[str, int]
    actual: float | None = None
    reference: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "indices": dict(self.indices),
            "actual": self.actual,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class OutputSlice:
    """The output cell where actual ≠ reference."""

    name: str
    indices: dict[str, int]
    actual: float
    reference: float
    abs_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "indices": dict(self.indices),
            "actual": self.actual,
            "reference": self.reference,
            "abs_error": self.abs_error,
        }


@dataclass(frozen=True)
class IRSlice:
    """The Recipe-IR op + tactic context that produced the bad output."""

    region_id: str
    op: str
    annotation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "op": self.op,
            "annotation": self.annotation,
        }


@dataclass(frozen=True)
class RemediationHint:
    """LLM-shaped suggestion drawn from the precomputed candidate set.

    ``suggest`` is the id of a candidate the Tactician already has in
    its candidate list (or ``None`` when the recommendation is to
    abandon the current tactic). The verifier never fabricates an
    edit — that is the structural guarantee P3.0's
    ``invent_candidate_not_in_input_list`` forbidden action protects.
    """

    kind: str
    suggest: str | None
    confidence: float
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.kind not in REMEDIATION_KINDS:
            raise CounterexampleError(
                f"remediation kind={self.kind!r} must be one of {REMEDIATION_KINDS}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise CounterexampleError(
                f"remediation confidence={self.confidence} must be in [0, 1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "suggest": self.suggest,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class Counterexample:
    """Full typed counterexample emitted by the verifier."""

    gate: str
    rejection_class: str
    input_slice: InputSlice
    output_slice: OutputSlice
    ir_slice: IRSlice
    likely_cause: str = ""
    remediation: RemediationHint | None = None

    def __post_init__(self) -> None:
        if self.rejection_class not in REJECTION_CLASSES:
            raise CounterexampleError(
                f"rejection_class={self.rejection_class!r} must be one of {REJECTION_CLASSES}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "rejection_class": self.rejection_class,
            "input_slice": self.input_slice.to_dict(),
            "output_slice": self.output_slice.to_dict(),
            "ir_slice": self.ir_slice.to_dict(),
            "likely_cause": self.likely_cause,
            "remediation": self.remediation.to_dict() if self.remediation else None,
        }


T = TypeVar("T")


def delta_debug_input(  # noqa: UP047  (keep TypeVar for Python 3.11 compat)
    failing_input: list[T],
    *,
    failing_predicate: Callable[[list[T]], bool],
    max_iterations: int = 64,
) -> list[T]:
    """Bisect ``failing_input`` to the smallest contiguous slice that
    still triggers ``failing_predicate``.

    The minimiser implements the classic delta-debugging halving step:
    at each iteration it tries the first half and the second half. If
    either still fails, it recurses on that half. If neither fails,
    the current ``failing_input`` is already minimal (modulo
    contiguousness — non-contiguous minimisation is a follow-up).

    The predicate must be deterministic: same input → same verdict.
    A ``max_iterations`` cap protects against pathological predicates.
    """

    if not failing_predicate(failing_input):
        raise CounterexampleError(
            "delta_debug_input called with an input that does not fail "
            "the predicate; nothing to minimise"
        )

    current = list(failing_input)
    iterations = 0
    while iterations < max_iterations and len(current) > 1:
        iterations += 1
        mid = len(current) // 2
        first_half = current[:mid]
        second_half = current[mid:]
        if first_half and failing_predicate(first_half):
            current = first_half
            continue
        if second_half and failing_predicate(second_half):
            current = second_half
            continue
        break  # neither half fails alone; current is minimal
    return current


def classify_rejection(
    *,
    legality_was_blocked: bool,
    numerical_only: bool,
    remediation_known: bool,
) -> str:
    """Decide which rung the rejection lands on (closed enum).

    Rules:

    * ``legality_was_blocked`` → ``tactic_fatal``: the verifier proved
      the tactic is illegal in principle. The Strategist drops the
      rung from the fallback ladder.
    * ``numerical_only and remediation_known`` → ``tactic_recoverable``:
      a known fix exists; the Tactician retries with the remediation
      hint.
    * everything else → ``surprising``: the Strategist escalates.
    """

    if legality_was_blocked:
        return "tactic_fatal"
    if numerical_only and remediation_known:
        return "tactic_recoverable"
    return "surprising"


__all__ = [
    "REJECTION_CLASSES",
    "REMEDIATION_KINDS",
    "Counterexample",
    "CounterexampleError",
    "IRSlice",
    "InputSlice",
    "OutputSlice",
    "RemediationHint",
    "classify_rejection",
    "delta_debug_input",
]
