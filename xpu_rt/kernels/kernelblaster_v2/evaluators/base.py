"""Evaluator protocol + a mock implementation.

The real evaluators (cross-compile + run on spike / chipyard sim, parse
cycle counts) live in sibling modules and currently raise
:class:`NotImplementedError`. They get wired up alongside the e2e
verification (task #10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from xpu_rt.kernels.kernelblaster_v2.generators import ProposeResponse


@dataclass(frozen=True)
class EvaluationReport:
    """Per-candidate scoring outcome.

    Attributes:
        correct: True iff the candidate passed structural + functional
            verification gates.
        score: Free-form quality score; higher is better. Cycle counts
            (lower is better) are converted to ``1 / cycles`` upstream.
        cycles: Estimated or measured cycle count; ``None`` when the
            evaluator does not measure cycles.
        compile_log: stderr/stdout from the compile step.
        runtime_log: stderr/stdout from the run / sim step.
        diff_summary: Short summary of numerical correctness diffs.
        metadata: Free-form per-evaluator extras.
    """

    correct: bool
    score: float
    cycles: int | None = None
    compile_log: str = ""
    runtime_log: str = ""
    diff_summary: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


class Evaluator(Protocol):
    """Score a candidate kernel; return one report per call."""

    name: str

    def evaluate(self, candidate: ProposeResponse) -> EvaluationReport: ...


@dataclass
class MockEvaluator:
    """Caller-supplied table; used by tests.

    The ``table`` callable receives the candidate and returns an
    :class:`EvaluationReport`. Useful for golden tests that don't need
    a real compiler.
    """

    name: str = "mock"
    table: Callable[[ProposeResponse], EvaluationReport] = field(
        default=lambda c: EvaluationReport(correct=True, score=1.0)
    )
    calls: list[ProposeResponse] = field(default_factory=list)

    def evaluate(self, candidate: ProposeResponse) -> EvaluationReport:
        self.calls.append(candidate)
        return self.table(candidate)
