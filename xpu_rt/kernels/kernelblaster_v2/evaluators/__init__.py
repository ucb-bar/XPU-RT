"""Evaluator backends — score a candidate kernel.

An :class:`Evaluator` returns an :class:`EvaluationReport` per
candidate. The agent loop uses ``correct`` + ``score`` to decide
whether to accept and how to feed back failures into the next propose
request.

Two concrete evaluators ship today:

* :class:`MockEvaluator` — deterministic, used by tests.

The two real evaluators (``TritonEvaluator`` for Triton-friendly
targets and ``CRiscvEvaluator`` for Saturn / Gemmini) are stubbed in
sibling modules with :class:`NotImplementedError` so the protocol +
agent loop are exercised end-to-end while the cross-compile / sim
plumbing is wired up incrementally.
"""

from __future__ import annotations

from xpu_rt.kernels.kernelblaster_v2.evaluators.base import (
    EvaluationReport,
    Evaluator,
    MockEvaluator,
)

__all__ = ["EvaluationReport", "Evaluator", "MockEvaluator"]
