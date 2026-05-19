"""Triton-target evaluator stub.

Real implementation: compile the candidate as a Triton kernel, diff its
output against the golden tensors that XPU-RT staged at contract
materialization time, and report cycles via the Triton profiler. This
module currently raises :class:`NotImplementedError` so the agent loop
can be wired and tested without the full compile/run pipeline; the
landing is tracked in task #10.
"""

from __future__ import annotations

from dataclasses import dataclass

from xpu_rt.kernels.kernelblaster_v2.evaluators.base import (
    EvaluationReport,
    Evaluator,
)
from xpu_rt.kernels.kernelblaster_v2.generators import ProposeResponse


@dataclass
class TritonEvaluator:
    name: str = "triton"

    def evaluate(self, candidate: ProposeResponse) -> EvaluationReport:
        raise NotImplementedError(
            "TritonEvaluator is not yet implemented; use MockEvaluator until task #10 lands"
        )
