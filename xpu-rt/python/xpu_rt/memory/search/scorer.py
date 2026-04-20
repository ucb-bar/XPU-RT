"""Composite scoring functions for candidates.

Each object kind has its own scoring formula with decomposed
components, so retrieval can later ask for "things that helped
memory-bound cases" or "things that reduced config cycles."
"""

from __future__ import annotations

from dataclasses import dataclass

from compgen.memory.schema import Evaluation


@dataclass(frozen=True)
class ScoreBreakdown:
    """Decomposed score for a candidate evaluation.

    Stored alongside the scalar score so retrieval can filter
    by component (e.g., "best for memory-bound" or "best proof bonus").
    """

    perf_gain: float = 0.0
    correctness_gate: float = 0.0
    portability_bonus: float = 0.0
    proof_bonus: float = 0.0
    reuse_bonus: float = 0.0
    compile_cost_penalty: float = 0.0
    complexity_penalty: float = 0.0
    total: float = 0.0


def score_kernel(eval_: Evaluation) -> ScoreBreakdown:
    """Score a kernel candidate.

    Formula::

        score = correctness_gate * (perf_gain + proof_bonus - complexity_penalty)

    where:
        correctness_gate = 1.0 if correct, 0.0 otherwise
        perf_gain = speedup proxy from latency (lower = better)
        proof_bonus = 0.1 if verifier passed
    """
    correctness_gate = 1.0 if eval_.correctness_ok else 0.0

    # Perf gain: normalize latency (lower is better, cap at 1.0)
    perf_gain = max(0.0, 1.0 - eval_.latency_us / 1000.0) if eval_.latency_us > 0 else 0.0

    proof_bonus = 0.1 if eval_.verifier_summary and "pass" in eval_.verifier_summary.lower() else 0.0
    compile_penalty = 0.0 if eval_.compile_ok else 0.5

    total = correctness_gate * (perf_gain + proof_bonus - compile_penalty)

    return ScoreBreakdown(
        perf_gain=perf_gain,
        correctness_gate=correctness_gate,
        proof_bonus=proof_bonus,
        compile_cost_penalty=compile_penalty,
        total=total,
    )


def score_pass(eval_: Evaluation) -> ScoreBreakdown:
    """Score a pass candidate.

    Formula::

        score = verification_gate * (perf_gain + reuse_bonus + proof_bonus)
    """
    verification_gate = 1.0 if eval_.correctness_ok else 0.0
    perf_gain = eval_.score  # use the evaluation's score directly
    proof_bonus = 0.2 if eval_.verifier_summary and "valid" in eval_.verifier_summary.lower() else 0.0

    total = verification_gate * (perf_gain + proof_bonus)

    return ScoreBreakdown(
        perf_gain=perf_gain,
        correctness_gate=verification_gate,
        proof_bonus=proof_bonus,
        total=total,
    )


def score_plan(eval_: Evaluation) -> ScoreBreakdown:
    """Score a backend plan candidate."""
    feasibility_gate = 1.0 if eval_.compile_ok else 0.0
    perf_gain = eval_.score
    total = feasibility_gate * perf_gain

    return ScoreBreakdown(
        perf_gain=perf_gain,
        correctness_gate=feasibility_gate,
        total=total,
    )


__all__ = ["ScoreBreakdown", "score_kernel", "score_pass", "score_plan"]
