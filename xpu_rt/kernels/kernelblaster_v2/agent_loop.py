"""KernelBlaster v2 propose → evaluate → repair loop.

Top-level orchestrator: takes a :class:`KernelContract`, a
:class:`TargetKnowledgeCard`, a :class:`KernelGenerator`, and an
:class:`Evaluator`; runs up to ``max_iterations`` rounds. Each round:

  1. :class:`PromptBuilder` stitches contract + card + lessons + prior
     attempts into a :class:`PromptBundle`.
  2. The generator proposes one candidate.
  3. The evaluator scores it.
  4. Best-so-far is updated. The strategy DB is bumped.
  5. If the candidate is correct *and* score >= acceptance threshold,
     a lesson is appended and the loop exits early.

The loop is deterministic in its mock test paths and resilient to
generator/evaluator failures (a single round error doesn't kill the
loop unless ``strict=True``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from xpu_rt.kernels.kernelblaster_v2.contract_state import StateVector, derive_state
from xpu_rt.kernels.kernelblaster_v2.evaluators.base import (
    EvaluationReport,
    Evaluator,
)
from xpu_rt.kernels.kernelblaster_v2.generators import (
    KernelGenerator,
    ProposeRequest,
    ProposeResponse,
)
from xpu_rt.kernels.kernelblaster_v2.lesson_writer import LessonWriter
from xpu_rt.kernels.kernelblaster_v2.prompt_builder import PromptBuilder
from xpu_rt.kernels.kernelblaster_v2.strategy_db import StrategyDB
from xpu_rt.kernels.provider import (
    ContractFeedback,
    KernelContract,
    KnowledgeExport,
    ProviderResult,
)
from xpu_rt.memory.target_knowledge import TargetKnowledgeCard

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentLoopConfig:
    """Tunables for one :class:`KernelBlasterV2.run` call.

    Attributes:
        max_iterations: Upper bound on propose calls.
        accept_threshold: Minimum :class:`EvaluationReport.score` for a
            correct candidate to short-circuit the loop. ``inf`` means
            "always run to the budget".
        write_lessons: When True (default), accepted candidates produce
            a lesson row on disk.
        strict: When True, an unhandled generator/evaluator exception
            aborts the loop and re-raises. When False (default), the
            loop logs and continues with the next iteration.
    """

    max_iterations: int = 4
    accept_threshold: float = 1.0
    write_lessons: bool = True
    strict: bool = False


@dataclass(frozen=True)
class Candidate:
    """One full attempt's record — kept in memory for the loop history."""

    attempt: int
    proposal: ProposeResponse
    report: EvaluationReport


@dataclass
class AgentLoopResult:
    """Outcome of the run: best candidate + the full history."""

    best: Candidate | None
    history: list[Candidate] = field(default_factory=list)
    contract_feedback: list[ContractFeedback] = field(default_factory=list)
    knowledge_exports: list[KnowledgeExport] = field(default_factory=list)
    state: StateVector | None = None
    aborted: bool = False
    abort_reason: str = ""

    def found(self) -> bool:
        return self.best is not None and self.best.report.correct

    def to_provider_result(self) -> ProviderResult:
        """Coerce into the legacy provider-protocol ProviderResult."""
        if self.best is None:
            return ProviderResult(
                found=False,
                iterations_used=len(self.history),
                total_candidates=len(self.history),
                contract_feedback=list(self.contract_feedback),
                knowledge_exports=list(self.knowledge_exports),
                metadata={
                    "aborted": self.aborted,
                    "abort_reason": self.abort_reason,
                },
            )
        report = self.best.report
        return ProviderResult(
            found=report.correct,
            kernel_code=self.best.proposal.kernel_code,
            language=self.best.proposal.language,
            correct=report.correct,
            speedup=report.score,
            latency_us=float(report.cycles) if report.cycles is not None else 0.0,
            plan=self.best.proposal.action,
            iterations_used=len(self.history),
            total_candidates=len(self.history),
            contract_feedback=list(self.contract_feedback),
            knowledge_exports=list(self.knowledge_exports),
            metadata={
                "best_attempt": self.best.attempt,
                "best_action": self.best.proposal.action,
                "state_hash": self.state.hash() if self.state else "",
                "aborted": self.aborted,
                "abort_reason": self.abort_reason,
            },
        )


@dataclass
class KernelBlasterV2:
    """The full agent loop wired against pluggable backends."""

    card: TargetKnowledgeCard
    generator: KernelGenerator
    evaluator: Evaluator
    config: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    # -------------------------------------------------------------- public

    def run(self, contract: KernelContract) -> AgentLoopResult:
        state = derive_state(contract, target_id=self.card.target_id)
        strategy_db = StrategyDB.for_card(self.card)
        builder = PromptBuilder(card=self.card, strategy_db=strategy_db)
        writer = LessonWriter(card=self.card) if self.config.write_lessons else None
        result = AgentLoopResult(best=None, state=state)
        prior_attempts: list[dict[str, Any]] = []

        for attempt in range(self.config.max_iterations):
            try:
                proposal, report = self._one_round(
                    builder=builder,
                    contract=contract,
                    state=state,
                    attempt=attempt,
                    prior_attempts=prior_attempts,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("kernelblaster_v2: round %d failed", attempt)
                if self.config.strict:
                    raise
                result.aborted = True
                result.abort_reason = f"{exc.__class__.__name__}: {exc}"
                break

            candidate = Candidate(attempt=attempt, proposal=proposal, report=report)
            result.history.append(candidate)
            result.contract_feedback.extend(proposal.contract_feedback)

            if _is_better(candidate, result.best):
                result.best = candidate

            strategy_db.record(
                state=state,
                action=proposal.action or "(unknown)",
                accepted=report.correct,
                speedup=report.score,
            )

            # Thread the compile_log into notes whenever a candidate
            # didn't make it past the compiler — the next prompt sees
            # the literal `error: '<macro>' undeclared / macro defined
            # here <name>(<args>)` line, which is enough for the LLM to
            # repair arity-mismatch / wrong-name bugs in one round.
            notes_parts: list[str] = []
            if report.diff_summary:
                notes_parts.append(report.diff_summary)
            if not report.correct and report.compile_log:
                # Compile log can be multi-KB; the LLM gets the most
                # actionable tail (the actual gcc error message).
                tail = report.compile_log[-800:]
                notes_parts.append("compile_log_tail: " + tail)
            elif not report.correct and report.runtime_log:
                notes_parts.append("runtime_log_tail: " + report.runtime_log[-400:])
            prior_attempts.append(
                {
                    "action": proposal.action or "(unknown)",
                    "accepted": report.correct,
                    "speedup": report.score,
                    "notes": "\n".join(notes_parts),
                }
            )

            if report.correct and report.score >= self.config.accept_threshold:
                if writer is not None and proposal.action:
                    writer.write(
                        state=state,
                        action=proposal.action,
                        measured_gain=report.score,
                        notes=report.diff_summary,
                    )
                break

        # If we ended the loop without crossing the threshold but the
        # best candidate is still correct, persist a lesson anyway so
        # the next run benefits.
        if (
            writer is not None
            and not result.aborted
            and result.best is not None
            and result.best.report.correct
            and result.best.proposal.action
            and (
                # avoid double-write when the best was the short-circuit candidate
                len(result.history) == 0
                or result.best is not result.history[-1]
                or result.best.report.score < self.config.accept_threshold
            )
        ):
            writer.write(
                state=state,
                action=result.best.proposal.action,
                measured_gain=result.best.report.score,
                notes="best-so-far",
            )

        # A simple knowledge export — useful for downstream registries.
        if result.best is not None:
            result.knowledge_exports.append(
                KnowledgeExport(
                    kind="kernelblaster_v2_best_kernel",
                    scope="target",
                    scope_key=state.target_id,
                    content=result.best.proposal.kernel_code,
                    metadata={
                        "state_hash": state.hash(),
                        "action": result.best.proposal.action,
                    },
                    confidence=min(max(result.best.report.score, 0.0), 1.0),
                )
            )

        return result

    # ------------------------------------------------------------- helpers

    def _one_round(
        self,
        *,
        builder: PromptBuilder,
        contract: KernelContract,
        state: StateVector,
        attempt: int,
        prior_attempts: list[dict[str, Any]],
    ) -> tuple[ProposeResponse, EvaluationReport]:
        bundle = builder.build(
            contract=contract,
            state=state,
            prior_attempts=tuple(prior_attempts),
        )
        request = ProposeRequest(
            bundle=bundle,
            attempt_index=attempt,
            state_hash=state.hash(),
        )
        proposal = self.generator.propose(request)
        report = self.evaluator.evaluate(proposal)
        return proposal, report


def _is_better(candidate: Candidate, current_best: Candidate | None) -> bool:
    if current_best is None:
        return True
    # Correctness wins over speed.
    if candidate.report.correct and not current_best.report.correct:
        return True
    if not candidate.report.correct and current_best.report.correct:
        return False
    # Both correct: higher score wins. Both incorrect: prefer a
    # candidate that at least produced cycle data over one that
    # didn't (compile_failed / runtime crash). This matters for the
    # cross-backend comparison — even an incorrect kernel that
    # compiled + ran on Spike contributes a cycle-source data point
    # to the harness-skew row, while compile_failed candidates
    # leave the cell empty. Cycle ties go to the smaller value.
    cand_cycles = candidate.report.cycles
    best_cycles = current_best.report.cycles
    if not candidate.report.correct and not current_best.report.correct:
        if cand_cycles is not None and best_cycles is None:
            return True
        if cand_cycles is None and best_cycles is not None:
            return False
        if cand_cycles is not None and best_cycles is not None:
            return cand_cycles < best_cycles
    return candidate.report.score > current_best.report.score
