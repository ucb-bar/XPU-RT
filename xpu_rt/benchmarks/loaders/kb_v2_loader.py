"""Load KB-v2 agent-loop results into the canonical row schema.

The KB-v2 agent loop returns :class:`AgentLoopResult` from
``xpu_rt.kernels.kernelblaster_v2.agent_loop:82``. The headline fields
we need:

  * ``best.report.correct``    → ``correctness``
  * ``best.report.cycles``     → ``cycles``
  * ``len(history)``           → ``rounds_used``
  * ``best.proposal.action``   → reported in ``notes``

For LLM cost we don't read it from the AgentLoopResult (the loop
doesn't track tokens). Instead we let the caller pass in a
``CostSnapshot`` derived from
:mod:`xpu_rt.observability.gemini_usage`'s pre/post cumulative
read — that gives per-cell cost without coupling the loader to
the Gemini SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xpu_rt.benchmarks.canonical_metrics import CanonicalCellRow


@dataclass(frozen=True)
class CostSnapshot:
    """Pre/post Gemini cumulative for one repeat. Caller computes
    these via :func:`xpu_rt.observability.gemini_usage.cumulative_usd`
    bracketing the agent-loop call."""

    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_s: float = 0.0


def cycle_source_for_target(target: str) -> str:
    """Map target id → the cycle counter the harness reads.

    Gemmini reads ``MAIN_LD_ST_EX_CYCLES``; Saturn reads ``mcycle``
    (the standard RISC-V CSR; matches the kb_saturn templates).
    """
    t = target.lower()
    if t.startswith("saturn") or t.startswith("opu"):
        return "mcycle"
    return "MAIN_LD_ST_EX_CYCLES"


def load_kb_v2_row(
    result: Any,
    *,
    target: str,
    workload: str,
    shape_id: str,
    repeat: int,
    cost: CostSnapshot,
) -> CanonicalCellRow:
    """Build a canonical row from one AgentLoopResult.

    ``result`` is typed ``Any`` so this loader stays import-light;
    it accesses ``result.best``, ``result.history``,
    ``result.aborted``, ``result.abort_reason``.
    """
    best = getattr(result, "best", None)
    history = list(getattr(result, "history", []))
    aborted = bool(getattr(result, "aborted", False))
    abort_reason = str(getattr(result, "abort_reason", ""))

    if best is None:
        return CanonicalCellRow(
            backend="kb-v2",
            target=target,
            workload=workload,
            shape_id=shape_id,
            repeat=repeat,
            correctness=False,
            cycles=None,
            rounds_used=len(history),
            tokens_in=cost.tokens_in,
            tokens_out=cost.tokens_out,
            cost_usd=cost.cost_usd,
            wall_s=cost.wall_s,
            cycle_source="none",
            notes=f"no candidate; aborted={aborted} reason={abort_reason!r}"
            if aborted
            else "no candidate accepted",
        )

    report = best.report
    correct = bool(report.correct)
    cycles_raw = getattr(report, "cycles", None)
    cycles = int(cycles_raw) if cycles_raw is not None else None
    cycle_source = cycle_source_for_target(target) if cycles is not None else "none"

    # Capture the *failure mode* (compile_failed / spike_timeout /
    # harness_unsupported_op_family / no_mismatch_line / ...) in notes
    # when correctness=False. Without this the canonical row gives the
    # report builder no signal about WHY a cell is zero-correct — and
    # the fair-comparison report's "why" column ends up blank.
    metadata = getattr(report, "metadata", {}) or {}
    reason = metadata.get("reason", "")
    action = getattr(best.proposal, "action", "")
    if correct:
        notes_str = f"action={action!r}"
    elif aborted:
        notes_str = f"aborted={aborted} reason={abort_reason!r}"
    else:
        # Failed candidate — surface the evaluator's failure-mode tag.
        diff = (getattr(report, "diff_summary", "") or "")[:120]
        notes_str = f"action={action!r} fail_reason={reason!r}"
        if diff:
            notes_str += f" diff={diff!r}"

    return CanonicalCellRow(
        backend="kb-v2",
        target=target,
        workload=workload,
        shape_id=shape_id,
        repeat=repeat,
        correctness=correct,
        cycles=cycles,
        rounds_used=int(getattr(best, "attempt", 0)) + 1,
        tokens_in=cost.tokens_in,
        tokens_out=cost.tokens_out,
        cost_usd=cost.cost_usd,
        wall_s=cost.wall_s,
        cycle_source=cycle_source,
        notes=notes_str,
    )


__all__ = ["CostSnapshot", "cycle_source_for_target", "load_kb_v2_row"]
