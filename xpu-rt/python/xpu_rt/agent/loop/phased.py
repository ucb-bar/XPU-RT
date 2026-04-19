"""Phased LLM drive loop (P2 skeleton).

Minimal iterator that drives the LLM through Phases 2 → 5 by calling
registered tools and exercising registered invent-slots. Every call is
recorded via ``ToolCallRecorder``.

This is **not** a replacement for ``compgen.agent.loop.core.AgenticCompilationLoop``
(1334 LOC, full iterative optimization with environment reset,
memory, refinement, and cost tracking). That remains the real
production loop. This module is the phased scaffold the
architecture docs describe — useful for:

- Unit testing the registry + recorder + invent-slot path end-to-end.
- Driving a minimal mock-LLM compilation through the phases.
- Providing the template that more specialized drivers (per-phase
  recipes) can reuse.

Expected compose flow once P7/P8 real pass ports land:
    AgenticCompilationLoop (or bespoke Phase 0–1 driver)
        → PhasedDriveLoop.run(phases=[2, 3, 4, 5], ...)
        → kernel contract emission
        → autocomp
        → runtime contract emission
        → verification + promotion
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from compgen.llm.recorder import ToolCallRecorder
from compgen.llm.registry import (
    InventSlot,
    Registry,
    Tool,
    get_registry,
)


# ---------------------------------------------------------------------------
# Types the driver consumes
# ---------------------------------------------------------------------------


#: A PhasePolicy decides what the LLM *would* do in a given phase.
#: Input: (phase, registry, context). Output: an ordered list of
#: (tool_or_slot_name, args). The driver runs them in order.
#: In a real integration the LLM returns this; for tests + scaffolding
#: we accept a plain callable.
PhasePolicy = Callable[[int, Registry, dict[str, Any]], list[tuple[str, dict[str, Any]]]]


@dataclass
class PhaseRunSummary:
    """What happened during one phase."""

    phase: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    invent_calls: list[dict[str, Any]] = field(default_factory=list)
    rejected_invents: int = 0
    elapsed_ms: float = 0.0


@dataclass
class DriveLoopResult:
    """What happened across all phases."""

    phase_summaries: list[PhaseRunSummary] = field(default_factory=list)
    transcript_path: Path | None = None
    total_elapsed_ms: float = 0.0

    @property
    def total_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.phase_summaries)

    @property
    def total_invent_calls(self) -> int:
        return sum(len(s.invent_calls) for s in self.phase_summaries)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


@dataclass
class PhasedDriveLoop:
    """Drive the registry through a sequence of phases.

    Attributes:
        registry: The LLM registry (defaults to the process-wide one).
        recorder: Optional ToolCallRecorder; if None, records nothing.
        context: Opaque shared state passed to every policy call (IR,
            target profile, dossier, ...).
        llm_turn_id: Base turn id; incremented each phase.
    """

    registry: Registry = field(default_factory=get_registry)
    recorder: ToolCallRecorder | None = None
    context: dict[str, Any] = field(default_factory=dict)
    llm_turn_id: str = "phased_drive_loop"

    def run(
        self,
        *,
        phases: Iterable[int],
        policy: PhasePolicy,
    ) -> DriveLoopResult:
        """Iterate the phases in order, applying ``policy`` at each step.

        ``policy`` returns a list of (name, args) tuples; each is
        looked up in the registry as a tool first, then as an invent-
        slot. Tool results are recorded as ``kind='tool_call'``;
        invent-slot results are recorded as ``kind='invent_proposal'``
        with whatever gate the slot implements.
        """
        total_t0 = time.perf_counter()
        summaries: list[PhaseRunSummary] = []

        for phase in phases:
            phase_t0 = time.perf_counter()
            summary = PhaseRunSummary(phase=phase)

            steps = policy(phase, self.registry, self.context)
            for step_idx, (name, args) in enumerate(steps):
                tool = self.registry.lookup_tool(name, phase=phase)
                if tool is not None:
                    self._invoke_tool(phase, step_idx, tool, args, summary)
                    continue
                slot = self.registry.lookup_invent_slot(name, phase=phase)
                if slot is not None:
                    self._invoke_invent_slot(phase, step_idx, slot, args, summary)
                    continue
                # Neither tool nor slot: record a miss.
                entry = {
                    "step": step_idx,
                    "name": name,
                    "status": "not_found",
                    "phase": phase,
                }
                summary.tool_calls.append(entry)
                if self.recorder is not None:
                    self.recorder.record(
                        phase=phase,
                        name=name,
                        kind="tool_call",
                        args=args,
                        result={"status": "not_found"},
                        llm_turn_id=f"{self.llm_turn_id}:p{phase}:s{step_idx}",
                    )

            summary.elapsed_ms = (time.perf_counter() - phase_t0) * 1000.0
            summaries.append(summary)

        result = DriveLoopResult(
            phase_summaries=summaries,
            transcript_path=self.recorder.log_path if self.recorder else None,
            total_elapsed_ms=(time.perf_counter() - total_t0) * 1000.0,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _invoke_tool(
        self,
        phase: int,
        step_idx: int,
        tool: Tool,
        args: dict[str, Any],
        summary: PhaseRunSummary,
    ) -> None:
        t0 = time.perf_counter()
        try:
            result = tool.invoke(**args)
        except Exception as e:   # noqa: BLE001
            result = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        elapsed_ms = int((time.perf_counter() - t0) * 1000.0)

        entry = {
            "step": step_idx,
            "phase": phase,
            "name": tool.name,
            "kind": "tool_call",
            "status": result.get("status", "ok") if isinstance(result, dict) else "ok",
            "elapsed_ms": elapsed_ms,
        }
        summary.tool_calls.append(entry)

        if self.recorder is not None:
            self.recorder.record(
                phase=phase,
                name=tool.name,
                kind="tool_call",
                args=args,
                result=result if isinstance(result, dict) else {"value": result},
                select_vs_invent="select",
                elapsed_ms=elapsed_ms,
                llm_turn_id=f"{self.llm_turn_id}:p{phase}:s{step_idx}",
            )

    def _invoke_invent_slot(
        self,
        phase: int,
        step_idx: int,
        slot: InventSlot,
        args: dict[str, Any],
        summary: PhaseRunSummary,
    ) -> None:
        t0 = time.perf_counter()
        # The policy may pass either a fully-formed proposal or a
        # request to use the baseline seed.
        use_seed = args.pop("use_baseline_seed", False)
        if use_seed:
            proposal = slot.propose_baseline(**args)
        else:
            proposal = args.get("proposal") or {}

        gate_result = slot.verify(proposal, **args.get("gate_ctx", {}))
        elapsed_ms = int((time.perf_counter() - t0) * 1000.0)

        entry = {
            "step": step_idx,
            "phase": phase,
            "name": slot.name,
            "kind": "invent_proposal",
            "status": gate_result.get("status", "deferred"),
            "elapsed_ms": elapsed_ms,
        }
        summary.invent_calls.append(entry)
        if gate_result.get("status") == "rejected":
            summary.rejected_invents += 1

        if self.recorder is not None:
            self.recorder.record(
                phase=phase,
                name=slot.name,
                kind="invent_proposal",
                args=args,
                result={"chosen": proposal.get("chosen", {}),
                        "candidates_count": len(proposal.get("candidates", []))},
                select_vs_invent="invent",
                gate_result=gate_result,
                elapsed_ms=elapsed_ms,
                llm_turn_id=f"{self.llm_turn_id}:p{phase}:s{step_idx}",
            )


__all__ = [
    "DriveLoopResult",
    "PhasePolicy",
    "PhaseRunSummary",
    "PhasedDriveLoop",
]
