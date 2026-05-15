"""LLM-driven runtime scheduling decisions.

Instead of fixed heuristics, the LLM decides when to re-solve,
which batch tier to use, whether to migrate ops, and how to handle
thermal throttling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.agent.prompts.runtime_adapt import (
    RuntimeContext,
    format_prompt,
    parse_response,
)
from compgen.llm.base import CompGenLLMProtocol, GenerationRequest, LLMConfig, Objective, PromptContext

log = structlog.get_logger()


@dataclass(frozen=True)
class SchedulingDecision:
    """A scheduling decision from the LLM."""

    action: str  # "keep", "re_solve", "change_batch_tier", "migrate_ops", "throttle"
    reason: str
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5


@dataclass
class LLMScheduler:
    """LLM-driven dynamic scheduling.

    Wraps the calibration loop and adaptive scheduler with LLM-based
    decision making for when and how to adapt.
    """

    llm_client: CompGenLLMProtocol
    decision_history: list[SchedulingDecision] = field(default_factory=list)

    def decide(
        self,
        measured_latency_us: float,
        estimated_latency_us: float,
        device_utilization: dict[str, float] | None = None,
        batch_size: int = 1,
        request_rate_rps: float = 0.0,
        thermal_headroom_pct: float = 100.0,
    ) -> SchedulingDecision:
        """Ask LLM whether to adapt the runtime schedule.

        Args:
            measured_latency_us: Actual measured latency.
            estimated_latency_us: Cost model estimate.
            device_utilization: Per-device utilization (0-100%).
            batch_size: Current batch size.
            request_rate_rps: Incoming request rate.
            thermal_headroom_pct: Remaining thermal budget.

        Returns:
            SchedulingDecision with action and reasoning.
        """
        drift = abs(measured_latency_us - estimated_latency_us) / max(estimated_latency_us, 1e-9) * 100

        ctx = RuntimeContext(
            measured_latency_us=measured_latency_us,
            estimated_latency_us=estimated_latency_us,
            drift_pct=drift,
            device_utilization=device_utilization or {},
            batch_size=batch_size,
            request_rate_rps=request_rate_rps,
            thermal_headroom_pct=thermal_headroom_pct,
        )

        prompt = format_prompt(ctx)

        try:
            request = GenerationRequest(
                prompt_template=prompt,
                context=PromptContext(
                    model_ir_summary="",
                    target_profile_summary="runtime",
                    available_transforms=[],
                    kernel_contracts=[],
                    objective=Objective.LATENCY,
                ),
                config=LLMConfig(
                    model=str(getattr(self.llm_client, "model", "default")),
                    temperature=0.2,
                ),
            )
            response = self.llm_client.generate(request)
            parsed = parse_response(response.raw_text)

            if parsed is not None:
                decision = SchedulingDecision(
                    action=parsed.decision,
                    reason=parsed.reason,
                    parameters=parsed.parameters,
                    confidence=0.8,
                )
            else:
                decision = SchedulingDecision(action="keep", reason="failed to parse LLM response")
        except Exception as e:
            decision = SchedulingDecision(action="keep", reason=f"LLM error: {e}")

        self.decision_history.append(decision)
        log.info("llm_scheduler.decision", action=decision.action, reason=decision.reason)
        return decision

    def should_re_solve(self, drift_pct: float) -> bool:
        """Quick heuristic check before invoking LLM."""
        return drift_pct > 20.0

    def reset_history(self) -> None:
        """Clear decision history."""
        self.decision_history.clear()
