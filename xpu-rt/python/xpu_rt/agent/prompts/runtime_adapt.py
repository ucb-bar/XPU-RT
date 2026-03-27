"""Prompt for runtime scheduling adaptation."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeContext:
    """Context for runtime adaptation prompt."""

    measured_latency_us: float
    estimated_latency_us: float
    drift_pct: float
    device_utilization: dict[str, float]
    batch_size: int
    request_rate_rps: float
    thermal_headroom_pct: float


RUNTIME_PROMPT = textwrap.dedent("""\
    You are a runtime scheduling advisor for a heterogeneous ML system.

    ## Current State
    - Measured latency: {measured_us:.1f} us (estimated: {estimated_us:.1f} us)
    - Cost model drift: {drift_pct:.1f}%
    - Batch size: {batch_size}
    - Request rate: {request_rate:.1f} req/s
    - Thermal headroom: {thermal_pct:.0f}%

    ## Device Utilization:
    {utilization}

    ## Task
    Should we adapt the runtime? Choose one:
    1. "keep" — current schedule is fine
    2. "re_solve" — re-run the solver with calibrated costs
    3. "change_batch_tier" — switch to a different pre-computed batch plan
    4. "migrate_ops" — move ops between devices for better load balance
    5. "throttle" — reduce throughput to manage thermal/power

    Respond as JSON:
    {{"decision": "...", "reason": "...", "parameters": {{}}}}
""")


def format_prompt(ctx: RuntimeContext) -> str:
    """Render runtime adaptation prompt."""
    util_lines = "\n".join(
        f"  {dev}: {util:.0f}%" for dev, util in ctx.device_utilization.items()
    ) or "  (no data)"

    return RUNTIME_PROMPT.format(
        measured_us=ctx.measured_latency_us,
        estimated_us=ctx.estimated_latency_us,
        drift_pct=ctx.drift_pct,
        batch_size=ctx.batch_size,
        request_rate=ctx.request_rate_rps,
        thermal_pct=ctx.thermal_headroom_pct,
        utilization=util_lines,
    )


@dataclass(frozen=True)
class RuntimeDecision:
    """Parsed runtime decision from LLM."""

    decision: str
    reason: str
    parameters: dict[str, str]


def parse_response(response_text: str) -> RuntimeDecision | None:
    """Parse runtime decision response."""
    try:
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return RuntimeDecision(
                decision=data.get("decision", "keep"),
                reason=data.get("reason", ""),
                parameters=data.get("parameters", {}),
            )
    except (json.JSONDecodeError, ValueError):
        pass
    return None
