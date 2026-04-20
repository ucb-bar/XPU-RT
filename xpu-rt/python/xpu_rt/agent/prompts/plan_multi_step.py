"""Prompt for multi-step optimization planning."""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlanContext:
    """Context for multi-step planning prompt."""

    observation_summary: str
    history_summary: str
    legal_actions_summary: str
    budget_remaining: int
    error_patterns: list[dict] = field(default_factory=list)


PLAN_PROMPT = textwrap.dedent("""\
    You are an expert ML compiler optimizer planning a multi-step optimization.

    ## Current state
    {observation_summary}

    ## History
    {history_summary}

    ## Budget remaining: {budget_remaining} iterations

    ## Known failure modes
    {error_patterns}

    ## Legal actions
    {legal_actions_summary}

    ## Task
    Plan a sequence of 3-5 optimization steps. For each step specify:
    1. action_type: one of "eqsat", "tile", "fuse", "assign_device", "generate_pass", "discover_ops", "request_verification", "noop"
    2. target: which ops/regions to target
    3. reason: why this step is needed
    4. depends_on_step: step index this depends on (-1 for independent)

    Return a JSON array of steps ordered by execution priority.
""")

PLAN_SCHEMA: dict[str, object] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "action_type": {"type": "string"},
            "target": {"type": "string"},
            "reason": {"type": "string"},
            "depends_on_step": {"type": "integer"},
        },
        "required": ["action_type", "target", "reason"],
    },
}


def format_prompt(ctx: PlanContext) -> str:
    """Format the multi-step planning prompt."""
    error_text = (
        "None"
        if not ctx.error_patterns
        else "\n".join(f"  - {p.get('action_type', '?')}: {p.get('failure_reason', '?')}" for p in ctx.error_patterns)
    )
    return PLAN_PROMPT.format(
        observation_summary=ctx.observation_summary,
        history_summary=ctx.history_summary,
        budget_remaining=ctx.budget_remaining,
        error_patterns=error_text,
        legal_actions_summary=ctx.legal_actions_summary,
    )


def parse_response(text: str) -> list[dict] | None:
    """Parse multi-step plan response."""
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
    return None
