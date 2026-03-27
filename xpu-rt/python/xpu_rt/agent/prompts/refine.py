"""Prompt for iterative refinement based on prior results."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class RefinementContext:
    """Context for refinement prompt."""

    iteration: int
    total_budget: int
    best_latency_us: float
    current_latency_us: float
    improvement_so_far_pct: float
    actions_tried: list[dict[str, str]]
    last_action_result: str
    remaining_bottlenecks: list[str]


REFINE_PROMPT = textwrap.dedent("""\
    You are iteratively optimizing a compiled model.

    ## Progress
    - Iteration: {iteration}/{total_budget}
    - Best latency: {best_latency_us:.1f} us
    - Current latency: {current_latency_us:.1f} us
    - Improvement so far: {improvement_pct:+.1f}%

    ## Actions tried:
    {actions_tried}

    ## Last result: {last_result}

    ## Remaining bottlenecks:
    {bottlenecks}

    ## Task
    What should we try next? Choose one action:
    1. "eqsat" — propose a new rewrite rule
    2. "tile" — change tile sizes for a region
    3. "fuse" — fuse two adjacent regions
    4. "assign_device" — move an op to a different device
    5. "generate_pass" — ask LLM to generate a new compiler pass
    6. "noop" — stop optimizing (no more improvements possible)

    Respond as JSON:
    {{"action_type": "...", "target_region": "...", "parameters": {{}}, "reasoning": "..."}}
""")


def format_prompt(ctx: RefinementContext) -> str:
    """Render refinement prompt."""
    actions = "\n".join(
        f"  Step {i+1}: {a.get('action_type', '?')} on {a.get('target', '?')} → {a.get('result', '?')}"
        for i, a in enumerate(ctx.actions_tried[-5:])
    ) or "  (none yet)"
    bottlenecks = "\n".join(f"  - {b}" for b in ctx.remaining_bottlenecks[:5]) or "  (none)"

    return REFINE_PROMPT.format(
        iteration=ctx.iteration,
        total_budget=ctx.total_budget,
        best_latency_us=ctx.best_latency_us,
        current_latency_us=ctx.current_latency_us,
        improvement_pct=ctx.improvement_so_far_pct,
        actions_tried=actions,
        last_result=ctx.last_action_result,
        bottlenecks=bottlenecks,
    )


@dataclass(frozen=True)
class RefinementAction:
    """Parsed refinement action from LLM."""

    action_type: str
    target_region: str
    parameters: dict[str, str]
    reasoning: str


def parse_response(response_text: str) -> RefinementAction | None:
    """Parse refinement response."""
    try:
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return RefinementAction(
                action_type=data.get("action_type", "noop"),
                target_region=data.get("target_region", ""),
                parameters=data.get("parameters", {}),
                reasoning=data.get("reasoning", ""),
            )
    except (json.JSONDecodeError, ValueError):
        pass
    return None
