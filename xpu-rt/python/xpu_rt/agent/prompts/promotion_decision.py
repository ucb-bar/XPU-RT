"""Prompt for LLM-guided promotion decisions."""
from __future__ import annotations
import json
import re
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionContext:
    """Context for promotion decision prompt."""
    improvement_pct: float
    verification_summary: str
    target_name: str
    similar_promoted_count: int
    iterations_run: int
    best_latency_us: float
    initial_latency_us: float


PROMOTION_PROMPT = textwrap.dedent("""\
    You are an expert compiler engineer deciding whether to promote an optimized recipe.

    ## Optimization result
    - Improvement: {improvement_pct:+.1f}%
    - Initial latency: {initial_latency_us:.1f} us
    - Best latency: {best_latency_us:.1f} us
    - Iterations run: {iterations_run}

    ## Verification
    {verification_summary}

    ## Context
    - Target: {target_name}
    - Similar promoted recipes: {similar_promoted_count}

    ## Task
    Decide whether this recipe should be promoted to the library. Consider:
    - Is the improvement significant enough to justify promotion?
    - Did verification pass for all critical checks?
    - Is this target already well-served by existing recipes?

    Respond as JSON:
    {{
        "promote": bool,
        "confidence": float (0.0-1.0),
        "reason": "..."
    }}
""")

PROMOTION_SCHEMA = {
    "type": "object",
    "properties": {
        "promote": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["promote", "confidence", "reason"],
}


def format_prompt(ctx: PromotionContext) -> str:
    """Format the promotion decision prompt."""
    return PROMOTION_PROMPT.format(
        improvement_pct=ctx.improvement_pct,
        initial_latency_us=ctx.initial_latency_us,
        best_latency_us=ctx.best_latency_us,
        iterations_run=ctx.iterations_run,
        verification_summary=ctx.verification_summary,
        target_name=ctx.target_name,
        similar_promoted_count=ctx.similar_promoted_count,
    )


def parse_response(text: str) -> dict | None:
    """Parse promotion decision response."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "promote" in data:
            return data
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, dict) and "promote" in data:
                    return data
            except json.JSONDecodeError:
                pass
    return None
