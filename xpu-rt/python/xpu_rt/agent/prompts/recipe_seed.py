"""Prompt for LLM-guided Recipe IR seed personalization."""
from __future__ import annotations
import json
import re
import textwrap
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RecipeSeedContext:
    """Context for recipe seed personalization prompt."""
    op_histogram: dict[str, int]
    target_name: str
    objective: str
    total_flops: int
    total_bytes: int
    num_devices: int
    prior_success_patterns: list[str] = field(default_factory=list)


RECIPE_SEED_PROMPT = textwrap.dedent("""\
    You are an expert ML compiler optimizer personalizing a compilation recipe.

    ## Model analysis
    - Total FLOPs: {total_flops:,}
    - Total bytes: {total_bytes:,}
    - Devices: {num_devices}
    - Objective: {objective}

    ## Op distribution
    {op_histogram}

    ## Target: {target_name}

    ## Prior successful patterns
    {prior_patterns}

    ## Task
    Personalize the seed recipe for this model. Decide:
    1. Which ops to prioritize for optimization (highest impact first)
    2. Default tile sizes per op family
    3. Whether to enable aggressive fusion (fuse adjacent ops)
    4. Which ops to skip (too small to benefit from optimization)

    Respond as JSON:
    {{
        "prioritize_ops": ["op_name", ...],
        "skip_ops": ["op_name", ...],
        "default_tile_sizes": {{"matmul": [128, 64, 32], "conv": [64, 64]}},
        "aggressive_fusion": bool,
        "reasoning": "..."
    }}
""")

RECIPE_SEED_SCHEMA = {
    "type": "object",
    "properties": {
        "prioritize_ops": {"type": "array", "items": {"type": "string"}},
        "skip_ops": {"type": "array", "items": {"type": "string"}},
        "default_tile_sizes": {"type": "object"},
        "aggressive_fusion": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["prioritize_ops", "aggressive_fusion"],
}


def format_prompt(ctx: RecipeSeedContext) -> str:
    """Format the recipe seed prompt."""
    hist = "\n".join(f"  {k}: {v}" for k, v in sorted(ctx.op_histogram.items(), key=lambda x: -x[1]))
    patterns = "\n".join(f"  - {p}" for p in ctx.prior_success_patterns) if ctx.prior_success_patterns else "  None"
    return RECIPE_SEED_PROMPT.format(
        total_flops=ctx.total_flops,
        total_bytes=ctx.total_bytes,
        num_devices=ctx.num_devices,
        objective=ctx.objective,
        op_histogram=hist,
        target_name=ctx.target_name,
        prior_patterns=patterns,
    )


def parse_response(text: str) -> dict | None:
    """Parse recipe seed response."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "prioritize_ops" in data:
            return data
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, dict) and "prioritize_ops" in data:
                    return data
            except json.JSONDecodeError:
                pass
    return None
