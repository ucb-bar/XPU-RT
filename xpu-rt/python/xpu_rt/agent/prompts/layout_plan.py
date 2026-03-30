"""Prompt for LLM-guided layout planning."""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutPlanContext:
    """Context for layout planning prompt."""

    op_name: str
    encoding_str: str
    target_name: str
    capabilities_summary: str
    tile_family_hint: str = ""
    current_inner_tiles: str = "[16, 16]"


LAYOUT_PLAN_PROMPT = textwrap.dedent("""\
    You are an expert compiler engineer planning data layouts for hardware targets.

    ## Operation: {op_name}
    - Current encoding: {encoding_str}
    - Tile family hint: {tile_family_hint}
    - Default inner tiles: {current_inner_tiles}

    ## Target: {target_name}
    {capabilities_summary}

    ## Task
    Suggest a pack specification for this operation. Consider:
    - Memory alignment requirements of the target
    - Tile sizes that match the compute unit (tensor cores need multiples of 16)
    - Whether prepacking is beneficial for this operation
    - Padding strategy (zero vs none)

    Respond as JSON:
    {{
        "inner_tiles": [int, int],
        "outer_perm": [int, int],
        "padding_value": "zero" | "none",
        "reasoning": "..."
    }}
""")

LAYOUT_PLAN_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "inner_tiles": {"type": "array", "items": {"type": "integer"}},
        "outer_perm": {"type": "array", "items": {"type": "integer"}},
        "padding_value": {"type": "string", "enum": ["zero", "none"]},
        "reasoning": {"type": "string"},
    },
    "required": ["inner_tiles", "reasoning"],
}


def format_prompt(ctx: LayoutPlanContext) -> str:
    """Format the layout planning prompt."""
    return LAYOUT_PLAN_PROMPT.format(
        op_name=ctx.op_name,
        encoding_str=ctx.encoding_str,
        tile_family_hint=ctx.tile_family_hint or "none",
        current_inner_tiles=ctx.current_inner_tiles,
        target_name=ctx.target_name,
        capabilities_summary=ctx.capabilities_summary,
    )


def parse_response(text: str) -> dict | None:
    """Parse layout plan response."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "inner_tiles" in data:
            return data
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, dict) and "inner_tiles" in data:
                    return data
            except json.JSONDecodeError:
                pass
    return None
