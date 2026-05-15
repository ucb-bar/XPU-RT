"""Prompt for EqSat segmentation hints."""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SegmentContext:
    """Context for segmentation hints prompt."""

    op_count: int
    op_types_summary: str
    dataflow_depth: int
    current_threshold: int
    region_summaries: list[str] = field(default_factory=list)


SEGMENT_PROMPT = textwrap.dedent("""\
    You are advising on segmentation for equality saturation.

    Segmentation divides a function into smaller segments for peephole optimization.
    Smaller segments = faster saturation but may miss cross-segment optimizations.

    ## Function structure
    - Total ops: {op_count}
    - Dataflow depth: {dataflow_depth}
    - Current segment threshold: {current_threshold} ops

    ## Op type distribution
    {op_types_summary}

    ## Current regions
    {region_summaries}

    ## Task
    Suggest segment configuration:
    1. An appropriate segment threshold (ops per segment)
    2. Forced boundary points (ops that should always start a new segment)

    Guidelines:
    - Threshold 50-100 for small functions, 200-500 for large ones
    - Force boundaries at device transitions, dtype changes, or between
      logically independent subgraphs (e.g., attention vs MLP)

    Respond as JSON:
    {{
        "threshold": int,
        "forced_boundaries": ["region_id", ...],
        "reasoning": "..."
    }}
""")

SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "threshold": {"type": "integer"},
        "forced_boundaries": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["threshold"],
}


def format_prompt(ctx: SegmentContext) -> str:
    """Format the segmentation hints prompt."""
    regions = "\n".join(f"  - {r}" for r in ctx.region_summaries) if ctx.region_summaries else "  Not available"
    return SEGMENT_PROMPT.format(
        op_count=ctx.op_count,
        dataflow_depth=ctx.dataflow_depth,
        current_threshold=ctx.current_threshold,
        op_types_summary=ctx.op_types_summary,
        region_summaries=regions,
    )


def parse_response(text: str) -> dict | None:
    """Parse segmentation hints response."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "threshold" in data:
            return data
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, dict) and "threshold" in data:
                    return data
            except json.JSONDecodeError:
                pass
    return None
