"""Prompt for model analysis — identify bottlenecks and optimization targets."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisContext:
    """Context for analysis prompt."""

    model_name: str
    op_count: int
    op_summary: dict[str, int]
    total_flops: int
    total_bytes: int
    num_devices: int
    device_names: list[str]
    bottleneck_ops: list[str]


ANALYSIS_PROMPT = textwrap.dedent("""\
    You are an expert ML compiler optimizer. Analyze this model and identify
    the top optimization opportunities.

    ## Model: {model_name}
    - Total ops: {op_count}
    - Total FLOPs: {total_flops:,}
    - Total bytes: {total_bytes:,}
    - Devices: {device_list}

    ## Op distribution:
    {op_summary}

    ## Current bottlenecks:
    {bottlenecks}

    ## Task
    Identify the top 3 optimization actions. For each, specify:
    1. action_type: one of "eqsat", "tile", "fuse", "assign_device", "generate_pass"
    2. target: which ops or regions
    3. reason: why this helps
    4. expected_improvement: estimated % improvement

    Respond as JSON array:
    [
      {{"action_type": "...", "target": "...", "reason": "...", "expected_improvement": 0.0}},
      ...
    ]
""")


def format_prompt(ctx: AnalysisContext) -> str:
    """Render the analysis prompt with model context."""
    op_lines = "\n".join(f"  {name}: {count}" for name, count in sorted(
        ctx.op_summary.items(), key=lambda x: -x[1]
    )[:15])
    bottleneck_lines = "\n".join(f"  - {op}" for op in ctx.bottleneck_ops[:5]) or "  (none identified)"

    return ANALYSIS_PROMPT.format(
        model_name=ctx.model_name,
        op_count=ctx.op_count,
        total_flops=ctx.total_flops,
        total_bytes=ctx.total_bytes,
        device_list=", ".join(ctx.device_names) or "cpu",
        op_summary=op_lines,
        bottlenecks=bottleneck_lines,
    )


@dataclass(frozen=True)
class ProposedOptimization:
    """A single optimization proposed by the LLM."""

    action_type: str
    target: str
    reason: str
    expected_improvement: float


def parse_response(response_text: str) -> list[ProposedOptimization]:
    """Parse LLM response into proposed optimizations."""
    try:
        # Try to extract JSON from the response
        text = response_text.strip()
        # Find JSON array in the text
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            items = json.loads(text[start:end])
            return [
                ProposedOptimization(
                    action_type=item.get("action_type", "noop"),
                    target=item.get("target", ""),
                    reason=item.get("reason", ""),
                    expected_improvement=float(item.get("expected_improvement", 0.0)),
                )
                for item in items
            ]
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return []
