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
    graph_break_count: int = 0
    guard_count: int = 0
    unsupported_ops: list[str] | None = None
    repeated_patterns: dict[str, int] | None = None
    critical_path: list[str] | None = None
    backend_viability: list[str] | None = None
    analysis_summary: str = ""
    legal_actions_summary: str = ""


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

    ## Frontend diagnostics:
    - Graph breaks: {graph_break_count}
    - Guards: {guard_count}
    - Unsupported ops: {unsupported_ops}

    ## Current bottlenecks:
    {bottlenecks}

    ## Repeated patterns:
    {repeated_patterns}

    ## Critical path:
    {critical_path}

    ## Backend viability:
    {backend_viability}

    ## Detailed analysis:
    {analysis_summary}

    ## Legal actions:
    {legal_actions}

    ## Task
    Identify the top 3 optimization actions. For each, specify:
    1. action_type: one of "eqsat", "tile", "fuse", "assign_device", "generate_pass",
       "discover_ops", "request_verification"
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
    op_lines = "\n".join(
        f"  {name}: {count}" for name, count in sorted(ctx.op_summary.items(), key=lambda x: -x[1])[:15]
    )
    bottleneck_lines = "\n".join(f"  - {op}" for op in ctx.bottleneck_ops[:5]) or "  (none identified)"
    repeated_lines = (
        "\n".join(
            f"  - {name}: {count}"
            for name, count in sorted((ctx.repeated_patterns or {}).items(), key=lambda x: (-x[1], x[0]))[:8]
        )
        or "  (none)"
    )
    critical_path_lines = "\n".join(f"  - {item}" for item in (ctx.critical_path or [])[:8]) or "  (none)"
    backend_lines = "\n".join(f"  - {item}" for item in (ctx.backend_viability or [])[:8]) or "  (none)"
    unsupported = ", ".join((ctx.unsupported_ops or [])[:10]) or "(none)"

    return ANALYSIS_PROMPT.format(
        model_name=ctx.model_name,
        op_count=ctx.op_count,
        total_flops=ctx.total_flops,
        total_bytes=ctx.total_bytes,
        device_list=", ".join(ctx.device_names) or "cpu",
        op_summary=op_lines,
        graph_break_count=ctx.graph_break_count,
        guard_count=ctx.guard_count,
        unsupported_ops=unsupported,
        bottlenecks=bottleneck_lines,
        repeated_patterns=repeated_lines,
        critical_path=critical_path_lines,
        backend_viability=backend_lines,
        analysis_summary=ctx.analysis_summary or "  (none)",
        legal_actions=ctx.legal_actions_summary or "  (none)",
    )


@dataclass(frozen=True)
class ProposedOptimization:
    """A single optimization proposed by the LLM."""

    action_type: str
    target: str
    reason: str
    expected_improvement: float


ANALYSIS_SCHEMA: dict[str, object] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "action_type": {"type": "string"},
            "target": {"type": "string"},
            "reason": {"type": "string"},
            "expected_improvement": {"type": "number"},
        },
        "required": ["action_type", "target", "reason", "expected_improvement"],
        "additionalProperties": False,
    },
}


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
