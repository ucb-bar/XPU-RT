"""Prompt for cross-module global optimization strategy."""
from __future__ import annotations
import json
import re
import textwrap
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GlobalStrategyContext:
    """Context for global strategy prompt."""
    module_count: int
    per_module_summaries: list[dict]  # [{name, op_count, flops, bottleneck}]
    target_name: str
    memory_budget_bytes: int = 0
    shared_patterns: list[str] = field(default_factory=list)


GLOBAL_STRATEGY_PROMPT = textwrap.dedent("""\
    You are an expert compiler engineer coordinating optimization across multiple modules.

    ## Modules ({module_count} total)
    {module_summaries}

    ## Target: {target_name}
    - Memory budget: {memory_budget} bytes

    ## Shared patterns
    {shared_patterns}

    ## Task
    Design a global optimization strategy. Consider:
    - Which modules should be optimized first (dependencies, impact)
    - Which optimizations can be shared across modules
    - Whether parallel or phased execution is better

    Respond as JSON:
    {{
        "strategy": "sequential" | "parallel" | "phased",
        "priority_order": ["module_name", ...],
        "shared_optimizations": ["optimization_description", ...],
        "reasoning": "..."
    }}
""")

GLOBAL_STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string", "enum": ["sequential", "parallel", "phased"]},
        "priority_order": {"type": "array", "items": {"type": "string"}},
        "shared_optimizations": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["strategy", "priority_order"],
}


def format_prompt(ctx: GlobalStrategyContext) -> str:
    """Format the global strategy prompt."""
    summaries = "\n".join(
        f"  - {m.get('name', '?')}: {m.get('op_count', 0)} ops, {m.get('flops', 0):,} FLOPs, bottleneck={m.get('bottleneck', 'unknown')}"
        for m in ctx.per_module_summaries
    )
    patterns = "\n".join(f"  - {p}" for p in ctx.shared_patterns) if ctx.shared_patterns else "  None detected"
    return GLOBAL_STRATEGY_PROMPT.format(
        module_count=ctx.module_count,
        module_summaries=summaries,
        target_name=ctx.target_name,
        memory_budget=ctx.memory_budget_bytes,
        shared_patterns=patterns,
    )


def parse_response(text: str) -> dict | None:
    """Parse global strategy response."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "strategy" in data:
            return data
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, dict) and "strategy" in data:
                    return data
            except json.JSONDecodeError:
                pass
    return None
