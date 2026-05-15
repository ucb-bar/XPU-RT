"""Prompt for LLM-guided solver configuration."""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class SolverConfigContext:
    """Context for solver configuration prompt."""

    num_regions: int
    num_devices: int
    problem_type: str  # "placement", "schedule", "both"
    estimated_complexity: str  # "simple", "moderate", "complex"
    current_timeout_ms: int = 10000
    memory_pressure_pct: float = 0.0
    has_multi_device: bool = False


SOLVER_CONFIG_PROMPT = textwrap.dedent("""\
    You are an expert in combinatorial optimization configuring a solver.

    ## Problem
    - Type: {problem_type}
    - Regions: {num_regions}
    - Devices: {num_devices}
    - Estimated complexity: {estimated_complexity}
    - Memory pressure: {memory_pressure_pct:.1f}%
    - Multi-device: {has_multi_device}
    - Current timeout: {current_timeout_ms}ms

    ## Task
    Configure the solver for this problem. Consider:
    - Small problems (<20 regions, 1 device) are typically easy
    - Multi-device placement with memory constraints is hard
    - High memory pressure needs tighter feasibility checking

    Respond as JSON:
    {{
        "timeout_ms": int,
        "predicted_hardness": "easy" | "medium" | "hard",
        "solver_hints": {{
            "symmetry_breaking": bool,
            "objective_gap_pct": float
        }},
        "reasoning": "..."
    }}
""")

SOLVER_CONFIG_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "timeout_ms": {"type": "integer"},
        "predicted_hardness": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "solver_hints": {
            "type": "object",
            "properties": {
                "symmetry_breaking": {"type": "boolean"},
                "objective_gap_pct": {"type": "number"},
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["timeout_ms", "predicted_hardness"],
}


def format_prompt(ctx: SolverConfigContext) -> str:
    """Format the solver configuration prompt."""
    return SOLVER_CONFIG_PROMPT.format(
        problem_type=ctx.problem_type,
        num_regions=ctx.num_regions,
        num_devices=ctx.num_devices,
        estimated_complexity=ctx.estimated_complexity,
        memory_pressure_pct=ctx.memory_pressure_pct,
        has_multi_device=ctx.has_multi_device,
        current_timeout_ms=ctx.current_timeout_ms,
    )


def parse_response(text: str) -> dict | None:
    """Parse solver configuration response."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "timeout_ms" in data:
            return data
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, dict) and "timeout_ms" in data:
                    return data
            except json.JSONDecodeError:
                pass
    return None
