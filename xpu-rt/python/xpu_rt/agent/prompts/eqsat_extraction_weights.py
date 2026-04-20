"""Prompt for EqSat extraction cost weight tuning."""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class WeightsContext:
    """Context for extraction weight tuning prompt."""

    egraph_summary: str
    target_description: str
    current_fusion_weight: float
    current_transfer_weight: float
    current_backend_match_weight: float
    objective: str = "latency"


EXTRACTION_WEIGHTS_PROMPT = textwrap.dedent("""\
    You are tuning extraction cost model weights for equality saturation.

    ## E-graph state
    {egraph_summary}

    ## Target
    {target_description}

    ## Objective: {objective}

    ## Current weights
    - fusion_weight: {fusion_weight:.3f} (higher = prefer fused subgraphs)
    - transfer_weight: {transfer_weight:.3f} (higher = penalize data movement)
    - backend_match_weight: {backend_match_weight:.3f} (higher = prefer ops with native backend)

    ## Task
    Adjust the extraction weights to better serve the optimization objective.
    Consider:
    - For latency: prioritize fusion and backend match
    - For memory: prioritize transfer cost reduction
    - For throughput: balance fusion with parallelism

    Respond as JSON:
    {{
        "fusion_weight": float,
        "transfer_weight": float,
        "backend_match_weight": float,
        "reasoning": "..."
    }}
""")

EXTRACTION_WEIGHTS_SCHEMA = {
    "type": "object",
    "properties": {
        "fusion_weight": {"type": "number"},
        "transfer_weight": {"type": "number"},
        "backend_match_weight": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["fusion_weight", "transfer_weight", "backend_match_weight"],
}


def format_prompt(ctx: WeightsContext) -> str:
    """Format the extraction weights prompt."""
    return EXTRACTION_WEIGHTS_PROMPT.format(
        egraph_summary=ctx.egraph_summary,
        target_description=ctx.target_description,
        objective=ctx.objective,
        fusion_weight=ctx.current_fusion_weight,
        transfer_weight=ctx.current_transfer_weight,
        backend_match_weight=ctx.current_backend_match_weight,
    )


def parse_response(text: str) -> dict | None:
    """Parse extraction weights response."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    if isinstance(data, dict) and "fusion_weight" in data:
        return {
            "fusion_weight": float(data.get("fusion_weight", 1.0)),
            "transfer_weight": float(data.get("transfer_weight", 1.0)),
            "backend_match_weight": float(data.get("backend_match_weight", 1.0)),
        }
    return None
