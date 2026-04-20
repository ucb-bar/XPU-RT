"""Prompt for LLM-guided guard expression synthesis."""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardProposeContext:
    """Context for guard synthesis prompt."""

    variable_names: list[str]
    variable_types: dict[str, str]  # e.g. {"M": "int", "batch": "int"}
    positive_examples_summary: str
    negative_examples_summary: str
    num_positives: int
    num_negatives: int


GUARD_PROPOSE_PROMPT = textwrap.dedent("""\
    You are an expert in program verification synthesizing guard expressions.

    ## Variables
    {variables}

    ## Positive examples ({num_positives} total, guard should ACCEPT these):
    {positive_examples}

    ## Negative examples ({num_negatives} total, guard should REJECT these):
    {negative_examples}

    ## Task
    Propose guard expression fragments that:
    - Accept all positive examples
    - Reject all negative examples
    - Are as simple as possible

    Available expression types:
    - Comparison: var >= value, var <= value, var == value
    - Modulo: var % divisor == remainder
    - Boolean: variable (true/false check)

    Respond as JSON:
    {{
        "fragments": [
            {{"var": "M", "op": ">=", "value": 128}},
            {{"var": "N", "op": "%", "divisor": 16, "remainder": 0}},
            ...
        ],
        "reasoning": "..."
    }}
""")

GUARD_PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "fragments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "var": {"type": "string"},
                    "op": {"type": "string"},
                    "value": {"type": "number"},
                    "divisor": {"type": "integer"},
                    "remainder": {"type": "integer"},
                },
                "required": ["var", "op"],
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["fragments"],
}


def format_prompt(ctx: GuardProposeContext) -> str:
    """Format the guard proposal prompt."""
    variables = "\n".join(f"  - {name}: {ctx.variable_types.get(name, 'int')}" for name in ctx.variable_names)
    return GUARD_PROPOSE_PROMPT.format(
        variables=variables,
        num_positives=ctx.num_positives,
        num_negatives=ctx.num_negatives,
        positive_examples=ctx.positive_examples_summary,
        negative_examples=ctx.negative_examples_summary,
    )


def parse_response(text: str) -> list[dict] | None:
    """Parse guard proposal response, returning list of fragment dicts."""
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

    if isinstance(data, dict) and "fragments" in data:
        return data["fragments"]
    return None
