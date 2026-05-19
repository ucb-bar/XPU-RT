"""Prompt for counterexample repair — fix a transform after TV failure."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class CounterexampleRepairContext:
    """Context for counterexample repair prompt.

    Attributes:
        region_id: The region where the transform failed.
        transform_applied: Description of the transform that was applied.
        counterexample: Structured counterexample from Z3.
        verification_error: The verification status message.
        available_alternatives: Alternative action types the agent can try.
    """

    region_id: str
    transform_applied: str
    counterexample: dict
    verification_error: str
    available_alternatives: list[str]


COUNTEREXAMPLE_REPAIR_PROMPT = textwrap.dedent("""\
    You are an expert compiler optimizer. A transform you applied failed
    formal verification (translation validation). The SMT solver found a
    concrete counterexample — inputs where the transformed program produces
    different results from the original.

    ## Failed Transform
    Region: {region_id}
    Transform: {transform_applied}
    Verification result: {verification_error}

    ## Counterexample
    {counterexample}

    ## Available Alternative Actions
    {alternatives}

    ## Task
    Analyze WHY the transform is incorrect for the given counterexample,
    and propose a FIXED action that avoids the issue.

    Respond as JSON:
    {{
      "diagnosis": "brief explanation of why the transform failed",
      "action_type": "one of the available alternatives",
      "params": {{}},
      "reasoning": "why this alternative avoids the issue"
    }}

    If no safe alternative exists, respond with action_type "noop".
""")


def format_prompt(ctx: CounterexampleRepairContext) -> str:
    """Render the counterexample repair prompt."""
    cex_lines = []
    if ctx.counterexample.get("inputs"):
        cex_lines.append("  Inputs:")
        for k, v in list(ctx.counterexample["inputs"].items())[:5]:
            cex_lines.append(f"    {k} = {v}")
    if ctx.counterexample.get("expected"):
        cex_lines.append("  Expected outputs:")
        for k, v in list(ctx.counterexample["expected"].items())[:3]:
            cex_lines.append(f"    {k} = {v}")
    if ctx.counterexample.get("actual"):
        cex_lines.append("  Actual outputs (from transformed):")
        for k, v in list(ctx.counterexample["actual"].items())[:3]:
            cex_lines.append(f"    {k} = {v}")
    if ctx.counterexample.get("summary"):
        cex_lines.append(f"  Summary: {ctx.counterexample['summary']}")

    cex_text = "\n".join(cex_lines) or "  (no counterexample details)"
    alt_text = "\n".join(f"  - {a}" for a in ctx.available_alternatives) or "  (none)"

    return COUNTEREXAMPLE_REPAIR_PROMPT.format(
        region_id=ctx.region_id,
        transform_applied=ctx.transform_applied,
        verification_error=ctx.verification_error,
        counterexample=cex_text,
        alternatives=alt_text,
    )


@dataclass(frozen=True)
class RepairProposal:
    """A proposed repair from the LLM."""

    diagnosis: str
    action_type: str
    params: dict
    reasoning: str


def parse_response(text: str) -> RepairProposal | None:
    """Parse LLM response into a repair proposal."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= 0:
            return None
        data = json.loads(text[start:end])
        return RepairProposal(
            diagnosis=data.get("diagnosis", ""),
            action_type=data.get("action_type", "noop"),
            params=data.get("params", {}),
            reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, KeyError):
        return None


__all__ = [
    "CounterexampleRepairContext",
    "RepairProposal",
    "format_prompt",
    "parse_response",
]
