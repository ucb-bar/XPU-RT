"""Prompt for verification strategy — decide what to verify and at what level."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class VerifyStrategyContext:
    """Context for verification strategy prompt.

    Attributes:
        regions: List of dicts with region_id, op_type, transform_applied.
        verification_budget_ms: Total time budget for verification.
        verifiable_ops: Op types that have defined semantics (can be TV'd).
        past_failures: Recent TV failures [{region_id, counterexample_summary}].
    """

    regions: list[dict]
    verification_budget_ms: int
    verifiable_ops: list[str]
    past_failures: list[dict]


VERIFY_STRATEGY_PROMPT = textwrap.dedent("""\
    You are an expert compiler verification strategist. Decide which
    transformed regions should receive formal translation validation (TV)
    versus cheaper differential testing.

    ## Verification Budget
    Total time budget: {budget_ms}ms

    ## Regions to verify
    {regions}

    ## Ops with defined formal semantics (TV-eligible)
    {verifiable_ops}

    ## Recent TV failures
    {past_failures}

    ## Guidelines
    - TV is expensive (~5-30s per region) but sound. Use for high-risk transforms.
    - Differential testing is cheap (<1s) but incomplete.
    - Prioritize TV for: tiles, fuses, vectorizes, reassociations.
    - Skip TV for: device placement, copy insertion, runtime config.
    - If a region was TV-failed before, try TV again after repair.

    ## Task
    For each region, assign a verification level.

    Respond as JSON array:
    [
      {{"region_id": "...", "level": "tv"}},
      {{"region_id": "...", "level": "differential"}},
      {{"region_id": "...", "level": "both"}}
    ]
""")


def format_prompt(ctx: VerifyStrategyContext) -> str:
    """Render the verification strategy prompt."""
    region_lines = (
        "\n".join(
            f"  - {r['region_id']}: {r.get('op_type', '?')} → {r.get('transform_applied', '?')}"
            for r in ctx.regions[:10]
        )
        or "  (none)"
    )

    ops_line = ", ".join(ctx.verifiable_ops[:15]) or "(none)"

    failure_lines = (
        "\n".join(
            f"  - {f['region_id']}: {f.get('counterexample_summary', 'no details')}" for f in ctx.past_failures[:5]
        )
        or "  (none)"
    )

    return VERIFY_STRATEGY_PROMPT.format(
        budget_ms=ctx.verification_budget_ms,
        regions=region_lines,
        verifiable_ops=ops_line,
        past_failures=failure_lines,
    )


@dataclass(frozen=True)
class VerificationAssignment:
    """Verification level assigned to a region."""

    region_id: str
    level: str  # "tv", "differential", "both"


def parse_response(text: str) -> list[VerificationAssignment]:
    """Parse LLM response into verification assignments."""
    try:
        # Find JSON array in response
        start = text.find("[")
        end = text.rfind("]") + 1
        if start < 0 or end <= 0:
            return []
        data = json.loads(text[start:end])
        return [
            VerificationAssignment(
                region_id=item.get("region_id", ""),
                level=item.get("level", "differential"),
            )
            for item in data
            if isinstance(item, dict)
        ]
    except (json.JSONDecodeError, KeyError):
        return []


__all__ = [
    "VerificationAssignment",
    "VerifyStrategyContext",
    "format_prompt",
    "parse_response",
]
