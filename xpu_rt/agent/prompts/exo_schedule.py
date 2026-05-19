"""LLM prompt templates for Exo schedule generation."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExoScheduleContext:
    """Context for Exo schedule generation prompts.

    Attributes:
        proc_source: Exo proc source code.
        target_kit_name: Name of the target hardware kit.
        target_hw_summary: Human-readable hardware description.
        available_schedule_ops: List of available Exo schedule operations.
        previous_attempts: History of previous schedule attempts.
    """

    proc_source: str
    target_kit_name: str
    target_hw_summary: str
    available_schedule_ops: list[str] = field(default_factory=list)
    previous_attempts: list[dict[str, Any]] = field(default_factory=list)


EXO_SCHEDULE_PROMPT = textwrap.dedent("""\
    You are an expert Exo schedule author. Given an Exo proc and a
    target hardware description, generate a scheduling script that
    optimizes the proc for the target.

    ## Exo Proc
    ```python
    {proc_source}
    ```

    ## Target
    {target_hw_summary}

    ## Available Schedule Operations
    {schedule_ops}

    ## Previous Attempts
    {previous_attempts}

    ## Instructions
    Write a Python scheduling script that applies Exo schedule
    operations to optimize the proc. Use cursor-based navigation
    and compose operations from the available list.

    Return ONLY the schedule code in a ```python``` code block.
""")


def format_prompt(ctx: ExoScheduleContext) -> str:
    """Format the Exo schedule generation prompt.

    Args:
        ctx: Schedule generation context.

    Returns:
        Formatted prompt string.
    """
    schedule_ops = "\n".join(f"- {op}" for op in ctx.available_schedule_ops)
    prev = ""
    if ctx.previous_attempts:
        for attempt in ctx.previous_attempts:
            prev += f"- Gen {attempt.get('generation', '?')}: latency={attempt.get('latency', '?')}us\n"
    else:
        prev = "None yet (first generation)"

    return EXO_SCHEDULE_PROMPT.format(
        proc_source=ctx.proc_source,
        target_hw_summary=ctx.target_hw_summary,
        schedule_ops=schedule_ops,
        previous_attempts=prev,
    )


def parse_response(text: str) -> list[str]:
    """Parse LLM response into schedule operation lines.

    Args:
        text: Raw LLM response text.

    Returns:
        List of schedule code lines extracted from code blocks.
    """
    lines = text.split("\n")
    in_code = False
    code_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
    return code_lines if code_lines else [text]


__all__ = ["ExoScheduleContext", "format_prompt", "parse_response"]
