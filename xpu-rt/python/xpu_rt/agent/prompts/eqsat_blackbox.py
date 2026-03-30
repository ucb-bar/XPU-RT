"""Prompt for EqSat blackbox frontier decisions."""
from __future__ import annotations
import json
import re
import textwrap
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BlackboxContext:
    """Context for blackbox frontier prompt."""
    op_types_counts: dict[str, int]  # op_type -> count
    current_open: list[str]
    current_closed: list[str]
    target_name: str


BLACKBOX_PROMPT = textwrap.dedent("""\
    You are directing the equality saturation blackbox frontier.

    Blackbox ops are treated as opaque — rewrites cannot look inside them.
    Open ops are fully visible to rewrites.

    ## Op types in the e-graph
    {op_types}

    ## Currently open (visible to rewrites)
    {current_open}

    ## Currently closed (blackboxed)
    {current_closed}

    ## Target: {target_name}

    ## Task
    Decide which ops to open (make visible to rewrites) and which to close
    (blackbox). Opening too many ops causes combinatorial explosion.
    Closing important ops misses optimization opportunities.

    Guidelines:
    - Arithmetic ops (addi, muli, addf, mulf) should usually be open
    - Complex ops (conv, attention) are usually better blackboxed
    - Ops with many instances benefit most from being open

    Respond as JSON:
    {{
        "open": ["op_type", ...],
        "close": ["op_type", ...],
        "reasoning": "..."
    }}
""")

BLACKBOX_SCHEMA = {
    "type": "object",
    "properties": {
        "open": {"type": "array", "items": {"type": "string"}},
        "close": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["open", "close"],
}


def format_prompt(ctx: BlackboxContext) -> str:
    """Format the blackbox frontier prompt."""
    op_types = "\n".join(
        f"  {op}: {count} instances" for op, count in sorted(ctx.op_types_counts.items(), key=lambda x: -x[1])
    )
    return BLACKBOX_PROMPT.format(
        op_types=op_types,
        current_open=", ".join(ctx.current_open) if ctx.current_open else "none",
        current_closed=", ".join(ctx.current_closed) if ctx.current_closed else "none",
        target_name=ctx.target_name,
    )


def parse_response(text: str) -> dict | None:
    """Parse blackbox frontier response."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "open" in data:
            return data
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, dict) and "open" in data:
                    return data
            except json.JSONDecodeError:
                pass
    return None
