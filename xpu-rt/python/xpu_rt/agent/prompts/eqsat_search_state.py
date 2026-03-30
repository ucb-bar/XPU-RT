"""Prompt for EqSat search state consultation."""
from __future__ import annotations
import json
import re
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchStateContext:
    """Context for eqsat search state prompt."""
    egraph_summary: str
    rule_stats: dict[str, int]  # rule_name -> match_count
    best_cost: float
    iteration: int
    total_eclasses: int = 0
    total_enodes: int = 0


SEARCH_STATE_PROMPT = textwrap.dedent("""\
    You are directing an equality saturation search.

    ## E-graph state
    {egraph_summary}

    ## Statistics
    - E-classes: {total_eclasses}
    - E-nodes: {total_enodes}
    - Best cost: {best_cost:.1f}
    - Iteration: {iteration}

    ## Rule application statistics
    {rule_stats}

    ## Task
    What should the equality saturation engine do next? Choose one:
    1. PROPOSE_RULE — generate a new rewrite rule (rules are stale/insufficient)
    2. CHANGE_BLACKBOX — open/close ops for optimization (wrong granularity)
    3. ADJUST_SEGMENTS — change segment boundaries (segments too large/small)
    4. CHANGE_WEIGHTS — adjust extraction cost model weights (wrong trade-offs)
    5. CONTINUE — keep running current rules (making progress)
    6. STOP — saturation reached or diminishing returns

    Respond as JSON:
    {{
        "action": "PROPOSE_RULE" | "CHANGE_BLACKBOX" | "ADJUST_SEGMENTS" | "CHANGE_WEIGHTS" | "CONTINUE" | "STOP",
        "parameters": {{}},
        "reasoning": "..."
    }}
""")

SEARCH_STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["PROPOSE_RULE", "CHANGE_BLACKBOX", "ADJUST_SEGMENTS", "CHANGE_WEIGHTS", "CONTINUE", "STOP"]},
        "parameters": {"type": "object"},
        "reasoning": {"type": "string"},
    },
    "required": ["action", "reasoning"],
}


def format_prompt(ctx: SearchStateContext) -> str:
    """Format the search state prompt."""
    stats = "\n".join(
        f"  {name}: {count} matches" for name, count in sorted(ctx.rule_stats.items(), key=lambda x: -x[1])
    ) if ctx.rule_stats else "  No rules applied yet"
    return SEARCH_STATE_PROMPT.format(
        egraph_summary=ctx.egraph_summary,
        total_eclasses=ctx.total_eclasses,
        total_enodes=ctx.total_enodes,
        best_cost=ctx.best_cost,
        iteration=ctx.iteration,
        rule_stats=stats,
    )


def parse_response(text: str) -> dict | None:
    """Parse search state response."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action" in data:
            return data
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, dict) and "action" in data:
                    return data
            except json.JSONDecodeError:
                pass
    return None
