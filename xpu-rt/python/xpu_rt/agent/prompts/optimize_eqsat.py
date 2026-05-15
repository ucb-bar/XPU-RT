"""Prompt for e-graph rewrite rule proposal."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class EqSatContext:
    """Context for eqsat optimization prompt."""

    egraph_summary: str
    target_name: str
    objective: str
    prior_rules: list[str]
    prior_improvements: list[float]


EQSAT_PROMPT = textwrap.dedent("""\
    You are an expert compiler optimizer using equality saturation.

    ## E-graph state
    {egraph_summary}

    ## Target: {target_name}
    ## Objective: {objective}

    ## Previously tried rules:
    {prior_rules}

    ## Task
    Propose a Python rewrite rule that adds an equivalent but cheaper
    alternative to an e-class. The rule must:
    1. Inherit from EqSatRewriteRule
    2. Have a unique `name` property
    3. Implement `match_and_add(self, module)` returning match count
    4. Use `add_alternative_to_eclass()` to add alternatives non-destructively

    Available imports:
    ```python
    from xdsl.dialects import arith, equivalence, linalg
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import OpResult
    from xdsl.rewriter import InsertPoint, Rewriter
    from compgen.eqsat.rules.python_rules import (
        EqSatRewriteRule, add_alternative_to_eclass, get_eclass_for_result,
    )
    ```

    Return ONLY the Python class definition.
""")


def format_prompt(ctx: EqSatContext) -> str:
    """Render the eqsat rule proposal prompt."""
    prior = (
        "\n".join(f"  - {name}: {imp:+.1f}%" for name, imp in zip(ctx.prior_rules, ctx.prior_improvements))
        or "  (none yet)"
    )
    return EQSAT_PROMPT.format(
        egraph_summary=ctx.egraph_summary,
        target_name=ctx.target_name,
        objective=ctx.objective,
        prior_rules=prior,
    )


def parse_response(response_text: str) -> str:
    """Extract Python code from LLM response."""
    text = response_text.strip()
    # Extract code block if present
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    # Try the whole text as code
    if "class " in text and "EqSatRewriteRule" in text:
        return text
    return ""
