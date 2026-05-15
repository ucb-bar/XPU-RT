"""Prompt for semantics generation — LLM generates op semantics definitions."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticsGenContext:
    """Context for semantics generation prompt.

    Attributes:
        op_type: The operation type to define semantics for.
        op_signature: Input/output type signature.
        op_description: Natural language description of the op.
        existing_semantics_examples: Code examples of existing semantics.
    """

    op_type: str
    op_signature: str
    op_description: str
    existing_semantics_examples: list[str]


SEMANTICS_GEN_PROMPT = textwrap.dedent("""\
    You are an expert in compiler semantics. Define the formal semantics
    for the following IR operation as a Python function that builds Z3
    bitvector expressions.

    ## Operation
    Type: {op_type}
    Signature: {op_signature}
    Description: {op_description}

    ## Format
    Write a Python function with this signature:

    ```python
    def lower_op(operands: list[z3.BitVecRef], width: int) -> z3.BitVecRef:
        \"\"\"Lower {op_type} to Z3 bitvector expression.

        Args:
            operands: Z3 bitvector expressions for each input.
            width: The bitwidth of the operation.

        Returns:
            Z3 bitvector expression for the result.
        \"\"\"
        import z3
        # ... your implementation ...
    ```

    ## Examples of existing semantics
    {examples}

    ## Guidelines
    - Use z3 bitvector operations (BitVecVal, +, -, *, UDiv, SRem, etc.)
    - Handle edge cases (division by zero → undefined behavior)
    - The function must be pure — no side effects
    - Return ONLY the Python function definition, no explanation

    Return ONLY the Python code block.
""")


def format_prompt(ctx: SemanticsGenContext) -> str:
    """Render the semantics generation prompt."""
    examples = "\n\n".join(f"```python\n{ex}\n```" for ex in ctx.existing_semantics_examples[:3]) or "(none available)"

    return SEMANTICS_GEN_PROMPT.format(
        op_type=ctx.op_type,
        op_signature=ctx.op_signature,
        op_description=ctx.op_description,
        examples=examples,
    )


def parse_response(text: str) -> str | None:
    """Extract Python code from LLM response.

    Returns the code string or None if extraction fails.
    """
    # Find code block
    markers = ["```python", "```py", "```"]
    for marker in markers:
        start = text.find(marker)
        if start >= 0:
            start += len(marker)
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

    # Fallback: look for def lower_op
    if "def lower_op" in text:
        start = text.find("def lower_op")
        return text[start:].strip()

    return None


__all__ = ["SemanticsGenContext", "format_prompt", "parse_response"]
