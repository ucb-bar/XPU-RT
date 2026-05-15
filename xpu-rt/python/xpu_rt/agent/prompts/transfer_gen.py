"""Prompt for transfer analysis generation — LLM designs verified dataflow analyses."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class TransferGenContext:
    """Context for transfer analysis generation prompt.

    Attributes:
        region_id: The region to analyze.
        region_properties: Properties of the region (shapes, dtypes, ops).
        analysis_type: Type of analysis ("tile_divisibility",
            "local_mem_fit", "contiguous_layout").
        target_profile: Hardware target summary.
    """

    region_id: str
    region_properties: dict
    analysis_type: str
    target_profile: str


TRANSFER_GEN_PROMPT = textwrap.dedent("""\
    You are an expert in abstract interpretation and compiler analysis.
    Design a transfer function for the specified analysis that will be
    formally verified for soundness via Z3.

    ## Region
    ID: {region_id}
    Properties: {region_properties}

    ## Analysis Type: {analysis_type}

    ## Target Hardware
    {target_profile}

    ## Task
    Define the transfer function as four Python callables:

    1. **concrete_fn**: The concrete operation semantics
       ``(operands: list[z3.BitVec]) -> z3.BitVec``

    2. **transfer_fn**: The abstract transfer function
       ``(abstract_inputs: list[tuple[z3.BitVec, z3.BitVec]]) -> tuple[z3.BitVec, z3.BitVec]``

    3. **abstract_constraint**: Consistency check
       ``(concrete: z3.BitVec, abstract: tuple[z3.BitVec, z3.BitVec]) -> z3.BoolRef``

    4. **instance_constraint**: Abstract domain validity
       ``(abstract: tuple[z3.BitVec, z3.BitVec]) -> z3.BoolRef``

    Respond as a JSON with the code for each function.

    Example for known-bits OR:
    ```json
    {{
      "concrete_fn": "lambda ops: ops[0] | ops[1]",
      "transfer_fn": "lambda abs_ops: (abs_ops[0][0] & abs_ops[1][0], abs_ops[0][1] | abs_ops[1][1])",
      "abstract_constraint": "lambda c, ab: (c & ab[0]) == z3.BitVecVal(0, c.size()) and (c | ~ab[1]) == ~z3.BitVecVal(0, c.size())",
      "instance_constraint": "lambda ab: (ab[0] & ab[1]) == z3.BitVecVal(0, ab[0].size())",
      "num_operands": 2,
      "description": "Known-bits transfer for OR: zeros = lhs_zeros & rhs_zeros, ones = lhs_ones | rhs_ones"
    }}
    ```
""")


def format_prompt(ctx: TransferGenContext) -> str:
    """Render the transfer analysis generation prompt."""
    props = "\n".join(f"  {k}: {v}" for k, v in ctx.region_properties.items()) or "  (none)"

    return TRANSFER_GEN_PROMPT.format(
        region_id=ctx.region_id,
        region_properties=props,
        analysis_type=ctx.analysis_type,
        target_profile=ctx.target_profile or "(none specified)",
    )


@dataclass(frozen=True)
class TransferGenResult:
    """Parsed transfer function generation result."""

    concrete_fn_code: str
    transfer_fn_code: str
    abstract_constraint_code: str
    instance_constraint_code: str
    num_operands: int
    description: str


def parse_response(text: str) -> TransferGenResult | None:
    """Parse LLM response into a transfer generation result."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= 0:
            return None
        data = json.loads(text[start:end])
        return TransferGenResult(
            concrete_fn_code=data.get("concrete_fn", ""),
            transfer_fn_code=data.get("transfer_fn", ""),
            abstract_constraint_code=data.get("abstract_constraint", ""),
            instance_constraint_code=data.get("instance_constraint", ""),
            num_operands=data.get("num_operands", 2),
            description=data.get("description", ""),
        )
    except (json.JSONDecodeError, KeyError):
        return None


__all__ = ["TransferGenContext", "TransferGenResult", "format_prompt", "parse_response"]
