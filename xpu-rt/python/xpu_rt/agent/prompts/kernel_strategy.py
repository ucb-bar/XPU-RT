"""Prompt for LLM-guided kernel strategy selection."""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class KernelStrategyContext:
    """Context for kernel strategy prompt."""

    op_name: str
    op_family: str
    flops: int
    input_shapes: str
    output_shapes: str
    dtype: str
    target_name: str
    has_gpu: bool
    available_strategies: list[str]
    autocomp_flop_threshold: int = 1000


KERNEL_STRATEGY_PROMPT = textwrap.dedent("""\
    You are an expert compiler engineer selecting kernel implementation strategies.

    ## Operation
    - Name: {op_name}
    - Family: {op_family}
    - FLOPs: {flops:,}
    - Input shapes: {input_shapes}
    - Output shapes: {output_shapes}
    - Dtype: {dtype}

    ## Target: {target_name}
    - GPU available: {has_gpu}
    - Default autocomp threshold: {autocomp_flop_threshold} FLOPs

    ## Available strategies
    {strategies}

    ## Strategy descriptions
    - native: Lowered natively by the compiler (best for elementwise/reshape ops)
    - library: Use vendor library (cuBLAS, cuDNN) — fast but inflexible
    - ukernel: Use registered micro-kernel — good for known patterns
    - autocomp: LLM-driven kernel search — best for compute-heavy ops worth optimizing
    - fallback: Generic correct implementation — safe but slow

    ## Task
    Select the best strategy for this operation. Consider:
    - Compute intensity vs overhead of search
    - Whether the op shape/type benefits from custom optimization
    - Target hardware capabilities

    Respond as JSON: {{"strategy": "...", "reason": "..."}}
""")

KERNEL_STRATEGY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string", "enum": ["native", "library", "ukernel", "autocomp", "fallback"]},
        "reason": {"type": "string"},
    },
    "required": ["strategy", "reason"],
}


def format_prompt(ctx: KernelStrategyContext) -> str:
    """Format the kernel strategy prompt."""
    strategies = "\n".join(f"  - {s}" for s in ctx.available_strategies)
    return KERNEL_STRATEGY_PROMPT.format(
        op_name=ctx.op_name,
        op_family=ctx.op_family,
        flops=ctx.flops,
        input_shapes=ctx.input_shapes,
        output_shapes=ctx.output_shapes,
        dtype=ctx.dtype,
        target_name=ctx.target_name,
        has_gpu=ctx.has_gpu,
        autocomp_flop_threshold=ctx.autocomp_flop_threshold,
        strategies=strategies,
    )


def parse_response(text: str) -> dict | None:
    """Parse kernel strategy response."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "strategy" in data:
            return data
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, dict) and "strategy" in data:
                    return data
            except json.JSONDecodeError:
                pass
    return None
