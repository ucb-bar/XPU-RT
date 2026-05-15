"""Pattern-recognition pass tool: matmul → bias_add → relu.

Receives a region summary (typically derived from a payload-IR
analysis snapshot) and emits a typed
:class:`PassToolResult` proposing the fusion via
``FuseElementwise`` + ``SetAccumulator`` Recipe-IR ops. The
verifier downstream decides whether to apply the delta.

The pass tool reads only the kwargs declared on its card and
returns the result; it never opens a file, never edits Payload IR,
and never bypasses the verifier.
"""

from __future__ import annotations

from typing import Any

from compgen.pass_tools.pass_tool_result import (
    PassToolResult,
    make_no_op,
    make_proposal,
)

TOOL_ID = "fuse_matmul_bias_relu"


def _matches_pattern(ops: tuple[str, ...]) -> bool:
    """Pure pattern check — no IR mutation."""

    if len(ops) < 3:
        return False
    return ops[:3] == ("matmul", "bias_add", "relu")


def run(
    *,
    region_id: str,
    ops: tuple[str, ...] | list[str],
    single_consumer: bool = True,
    **_: Any,
) -> PassToolResult:
    """Propose a fused-elementwise Recipe-IR delta when the
    matmul→bias_add→relu pattern matches and the producer has a
    single consumer."""

    ops_tuple = tuple(ops)
    if not _matches_pattern(ops_tuple):
        return make_no_op(
            tool_id=TOOL_ID,
            refinement_claim="tolerance_eps",
            detail=f"region {region_id} ops {ops_tuple} do not match the pattern",
        )
    if not single_consumer:
        return make_no_op(
            tool_id=TOOL_ID,
            refinement_claim="tolerance_eps",
            detail=f"region {region_id} matmul has multiple consumers",
        )

    delta = [
        {
            "op": "FuseElementwise",
            "region": region_id,
            "ops": ["matmul_0", "bias_add_0", "relu_0"],
        },
        {
            "op": "SetAccumulator",
            "region": region_id,
            "accumulator_dtype": "fp32",
        },
    ]
    return make_proposal(
        tool_id=TOOL_ID,
        recipe_delta=delta,
        refinement_claim="tolerance_eps",
        evidence={
            "matched_pattern": "matmul_bias_relu",
            "single_consumer": single_consumer,
        },
    )
