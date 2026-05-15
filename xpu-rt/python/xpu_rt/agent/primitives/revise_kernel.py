"""P3.4 — revise_kernel primitive.

Given a kernel contract, target envelope, prior attempt source, and
the typed counterexample the verifier produced, the LLM revises the
kernel toward a fix. Deterministic fallback: rotate through a fixed
sequence of tile shrinks.

The primary delegates to the fallback until the P2 typed-CEX feedback
loop is wired through Tactician.revise_kernel — at which point the
LLM consumes the Counterexample directly.
"""

from __future__ import annotations

from typing import Any

from compgen.llm.call_site import llm_call_site, register_fallback

# Deterministic shrink ladder: each entry halves the inner-most tile
# dim. The fallback walks the ladder one rung per revision.
_TILE_SHRINK_LADDER: tuple[int, ...] = (256, 128, 64, 32, 16)

REVISE_KERNEL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["next_kernel_source", "applied_tactic", "fallback_used"],
    "properties": {
        "next_kernel_source": {"type": "string"},
        "applied_tactic": {"type": "string"},
        "new_tile_size": {"type": ["integer", "null"], "minimum": 1},
        "fallback_used": {"type": "boolean"},
    },
    "additionalProperties": False,
}


@register_fallback("revise_kernel_shrink_tile")
def _revise_fallback(
    kernel_contract: dict[str, Any],
    target_envelope: dict[str, Any],
    prev_attempt: str,
    typed_failure: dict[str, Any],
) -> dict[str, Any]:
    """Walk the tile shrink ladder by one rung relative to the previous attempt.

    The previous attempt's tile size is read from ``typed_failure``'s
    ``ir_slice.annotation`` if present; otherwise we fall back to the
    largest tile in the ladder.
    """

    prev_tile: int | None = None
    ir_slice = typed_failure.get("ir_slice") or {}
    annotation = str(ir_slice.get("annotation") or "")
    for size in _TILE_SHRINK_LADDER:
        if f"tile={size}" in annotation or f"tile_size={size}" in annotation:
            prev_tile = size
            break

    if prev_tile is None:
        new_tile = _TILE_SHRINK_LADDER[0]
    else:
        idx = _TILE_SHRINK_LADDER.index(prev_tile)
        new_tile = (
            _TILE_SHRINK_LADDER[idx + 1]
            if idx + 1 < len(_TILE_SHRINK_LADDER)
            else _TILE_SHRINK_LADDER[-1]
        )

    return {
        "next_kernel_source": (
            f"# revised by revise_kernel_shrink_tile fallback\n"
            f"# tile_size={new_tile}\n{prev_attempt}"
        ),
        "applied_tactic": f"shrink_tile_to_{new_tile}",
        "new_tile_size": new_tile,
        "fallback_used": True,
    }


@llm_call_site(
    site_id="revise_kernel",
    leverage="Apply a single known fix per turn to a kernel that failed "
    "verification, guided by the typed counterexample.",
    inputs=[
        "kernel_contract:dict",
        "target_envelope:dict",
        "prev_attempt:str",
        "typed_failure:dict",
    ],
    output_schema=REVISE_KERNEL_OUTPUT_SCHEMA,
    forbidden=["invent_numerical_threshold", "be_sole_correctness_decider"],
    fallback="revise_kernel_shrink_tile",
)
def revise_kernel(
    kernel_contract: dict[str, Any],
    target_envelope: dict[str, Any],
    prev_attempt: str,
    typed_failure: dict[str, Any],
) -> dict[str, Any]:
    return _revise_fallback(kernel_contract, target_envelope, prev_attempt, typed_failure)


__all__ = ["REVISE_KERNEL_OUTPUT_SCHEMA", "revise_kernel"]
