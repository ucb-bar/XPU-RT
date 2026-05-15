"""P3.6 — explain_counterexample primitive.

Turns a verifier counterexample + IR slice into prose + a structured
``suggested_edit`` that the Tactician can route back through
:func:`compgen.agent.primitives.rank_candidates.rank_candidates`.

Hard rule (the most important contract in this layer): the LLM is
NEVER the verifier. The ``suggested_edit`` must reference a
candidate id from the candidate set the Tactician already has —
the primitive never invents an edit out of thin air.

Deterministic fallback: emits typed prose summarising the verifier's
rejection reason and proposes the safest fallback edit (
``abandon_tactic``).
"""

from __future__ import annotations

from typing import Any

from compgen.llm.call_site import llm_call_site, register_fallback

EXPLAIN_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["prose", "suggested_edit", "fallback_used"],
    "properties": {
        "prose": {"type": "string", "minLength": 1},
        "suggested_edit": {
            "type": "object",
            "required": ["kind", "rationale"],
            "properties": {
                "kind": {
                    "enum": ["tactic_change", "param_change", "abandon_tactic"]
                },
                "candidate_id": {"type": ["string", "null"]},
                "rationale": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "fallback_used": {"type": "boolean"},
    },
    "additionalProperties": False,
}


@register_fallback("explain_counterexample_typed_summary")
def _explain_fallback(
    counterexample: dict[str, Any],
    ir_slice: dict[str, Any],
    refinement_spec: dict[str, Any],
) -> dict[str, Any]:
    likely_cause = counterexample.get("likely_cause") or "unspecified"
    return {
        "prose": (
            f"verifier rejected the proposal; likely_cause={likely_cause!r}. "
            f"No LLM was available to elaborate — falling back to abandon_tactic. "
            f"The Strategist should drop this rung of the fallback ladder."
        ),
        "suggested_edit": {
            "kind": "abandon_tactic",
            "candidate_id": None,
            "rationale": "safe deterministic fallback when no LLM is available",
        },
        "fallback_used": True,
    }


@llm_call_site(
    site_id="explain_counterexample",
    leverage="Turn a typed verifier counterexample into prose + a "
    "structured edit drawn from the existing candidate set.",
    inputs=[
        "counterexample:dict",
        "ir_slice:dict",
        "refinement_spec:dict",
    ],
    output_schema=EXPLAIN_OUTPUT_SCHEMA,
    forbidden=["be_sole_correctness_decider", "emit_certificate"],
    fallback="explain_counterexample_typed_summary",
)
def explain_counterexample(
    counterexample: dict[str, Any],
    ir_slice: dict[str, Any],
    refinement_spec: dict[str, Any],
) -> dict[str, Any]:
    return _explain_fallback(counterexample, ir_slice, refinement_spec)


__all__ = ["EXPLAIN_OUTPUT_SCHEMA", "explain_counterexample"]
