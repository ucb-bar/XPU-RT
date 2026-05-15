"""P3.1 — recognize_python_pattern primitive.

Given a Python module's source and its FX graph, the LLM proposes a
list of ``(subgraph_id, suggested_lift)`` pairs naming algorithmic
concepts (RoPE, RMSNorm, GLU, ...) it sees in the source. The
*suggestion* is non-binding; the recipe library's structural verifier
decides whether to accept the lift.

Hard rule: the LLM only proposes; it never authorises. The
deterministic fallback returns the empty list (no recognition), which
is the honest baseline — without an LLM the compiler relies on its
existing pattern table.
"""

from __future__ import annotations

from typing import Any

from compgen.llm.call_site import llm_call_site, register_fallback

RECOGNIZE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["proposals", "fallback_used"],
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["subgraph_id", "suggested_lift"],
                "properties": {
                    "subgraph_id": {"type": "string"},
                    "suggested_lift": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "additionalProperties": False,
            },
        },
        "fallback_used": {"type": "boolean"},
    },
    "additionalProperties": False,
}


@register_fallback("recognize_python_pattern_deterministic")
def _recognize_fallback(python_source: str, fx_graph_summary: dict[str, Any]) -> dict[str, Any]:
    """Empty-list fallback — honest 'no recognition without an LLM'."""

    return {"proposals": [], "fallback_used": True}


@llm_call_site(
    site_id="recognize_python_pattern",
    leverage="Lift Python source soup into named algorithmic concepts "
    "(RoPE, RMSNorm, GLU, …). LLM proposes; verifier decides.",
    inputs=["python_source:str", "fx_graph_summary:dict"],
    output_schema=RECOGNIZE_OUTPUT_SCHEMA,
    forbidden=["be_sole_correctness_decider"],
    fallback="recognize_python_pattern_deterministic",
)
def recognize_python_pattern(
    python_source: str, fx_graph_summary: dict[str, Any]
) -> dict[str, Any]:
    return _recognize_fallback(python_source, fx_graph_summary)


__all__ = ["RECOGNIZE_OUTPUT_SCHEMA", "recognize_python_pattern"]
