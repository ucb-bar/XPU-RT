"""P3.2 — name_cluster primitive.

Reads a region dossier and emits an algorithmic label + confidence +
closest known recipe id. The label is a *query* for the recipe
library, not authoritative. Deterministic fallback: ``("unknown",
0.0, None)``.
"""

from __future__ import annotations

from typing import Any

from compgen.llm.call_site import llm_call_site, register_fallback

NAME_CLUSTER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["label", "confidence", "fallback_used"],
    "properties": {
        "label": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "closest_known_recipe": {"type": ["string", "null"]},
        "fallback_used": {"type": "boolean"},
    },
    "additionalProperties": False,
}


@register_fallback("name_cluster_deterministic")
def _name_cluster_fallback(region_dossier: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": "unknown",
        "confidence": 0.0,
        "closest_known_recipe": None,
        "fallback_used": True,
    }


@llm_call_site(
    site_id="name_cluster",
    leverage="Tag a region with the algorithmic concept it implements "
    "(MHA-block, MoE-expert-routing, …). The label is a recipe-library query.",
    inputs=["region_dossier:dict"],
    output_schema=NAME_CLUSTER_OUTPUT_SCHEMA,
    forbidden=["be_sole_correctness_decider"],
    fallback="name_cluster_deterministic",
)
def name_cluster(region_dossier: dict[str, Any]) -> dict[str, Any]:
    return _name_cluster_fallback(region_dossier)


__all__ = ["NAME_CLUSTER_OUTPUT_SCHEMA", "name_cluster"]
