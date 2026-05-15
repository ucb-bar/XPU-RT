"""P3.7 — compare_recipes primitive.

Library-curation helper: given two recipes and a target class, the
LLM judges whether they are duplicates, parametric variants, distinct
behaviours, or one dominates the other. Every proposal is gated by a
real differential run before the recipe library acts on it — the LLM
never *decides* the merge.

Deterministic fallback: ``relation=distinct`` (no merge proposed).
"""

from __future__ import annotations

from typing import Any, Final

from compgen.llm.call_site import llm_call_site, register_fallback

RECIPE_RELATIONS: Final[tuple[str, ...]] = (
    "duplicate",
    "parametric_variant",
    "distinct",
    "a_dominates_b",
    "b_dominates_a",
)

COMPARE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["relation", "fallback_used"],
    "properties": {
        "relation": {"enum": list(RECIPE_RELATIONS)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
        "fallback_used": {"type": "boolean"},
    },
    "additionalProperties": False,
}


@register_fallback("compare_recipes_distinct")
def _compare_fallback(
    recipe_a: dict[str, Any],
    recipe_b: dict[str, Any],
    target_class: str,
) -> dict[str, Any]:
    return {
        "relation": "distinct",
        "confidence": 0.0,
        "rationale": "deterministic fallback: no merge proposed without LLM",
        "fallback_used": True,
    }


@llm_call_site(
    site_id="compare_recipes",
    leverage="Judge the structural relation between two recipes so the "
    "library can deduplicate / mark parametric variants / drop obsoletes.",
    inputs=["recipe_a:dict", "recipe_b:dict", "target_class:str"],
    output_schema=COMPARE_OUTPUT_SCHEMA,
    forbidden=["be_sole_correctness_decider", "emit_certificate"],
    fallback="compare_recipes_distinct",
)
def compare_recipes(
    recipe_a: dict[str, Any],
    recipe_b: dict[str, Any],
    target_class: str,
) -> dict[str, Any]:
    return _compare_fallback(recipe_a, recipe_b, target_class)


__all__ = ["COMPARE_OUTPUT_SCHEMA", "RECIPE_RELATIONS", "compare_recipes"]
