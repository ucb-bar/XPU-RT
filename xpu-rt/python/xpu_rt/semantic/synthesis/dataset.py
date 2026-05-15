"""Dataset helpers for synthesized guard search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.ir import Operation

from compgen.semantic.synthesis.facts import RecipeFactIndex, build_candidate_env
from compgen.semantic.synthesis.specs import GuardFamilySpec, get_family_spec


@dataclass(frozen=True)
class SynthesisExample:
    """Single training/evaluation example for guard search."""

    transform_family: str
    env: dict[str, Any]
    safe: bool
    profitable: bool
    candidate_symbol: str = ""
    failure_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def example_from_candidate(
    op: Operation,
    fact_index: RecipeFactIndex,
    family_spec: GuardFamilySpec,
) -> SynthesisExample | None:
    """Convert a candidate op into a synthesis example for one family."""

    if not family_spec.matches_candidate(op):
        return None
    env = build_candidate_env(op, fact_index)
    safe, profitable = family_spec.label(env)
    candidate_symbol = ""
    if hasattr(op, "sym_name") and getattr(op, "sym_name") is not None:
        candidate_symbol = getattr(op, "sym_name").data
    return SynthesisExample(
        transform_family=family_spec.family,
        env=env,
        safe=safe,
        profitable=profitable,
        candidate_symbol=candidate_symbol,
        failure_reason="" if safe else "observed_unsafe",
    )


def build_examples_for_family(
    module: Any,
    fact_index: RecipeFactIndex,
    family: str,
) -> list[SynthesisExample]:
    """Build synthesis examples for one family from a Recipe module."""

    family_spec = get_family_spec(family)
    examples: list[SynthesisExample] = []
    for op in module.walk():
        example = example_from_candidate(op, fact_index, family_spec)
        if example is not None:
            examples.append(example)
    return examples


__all__ = ["SynthesisExample", "build_examples_for_family", "example_from_candidate"]
