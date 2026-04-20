"""Suggester for ``propose_buffer_lifetime_plan``.

Suggests a coloring policy based on the recipe's matmul + activation
chain. Uses 'first-fit' for small recipes, 'minimum-makespan' for large.
"""

from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ModuleOp

from compgen.agent.suggest._candidate import ProposalCandidate
from compgen.agent.suggest._dispatch import register_suggester
from compgen.agent.suggest._recipe_index import build_recipe_index


@register_suggester("propose_buffer_lifetime_plan")
def suggest_buffer_lifetime(
    *,
    recipe: ModuleOp,
    dossier: Any,
    target: Any,
    k: int = 5,
) -> list[ProposalCandidate]:
    idx = build_recipe_index(recipe)
    n = len(idx.regions)
    if n == 0:
        return []
    if n <= 12:
        policy = "first_fit"
        impact = 0.6
        why = "small recipe — first-fit colouring is near-optimal"
    elif n <= 64:
        policy = "minimum_makespan"
        impact = 0.7
        why = "medium recipe — makespan minimisation pays off"
    else:
        policy = "ilp_window_8"
        impact = 0.8
        why = "large recipe — windowed ILP keeps allocator runtime bounded"
    return [
        ProposalCandidate(
            chosen={"policy": policy, "window_size": 8, "region_count": n},
            rationale=why,
            expected_impact=impact,
            target_feature_justification="recipe-size heuristic",
            metadata={"region_count": n},
        )
    ]


__all__ = ["suggest_buffer_lifetime"]
