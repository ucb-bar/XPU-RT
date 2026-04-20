"""Suggester for ``propose_rematerialization_plan``.

Targets cheap-to-recompute regions (norms, views, casts) for remat
when the recipe is large enough to suggest memory pressure.
"""

from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ModuleOp

from compgen.agent.suggest._candidate import ProposalCandidate
from compgen.agent.suggest._dispatch import register_suggester
from compgen.agent.suggest._recipe_index import build_recipe_index

_CHEAP_ROLES = frozenset(
    {
        "view",
        "permute",
        "transpose",
        "expand",
        "unsqueeze",
        "neg",
        "cat",
        "rsqrt",
        "rmsnorm",
        "layer_norm",
    }
)


@register_suggester("propose_rematerialization_plan")
def suggest_rematerialization(
    *,
    recipe: ModuleOp,
    dossier: Any,
    target: Any,
    k: int = 5,
) -> list[ProposalCandidate]:
    idx = build_recipe_index(recipe)
    cheap = [s for s in idx.regions if idx.role_by_region.get(s, "") in _CHEAP_ROLES]
    if not cheap:
        return []
    return [
        ProposalCandidate(
            chosen={
                "remat_region_refs": cheap[:16],
                "recompute_cost_tolerance": 0.05,
            },
            rationale=(f"Remat {len(cheap)} cheap-to-recompute regions (view/permute/norm) to reduce live-set memory"),
            expected_impact=0.55,
            target_feature_justification="cheap-recompute heuristic",
            metadata={"candidate_count": len(cheap)},
        )
    ]


__all__ = ["suggest_rematerialization"]
