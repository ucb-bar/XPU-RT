"""Suggester for ``propose_peephole_pattern``.

Finds adjacent role pairs that match well-known peephole rewrites
(double-transpose elimination, view→view collapse, neg→add fold).
"""

from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ModuleOp

from compgen.agent.suggest._candidate import ProposalCandidate
from compgen.agent.suggest._dispatch import register_suggester
from compgen.agent.suggest._recipe_index import build_recipe_index

_PEEPHOLE_PATTERNS: dict[tuple[str, str], tuple[str, str, float]] = {
    ("transpose", "transpose"): (
        "double_transpose_elim",
        "two transposes cancel out",
        0.9,
    ),
    ("permute", "permute"): (
        "double_permute_elim",
        "two permutes cancel out",
        0.85,
    ),
    ("view", "view"): ("view_collapse", "consecutive views collapse", 0.7),
    ("expand", "view"): ("expand_view_canon", "expand+view canonicalisation", 0.5),
    ("neg", "neg"): ("double_neg_elim", "two negations cancel", 0.9),
}


@register_suggester("propose_peephole_pattern")
def suggest_peephole_pattern(
    *,
    recipe: ModuleOp,
    dossier: Any,
    target: Any,
    k: int = 5,
) -> list[ProposalCandidate]:
    idx = build_recipe_index(recipe)
    out: list[ProposalCandidate] = []
    for prod, cons in idx.adjacency:
        prod_role = idx.role_by_region.get(prod, "")
        cons_role = idx.role_by_region.get(cons, "")
        match = _PEEPHOLE_PATTERNS.get((prod_role, cons_role))
        if match is None:
            continue
        pattern_class, rationale, score = match
        out.append(
            ProposalCandidate(
                chosen={
                    "region_ref": prod,
                    "pattern_class": pattern_class,
                    "consumed_region_ref": cons,
                },
                rationale=f"{rationale} ({prod} → {cons})",
                expected_impact=score,
                target_feature_justification="structural rewrite, target-agnostic",
                metadata={"pair": [prod_role, cons_role]},
            )
        )
    out.sort(key=lambda c: c.expected_impact, reverse=True)
    return out[:k]


__all__ = ["suggest_peephole_pattern"]
