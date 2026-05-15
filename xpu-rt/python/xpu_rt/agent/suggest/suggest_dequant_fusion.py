"""Suggester for ``propose_dequant_fusion``.

Find regions whose role indicates a dequant op (``dequantize_per_*``,
``_weight_int4pack_*``, ``_weight_int8pack_*``) adjacent to a matmul.
"""

from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ModuleOp

from compgen.agent.suggest._candidate import ProposalCandidate
from compgen.agent.suggest._dispatch import register_suggester
from compgen.agent.suggest._recipe_index import build_recipe_index

_DEQUANT_ROLES = frozenset(
    {
        "dequantize_per_channel",
        "dequantize_per_tensor",
        "dequantize_per_group_along_last_dim",
        "_weight_int4pack_mm",
        "_weight_int8pack_mm",
    }
)
_MATMUL_ROLES = frozenset({"matmul", "mm", "addmm", "linear", "bmm"})


@register_suggester("propose_dequant_fusion")
def suggest_dequant_fusion(
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
        if prod_role in _DEQUANT_ROLES and cons_role in _MATMUL_ROLES:
            out.append(
                ProposalCandidate(
                    chosen={
                        "region_ref": prod,
                        "matmul_region_ref": cons,
                        "pattern": f"{prod_role}_into_{cons_role}",
                        "tolerance_hint": 3e-2,
                    },
                    rationale=f"Dequant-into-matmul: {prod} ({prod_role}) → {cons} ({cons_role})",
                    expected_impact=0.75,
                    target_feature_justification="quantized-weight matmul fusion",
                    metadata={"prod_role": prod_role, "cons_role": cons_role},
                )
            )
    return out[:k]


__all__ = ["suggest_dequant_fusion"]
