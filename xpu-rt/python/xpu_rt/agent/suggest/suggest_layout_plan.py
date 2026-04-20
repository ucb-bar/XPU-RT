"""Suggester for ``propose_layout_plan``.

For each region with a matmul-family role, propose a target-aligned
layout. Reads the target's first compute unit's ``tile_mn`` to decide
``blocked_<M>x<N>`` vs ``row_major``.
"""

from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ModuleOp

from compgen.agent.suggest._candidate import ProposalCandidate
from compgen.agent.suggest._dispatch import register_suggester
from compgen.agent.suggest._recipe_index import build_recipe_index


_MATMUL_ROLES = frozenset({
    "matmul", "mm", "addmm", "linear", "bmm", "batch_matmul",
    "_weight_int4pack_mm", "_weight_int8pack_mm",
})


def _target_layout(target: Any) -> tuple[str, str]:
    """Return ``(layout_name, hardware_justification)``."""
    if target is None or not getattr(target, "devices", None):
        return "row_major", "no target devices declared"
    dev = target.devices[0]
    cus = getattr(dev, "compute_units", []) or []
    for cu in cus:
        tile = getattr(cu, "tile_mn", None)
        if tile and len(tile) >= 2:
            return (
                f"blocked_{tile[0]}x{tile[1]}",
                f"{getattr(dev, 'name', '?')}.{cu.name} tile={tile}",
            )
    return "row_major", f"{getattr(dev, 'name', 'device')} no tile_mn declared"


@register_suggester("propose_layout_plan")
def suggest_layout_plan(
    *,
    recipe: ModuleOp,
    dossier: Any,
    target: Any,
    k: int = 5,
) -> list[ProposalCandidate]:
    idx = build_recipe_index(recipe)
    layout, just = _target_layout(target)

    out: list[ProposalCandidate] = []
    matmul_regions = [s for s in idx.regions
                      if idx.role_by_region.get(s, "") in _MATMUL_ROLES]
    for sym in matmul_regions[:k]:
        role = idx.role_by_region.get(sym, "matmul")
        out.append(ProposalCandidate(
            chosen={"region_ref": sym, "layout": layout},
            rationale=f"Tile-aligned layout {layout} on {role} region {sym}",
            expected_impact=0.7 if "blocked" in layout else 0.4,
            target_feature_justification=just,
            metadata={"role": role, "layout": layout},
        ))
    return out


__all__ = ["suggest_layout_plan"]
