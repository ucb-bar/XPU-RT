"""One-pass walk over a recipe ModuleOp building the indices every
suggester needs: regions by role, payload-region map, adjacency.

Adjacency is approximated from sequential block order in the recipe
(the seed walks payload ops in order, so adjacent recipe regions
correspond to adjacent payload ops). This is good enough for the
producer-consumer fusion suggester; a precise SSA edge map would
require walking the payload module too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import ModuleOp, StringAttr


@dataclass
class RecipeIndex:
    regions: list[str] = field(default_factory=list)  # in declaration order
    role_by_region: dict[str, str] = field(default_factory=dict)  # sym -> role
    payload_by_region: dict[str, str] = field(default_factory=dict)  # sym -> payload_id
    regions_by_role: dict[str, list[str]] = field(default_factory=dict)
    # Pairs of (producer_sym, consumer_sym) where consumer is the next
    # region in declaration order with a known role. NOT a precise SSA
    # edge — a sequence proxy.
    adjacency: list[tuple[str, str]] = field(default_factory=list)


def build_recipe_index(recipe: ModuleOp) -> RecipeIndex:
    """Walk ``recipe.body.block.ops`` once; build every index."""
    idx = RecipeIndex()
    last_region: str | None = None
    for op in recipe.body.block.ops:
        if op.name != "recipe.region":
            continue
        sym = op.properties.get("sym_name")
        pid = op.properties.get("payload_region_id")
        role = op.properties.get("role")
        if not isinstance(sym, StringAttr):
            continue
        sym_str = sym.data
        idx.regions.append(sym_str)
        if isinstance(pid, StringAttr):
            idx.payload_by_region[sym_str] = pid.data
        if isinstance(role, StringAttr) and role.data:
            idx.role_by_region[sym_str] = role.data
            idx.regions_by_role.setdefault(role.data, []).append(sym_str)
        if last_region is not None:
            idx.adjacency.append((last_region, sym_str))
        last_region = sym_str
    return idx


def critical_path_recipe_syms(idx: RecipeIndex, dossier: Any) -> list[str]:
    """Translate the dossier's payload-id critical path back to recipe syms."""
    if dossier is None:
        return []
    cp_payload_ids = list(getattr(dossier, "critical_path", ()))
    if not cp_payload_ids:
        return []
    inv = {pid: sym for sym, pid in idx.payload_by_region.items()}
    return [inv[p] for p in cp_payload_ids if p in inv]


__all__ = [
    "RecipeIndex",
    "build_recipe_index",
    "critical_path_recipe_syms",
]
