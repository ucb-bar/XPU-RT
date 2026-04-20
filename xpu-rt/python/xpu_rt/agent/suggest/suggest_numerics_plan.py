"""Suggester for ``propose_numerics_plan``.

Heuristic: for each compute-heavy region (matmul/softmax), suggest a
dtype plan using the target's preferred dtypes. Falls back to fp32.
"""

from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ModuleOp

from compgen.agent.suggest._candidate import ProposalCandidate
from compgen.agent.suggest._dispatch import register_suggester
from compgen.agent.suggest._recipe_index import build_recipe_index


_COMPUTE_ROLES = frozenset({
    "matmul", "mm", "addmm", "linear", "bmm", "batch_matmul", "softmax",
})


def _preferred_dtype(target: Any) -> str:
    if target is None or not getattr(target, "devices", None):
        return "f32"
    dev = target.devices[0]
    for cu in getattr(dev, "compute_units", []) or []:
        prefs = getattr(cu, "preferred_dtypes", None) or []
        if prefs:
            # Use the FIRST preferred dtype that's a real fp / int type.
            for p in prefs:
                p = str(p)
                if any(t in p for t in ("float", "int", "bf16", "f16", "f32")):
                    return p
    return "f32"


@register_suggester("propose_numerics_plan")
def suggest_numerics_plan(
    *,
    recipe: ModuleOp,
    dossier: Any,
    target: Any,
    k: int = 5,
) -> list[ProposalCandidate]:
    idx = build_recipe_index(recipe)
    dtype = _preferred_dtype(target)
    out: list[ProposalCandidate] = []
    for sym in idx.regions:
        role = idx.role_by_region.get(sym, "")
        if role not in _COMPUTE_ROLES:
            continue
        out.append(ProposalCandidate(
            chosen={
                "region_ref": sym, "compute_dtype": dtype,
                "accumulator_dtype": "f32",
            },
            rationale=f"{role} on {sym} → {dtype} compute, f32 accumulator",
            expected_impact=0.6,
            target_feature_justification=f"target preferred dtype: {dtype}",
            metadata={"role": role, "compute_dtype": dtype},
        ))
        if len(out) >= k:
            break
    return out


__all__ = ["suggest_numerics_plan"]
