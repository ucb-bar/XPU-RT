"""Suggester for the ``propose_fusion`` invent slot.

Strategy: walk the recipe's adjacency list; emit one candidate per
adjacent (producer, consumer) pair whose roles are compatible for
producer-consumer fusion. Score each by:
    + bonus if both regions are on the dossier's critical path
    + bonus if the role pair is one of the canonical ones
      (norm→matmul, matmul→activation, silu→mul, etc.)
"""

from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ModuleOp

from compgen.agent.suggest._candidate import ProposalCandidate
from compgen.agent.suggest._dispatch import register_suggester
from compgen.agent.suggest._recipe_index import (
    build_recipe_index,
    critical_path_recipe_syms,
)

# Coarse role-compatibility table for producer-consumer fusion.
_FUSION_ROLE_PAIRS: set[tuple[str, str]] = set()


def _canonical_pairs() -> set[tuple[str, str]]:
    if _FUSION_ROLE_PAIRS:
        return _FUSION_ROLE_PAIRS
    pairs = [
        # Norm → projection
        ("rsqrt", "mul"),
        ("rsqrt", "matmul"),
        ("rmsnorm", "matmul"),
        ("layer_norm", "matmul"),
        ("native_layer_norm", "matmul"),
        # Linear → activation (any of)
        ("matmul", "silu"),
        ("matmul", "gelu"),
        ("matmul", "sigmoid"),
        ("matmul", "tanh"),
        ("matmul", "relu"),
        ("matmul", "neg"),
        ("matmul", "mul"),
        ("matmul", "add"),
        ("mm", "silu"),
        ("mm", "gelu"),
        ("mm", "sigmoid"),
        ("addmm", "silu"),
        ("addmm", "gelu"),
        ("addmm", "sigmoid"),
        # Activation pairings (SwiGLU: silu(gate) * up)
        ("silu", "mul"),
        ("sigmoid", "mul"),
        ("gelu", "mul"),
        # Elementwise chains
        ("mul", "add"),
        ("mul", "mul"),
        ("add", "mul"),
        ("sub", "mul"),
        # Layout-prep next to compute
        ("view", "matmul"),
        ("permute", "matmul"),
        ("transpose", "matmul"),
        # Softmax + value-projection
        ("softmax", "matmul"),
        ("softmax", "bmm"),
        ("softmax", "batch_matmul"),
    ]
    _FUSION_ROLE_PAIRS.update(pairs)
    return _FUSION_ROLE_PAIRS


def _score(prod_role: str, cons_role: str, in_critical: bool) -> float:
    base = 0.4
    if (prod_role, cons_role) in _canonical_pairs():
        base += 0.4
    if in_critical:
        base += 0.2
    return min(base, 1.0)


@register_suggester("propose_fusion")
def suggest_fusion(
    *,
    recipe: ModuleOp,
    dossier: Any,
    target: Any,
    k: int = 5,
    lookahead: int = 4,
) -> list[ProposalCandidate]:
    """Walk regions; for each role-bearing region look ahead up to
    ``lookahead`` neighbours for a role-compatible consumer.

    Pure-adjacency (lookahead=1) misses pairs separated by structural
    ops the seed picks up (yield / empty / fact ops between
    transpose and matmul). The lookahead window keeps us robust to
    that without needing the full SSA edge graph.
    """
    idx = build_recipe_index(recipe)
    crit = set(critical_path_recipe_syms(idx, dossier))
    canonical = _canonical_pairs()
    already = _existing_grouped_pairs(recipe)

    role_regions = [
        (i, sym, idx.role_by_region.get(sym, "")) for i, sym in enumerate(idx.regions) if idx.role_by_region.get(sym)
    ]

    # Group instances by (prod_role, cons_role). Multiple occurrences
    # of the same role-pair (e.g. 3 RMSNorm rsqrt→mul fusions in a
    # transformer block) collapse into ONE candidate carrying every
    # instance under .members — the agent gets multiplicity without
    # the candidate list being eaten by structural near-duplicates.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_pairs: set[tuple[str, str]] = set()

    for ai, (i, prod, prod_role) in enumerate(role_regions):
        for j, cons, cons_role in role_regions[ai + 1 : ai + 1 + lookahead]:
            if not prod_role or not cons_role:
                continue
            sym_pair = (prod, cons)
            if sym_pair in seen_pairs:
                continue
            # Skip pairs the agent has already proposed (idempotent
            # re-suggestion shouldn't pile up duplicates).
            if frozenset(sym_pair) in already:
                continue
            in_critical = (prod in crit) or (cons in crit)
            score = _score(prod_role, cons_role, in_critical)
            distance = j - i
            score = max(0.0, score - 0.05 * (distance - 1))
            role_key = (prod_role, cons_role)
            if role_key not in canonical and score < 0.5:
                continue
            seen_pairs.add(sym_pair)
            grouped.setdefault(role_key, []).append(
                {
                    "chosen": {
                        "grouped_regions": [prod, cons],
                        "fusion_kind": "producer_consumer",
                        "producer_role": prod_role,
                        "consumer_role": cons_role,
                    },
                    "score": score,
                    "in_critical_path": in_critical,
                    "distance": distance,
                    "regions": [prod, cons],
                }
            )

    target_just = (
        f"{target.name} compute lane"
        if target is not None and getattr(target, "name", "")
        else "agent-discovered adjacency"
    )

    candidates: list[ProposalCandidate] = []
    for (prod_role, cons_role), instances in grouped.items():
        instances.sort(key=lambda m: m["score"], reverse=True)
        head = instances[0]
        # Multiplicity boost: repeated structure is itself a signal.
        boost = min(0.1, 0.02 * (len(instances) - 1))
        score = min(1.0, head["score"] + boost)
        if len(instances) > 1:
            extra_pairs = ", ".join(f"({m['regions'][0]}, {m['regions'][1]})" for m in instances[1:4])
            tail = "" if len(instances) <= 4 else f" (+{len(instances) - 4} more)"
            rationale = (
                f"{prod_role} → {cons_role} producer-consumer fusion "
                f"(× {len(instances)} occurrences). Apply via batch_propose. "
                f"Head: ({head['regions'][0]}, {head['regions'][1]}); "
                f"others: {extra_pairs}{tail}."
            )
        else:
            rationale = (
                f"{prod_role} → {cons_role} producer-consumer fusion "
                f"(regions {head['regions'][0]}, {head['regions'][1]}, "
                f"distance={head['distance']})" + (" [on critical path]" if head["in_critical_path"] else "")
            )
        candidates.append(
            ProposalCandidate(
                chosen=dict(head["chosen"]),
                rationale=rationale,
                expected_impact=score,
                target_feature_justification=target_just,
                members=[
                    {"chosen": dict(m["chosen"]), "regions": list(m["regions"]), "score": m["score"]} for m in instances
                ],
                metadata={
                    "pair": [prod_role, cons_role],
                    "instance_count": len(instances),
                    "in_critical_path": head["in_critical_path"],
                    "distance": head["distance"],
                },
            )
        )

    candidates.sort(key=lambda c: c.expected_impact, reverse=True)
    return candidates[:k]


def _existing_grouped_pairs(recipe: ModuleOp) -> set[frozenset]:
    """Walk the recipe for already-proposed propose_fusion ops; return
    the set of grouped_regions frozensets the agent has already
    submitted (so suggest_proposals skips them on re-call)."""
    out: set[frozenset] = set()
    for op in recipe.body.block.ops:
        if op.name != "recipe.propose_fusion":
            continue
        ar = op.properties.get("grouped_regions")
        if ar is None:
            continue
        try:
            syms = []
            for ref in ar.data:
                root = getattr(ref, "root_reference", None)
                if root is not None:
                    syms.append(root.data)
            if syms:
                out.add(frozenset(syms))
        except Exception:  # noqa: BLE001
            continue
    return out


__all__ = ["suggest_fusion"]
