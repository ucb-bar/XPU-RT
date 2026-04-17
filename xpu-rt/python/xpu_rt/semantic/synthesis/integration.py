"""Integration helpers for synthesizing and attaching guards to Recipe IR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr

from compgen.ir.recipe.ops_scope import RecipeGuardOp
from compgen.semantic.synthesis.dataset import build_examples_for_family
from compgen.semantic.synthesis.facts import RecipeFactIndex, build_fact_index
from compgen.semantic.synthesis.guard_lang import Const
from compgen.semantic.synthesis.promote import promote_guard
from compgen.semantic.synthesis.registry import GuardRegistry
from compgen.semantic.synthesis.search import GuardSearchConfig, search_guard_fragments
from compgen.semantic.synthesis.specs import PROMOTED_FAMILIES, get_soundness_spec
from compgen.semantic.synthesis.verify import prove_guard_soundness


def synthesize_and_attach_guards(
    module: ModuleOp,
    *,
    out_dir: str | Path,
    target_class: str = "",
    cfg: GuardSearchConfig | None = None,
) -> tuple[GuardRegistry, RecipeFactIndex, dict[str, Any]]:
    """Synthesize promoted guards, persist them, and attach guard refs to the recipe."""

    guard_dir = Path(out_dir)
    fact_index = build_fact_index(module, target_class=target_class)
    registry = GuardRegistry()
    summary: dict[str, Any] = {"families": {}}

    for family_name, family_spec in PROMOTED_FAMILIES.items():
        examples = build_examples_for_family(module, fact_index, family_name)
        if not examples:
            continue

        result = search_guard_fragments(examples, cfg)
        promoted = bool(result.promoted_fragments) and not (
            len(result.promoted_fragments) == 1 and isinstance(result.promoted_fragments[0], Const) and result.promoted_fragments[0].value is False
        )
        proof = None
        proof_spec = get_soundness_spec(family_name)
        if promoted and proof_spec is not None:
            proof = prove_guard_soundness(result.promoted_fragments, proof_spec)

        family_summary = {
            "examples": len(examples),
            "fragments_proposed": len(result.sound_fragments) + len(result.precise_unsound_fragments),
            "sound_on_first_attempt": len(result.sound_fragments),
            "precise_unsound": len(result.precise_unsound_fragments),
            "repaired_by_guard": len(result.repaired_fragments),
            "promoted": int(promoted),
            "average_guard_terms": float(len(result.promoted_fragments)) if promoted else 0.0,
            "average_proof_time_ms": float(proof.verification_time_ms) if proof is not None else 0.0,
            "proof_status": proof.status if proof is not None else "skipped",
        }
        summary["families"][family_name] = family_summary

        if not promoted:
            continue

        artifact = promote_guard(
            guard_dir,
            transform_family=family_name,
            guard_kind=family_spec.guard_kind,
            target_class=target_class,
            fragments=result.promoted_fragments,
            sound_fragments=len(result.sound_fragments),
            precise_unsound_fragments=len(result.precise_unsound_fragments),
            repaired_fragments=len(result.repaired_fragments),
            proved_sound=bool(proof and proof.proved_sound),
            proof_status=proof.status if proof is not None else "skipped",
            verification_time_ms=proof.verification_time_ms if proof is not None else 0.0,
            metadata={"examples": len(examples)},
        )
        registry.register(artifact)

        guard_symbol = f"guard_{family_name}"
        existing_guard = next(
            (
                op for op in module.body.block.ops
                if isinstance(op, RecipeGuardOp) and op.sym_name.data == guard_symbol
            ),
            None,
        )
        if existing_guard is None:
            module.body.block.add_op(RecipeGuardOp.build(properties={
                "sym_name": StringAttr(guard_symbol),
                "guard_key": StringAttr(artifact.guard_key),
                "transform_family": StringAttr(family_name),
                "guard_kind": StringAttr(family_spec.guard_kind),
                **({"target_class": StringAttr(target_class)} if target_class else {}),
            }))

        for op in module.walk():
            if family_spec.matches_candidate(op):
                op.properties["guard_refs"] = ArrayAttr([SymbolRefAttr(guard_symbol)])

    total_promoted = sum(family["promoted"] for family in summary["families"].values())
    total_fragments = sum(family["fragments_proposed"] for family in summary["families"].values())
    summary["fragments_proposed"] = total_fragments
    summary["sound_on_first_attempt"] = sum(family["sound_on_first_attempt"] for family in summary["families"].values())
    summary["precise_unsound"] = sum(family["precise_unsound"] for family in summary["families"].values())
    summary["repaired_by_guard"] = sum(family["repaired_by_guard"] for family in summary["families"].values())
    summary["promoted"] = total_promoted
    summary["average_guard_terms"] = (
        sum(family["average_guard_terms"] for family in summary["families"].values()) / max(total_promoted, 1)
        if summary["families"] else 0.0
    )
    summary["average_proof_time_ms"] = (
        sum(family["average_proof_time_ms"] for family in summary["families"].values()) / max(total_promoted, 1)
        if summary["families"] else 0.0
    )
    return registry, fact_index, summary


__all__ = ["synthesize_and_attach_guards"]
