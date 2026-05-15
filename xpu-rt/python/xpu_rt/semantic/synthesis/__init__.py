"""Synthesized guard and analysis support."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Add": "compgen.semantic.synthesis.guard_lang",
    "BoolN": "compgen.semantic.synthesis.guard_lang",
    "BoolOp": "compgen.semantic.synthesis.guard_lang",
    "Cmp": "compgen.semantic.synthesis.guard_lang",
    "CmpOp": "compgen.semantic.synthesis.guard_lang",
    "Const": "compgen.semantic.synthesis.guard_lang",
    "Div": "compgen.semantic.synthesis.guard_lang",
    "EXPERIMENTAL_FAMILIES": "compgen.semantic.synthesis.specs",
    "Expr": "compgen.semantic.synthesis.guard_lang",
    "FUSION_FAMILY": "compgen.semantic.synthesis.specs",
    "FusionGuardSpec": "compgen.semantic.synthesis.specs",
    "FusionSoundnessSpec": "compgen.semantic.synthesis.specs",
    "GuardArtifact": "compgen.semantic.synthesis.promote",
    "GuardFamilySpec": "compgen.semantic.synthesis.specs",
    "GuardProofResult": "compgen.semantic.synthesis.verify",
    "GuardRegistry": "compgen.semantic.synthesis.registry",
    "GuardRuntime": "compgen.semantic.synthesis.runtime",
    "GuardSearchConfig": "compgen.semantic.synthesis.search",
    "GuardSearchResult": "compgen.semantic.synthesis.search",
    "GuardVerdict": "compgen.semantic.synthesis.runtime",
    "LOCAL_MEM_FAMILY": "compgen.semantic.synthesis.specs",
    "LocalMemGuardSpec": "compgen.semantic.synthesis.specs",
    "LocalMemSoundnessSpec": "compgen.semantic.synthesis.specs",
    "ModEq": "compgen.semantic.synthesis.guard_lang",
    "Mul": "compgen.semantic.synthesis.guard_lang",
    "Not": "compgen.semantic.synthesis.guard_lang",
    "PROMOTED_FAMILIES": "compgen.semantic.synthesis.specs",
    "QuantizationLegalitySpec": "compgen.semantic.synthesis.specs",
    "RangeNoWrapSpec": "compgen.semantic.synthesis.specs",
    "RecipeFactIndex": "compgen.semantic.synthesis.facts",
    "RegionFacts": "compgen.semantic.synthesis.facts",
    "SoundnessFormulaSpec": "compgen.semantic.synthesis.verify",
    "Sub": "compgen.semantic.synthesis.guard_lang",
    "SynthesisExample": "compgen.semantic.synthesis.dataset",
    "Var": "compgen.semantic.synthesis.guard_lang",
    "VectorizationGuardSpec": "compgen.semantic.synthesis.specs",
    "and_": "compgen.semantic.synthesis.guard_lang",
    "build_candidate_env": "compgen.semantic.synthesis.facts",
    "build_examples_for_family": "compgen.semantic.synthesis.dataset",
    "build_fact_index": "compgen.semantic.synthesis.facts",
    "eval_guard": "compgen.semantic.synthesis.guard_lang",
    "expr_from_json": "compgen.semantic.synthesis.guard_lang",
    "expr_to_json": "compgen.semantic.synthesis.guard_lang",
    "get_family_spec": "compgen.semantic.synthesis.specs",
    "get_soundness_spec": "compgen.semantic.synthesis.specs",
    "load_guard_artifact": "compgen.semantic.synthesis.promote",
    "or_": "compgen.semantic.synthesis.guard_lang",
    "promote_guard": "compgen.semantic.synthesis.promote",
    "prove_guard_soundness": "compgen.semantic.synthesis.verify",
    "search_guard_fragments": "compgen.semantic.synthesis.search",
    "synthesize_and_attach_guards": "compgen.semantic.synthesis.integration",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name])
    return getattr(module, name)
