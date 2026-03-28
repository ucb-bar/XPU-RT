"""Synthesized guard and analysis support."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Add": "compgen.synthesis.guard_lang",
    "BoolN": "compgen.synthesis.guard_lang",
    "BoolOp": "compgen.synthesis.guard_lang",
    "Cmp": "compgen.synthesis.guard_lang",
    "CmpOp": "compgen.synthesis.guard_lang",
    "Const": "compgen.synthesis.guard_lang",
    "Div": "compgen.synthesis.guard_lang",
    "EXPERIMENTAL_FAMILIES": "compgen.synthesis.specs",
    "Expr": "compgen.synthesis.guard_lang",
    "FUSION_FAMILY": "compgen.synthesis.specs",
    "FusionGuardSpec": "compgen.synthesis.specs",
    "FusionSoundnessSpec": "compgen.synthesis.specs",
    "GuardArtifact": "compgen.synthesis.promote",
    "GuardFamilySpec": "compgen.synthesis.specs",
    "GuardProofResult": "compgen.synthesis.verify",
    "GuardRegistry": "compgen.synthesis.registry",
    "GuardRuntime": "compgen.synthesis.runtime",
    "GuardSearchConfig": "compgen.synthesis.search",
    "GuardSearchResult": "compgen.synthesis.search",
    "GuardVerdict": "compgen.synthesis.runtime",
    "LOCAL_MEM_FAMILY": "compgen.synthesis.specs",
    "LocalMemGuardSpec": "compgen.synthesis.specs",
    "LocalMemSoundnessSpec": "compgen.synthesis.specs",
    "ModEq": "compgen.synthesis.guard_lang",
    "Mul": "compgen.synthesis.guard_lang",
    "Not": "compgen.synthesis.guard_lang",
    "PROMOTED_FAMILIES": "compgen.synthesis.specs",
    "QuantizationLegalitySpec": "compgen.synthesis.specs",
    "RangeNoWrapSpec": "compgen.synthesis.specs",
    "RecipeFactIndex": "compgen.synthesis.facts",
    "RegionFacts": "compgen.synthesis.facts",
    "SoundnessFormulaSpec": "compgen.synthesis.verify",
    "Sub": "compgen.synthesis.guard_lang",
    "SynthesisExample": "compgen.synthesis.dataset",
    "Var": "compgen.synthesis.guard_lang",
    "VectorizationGuardSpec": "compgen.synthesis.specs",
    "and_": "compgen.synthesis.guard_lang",
    "build_candidate_env": "compgen.synthesis.facts",
    "build_examples_for_family": "compgen.synthesis.dataset",
    "build_fact_index": "compgen.synthesis.facts",
    "eval_guard": "compgen.synthesis.guard_lang",
    "expr_from_json": "compgen.synthesis.guard_lang",
    "expr_to_json": "compgen.synthesis.guard_lang",
    "get_family_spec": "compgen.synthesis.specs",
    "get_soundness_spec": "compgen.synthesis.specs",
    "load_guard_artifact": "compgen.synthesis.promote",
    "or_": "compgen.synthesis.guard_lang",
    "promote_guard": "compgen.synthesis.promote",
    "prove_guard_soundness": "compgen.synthesis.verify",
    "search_guard_fragments": "compgen.synthesis.search",
    "synthesize_and_attach_guards": "compgen.synthesis.integration",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name])
    return getattr(module, name)
