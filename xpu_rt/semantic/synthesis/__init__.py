"""Synthesized guard and analysis support."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Add": "xpu_rt.semantic.synthesis.guard_lang",
    "BoolN": "xpu_rt.semantic.synthesis.guard_lang",
    "BoolOp": "xpu_rt.semantic.synthesis.guard_lang",
    "Cmp": "xpu_rt.semantic.synthesis.guard_lang",
    "CmpOp": "xpu_rt.semantic.synthesis.guard_lang",
    "Const": "xpu_rt.semantic.synthesis.guard_lang",
    "Div": "xpu_rt.semantic.synthesis.guard_lang",
    "EXPERIMENTAL_FAMILIES": "xpu_rt.semantic.synthesis.specs",
    "Expr": "xpu_rt.semantic.synthesis.guard_lang",
    "FUSION_FAMILY": "xpu_rt.semantic.synthesis.specs",
    "FusionGuardSpec": "xpu_rt.semantic.synthesis.specs",
    "FusionSoundnessSpec": "xpu_rt.semantic.synthesis.specs",
    "GuardArtifact": "xpu_rt.semantic.synthesis.promote",
    "GuardFamilySpec": "xpu_rt.semantic.synthesis.specs",
    "GuardProofResult": "xpu_rt.semantic.synthesis.verify",
    "GuardRegistry": "xpu_rt.semantic.synthesis.registry",
    "GuardRuntime": "xpu_rt.semantic.synthesis.runtime",
    "GuardSearchConfig": "xpu_rt.semantic.synthesis.search",
    "GuardSearchResult": "xpu_rt.semantic.synthesis.search",
    "GuardVerdict": "xpu_rt.semantic.synthesis.runtime",
    "LOCAL_MEM_FAMILY": "xpu_rt.semantic.synthesis.specs",
    "LocalMemGuardSpec": "xpu_rt.semantic.synthesis.specs",
    "LocalMemSoundnessSpec": "xpu_rt.semantic.synthesis.specs",
    "ModEq": "xpu_rt.semantic.synthesis.guard_lang",
    "Mul": "xpu_rt.semantic.synthesis.guard_lang",
    "Not": "xpu_rt.semantic.synthesis.guard_lang",
    "PROMOTED_FAMILIES": "xpu_rt.semantic.synthesis.specs",
    "QuantizationLegalitySpec": "xpu_rt.semantic.synthesis.specs",
    "RangeNoWrapSpec": "xpu_rt.semantic.synthesis.specs",
    "RecipeFactIndex": "xpu_rt.semantic.synthesis.facts",
    "RegionFacts": "xpu_rt.semantic.synthesis.facts",
    "SoundnessFormulaSpec": "xpu_rt.semantic.synthesis.verify",
    "Sub": "xpu_rt.semantic.synthesis.guard_lang",
    "SynthesisExample": "xpu_rt.semantic.synthesis.dataset",
    "Var": "xpu_rt.semantic.synthesis.guard_lang",
    "VectorizationGuardSpec": "xpu_rt.semantic.synthesis.specs",
    "and_": "xpu_rt.semantic.synthesis.guard_lang",
    "build_candidate_env": "xpu_rt.semantic.synthesis.facts",
    "build_examples_for_family": "xpu_rt.semantic.synthesis.dataset",
    "build_fact_index": "xpu_rt.semantic.synthesis.facts",
    "eval_guard": "xpu_rt.semantic.synthesis.guard_lang",
    "expr_from_json": "xpu_rt.semantic.synthesis.guard_lang",
    "expr_to_json": "xpu_rt.semantic.synthesis.guard_lang",
    "get_family_spec": "xpu_rt.semantic.synthesis.specs",
    "get_soundness_spec": "xpu_rt.semantic.synthesis.specs",
    "load_guard_artifact": "xpu_rt.semantic.synthesis.promote",
    "or_": "xpu_rt.semantic.synthesis.guard_lang",
    "promote_guard": "xpu_rt.semantic.synthesis.promote",
    "prove_guard_soundness": "xpu_rt.semantic.synthesis.verify",
    "search_guard_fragments": "xpu_rt.semantic.synthesis.search",
    "synthesize_and_attach_guards": "xpu_rt.semantic.synthesis.integration",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name])
    return getattr(module, name)
