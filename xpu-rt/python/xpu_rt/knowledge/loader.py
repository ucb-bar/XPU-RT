"""Knowledge base YAML serialization and loading.

Dumps the entire KnowledgeBase to a YAML file in ``data/`` and
loads it back. This is the persistence layer for the optimization
knowledge database.
"""

from __future__ import annotations

import enum
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from compgen.knowledge.base import (
    AntiPattern,
    BackendGuidance,
    CompilerHeuristic,
    Confidence,
    FusionOpportunity,
    KernelLibraryWisdom,
    KnowledgeBase,
    LayoutPreference,
    OpWisdom,
    TargetPattern,
    TilingGuidance,
    TransformRecipe,
    TransformStep,
)


def save_kb(kb: KnowledgeBase, path: str | Path) -> Path:
    """Save a KnowledgeBase to a YAML file.

    Args:
        kb: KnowledgeBase to serialize.
        path: Output file path.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _kb_to_dict(kb)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False, width=120))
    return path


def load_kb(path: str | Path) -> KnowledgeBase:
    """Load a KnowledgeBase from a YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        Populated KnowledgeBase.
    """
    data = yaml.safe_load(Path(path).read_text())
    return _dict_to_kb(data)


def _confidence(val: str) -> Confidence:
    return Confidence(val)


def _kb_to_dict(kb: KnowledgeBase) -> dict[str, Any]:
    """Serialize entire KB to a plain dict."""
    return {
        "schema_version": "1.0",
        "summary": kb.summary(),
        "op_wisdom": {
            name: _op_wisdom_to_dict(w) for name, w in kb.op_wisdom.items()
        },
        "target_patterns": {
            tc: [_target_pattern_to_dict(p) for p in patterns]
            for tc, patterns in kb.target_patterns.items()
        },
        "transform_recipes": [_recipe_to_dict(r) for r in kb.transform_recipes],
        "kernel_wisdom": [_enum_to_str(asdict(w)) for w in kb.kernel_wisdom],
        "compiler_heuristics": [_enum_to_str(asdict(h)) for h in kb.compiler_heuristics],
        "anti_patterns": [_enum_to_str(asdict(a)) for a in kb.anti_patterns],
    }


def _enum_to_str(obj: Any) -> Any:
    """Recursively convert enum values to strings in dicts/lists."""
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _enum_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_enum_to_str(v) for v in obj]
    return obj


def _op_wisdom_to_dict(w: OpWisdom) -> dict[str, Any]:
    return {
        "op_family": w.op_family,
        "tiling_guidance": [_enum_to_str(asdict(t)) for t in w.tiling_guidance],
        "fusion_opportunities": [_enum_to_str(asdict(f)) for f in w.fusion_opportunities],
        "layout_preferences": [_enum_to_str(asdict(lp)) for lp in w.layout_preferences],
        "backend_guidance": [_enum_to_str(asdict(b)) for b in w.backend_guidance],
        "pitfalls": w.pitfalls,
        "performance_bounds": w.performance_bounds,
    }


def _target_pattern_to_dict(p: TargetPattern) -> dict[str, Any]:
    return _enum_to_str(asdict(p))


def _recipe_to_dict(r: TransformRecipe) -> dict[str, Any]:
    return _enum_to_str(asdict(r))


def _dict_to_kb(data: dict[str, Any]) -> KnowledgeBase:
    """Deserialize a dict into a KnowledgeBase."""
    op_wisdom: dict[str, OpWisdom] = {}
    for name, wd in data.get("op_wisdom", {}).items():
        op_wisdom[name] = _dict_to_op_wisdom(wd)

    target_patterns: dict[str, list[TargetPattern]] = {}
    for tc, patterns in data.get("target_patterns", {}).items():
        target_patterns[tc] = [_dict_to_target_pattern(p) for p in patterns]

    transform_recipes = [_dict_to_recipe(r) for r in data.get("transform_recipes", [])]
    kernel_wisdom = [_dict_to_kernel_wisdom(w) for w in data.get("kernel_wisdom", [])]
    compiler_heuristics = [_dict_to_heuristic(h) for h in data.get("compiler_heuristics", [])]
    anti_patterns = [_dict_to_anti_pattern(a) for a in data.get("anti_patterns", [])]

    return KnowledgeBase(
        op_wisdom=op_wisdom,
        target_patterns=target_patterns,
        transform_recipes=transform_recipes,
        kernel_wisdom=kernel_wisdom,
        compiler_heuristics=compiler_heuristics,
        anti_patterns=anti_patterns,
    )


def _dict_to_op_wisdom(d: dict[str, Any]) -> OpWisdom:
    return OpWisdom(
        op_family=d["op_family"],
        tiling_guidance=[
            TilingGuidance(
                target_class=t["target_class"],
                tile_sizes=t["tile_sizes"],
                rationale=t["rationale"],
                source=t["source"],
                confidence=_confidence(t["confidence"]),
            )
            for t in d.get("tiling_guidance", [])
        ],
        fusion_opportunities=[
            FusionOpportunity(
                pattern=f["pattern"],
                description=f["description"],
                conditions=f["conditions"],
                source=f["source"],
                confidence=_confidence(f["confidence"]),
            )
            for f in d.get("fusion_opportunities", [])
        ],
        layout_preferences=[
            LayoutPreference(
                target_class=lp["target_class"],
                preferred_layout=lp["preferred_layout"],
                rationale=lp["rationale"],
                source=lp["source"],
            )
            for lp in d.get("layout_preferences", [])
        ],
        backend_guidance=[
            BackendGuidance(
                target_class=b["target_class"],
                recommended_backend=b["recommended_backend"],
                conditions=b["conditions"],
                rationale=b["rationale"],
                source=b["source"],
            )
            for b in d.get("backend_guidance", [])
        ],
        pitfalls=d.get("pitfalls", []),
        performance_bounds=d.get("performance_bounds", []),
    )


def _dict_to_target_pattern(d: dict[str, Any]) -> TargetPattern:
    return TargetPattern(
        target_class=d["target_class"],
        category=d["category"],
        pattern_name=d["pattern_name"],
        description=d["description"],
        implementation_notes=d.get("implementation_notes", []),
        source=d["source"],
        confidence=_confidence(d["confidence"]),
    )


def _dict_to_recipe(d: dict[str, Any]) -> TransformRecipe:
    return TransformRecipe(
        name=d["name"],
        op_family=d["op_family"],
        target_class=d["target_class"],
        steps=[
            TransformStep(
                action=s["action"],
                parameters=s.get("parameters", {}),
                rationale=s["rationale"],
            )
            for s in d.get("steps", [])
        ],
        expected_speedup=d["expected_speedup"],
        source=d["source"],
        confidence=_confidence(d["confidence"]),
    )


def _dict_to_kernel_wisdom(d: dict[str, Any]) -> KernelLibraryWisdom:
    return KernelLibraryWisdom(
        library=d["library"],
        topic=d["topic"],
        insight=d["insight"],
        conditions=d.get("conditions", []),
        confidence=_confidence(d["confidence"]),
    )


def _dict_to_heuristic(d: dict[str, Any]) -> CompilerHeuristic:
    return CompilerHeuristic(
        compiler=d["compiler"],
        topic=d["topic"],
        heuristic=d["heuristic"],
        conditions=d.get("conditions", []),
        confidence=_confidence(d["confidence"]),
    )


def _dict_to_anti_pattern(d: dict[str, Any]) -> AntiPattern:
    return AntiPattern(
        name=d["name"],
        description=d["description"],
        symptoms=d.get("symptoms", []),
        fix=d["fix"],
        source=d["source"],
    )


def create_default_kb() -> KnowledgeBase:
    """Create a KnowledgeBase populated with all default knowledge.

    This is the primary entry point. Consolidates optimization wisdom
    from CUTLASS, cuDNN, oneDNN, Triton, Exo, TVM, Halide, IREE, XLA.

    Returns:
        A fully populated KnowledgeBase.
    """
    from compgen.knowledge.anti_patterns import build_default_anti_patterns
    from compgen.knowledge.compiler_heuristics import build_default_compiler_heuristics
    from compgen.knowledge.kernel_wisdom import build_default_kernel_wisdom
    from compgen.knowledge.ops_wisdom import build_default_op_wisdom
    from compgen.knowledge.target_patterns import build_default_target_patterns
    from compgen.knowledge.transform_recipes import build_default_recipes

    return KnowledgeBase(
        op_wisdom=build_default_op_wisdom(),
        target_patterns=build_default_target_patterns(),
        transform_recipes=build_default_recipes(),
        kernel_wisdom=build_default_kernel_wisdom(),
        compiler_heuristics=build_default_compiler_heuristics(),
        anti_patterns=build_default_anti_patterns(),
    )


__all__ = ["create_default_kb", "load_kb", "save_kb"]
