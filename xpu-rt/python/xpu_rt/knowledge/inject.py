"""Knowledge injection into LLM prompts.

Formats QueryResult into text blocks suitable for insertion into
LLM prompts at various decision points.
"""

from __future__ import annotations

from compgen.knowledge.base import (
    AntiPattern,
    KernelLibraryWisdom,
    KnowledgeBase,
    OpWisdom,
    TargetPattern,
    TransformRecipe,
)
from compgen.knowledge.query import QueryContext, query


def inject_knowledge(
    kb: KnowledgeBase,
    ctx: QueryContext,
    *,
    max_items_per_section: int = 5,
    include_anti_patterns: bool = True,
    include_recipes: bool = True,
    include_kernel_wisdom: bool = True,
) -> str:
    """Query KB and format results for LLM prompt injection.

    Returns a formatted text block ready to insert into an LLM prompt.
    Sections are only included if they have content.

    Args:
        kb: The knowledge base to query.
        ctx: Context describing the current compilation situation.
        max_items_per_section: Maximum entries per output section.
        include_anti_patterns: Whether to include anti-pattern warnings.
        include_recipes: Whether to include transform recipes.
        include_kernel_wisdom: Whether to include kernel library insights.

    Returns:
        Formatted text block, or empty string if no relevant knowledge.
    """
    result = query(kb, ctx)
    if result.is_empty:
        return ""

    sections: list[str] = []

    # Op-level guidance
    if result.op_wisdom:
        sections.append(_format_op_wisdom(result.op_wisdom, ctx.target_class, max_items_per_section))

    # Target optimization patterns
    if result.target_patterns:
        sections.append(_format_target_patterns(result.target_patterns[:max_items_per_section]))

    # Transform recipes
    if include_recipes and result.recipes:
        sections.append(_format_recipes(result.recipes[:max_items_per_section]))

    # Kernel library insights
    if include_kernel_wisdom and result.kernel_wisdom:
        sections.append(_format_kernel_wisdom(result.kernel_wisdom[:max_items_per_section]))

    # Anti-patterns
    if include_anti_patterns and result.anti_patterns:
        sections.append(_format_anti_patterns(result.anti_patterns[:max_items_per_section]))

    return "\n\n".join(sections)


def _format_op_wisdom(wisdom_list: list[OpWisdom], target_class: str, max_items: int) -> str:
    """Format op-level optimization guidance."""
    lines = ["## Optimization Knowledge"]
    for w in wisdom_list[:max_items]:
        lines.append(f"\n### {w.op_family.upper()}")
        # Tiling guidance for this target
        for tg in w.tiling_guidance:
            if tg.target_class in (target_class, "any"):
                sizes = ", ".join(str(s) for s in tg.tile_sizes)
                lines.append(
                    f"- **Tiling [{sizes}]**: {tg.rationale}"
                    f" (source: {tg.source}, confidence: {tg.confidence.value})"
                )
        # Fusion opportunities
        for fo in w.fusion_opportunities[:3]:
            lines.append(f"- **Fusion [{fo.pattern}]**: {fo.description} (source: {fo.source})")
        # Backend guidance for this target
        for bg in w.backend_guidance:
            if bg.target_class in (target_class, "any"):
                lines.append(
                    f"- **Backend**: Use {bg.recommended_backend}. {bg.rationale} (source: {bg.source})"
                )
        # Pitfalls
        if w.pitfalls:
            lines.append(f"- **Pitfalls**: {'; '.join(w.pitfalls[:3])}")
    return "\n".join(lines)


def _format_target_patterns(patterns: list[TargetPattern]) -> str:
    """Format target-class optimization patterns."""
    lines = ["## Target Optimization Patterns"]
    for p in patterns:
        lines.append(f"- **{p.pattern_name}** ({p.category}): {p.description}")
        for note in p.implementation_notes[:2]:
            lines.append(f"  - {note}")
    return "\n".join(lines)


def _format_recipes(recipes: list[TransformRecipe]) -> str:
    """Format transform recipes."""
    lines = ["## Recommended Transform Recipes"]
    for r in recipes:
        lines.append(f"\n### {r.name} ({r.op_family}, {r.target_class})")
        for i, step in enumerate(r.steps, 1):
            params = ", ".join(f"{k}={v}" for k, v in step.parameters.items()) if step.parameters else ""
            lines.append(f"  {i}. **{step.action}**({params}): {step.rationale}")
        lines.append(f"  Expected: {r.expected_speedup} (source: {r.source})")
    return "\n".join(lines)


def _format_kernel_wisdom(wisdom: list[KernelLibraryWisdom]) -> str:
    """Format kernel library insights."""
    lines = ["## Kernel Library Insights"]
    for w in wisdom:
        lines.append(f"- **[{w.library}]** {w.topic}: {w.insight}")
    return "\n".join(lines)


def _format_anti_patterns(patterns: list[AntiPattern]) -> str:
    """Format anti-patterns as warnings."""
    lines = ["## Common Pitfalls to Avoid"]
    for p in patterns:
        lines.append(f"- **{p.name}**: {p.description}")
        lines.append(f"  Fix: {p.fix}")
    return "\n".join(lines)


def inject_for_analysis(kb: KnowledgeBase, op_families: list[str], target_class: str = "gpu") -> str:
    """Inject knowledge for model analysis prompt.

    Args:
        kb: The knowledge base to query.
        op_families: Op families present in the model.
        target_class: Hardware target class (default ``"gpu"``).

    Returns:
        Formatted knowledge text for analysis prompts.
    """
    return inject_knowledge(
        kb,
        QueryContext(
            op_families=op_families,
            target_class=target_class,
            current_stage="analysis",
        ),
    )


def inject_for_eqsat(kb: KnowledgeBase, op_families: list[str], target_class: str = "gpu") -> str:
    """Inject knowledge for eqsat rule suggestion.

    Args:
        kb: The knowledge base to query.
        op_families: Op families relevant to the rewrite.
        target_class: Hardware target class (default ``"gpu"``).

    Returns:
        Formatted knowledge text for eqsat prompts (no kernel wisdom).
    """
    return inject_knowledge(
        kb,
        QueryContext(
            op_families=op_families,
            target_class=target_class,
            current_stage="eqsat",
        ),
        include_kernel_wisdom=False,
    )


def inject_for_kernel_search(kb: KnowledgeBase, op_family: str, target_class: str = "gpu") -> str:
    """Inject knowledge for kernel search/scheduling.

    Args:
        kb: The knowledge base to query.
        op_family: The op family being searched.
        target_class: Hardware target class (default ``"gpu"``).

    Returns:
        Formatted knowledge text for kernel search prompts (no anti-patterns).
    """
    return inject_knowledge(
        kb,
        QueryContext(
            op_families=[op_family],
            target_class=target_class,
            current_stage="kernel_search",
        ),
        include_anti_patterns=False,
    )


def inject_for_scheduling(kb: KnowledgeBase, target_class: str = "gpu") -> str:
    """Inject knowledge for dispatch/scheduling decisions.

    Args:
        kb: The knowledge base to query.
        target_class: Hardware target class (default ``"gpu"``).

    Returns:
        Formatted knowledge text for scheduling prompts.
    """
    return inject_knowledge(
        kb,
        QueryContext(
            target_class=target_class,
            current_stage="scheduling",
        ),
        include_recipes=False,
        include_kernel_wisdom=False,
    )


__all__ = [
    "inject_for_analysis",
    "inject_for_eqsat",
    "inject_for_kernel_search",
    "inject_for_scheduling",
    "inject_knowledge",
]
