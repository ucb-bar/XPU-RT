"""Knowledge base query API.

Provides context-aware queries that return relevant knowledge slices
for specific compilation situations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compgen.knowledge.base import (
    AntiPattern,
    CompilerHeuristic,
    KernelLibraryWisdom,
    KnowledgeBase,
    OpWisdom,
    TargetPattern,
    TransformRecipe,
)


@dataclass(frozen=True)
class QueryContext:
    """Context for a knowledge query.

    Attributes:
        op_families: Op families in the current workload (e.g., ["matmul", "relu"]).
        target_class: Target hardware class (e.g., "gpu", "cpu", "npu").
        objective: Optimization objective (e.g., "latency", "throughput").
        bottleneck_kind: If known, current bottleneck type ("compute_bound", "memory_bound").
        current_stage: Pipeline stage making the query ("analysis", "eqsat", "kernel_search", "scheduling").
    """

    op_families: list[str] = field(default_factory=list)
    target_class: str = "gpu"
    objective: str = "latency"
    bottleneck_kind: str = ""
    current_stage: str = ""


@dataclass(frozen=True)
class QueryResult:
    """Result of a knowledge query.

    Contains ranked, filtered knowledge relevant to the query context.
    """

    op_wisdom: list[OpWisdom] = field(default_factory=list)
    target_patterns: list[TargetPattern] = field(default_factory=list)
    recipes: list[TransformRecipe] = field(default_factory=list)
    kernel_wisdom: list[KernelLibraryWisdom] = field(default_factory=list)
    compiler_heuristics: list[CompilerHeuristic] = field(default_factory=list)
    anti_patterns: list[AntiPattern] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Return True if all result lists are empty."""
        return (
            not self.op_wisdom
            and not self.target_patterns
            and not self.recipes
            and not self.kernel_wisdom
            and not self.compiler_heuristics
            and not self.anti_patterns
        )


def query(kb: KnowledgeBase, ctx: QueryContext) -> QueryResult:
    """Query the knowledge base with context-aware filtering and ranking.

    Returns knowledge relevant to the current compilation context:
    - Op wisdom for ops in the workload
    - Target patterns for the current target class
    - Transform recipes matching ops + target
    - Kernel library wisdom relevant to the target
    - Compiler heuristics relevant to the current stage
    - Anti-patterns (always included as warnings)

    Args:
        kb: The knowledge base to query.
        ctx: Context describing the current compilation situation.

    Returns:
        A ``QueryResult`` containing filtered, relevant knowledge.
    """
    # Get op wisdom for requested families
    op_wisdom: list[OpWisdom] = []
    for family in ctx.op_families:
        w = kb.query_op(family)
        if w is not None:
            op_wisdom.append(w)

    # Get target patterns
    target_patterns = kb.query_target(ctx.target_class)

    # Get recipes matching ops + target
    recipes: list[TransformRecipe] = []
    for family in ctx.op_families:
        recipes.extend(kb.query_recipes(op_family=family, target_class=ctx.target_class))
    # Also get target-wide recipes
    recipes.extend(kb.query_recipes(target_class=ctx.target_class))
    # Deduplicate by name
    seen: set[str] = set()
    unique_recipes: list[TransformRecipe] = []
    for r in recipes:
        if r.name not in seen:
            seen.add(r.name)
            unique_recipes.append(r)
    recipes = unique_recipes

    # Get kernel wisdom based on target class
    kernel_wisdom: list[KernelLibraryWisdom] = []
    if ctx.target_class == "gpu":
        kernel_wisdom.extend(kb.query_kernel_wisdom(library="cutlass"))
        kernel_wisdom.extend(kb.query_kernel_wisdom(library="triton"))
        kernel_wisdom.extend(kb.query_kernel_wisdom(library="cudnn"))
    elif ctx.target_class == "cpu":
        kernel_wisdom.extend(kb.query_kernel_wisdom(library="onednn"))
        kernel_wisdom.extend(kb.query_kernel_wisdom(library="exo"))
    elif ctx.target_class in ("npu", "accelerator"):
        kernel_wisdom.extend(kb.query_kernel_wisdom(library="exo"))

    # Get compiler heuristics relevant to current stage
    compiler_heuristics = kb.query_heuristics()
    if ctx.current_stage == "eqsat":
        compiler_heuristics = [
            h for h in compiler_heuristics if h.topic in ("fusion", "simplification", "rewrite")
        ]
    elif ctx.current_stage == "kernel_search":
        compiler_heuristics = [
            h for h in compiler_heuristics if h.topic in ("autotuning", "code_generation", "scheduling")
        ]

    # Anti-patterns always included
    anti_patterns = kb.query_anti_patterns()

    return QueryResult(
        op_wisdom=op_wisdom,
        target_patterns=target_patterns,
        recipes=recipes,
        kernel_wisdom=kernel_wisdom,
        compiler_heuristics=compiler_heuristics,
        anti_patterns=anti_patterns,
    )


def query_for_op(kb: KnowledgeBase, op_family: str, target_class: str = "gpu") -> QueryResult:
    """Convenience: query for a single op family.

    Args:
        kb: The knowledge base to query.
        op_family: The op family to look up (e.g. ``"matmul"``).
        target_class: Hardware target class (default ``"gpu"``).

    Returns:
        A ``QueryResult`` scoped to the given op and target.
    """
    return query(kb, QueryContext(op_families=[op_family], target_class=target_class))


def query_for_stage(
    kb: KnowledgeBase,
    stage: str,
    target_class: str = "gpu",
    op_families: list[str] | None = None,
) -> QueryResult:
    """Convenience: query for a pipeline stage.

    Args:
        kb: The knowledge base to query.
        stage: Pipeline stage name (e.g. ``"eqsat"``, ``"kernel_search"``).
        target_class: Hardware target class (default ``"gpu"``).
        op_families: Optional list of op families in the workload.

    Returns:
        A ``QueryResult`` scoped to the given stage and target.
    """
    return query(
        kb,
        QueryContext(
            op_families=op_families or [],
            target_class=target_class,
            current_stage=stage,
        ),
    )


__all__ = [
    "QueryContext",
    "QueryResult",
    "query",
    "query_for_op",
    "query_for_stage",
]
