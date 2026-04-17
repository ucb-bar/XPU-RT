"""Core data types and container for the compiler optimization knowledge base.

Defines the foundational types -- OpWisdom, TargetPattern, TransformRecipe,
KernelLibraryWisdom, CompilerHeuristic, AntiPattern -- and the queryable
KnowledgeBase container that indexes them by op family, target class, and topic.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Confidence(enum.Enum):
    """Confidence level for a knowledge entry."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TargetClass(enum.Enum):
    """Broad hardware target classification.

    String-based for flexibility -- mirrors the concept from capability.py
    without hard-coupling to it.
    """

    GPU = "gpu"
    CPU = "cpu"
    NPU = "npu"
    ACCELERATOR = "accelerator"
    SOC = "soc"
    ANY = "any"


# ---------------------------------------------------------------------------
# Knowledge entry types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TilingGuidance:
    """Recommended tile sizes for a given target class.

    Attributes:
        target_class: Hardware family this guidance applies to.
        tile_sizes: Recommended tile dimensions.
        rationale: Why these sizes work well.
        source: Compiler / library the guidance originates from.
        confidence: How reliable this guidance is.
    """

    target_class: str
    tile_sizes: list[int]
    rationale: str
    source: str
    confidence: Confidence


@dataclass(frozen=True)
class FusionOpportunity:
    """A known-profitable fusion pattern.

    Attributes:
        pattern: Short description of the fused op chain (e.g. ``"matmul+bias+relu"``).
        description: Longer explanation of the fusion.
        conditions: When the fusion is applicable.
        source: Compiler / library where this pattern is well-known.
        confidence: How reliable this guidance is.
    """

    pattern: str
    description: str
    conditions: list[str]
    source: str
    confidence: Confidence


@dataclass(frozen=True)
class LayoutPreference:
    """Preferred data layout for a target class.

    Attributes:
        target_class: Hardware family this preference applies to.
        preferred_layout: Layout string (e.g. ``"NHWC"``, ``"row-major"``).
        rationale: Why this layout is preferred.
        source: Compiler / library the preference originates from.
    """

    target_class: str
    preferred_layout: str
    rationale: str
    source: str


@dataclass(frozen=True)
class BackendGuidance:
    """Recommended backend for an op family on a given target.

    Attributes:
        target_class: Hardware family.
        recommended_backend: E.g. ``"triton"``, ``"cutlass"``, ``"onednn"``.
        conditions: When this backend is the right choice.
        rationale: Why it is recommended.
        source: Origin of the recommendation.
    """

    target_class: str
    recommended_backend: str
    conditions: str
    rationale: str
    source: str


@dataclass(frozen=True)
class TransformStep:
    """One step inside a curated transform sequence.

    Attributes:
        action: Transform verb (e.g. ``"tile"``, ``"vectorize"``, ``"fuse"``).
        parameters: Action-specific parameters.
        rationale: Why this step is taken.
    """

    action: str
    parameters: dict[str, Any]
    rationale: str


# ---------------------------------------------------------------------------
# Composite knowledge entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpWisdom:
    """Optimization knowledge for one op family.

    Aggregates tiling, fusion, layout, backend, and general guidance for a
    single op family such as ``"matmul"`` or ``"conv2d"``.

    Attributes:
        op_family: Canonical name (e.g. ``"matmul"``, ``"conv2d"``).
        tiling_guidance: Per-target tiling recommendations.
        fusion_opportunities: Known-profitable fusion patterns.
        layout_preferences: Per-target layout recommendations.
        backend_guidance: Per-target backend recommendations.
        pitfalls: Common mistakes / things to watch out for.
        performance_bounds: Roofline / theoretical bounds notes.
    """

    op_family: str
    tiling_guidance: list[TilingGuidance]
    fusion_opportunities: list[FusionOpportunity]
    layout_preferences: list[LayoutPreference]
    backend_guidance: list[BackendGuidance]
    pitfalls: list[str]
    performance_bounds: list[str]


@dataclass(frozen=True)
class TargetPattern:
    """Optimization pattern for a hardware target class.

    Attributes:
        target_class: Hardware family (e.g. ``"gpu"``, ``"cpu"``).
        category: Pattern category (e.g. ``"memory_hierarchy"``, ``"parallelism"``).
        pattern_name: Short identifier.
        description: What the pattern achieves.
        implementation_notes: Practical tips for applying the pattern.
        source: Compiler / library the pattern comes from.
        confidence: How reliable this pattern is.
    """

    target_class: str
    category: str
    pattern_name: str
    description: str
    implementation_notes: list[str]
    source: str
    confidence: Confidence


@dataclass(frozen=True)
class TransformRecipe:
    """Curated sequence of transforms for an (op, target) pair.

    Attributes:
        name: Human-readable recipe name.
        op_family: Op family this recipe targets.
        target_class: Hardware family this recipe targets.
        steps: Ordered list of transform steps.
        expected_speedup: Qualitative or quantitative expected gain.
        source: Origin of the recipe.
        confidence: How reliable the recipe is.
    """

    name: str
    op_family: str
    target_class: str
    steps: list[TransformStep]
    expected_speedup: str
    source: str
    confidence: Confidence


@dataclass(frozen=True)
class KernelLibraryWisdom:
    """Insight distilled from a kernel library.

    Attributes:
        library: Library name (e.g. ``"cutlass"``, ``"cudnn"``).
        topic: What aspect the insight covers.
        insight: The actual wisdom.
        conditions: When the insight applies.
        confidence: How reliable it is.
    """

    library: str
    topic: str
    insight: str
    conditions: list[str]
    confidence: Confidence


@dataclass(frozen=True)
class CompilerHeuristic:
    """Heuristic extracted from a production compiler.

    Attributes:
        compiler: Compiler name (e.g. ``"tvm"``, ``"xla"``).
        topic: What the heuristic covers.
        heuristic: The rule itself.
        conditions: When the heuristic applies.
        confidence: How reliable it is.
    """

    compiler: str
    topic: str
    heuristic: str
    conditions: list[str]
    confidence: Confidence


@dataclass(frozen=True)
class AntiPattern:
    """A known-bad optimization pattern to avoid.

    Attributes:
        name: Short identifier for the anti-pattern.
        description: What goes wrong.
        symptoms: Observable signs that the anti-pattern is in play.
        fix: How to avoid or correct it.
        source: Where the lesson was learned.
    """

    name: str
    description: str
    symptoms: list[str]
    fix: str
    source: str


# ---------------------------------------------------------------------------
# KnowledgeBase container
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeBase:
    """Centralized compiler optimization knowledge base.

    Stores structured optimization wisdom from real-world compilers
    and kernel libraries.  Queryable by op family, target class, or topic.

    Attributes:
        op_wisdom: Op-family-keyed optimization wisdom.
        target_patterns: Target-class-keyed optimization patterns.
        transform_recipes: Curated transform sequences.
        kernel_wisdom: Insights from kernel libraries.
        compiler_heuristics: Heuristics from production compilers.
        anti_patterns: Known-bad patterns to avoid.
    """

    op_wisdom: dict[str, OpWisdom] = field(default_factory=dict)
    target_patterns: dict[str, list[TargetPattern]] = field(default_factory=dict)
    transform_recipes: list[TransformRecipe] = field(default_factory=list)
    kernel_wisdom: list[KernelLibraryWisdom] = field(default_factory=list)
    compiler_heuristics: list[CompilerHeuristic] = field(default_factory=list)
    anti_patterns: list[AntiPattern] = field(default_factory=list)

    # -- query helpers -----------------------------------------------------

    def query_op(self, op_family: str) -> OpWisdom | None:
        """Look up optimization wisdom for an op family.

        Args:
            op_family: Canonical op family name (e.g. ``"matmul"``).

        Returns:
            The matching ``OpWisdom`` entry, or ``None`` if not found.
        """
        return self.op_wisdom.get(op_family)

    def query_target(self, target_class: str, category: str | None = None) -> list[TargetPattern]:
        """Retrieve optimization patterns for a target class.

        Args:
            target_class: Hardware family (e.g. ``"gpu"``).
            category: Optional category filter (e.g. ``"memory_hierarchy"``).

        Returns:
            List of matching ``TargetPattern`` entries.
        """
        patterns = self.target_patterns.get(target_class, [])
        if category is not None:
            patterns = [p for p in patterns if p.category == category]
        return patterns

    def query_recipes(
        self,
        op_family: str | None = None,
        target_class: str | None = None,
    ) -> list[TransformRecipe]:
        """Find curated transform recipes matching the given filters.

        Args:
            op_family: Optional op family filter.
            target_class: Optional target class filter.

        Returns:
            List of matching ``TransformRecipe`` entries.
        """
        results = self.transform_recipes
        if op_family is not None:
            results = [r for r in results if r.op_family == op_family]
        if target_class is not None:
            results = [r for r in results if r.target_class == target_class]
        return results

    def query_kernel_wisdom(
        self,
        library: str | None = None,
        topic: str | None = None,
    ) -> list[KernelLibraryWisdom]:
        """Query kernel library insights.

        Args:
            library: Optional library name filter (e.g. ``"cutlass"``).
            topic: Optional topic filter.

        Returns:
            List of matching ``KernelLibraryWisdom`` entries.
        """
        results = self.kernel_wisdom
        if library is not None:
            results = [w for w in results if w.library == library]
        if topic is not None:
            results = [w for w in results if w.topic == topic]
        return results

    def query_heuristics(
        self,
        compiler: str | None = None,
        topic: str | None = None,
    ) -> list[CompilerHeuristic]:
        """Query compiler heuristics.

        Args:
            compiler: Optional compiler name filter (e.g. ``"tvm"``).
            topic: Optional topic filter.

        Returns:
            List of matching ``CompilerHeuristic`` entries.
        """
        results = self.compiler_heuristics
        if compiler is not None:
            results = [h for h in results if h.compiler == compiler]
        if topic is not None:
            results = [h for h in results if h.topic == topic]
        return results

    def query_anti_patterns(self) -> list[AntiPattern]:
        """Return all known anti-patterns.

        Returns:
            List of all ``AntiPattern`` entries.
        """
        return list(self.anti_patterns)

    def summary(self) -> dict[str, int]:
        """Return a summary of knowledge base contents.

        Returns:
            Dictionary mapping category names to entry counts.
        """
        total_target_patterns = sum(len(v) for v in self.target_patterns.values())
        return {
            "op_wisdom": len(self.op_wisdom),
            "target_patterns": total_target_patterns,
            "transform_recipes": len(self.transform_recipes),
            "kernel_wisdom": len(self.kernel_wisdom),
            "compiler_heuristics": len(self.compiler_heuristics),
            "anti_patterns": len(self.anti_patterns),
        }
