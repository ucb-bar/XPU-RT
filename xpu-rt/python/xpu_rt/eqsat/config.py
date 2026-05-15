"""EqSat configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EqSatConfig:
    """Configuration for the equality saturation pass.

    Attributes:
        max_iterations: Maximum rewrite iterations before stopping.
        segment_threshold: Max non-blackboxed ops per segment (τ).
        default_cost: Default per-op cost when no cost model entry exists.
        enable_algebraic: Enable algebraic rewrite rules.
        enable_layout: Enable layout normalization rules.
        enable_fusion: Enable fusion-enabling rules.
        cost_file: Optional path to JSON cost file for EqsatAddCostsPass.
    """

    max_iterations: int = 10
    segment_threshold: int = 200
    default_cost: int = 1
    enable_algebraic: bool = True
    enable_layout: bool = True
    enable_fusion: bool = True
    cost_file: str | None = None
    rule_categories: tuple[str, ...] = ("algebraic",)
    custom_rules: list[object] = field(default_factory=list)
