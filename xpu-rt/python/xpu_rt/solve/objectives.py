"""Cost function definitions for solver objectives.

Defines how different optimization objectives (latency, throughput,
memory, energy) are expressed as solver cost terms.

Invariants:
    - Cost functions are composable (weighted sum for multi-objective).
    - Every cost term is traceable to a source (op duration, transfer cost, etc.).

TODO: Implement cost function evaluation from profiled data.
TODO: Support composite objectives (e.g., minimize latency subject to memory cap).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LatencyCost:
    """Latency cost term: total makespan.

    Attributes:
        weight: Weight in composite objective.
    """

    weight: float = 1.0


@dataclass(frozen=True)
class ThroughputCost:
    """Throughput cost term: inverse of sustained throughput.

    Attributes:
        weight: Weight in composite objective.
        batch_size: Batch size for throughput calculation.
    """

    weight: float = 1.0
    batch_size: int = 1


@dataclass(frozen=True)
class MemoryCost:
    """Memory cost term: peak memory usage.

    Attributes:
        weight: Weight in composite objective.
    """

    weight: float = 1.0


@dataclass(frozen=True)
class EnergyCost:
    """Energy cost term: estimated energy consumption.

    Attributes:
        weight: Weight in composite objective.
    """

    weight: float = 1.0


@dataclass(frozen=True)
class CompositeCost:
    """Composite cost combining multiple objectives.

    Attributes:
        terms: List of cost terms with their weights.
    """

    terms: list[LatencyCost | ThroughputCost | MemoryCost | EnergyCost] = field(default_factory=list)


__all__ = ["CompositeCost", "EnergyCost", "LatencyCost", "MemoryCost", "ThroughputCost"]
