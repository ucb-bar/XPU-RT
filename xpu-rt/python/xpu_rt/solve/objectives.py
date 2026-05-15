"""Cost function definitions for solver objectives.

Defines how different optimization objectives (latency, throughput,
memory, energy) are expressed as solver cost terms.

Invariants:
    - Cost functions are composable (weighted sum for multi-objective).
    - Every cost term is traceable to a source (op duration, transfer cost, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compgen.solve.partition import Partition


@dataclass(frozen=True)
class LatencyCost:
    """Latency cost term: total makespan.

    Attributes:
        weight: Weight in composite objective.
    """

    weight: float = 1.0

    def evaluate(self, partitions: list[Partition], cost_data: dict[str, float] | None = None) -> float:
        """Evaluate total latency cost across partitions."""
        total = sum(p.estimated_cost_us for p in partitions)
        return total * self.weight


@dataclass(frozen=True)
class ThroughputCost:
    """Throughput cost term: inverse of sustained throughput.

    Attributes:
        weight: Weight in composite objective.
        batch_size: Batch size for throughput calculation.
    """

    weight: float = 1.0
    batch_size: int = 1

    def evaluate(self, partitions: list[Partition], cost_data: dict[str, float] | None = None) -> float:
        """Evaluate throughput cost (lower is better)."""
        total_latency = sum(p.estimated_cost_us for p in partitions)
        if total_latency <= 0:
            return 0.0
        # Inverse throughput: higher latency → higher cost
        return (total_latency / max(self.batch_size, 1)) * self.weight


@dataclass(frozen=True)
class MemoryCost:
    """Memory cost term: peak memory usage.

    Attributes:
        weight: Weight in composite objective.
    """

    weight: float = 1.0

    def evaluate(self, partitions: list[Partition], cost_data: dict[str, float] | None = None) -> float:
        """Evaluate peak memory cost across partitions."""
        if not partitions:
            return 0.0
        peak = max(p.memory_bytes for p in partitions)
        return peak * self.weight


@dataclass(frozen=True)
class EnergyCost:
    """Energy cost term: estimated energy consumption.

    Attributes:
        weight: Weight in composite objective.
        energy_per_us: Energy coefficient (joules per microsecond of compute).
    """

    weight: float = 1.0
    energy_per_us: float = 1e-6

    def evaluate(self, partitions: list[Partition], cost_data: dict[str, float] | None = None) -> float:
        """Evaluate energy cost (proportional to total compute time)."""
        total_us = sum(p.estimated_cost_us for p in partitions)
        return total_us * self.energy_per_us * self.weight


CostTerm = LatencyCost | ThroughputCost | MemoryCost | EnergyCost


@dataclass(frozen=True)
class CompositeCost:
    """Composite cost combining multiple objectives.

    Attributes:
        terms: List of cost terms with their weights.
    """

    terms: list[CostTerm] = field(default_factory=list)

    def evaluate(self, partitions: list[Partition], cost_data: dict[str, float] | None = None) -> float:
        """Evaluate composite cost as weighted sum of all terms."""
        return sum(term.evaluate(partitions, cost_data) for term in self.terms)

    @classmethod
    def from_learned(cls, weights: dict[str, float]) -> CompositeCost:
        """Create a CompositeCost from learned weight dict.

        Args:
            weights: Dict with keys like 'latency_weight', 'throughput_weight',
                    'memory_weight', 'energy_weight', 'fusion_weight', etc.

        Returns:
            CompositeCost with terms weighted by learned values.
        """
        terms: list[CostTerm] = []
        terms.append(LatencyCost(weight=weights.get("latency_weight", weights.get("fusion_weight", 1.0))))
        if "throughput_weight" in weights:
            terms.append(ThroughputCost(weight=weights["throughput_weight"]))
        if "memory_weight" in weights or "transfer_weight" in weights:
            terms.append(MemoryCost(weight=weights.get("memory_weight", weights.get("transfer_weight", 0.0))))
        if "energy_weight" in weights:
            terms.append(EnergyCost(weight=weights["energy_weight"]))
        return cls(terms=terms)


__all__ = ["CompositeCost", "CostTerm", "EnergyCost", "LatencyCost", "MemoryCost", "ThroughputCost"]
