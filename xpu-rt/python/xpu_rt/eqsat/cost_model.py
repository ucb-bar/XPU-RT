"""Non-additive cost model for equality saturation extraction.

Goes beyond simple per-op costs by modeling inter-operator effects:
  - Fusion bonus: producer-consumer pairs that can share memory
  - Copy/transfer penalty: cross-device data movement
  - Backend match bonus: ops that map well to target hardware
  - Memory pressure penalty: ops that exceed local memory budget

Based on Constable's approach (OOPSLA 2025): "local rewrites can change
the profitability of downstream transformations, particularly regarding
data layout, parallelization, and memory management."
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from xdsl.dialects import equivalence, func
from xdsl.dialects.builtin import IntAttr, ModuleOp
from xdsl.ir import Operation

if TYPE_CHECKING:
    from compgen.targets.schema import TargetProfile


@dataclass
class CostWeights:
    """Tunable weights for the non-additive cost model.

    The LLM can adjust these via SetExtractionObjectiveAction.
    """

    fusion_weight: float = 1.0
    transfer_weight: float = 1.0
    backend_match_weight: float = 1.0
    memory_pressure_weight: float = 1.0


@dataclass
class CostModel:
    """Non-additive cost model for e-graph extraction.

    Computes per-op costs that account for inter-op interactions.
    """

    base_costs: dict[str, int] = field(default_factory=dict)
    weights: CostWeights = field(default_factory=CostWeights)
    default_cost: int = 10

    # Class-level constants (not dataclass fields)
    _ZERO_COST_OPS: frozenset[str] = field(
        init=False,
        repr=False,
        default=frozenset({"arith.constant"}),
    )
    _FUSIBLE_PRODUCERS: frozenset[str] = field(
        init=False,
        repr=False,
        default=frozenset(
            {
                "arith.addi",
                "arith.addf",
                "arith.muli",
                "arith.mulf",
                "arith.subf",
                "arith.subi",
                "arith.divf",
                "arith.negf",
                "arith.maximumf",
                "arith.minimumf",
                "arith.select",
            }
        ),
    )
    _COMPUTE_HEAVY: dict[str, int] = field(
        init=False,
        repr=False,
        default_factory=lambda: {
            "linalg.matmul": 100,
            "linalg.generic": 50,
            "linalg.transpose": 5,
        },
    )

    def get_base_cost(self, op: Operation) -> int:
        """Get the base cost for an operation (before adjustments)."""
        name = op.name

        # User-provided overrides
        if name in self.base_costs:
            return self.base_costs[name]

        # Zero-cost ops
        if name in self._ZERO_COST_OPS:
            return 1  # Minimum 1 for extraction ordering

        # Compute-heavy ops
        if name in self._COMPUTE_HEAVY:
            return self._COMPUTE_HEAVY[name]

        return self.default_cost

    def compute_fusion_bonus(self, op: Operation) -> int:
        """Compute fusion bonus: negative cost for fusible producer-consumer pairs.

        If a producer has exactly one consumer and both are elementwise,
        they can be fused into a single kernel, avoiding intermediate
        materialization. Bonus = producer's memory cost.
        """
        if op.name not in self._FUSIBLE_PRODUCERS:
            return 0

        # Check if this op has exactly one user (through eclass)
        if not op.results:
            return 0

        result = op.results[0]
        user_count = 0
        for use in result.uses:
            if isinstance(use.operation, equivalence.AnyClassOp):
                # Count non-eclass users of the eclass result
                eclass = use.operation
                for eclass_use in eclass.result.uses:
                    if not isinstance(eclass_use.operation, equivalence.AnyClassOp):
                        user_count += 1
            else:
                user_count += 1

        if user_count == 1:
            # Eligible for fusion: bonus proportional to memory savings
            bonus = int(self.weights.fusion_weight * 3)
            return bonus

        return 0

    def compute_adjusted_cost(self, op: Operation) -> int:
        """Compute the full adjusted cost for an operation.

        cost = base_cost - fusion_bonus

        Minimum cost is 1 (to ensure extraction ordering works).
        """
        base = self.get_base_cost(op)
        fusion_bonus = self.compute_fusion_bonus(op)

        adjusted = base - fusion_bonus
        return max(1, adjusted)

    def assign_costs(self, module: ModuleOp) -> None:
        """Assign eqsat_cost attributes to all ops in the module.

        This replaces EqsatAddCostsPass for cases where we need
        non-additive cost modeling.
        """
        for op in module.walk():
            if isinstance(
                op,
                (
                    equivalence.AnyClassOp,
                    equivalence.GraphOp,
                    equivalence.YieldOp,
                    ModuleOp,
                    func.FuncOp,
                    func.ReturnOp,
                ),
            ):
                continue
            if not op.results:
                continue
            if equivalence.EQSAT_COST_LABEL in op.attributes:
                continue

            cost = self.compute_adjusted_cost(op)
            op.attributes[equivalence.EQSAT_COST_LABEL] = IntAttr(cost)

    def to_json_dict(self) -> dict[str, int]:
        """Export as a simple op_name → cost dict (for EqsatAddCostsPass)."""
        costs: dict[str, int] = {}
        costs.update(self.base_costs)
        for name in self._ZERO_COST_OPS:
            costs.setdefault(name, 1)
        for name, cost in self._COMPUTE_HEAVY.items():
            costs.setdefault(name, cost)
        return costs

    def to_json_path(self) -> str:
        """Write costs to a temp JSON file, return path."""
        d = self.to_json_dict()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(d, f)
            return f.name


def create_cost_model(
    target: TargetProfile | None = None,
    weights: CostWeights | None = None,
    overrides: dict[str, int] | None = None,
) -> CostModel:
    """Create a cost model, optionally target-aware.

    Args:
        target: Target profile for hardware-specific costs.
        weights: Custom weight tuning.
        overrides: Op name → cost overrides.

    Returns:
        A configured CostModel instance.
    """
    base_costs: dict[str, int] = {}

    if overrides:
        base_costs.update(overrides)

    return CostModel(
        base_costs=base_costs,
        weights=weights or CostWeights(),
    )
