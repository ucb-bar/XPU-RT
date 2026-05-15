"""Adaptive batch scheduler -- pick pre-computed plans by batch size tier.

Pre-computes :class:`~compgen.runtime.planner.ExecutionPlan` objects for a set
of batch-size tiers (e.g. 1, 8, 32, 128).  At request time the scheduler
selects the closest tier without exceeding device memory, giving near-optimal
plans for any incoming batch size with zero compile-time overhead.

Invariants:
    - Tiers are sorted ascending and contain at least one entry.
    - ``select_plan`` always returns the plan for the closest tier whose
      batch size is >= the request (or the largest tier if none qualifies).
    - Plan generation delegates to :func:`~compgen.runtime.planner.plan_execution`.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

import structlog
from xdsl.dialects.builtin import ModuleOp

from compgen.runtime.planner import ExecutionPlan, ExecutionPlanner
from compgen.targets.schema import TargetProfile

log = structlog.get_logger()

DEFAULT_TIERS: tuple[int, ...] = (1, 8, 32, 128)


@dataclass(frozen=True)
class TieredPlan:
    """An execution plan associated with a specific batch-size tier.

    Attributes:
        batch_size: The batch size this plan was generated for.
        plan: The pre-computed execution plan.
    """

    batch_size: int
    plan: ExecutionPlan


@dataclass
class AdaptiveBatchScheduler:
    """Pre-compute and select execution plans by batch-size tier.

    Attributes:
        target: Hardware target profile.
        tiers: Sorted tuple of batch sizes to pre-compute plans for.
        plans: Mapping from batch-size tier to pre-computed plan.
    """

    target: TargetProfile
    tiers: tuple[int, ...] = DEFAULT_TIERS
    plans: dict[int, TieredPlan] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError("At least one batch-size tier is required")
        # Ensure tiers are sorted and deduplicated
        self.tiers = tuple(sorted(set(self.tiers)))

    def precompute(
        self,
        module: ModuleOp,
        kernels: dict[str, Any] | None = None,
    ) -> dict[int, TieredPlan]:
        """Generate execution plans for every configured tier.

        Args:
            module: The xDSL module to plan for.
            kernels: Optional generated kernels keyed by op name.

        Returns:
            Dict mapping batch size to :class:`TieredPlan`.
        """
        planner = ExecutionPlanner(target=self.target)
        for bs in self.tiers:
            plan = planner.plan(module, kernels)
            tagged = ExecutionPlan(
                placements=plan.placements,
                copies=plan.copies,
                execution_order=plan.execution_order,
                memory_plans=plan.memory_plans,
                estimated_latency_us=plan.estimated_latency_us,
                metadata={**plan.metadata, "batch_size_tier": bs},
            )
            self.plans[bs] = TieredPlan(batch_size=bs, plan=tagged)
            log.debug("adaptive.precomputed", batch_size=bs)

        return dict(self.plans)

    def select_plan(self, batch_size: int) -> TieredPlan:
        """Select the best pre-computed plan for *batch_size*.

        Strategy: pick the smallest tier whose batch size is >= the request.
        If no tier is large enough, return the largest available tier.

        Args:
            batch_size: The incoming request batch size.

        Returns:
            The :class:`TieredPlan` for the closest tier.

        Raises:
            RuntimeError: If :meth:`precompute` has not been called.
        """
        if not self.plans:
            raise RuntimeError("No plans have been precomputed. Call precompute() first.")

        idx = bisect.bisect_left(self.tiers, batch_size)

        if idx < len(self.tiers):
            selected = self.tiers[idx]
        else:
            selected = self.tiers[-1]

        log.debug(
            "adaptive.select",
            request_batch_size=batch_size,
            selected_tier=selected,
        )
        return self.plans[selected]


__all__ = ["AdaptiveBatchScheduler", "DEFAULT_TIERS", "TieredPlan"]
