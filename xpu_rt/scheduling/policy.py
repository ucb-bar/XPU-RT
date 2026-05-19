"""Solver-selection policy for the compile-time scheduler and memory planner.

Codifies the empirical thresholds measured in `build/experiments/`:

* MOSEK is feasible only up to ~60 partitions for joint placement+ordering
  (Exp 1.5, Exp 7: first MOSEK skip at n=66).
* CP-SAT solves the same problem cleanly up to ~200 partitions within a 30s
  budget (Exp 7b, A3); timeouts past that.
* Greedy is the only viable fallback above 200.

For memory planning, greedy first-fit-decreasing is structurally near-optimal
on real models (Exp 2, M12); MILP only wins when small-tier feasibility is
tight. MILP's ``_canonicalize`` post-pass scales linearly in
bytes/alignment and hangs on offsets above ~64 MiB (M3 finding) — so MILP is
skipped entirely above that byte budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import structlog

log = structlog.get_logger(__name__)


class SolverChoice(StrEnum):
    """Discrete scheduling-solver dispatch labels."""

    MOSEK = "mosek"
    CPSAT = "cpsat"
    GREEDY = "greedy"


class MemoryPlannerChoice(StrEnum):
    """Discrete memory-planner dispatch labels."""

    MILP = "milp"
    GREEDY = "greedy"


@dataclass(frozen=True)
class SchedulerPolicy:
    """Choose a scheduling solver from problem size.

    Attributes:
        mosek_max_partitions: Largest partition count where MOSEK is feasible.
            Default 60, derived from Exp 1.5 / Exp 7 (first MOSEK skip at n=66).
        cpsat_max_partitions: Largest partition count where CP-SAT solves
            within the 30s budget. Default 200, from Exp 7b / A3.
    """

    mosek_max_partitions: int = 60
    cpsat_max_partitions: int = 200

    def choose(self, n_partitions: int, problem_kind: str = "schedule") -> SolverChoice:
        """Pick a solver based on partition count.

        Args:
            n_partitions: Number of partitions in the joint placement+ordering
                problem.
            problem_kind: Reserved for future use (e.g. distinguishing
                placement-only vs joint). Currently informational; passed
                through to logging only.

        Returns:
            The selected :class:`SolverChoice`.
        """
        if n_partitions <= self.mosek_max_partitions:
            choice = SolverChoice.MOSEK
        elif n_partitions <= self.cpsat_max_partitions:
            choice = SolverChoice.CPSAT
        else:
            choice = SolverChoice.GREEDY
        log.debug(
            "scheduler_policy.choose",
            n_partitions=n_partitions,
            problem_kind=problem_kind,
            choice=str(choice),
        )
        return choice

    def reason(self, n_partitions: int) -> str:
        """Return a single-line human-readable rationale for the choice.

        Args:
            n_partitions: Partition count that drove the decision.

        Returns:
            A short audit-log-friendly string mentioning both thresholds and
            the resulting solver.
        """
        choice = self.choose(n_partitions)
        m = self.mosek_max_partitions
        c = self.cpsat_max_partitions
        if choice is SolverChoice.MOSEK:
            return f"n={n_partitions} <= mosek_max({m}) -> MOSEK (cpsat_max={c})"
        if choice is SolverChoice.CPSAT:
            return f"n={n_partitions} > mosek_max({m}), n <= cpsat_max({c}) -> CPSAT"
        return f"n={n_partitions} > cpsat_max({c}) (mosek_max={m}) -> GREEDY"


@dataclass(frozen=True)
class MemoryPlannerPolicy:
    """Choose between MILP and greedy memory planning.

    Attributes:
        milp_tier_tightness_threshold: If projected small-tier usage / capacity
            exceeds this, MILP's feasibility win justifies its cost.
        milp_canonicalize_max_bytes: Skip MILP entirely above this total-bytes
            budget — MILP's ``_canonicalize`` post-pass hangs at larger offsets
            (M3 finding).
    """

    milp_tier_tightness_threshold: float = 0.85
    milp_canonicalize_max_bytes: int = 64 * 2**20

    # Past this many buffers, MILP search time dominates the feasibility win
    # (matches the scheduler's CP-SAT ceiling — same problem-size regime).
    _milp_max_buffers: int = 200

    def choose(
        self,
        n_buffers: int,
        projected_tier_usage_ratio: float,
        projected_total_bytes: int,
    ) -> MemoryPlannerChoice:
        """Pick a memory planner.

        Args:
            n_buffers: Number of live buffers requiring placement.
            projected_tier_usage_ratio: Projected ``peak_usage / capacity`` for
                the tightest memory tier (typically SRAM / on-chip).
            projected_total_bytes: Projected total allocation extent in bytes
                (used to gate MILP's canonicalize post-pass).

        Returns:
            The selected :class:`MemoryPlannerChoice`.
        """
        if projected_total_bytes > self.milp_canonicalize_max_bytes:
            choice = MemoryPlannerChoice.GREEDY
        elif (
            projected_tier_usage_ratio >= self.milp_tier_tightness_threshold
            and n_buffers <= self._milp_max_buffers
        ):
            choice = MemoryPlannerChoice.MILP
        else:
            choice = MemoryPlannerChoice.GREEDY
        log.debug(
            "memory_planner_policy.choose",
            n_buffers=n_buffers,
            projected_tier_usage_ratio=projected_tier_usage_ratio,
            projected_total_bytes=projected_total_bytes,
            choice=str(choice),
        )
        return choice

    def reason(
        self,
        n_buffers: int,
        projected_tier_usage_ratio: float,
        projected_total_bytes: int,
    ) -> str:
        """Return a single-line human-readable rationale for the choice.

        Args:
            n_buffers: Number of live buffers.
            projected_tier_usage_ratio: Projected ``peak / capacity`` for the
                tightest tier.
            projected_total_bytes: Projected total allocation extent.

        Returns:
            A short audit-log-friendly string.
        """
        choice = self.choose(
            n_buffers, projected_tier_usage_ratio, projected_total_bytes
        )
        cap = self.milp_canonicalize_max_bytes
        thr = self.milp_tier_tightness_threshold
        if projected_total_bytes > cap:
            return (
                f"total_bytes={projected_total_bytes} > canonicalize_max({cap}) "
                f"-> GREEDY (canonicalize guard)"
            )
        if choice is MemoryPlannerChoice.MILP:
            return (
                f"ratio={projected_tier_usage_ratio:.3f} >= tight({thr}) and "
                f"n_buffers={n_buffers} <= {self._milp_max_buffers} -> MILP"
            )
        return (
            f"ratio={projected_tier_usage_ratio:.3f} < tight({thr}) or "
            f"n_buffers={n_buffers} > {self._milp_max_buffers} -> GREEDY"
        )
