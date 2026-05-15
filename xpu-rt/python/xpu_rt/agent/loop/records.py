"""Iteration records for the agentic compilation loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.agent.env import Observation


@dataclass(frozen=True)
class IterationRecord:
    """Record of one optimization iteration."""

    iteration: int
    action_type: str
    target: str
    applied: bool
    cost_before_us: float
    cost_after_us: float
    improvement_pct: float
    reasoning: str


@dataclass(frozen=True)
class CompilationResult:
    """Result of the full agentic compilation loop."""

    initial_cost_us: float
    final_cost_us: float
    total_improvement_pct: float
    iterations_run: int
    iterations_improved: int
    history: list[IterationRecord]
    best_observation: Observation | None = None
    runtime_artifacts: dict[str, Any] = field(default_factory=dict)
