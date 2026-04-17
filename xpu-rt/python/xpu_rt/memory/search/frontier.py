"""Graph frontier for cross-task search (KernelEvolve-style).

Manages which optimization tasks to expand next using a UCB1 bandit
policy. Prioritizes tasks with highest expected value while maintaining
exploration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import structlog

from compgen.memory.store import CompilerMemory
from compgen.memory.search.task import SearchTask

log = structlog.get_logger()


@dataclass
class FrontierEntry:
    """One task in the frontier with bandit statistics."""

    task: SearchTask
    pulls: int = 0
    total_reward: float = 0.0
    best_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls > 0 else 0.0


class SearchFrontier:
    """Manages which tasks to expand next using UCB1.

    The frontier is the global controller that decides, across ALL
    pending optimization tasks, which one to spend compute on next.
    This is the KernelEvolve-style graph frontier.

    Attributes:
        memory: CompilerMemory for persistence.
        exploration_weight: UCB1 exploration parameter (c).
    """

    def __init__(
        self,
        memory: CompilerMemory,
        exploration_weight: float = 1.414,
    ) -> None:
        self.memory = memory
        self.exploration_weight = exploration_weight
        self._entries: list[FrontierEntry] = []
        self._total_pulls = 0

    def add_task(self, task: SearchTask) -> None:
        """Add a task to the frontier."""
        self._entries.append(FrontierEntry(task=task))

    def next_task(self) -> SearchTask | None:
        """Select the next task to expand using UCB1.

        Returns the task with highest UCB1 score, or None if
        the frontier is empty.
        """
        if not self._entries:
            return None

        # Unpulled tasks get infinite priority
        for entry in self._entries:
            if entry.pulls == 0:
                return entry.task

        # UCB1 selection
        best_score = -float("inf")
        best_entry: FrontierEntry | None = None

        for entry in self._entries:
            ucb = entry.mean_reward + self.exploration_weight * math.sqrt(
                math.log(self._total_pulls) / entry.pulls
            )
            if ucb > best_score:
                best_score = ucb
                best_entry = entry

        return best_entry.task if best_entry else None

    def update(self, task_id: str, reward: float) -> None:
        """Update frontier statistics after expanding a task.

        Args:
            task_id: The task that was expanded.
            reward: The reward obtained (higher = better).
        """
        for entry in self._entries:
            if entry.task.task_id == task_id:
                entry.pulls += 1
                entry.total_reward += reward
                entry.best_reward = max(entry.best_reward, reward)
                self._total_pulls += 1
                return

    def remove_task(self, task_id: str) -> None:
        """Remove a completed/retired task from the frontier."""
        self._entries = [e for e in self._entries if e.task.task_id != task_id]

    @property
    def size(self) -> int:
        """Number of tasks in the frontier."""
        return len(self._entries)

    def summary(self) -> list[dict[str, float | str]]:
        """Get a summary of frontier state for logging."""
        return [
            {
                "task_id": e.task.task_id,
                "kind": e.task.kind.value,
                "pulls": e.pulls,
                "mean_reward": round(e.mean_reward, 4),
                "best_reward": round(e.best_reward, 4),
            }
            for e in sorted(self._entries, key=lambda x: x.mean_reward, reverse=True)
        ]


__all__ = ["SearchFrontier"]
