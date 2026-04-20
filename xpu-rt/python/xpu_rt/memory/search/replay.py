"""Replay buffer for search trajectories (L0/L1 memory).

Records every step of every search episode so future tasks can
replay what worked before. This is the KernelBlaster-style persistent
replay component.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from compgen.memory.schema import EpisodeStep
from compgen.memory.store import CompilerMemory

log = structlog.get_logger()


@dataclass(frozen=True)
class Trajectory:
    """A complete search trajectory for one task.

    Attributes:
        task_id: The task these steps belong to.
        steps: Ordered list of episode steps.
        total_reward: Sum of all step rewards.
        best_reward: Maximum single-step reward.
    """

    task_id: str
    steps: list[EpisodeStep] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    @property
    def best_reward(self) -> float:
        return max((s.reward for s in self.steps), default=0.0)

    @property
    def length(self) -> int:
        return len(self.steps)


class ReplayBuffer:
    """Persistent replay buffer backed by CompilerMemory.

    Records search trajectories and retrieves the best past
    trajectories for similar tasks.

    Attributes:
        memory: The unified CompilerMemory.
    """

    def __init__(self, memory: CompilerMemory) -> None:
        self.memory = memory

    def record_step(
        self,
        task_id: str,
        action: str,
        reward: float,
        candidate_id: str = "",
        step_number: int = 0,
        metadata: dict[str, str] | None = None,
    ) -> EpisodeStep:
        """Record one step in a search episode."""
        return self.memory.record_episode_step(
            task_id=task_id,
            action=action,
            reward=reward,
            candidate_id=candidate_id,
            step_number=step_number,
            metadata=metadata,
        )

    def replay(self, task_id: str) -> Trajectory:
        """Replay all steps for a task."""
        steps = self.memory.replay_task(task_id)
        return Trajectory(task_id=task_id, steps=steps)

    def best_trajectory_for_kind(
        self,
        task_kind: str,
        top_k: int = 3,
    ) -> list[Trajectory]:
        """Find the best trajectories for a task kind.

        Args:
            task_kind: The ObjectKind value (e.g., "kernel", "pass").
            top_k: Number of trajectories to return.

        Returns:
            Trajectories sorted by total reward (best first).
        """
        # Get all tasks of this kind
        rows = self.memory.db.fetchall("SELECT task_id FROM tasks WHERE task_kind = ?", (task_kind,))

        trajectories: list[Trajectory] = []
        for row in rows:
            traj = self.replay(row["task_id"])
            if traj.length > 0:
                trajectories.append(traj)

        trajectories.sort(key=lambda t: t.total_reward, reverse=True)
        return trajectories[:top_k]


__all__ = ["ReplayBuffer", "Trajectory"]
