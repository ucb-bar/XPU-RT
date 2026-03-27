"""Global priority scheduler with per-device ready queues.

Implements the Level-2 (runtime) scheduling layer described in the runtime
``__init__.py``:  admission control, priority ordering, and cooperative
preemption at partition boundaries.

Workloads are submitted with a :class:`Priority` and placed into per-device
ready queues.  :meth:`PriorityScheduler.dequeue` returns the highest-priority
workload for a given device.  Cooperative preemption is modeled as a
checkpoint: at each partition boundary the running workload checks whether a
higher-priority item has arrived and yields if so.

Invariants:
    - Dequeue returns workloads strictly in priority order (HIGH > NORMAL > LOW).
    - Within the same priority, FIFO ordering is preserved.
    - Preemption is cooperative (checked at partition boundaries only).
"""

from __future__ import annotations

import enum
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.runtime.planner import ExecutionPlan

log = structlog.get_logger()


class Priority(enum.IntEnum):
    """Workload priority levels (higher value = higher priority)."""

    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass
class Workload:
    """A scheduled unit of work.

    Attributes:
        workload_id: Unique identifier.
        plan: The execution plan to run.
        priority: Scheduling priority.
        submitted_at: Monotonic timestamp when the workload was submitted.
        metadata: Arbitrary caller metadata.
    """

    workload_id: str
    plan: ExecutionPlan
    priority: Priority = Priority.NORMAL
    submitted_at: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PriorityScheduler:
    """Global scheduler with per-device priority queues.

    Attributes:
        num_devices: Number of devices managed by this scheduler.
    """

    num_devices: int = 1
    _queues: dict[int, list[deque[Workload]]] = field(
        default_factory=dict, init=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        # Per device: one deque per priority level (indexed by Priority value)
        for device in range(self.num_devices):
            self._queues[device] = [
                deque() for _ in range(len(Priority))
            ]

    def submit(self, workload: Workload, device: int = 0) -> None:
        """Submit a workload to a device's ready queue.

        Args:
            workload: The workload to schedule.
            device: Target device index.

        Raises:
            ValueError: If *device* is out of range.
        """
        if device < 0 or device >= self.num_devices:
            raise ValueError(
                f"Device {device} out of range [0, {self.num_devices})"
            )

        with self._lock:
            self._queues[device][workload.priority].append(workload)

        log.debug(
            "scheduler.submit",
            workload_id=workload.workload_id,
            priority=workload.priority.name,
            device=device,
        )

    def dequeue(self, device: int = 0) -> Workload | None:
        """Dequeue the highest-priority workload for *device*.

        Returns ``None`` if the device queue is empty.

        Args:
            device: Device index to dequeue from.

        Returns:
            The next :class:`Workload`, or ``None``.
        """
        if device < 0 or device >= self.num_devices:
            raise ValueError(
                f"Device {device} out of range [0, {self.num_devices})"
            )

        with self._lock:
            # Iterate from highest to lowest priority
            for prio in reversed(Priority):
                q = self._queues[device][prio]
                if q:
                    workload = q.popleft()
                    log.debug(
                        "scheduler.dequeue",
                        workload_id=workload.workload_id,
                        priority=workload.priority.name,
                        device=device,
                    )
                    return workload
        return None

    def pending_count(self, device: int = 0) -> int:
        """Return the number of pending workloads on *device*.

        Args:
            device: Device index.

        Returns:
            Total pending workload count across all priority levels.
        """
        with self._lock:
            return sum(len(q) for q in self._queues[device])

    def should_preempt(self, current_priority: Priority, device: int = 0) -> bool:
        """Check whether a higher-priority workload is waiting.

        Called at partition boundaries for cooperative preemption.

        Args:
            current_priority: Priority of the currently running workload.
            device: Device index.

        Returns:
            ``True`` if a workload with strictly higher priority is pending.
        """
        with self._lock:
            for prio in reversed(Priority):
                if prio <= current_priority:
                    break
                if self._queues[device][prio]:
                    log.debug(
                        "scheduler.preempt",
                        current=current_priority.name,
                        preempting=prio.name,
                        device=device,
                    )
                    return True
        return False

    def drain(self, device: int = 0) -> list[Workload]:
        """Remove and return all pending workloads for *device* in priority order.

        Args:
            device: Device index.

        Returns:
            List of workloads, highest priority first.
        """
        result: list[Workload] = []
        with self._lock:
            for prio in reversed(Priority):
                q = self._queues[device][prio]
                while q:
                    result.append(q.popleft())
        return result


__all__ = ["Priority", "PriorityScheduler", "Workload"]
