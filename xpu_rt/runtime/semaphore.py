"""Timeline semaphore for cross-device ordering.

A monotonically increasing counter used to synchronize work across devices.
Waiters block until the counter reaches or exceeds a target value, enabling
produce/consume ordering between asynchronous device streams.

Invariants:
    - The counter only moves forward (monotonically non-decreasing).
    - Multiple waiters at different target values are supported.
    - Signal wakes all waiters whose target has been reached.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()


@dataclass
class TimelineSemaphore:
    """Monotonically increasing counter for cross-device ordering.

    Attributes:
        name: Human-readable identifier for logging.
        _value: Current counter value.
    """

    name: str = "timeline"
    _value: int = field(default=0, init=False)
    _condition: asyncio.Condition = field(default_factory=asyncio.Condition, init=False, repr=False)

    @property
    def value(self) -> int:
        """Current counter value."""
        return self._value

    async def signal(self, target: int) -> None:
        """Advance the counter to *target*.

        Args:
            target: New counter value. Must be >= current value.

        Raises:
            ValueError: If *target* is less than the current value.
        """
        async with self._condition:
            if target < self._value:
                raise ValueError(f"TimelineSemaphore({self.name}): cannot go backwards ({self._value} -> {target})")
            self._value = target
            log.debug("timeline.signal", name=self.name, value=target)
            self._condition.notify_all()

    async def wait(self, target: int) -> None:
        """Block until the counter reaches or exceeds *target*.

        Args:
            target: Value to wait for.
        """
        async with self._condition:
            while self._value < target:
                log.debug("timeline.wait", name=self.name, target=target, current=self._value)
                await self._condition.wait()

    def reset(self) -> None:
        """Reset the counter to zero (for test reuse only)."""
        self._value = 0


__all__ = ["TimelineSemaphore"]
