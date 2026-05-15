"""Ray-based transport for distributed inter-node communication.

Uses ``ray.util.queue.Queue`` for message passing between Ray actors/tasks
on different nodes.  Only imported when explicitly requested via
``create_transport("ray")``.

Invariants:
    - Ray is imported at call time, not at module level (optional dep).
    - Falls back to a clear ``ImportError`` if Ray is not installed.
    - Behaves like ``LocalTransport`` for testing when Ray is local.
"""

from __future__ import annotations

from typing import Any

import structlog

from compgen.runtime.transport import TransportMessage

log = structlog.get_logger()

_INSTALL_MSG = "Ray is required for RayTransport. Install with: pip install 'compgen[ray]'"


def _require_ray() -> Any:
    try:
        import ray

        return ray
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc


class RayTransport:
    """Transport using Ray's distributed queue for cross-node messaging.

    At the Python level this wraps ``ray.util.queue.Queue``.  The real
    value is that messages can flow between Ray workers on different
    machines transparently.
    """

    def __init__(self) -> None:
        self._open_flag = False
        self._queue: Any = None  # ray.util.queue.Queue
        self._max_depth: int = 0

    @property
    def name(self) -> str:
        return "ray"

    @property
    def is_open(self) -> bool:
        return self._open_flag

    def open(self, **kwargs: Any) -> None:
        """Open the Ray transport.

        Args:
            max_depth: Maximum queue depth (0 = unbounded).
            actor_name: Optional named actor for the queue endpoint.
        """
        ray = _require_ray()
        if not ray.is_initialized():
            ray.init(namespace="compgen")

        from ray.util.queue import Queue

        self._max_depth = kwargs.get("max_depth", 0)
        maxsize = self._max_depth if self._max_depth > 0 else 0
        self._queue = Queue(maxsize=maxsize)
        self._open_flag = True
        log.debug("transport.ray.open", max_depth=self._max_depth)

    def close(self) -> None:
        self._open_flag = False
        self._queue = None
        log.debug("transport.ray.close")

    def send(self, msg: TransportMessage, timeout_us: float | None = None) -> bool:
        if not self._open_flag or self._queue is None:
            return False
        try:
            timeout_s = timeout_us / 1_000_000 if timeout_us is not None else None
            self._queue.put(msg, timeout=timeout_s)
            return True
        except Exception:
            return False

    def recv(self, timeout_us: float | None = None) -> TransportMessage | None:
        if not self._open_flag or self._queue is None:
            return None
        try:
            timeout_s = timeout_us / 1_000_000 if timeout_us is not None else None
            return self._queue.get(timeout=timeout_s)
        except Exception:
            return None

    def barrier(self) -> None:
        # Single-queue transport: barrier is a no-op
        pass


__all__ = ["RayTransport"]
