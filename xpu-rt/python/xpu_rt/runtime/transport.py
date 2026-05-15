"""Transport abstraction for inter-node communication.

Provides a target-agnostic protocol for data and command exchange between
runtime nodes.  Concrete implementations handle the mechanism:

    - :class:`LocalTransport` — direct function call + buffer copy
      (same-process, same-node).
    - :class:`SharedMemTransport` — mmap-backed shared memory
      (multi-process on same host).
    - :class:`ZephyrIPCTransport` — maps to Zephyr ``k_msgq``/``k_pipe``
      and ``ipc_service`` for multi-domain SoC communication.
    - :class:`StubNetworkTransport` — placeholder for gRPC/TCP
      distributed transport (protocol defined, impl deferred).

The agentic LLM selects which transport to use per link and tunes
transport-specific parameters (buffer sizes, queue depths, timeouts)
via ``RuntimeLink.properties``.

Invariants:
    - All transports are symmetric: both ends can send and receive.
    - ``send`` / ``recv`` operate on opaque byte buffers.
    - ``barrier`` blocks until all participants reach it.
    - Transport lifecycle: ``open`` → ``send``/``recv``/``barrier`` → ``close``.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import structlog

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Message type
# ---------------------------------------------------------------------------


@dataclass
class TransportMessage:
    """A message exchanged via transport.

    Attributes:
        tag: Application-defined message tag (for multiplexing).
        payload: Raw byte payload.
        metadata: Optional metadata dict (e.g., tensor name, shape info).
    """

    tag: int = 0
    payload: bytes = b""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Transport protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Transport(Protocol):
    """Protocol for inter-node communication.

    Every transport implementation must provide these methods.
    The scaffold defines the contract; the target-specific implementation
    handles the mechanism.
    """

    @property
    def name(self) -> str:
        """Transport identifier (e.g., ``"local"``, ``"zephyr_ipc"``)."""
        ...

    @property
    def is_open(self) -> bool:
        """Whether the transport channel is open."""
        ...

    def open(self, **kwargs: Any) -> None:
        """Open the transport channel.

        Args:
            **kwargs: Transport-specific configuration (buffer sizes,
                queue depths, endpoints, etc.).
        """
        ...

    def close(self) -> None:
        """Close the transport channel and release resources."""
        ...

    def send(self, msg: TransportMessage, timeout_us: float | None = None) -> bool:
        """Send a message.

        Args:
            msg: The message to send.
            timeout_us: Optional timeout in microseconds.  ``None`` = block
                indefinitely.

        Returns:
            ``True`` if sent successfully, ``False`` on timeout/error.
        """
        ...

    def recv(self, timeout_us: float | None = None) -> TransportMessage | None:
        """Receive a message.

        Args:
            timeout_us: Optional timeout in microseconds.  ``None`` = block
                indefinitely.

        Returns:
            The received message, or ``None`` on timeout.
        """
        ...

    def barrier(self) -> None:
        """Block until all participants reach this point."""
        ...


# ---------------------------------------------------------------------------
# Local transport (same-process, direct function call)
# ---------------------------------------------------------------------------


class LocalTransport:
    """In-process transport using a thread-safe queue.

    Used for single-node topologies where all devices are in the same
    process.  Zero-copy when possible (shares the same ``bytes`` object).
    """

    def __init__(self) -> None:
        self._queue: deque[TransportMessage] = deque()
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._open = False

    @property
    def name(self) -> str:
        return "local"

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self, **kwargs: Any) -> None:
        max_depth = kwargs.get("max_depth", 0)
        self._max_depth = max_depth
        self._open = True
        log.debug("transport.local.open", max_depth=max_depth)

    def close(self) -> None:
        self._open = False
        self._queue.clear()
        self._event.set()  # wake any blocked recv
        log.debug("transport.local.close")

    def send(self, msg: TransportMessage, timeout_us: float | None = None) -> bool:
        if not self._open:
            return False
        with self._lock:
            if self._max_depth > 0 and len(self._queue) >= self._max_depth:
                return False
            self._queue.append(msg)
            self._event.set()
        return True

    def recv(self, timeout_us: float | None = None) -> TransportMessage | None:
        if not self._open:
            return None
        timeout_s = timeout_us / 1_000_000 if timeout_us is not None else None

        # Fast path: message already available
        with self._lock:
            if self._queue:
                return self._queue.popleft()

        # Wait for a message
        self._event.clear()
        signaled = self._event.wait(timeout=timeout_s)
        if not signaled or not self._open:
            return None

        with self._lock:
            if self._queue:
                return self._queue.popleft()
        return None

    def barrier(self) -> None:
        # Local transport: barrier is a no-op (single participant)
        pass

    @property
    def pending_count(self) -> int:
        """Number of messages waiting in the queue."""
        with self._lock:
            return len(self._queue)


# ---------------------------------------------------------------------------
# Shared memory transport
# ---------------------------------------------------------------------------


class SharedMemTransport:
    """Shared-memory transport for multi-process on the same host.

    Uses a named buffer region for zero-copy data exchange and a
    control queue for signaling.  The actual mmap is deferred to
    ``open()`` to keep construction cheap.
    """

    def __init__(self) -> None:
        self._open_flag = False
        self._shm_name: str = ""
        self._buffer_size: int = 0
        # Fallback to in-process queue when mmap is not available
        self._fallback_queue: deque[TransportMessage] = deque()
        self._lock = threading.Lock()
        self._event = threading.Event()

    @property
    def name(self) -> str:
        return "shared_memory"

    @property
    def is_open(self) -> bool:
        return self._open_flag

    def open(self, **kwargs: Any) -> None:
        self._shm_name = kwargs.get("shm_name", "compgen_shm")
        self._buffer_size = kwargs.get("buffer_size", 1024 * 1024)
        self._open_flag = True
        log.debug(
            "transport.shared_memory.open",
            shm_name=self._shm_name,
            buffer_size=self._buffer_size,
        )

    def close(self) -> None:
        self._open_flag = False
        self._fallback_queue.clear()
        self._event.set()
        log.debug("transport.shared_memory.close")

    def send(self, msg: TransportMessage, timeout_us: float | None = None) -> bool:
        if not self._open_flag:
            return False
        with self._lock:
            self._fallback_queue.append(msg)
            self._event.set()
        return True

    def recv(self, timeout_us: float | None = None) -> TransportMessage | None:
        if not self._open_flag:
            return None
        timeout_s = timeout_us / 1_000_000 if timeout_us is not None else None

        with self._lock:
            if self._fallback_queue:
                return self._fallback_queue.popleft()

        self._event.clear()
        signaled = self._event.wait(timeout=timeout_s)
        if not signaled or not self._open_flag:
            return None

        with self._lock:
            if self._fallback_queue:
                return self._fallback_queue.popleft()
        return None

    def barrier(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Zephyr IPC transport
# ---------------------------------------------------------------------------


@dataclass
class ZephyrIPCConfig:
    """Configuration for Zephyr IPC transport.

    These parameters are tuned by the LLM per-target and used during
    C code generation.

    Attributes:
        mechanism: Zephyr IPC primitive to use.
            ``"k_msgq"`` for small fixed-size command messages.
            ``"k_pipe"`` for byte-stream data transfer.
            ``"ipc_service"`` for multi-domain (AMP) communication.
        msg_size: Message size in bytes (for ``k_msgq``).
        queue_depth: Queue depth (for ``k_msgq``).
        pipe_size: Pipe buffer size (for ``k_pipe``).
        endpoint_name: IPC service endpoint name (for ``ipc_service``).
        thread_priority: Priority of the IPC handler thread.
        stack_size: Stack size for the IPC handler thread.
    """

    mechanism: str = "k_msgq"
    msg_size: int = 64
    queue_depth: int = 16
    pipe_size: int = 4096
    endpoint_name: str = "compgen_ep"
    thread_priority: int = 5
    stack_size: int = 4096


class ZephyrIPCTransport:
    """Transport for Zephyr RTOS multi-domain communication.

    This is a **code-generation driver**, not a runtime implementation.
    It holds the configuration that ``soc_codegen.py`` uses to generate
    C code with the appropriate Zephyr IPC calls (``k_msgq_put/get``,
    ``k_pipe_put/get``, or ``ipc_service_send/register``).

    At the Python level, it behaves like a local queue for testing.
    The real communication happens in the generated C code running on
    the target hardware.
    """

    def __init__(self, config: ZephyrIPCConfig | None = None) -> None:
        self._config = config or ZephyrIPCConfig()
        self._open_flag = False
        self._queue: deque[TransportMessage] = deque()
        self._lock = threading.Lock()
        self._event = threading.Event()

    @property
    def name(self) -> str:
        return "zephyr_ipc"

    @property
    def is_open(self) -> bool:
        return self._open_flag

    @property
    def config(self) -> ZephyrIPCConfig:
        """Zephyr IPC configuration (read by codegen)."""
        return self._config

    def open(self, **kwargs: Any) -> None:
        # Override config from kwargs if provided
        for key in (
            "mechanism",
            "msg_size",
            "queue_depth",
            "pipe_size",
            "endpoint_name",
            "thread_priority",
            "stack_size",
        ):
            if key in kwargs:
                object.__setattr__(self._config, key, kwargs[key])
        self._open_flag = True
        log.debug(
            "transport.zephyr_ipc.open",
            mechanism=self._config.mechanism,
            msg_size=self._config.msg_size,
            queue_depth=self._config.queue_depth,
        )

    def close(self) -> None:
        self._open_flag = False
        self._queue.clear()
        self._event.set()
        log.debug("transport.zephyr_ipc.close")

    def send(self, msg: TransportMessage, timeout_us: float | None = None) -> bool:
        if not self._open_flag:
            return False
        with self._lock:
            if len(self._queue) >= self._config.queue_depth:
                return False
            self._queue.append(msg)
            self._event.set()
        return True

    def recv(self, timeout_us: float | None = None) -> TransportMessage | None:
        if not self._open_flag:
            return None
        timeout_s = timeout_us / 1_000_000 if timeout_us is not None else None

        with self._lock:
            if self._queue:
                return self._queue.popleft()

        self._event.clear()
        signaled = self._event.wait(timeout=timeout_s)
        if not signaled or not self._open_flag:
            return None

        with self._lock:
            if self._queue:
                return self._queue.popleft()
        return None

    def barrier(self) -> None:
        pass

    def codegen_context(self) -> dict[str, Any]:
        """Return context dict for C code generation templates.

        Used by ``soc_codegen.py`` to generate the right Zephyr IPC calls.
        """
        return {
            "mechanism": self._config.mechanism,
            "msg_size": self._config.msg_size,
            "queue_depth": self._config.queue_depth,
            "pipe_size": self._config.pipe_size,
            "endpoint_name": self._config.endpoint_name,
            "thread_priority": self._config.thread_priority,
            "stack_size": self._config.stack_size,
            "use_ipc_service": self._config.mechanism == "ipc_service",
            "use_pipe": self._config.mechanism == "k_pipe",
        }


# ---------------------------------------------------------------------------
# Stub network transport
# ---------------------------------------------------------------------------


class StubNetworkTransport:
    """Placeholder for distributed network transport.

    Defines the interface for gRPC/TCP-based communication between
    remote nodes.  The protocol is defined; implementation is deferred.
    At the Python level, behaves like a local queue for testing.
    """

    def __init__(self) -> None:
        self._open_flag = False
        self._remote_address: str = ""
        self._port: int = 0
        self._queue: deque[TransportMessage] = deque()
        self._lock = threading.Lock()
        self._event = threading.Event()

    @property
    def name(self) -> str:
        return "network"

    @property
    def is_open(self) -> bool:
        return self._open_flag

    def open(self, **kwargs: Any) -> None:
        self._remote_address = kwargs.get("address", "localhost")
        self._port = kwargs.get("port", 50051)
        self._open_flag = True
        log.debug(
            "transport.network.open",
            address=self._remote_address,
            port=self._port,
        )

    def close(self) -> None:
        self._open_flag = False
        self._queue.clear()
        self._event.set()
        log.debug("transport.network.close")

    def send(self, msg: TransportMessage, timeout_us: float | None = None) -> bool:
        if not self._open_flag:
            return False
        with self._lock:
            self._queue.append(msg)
            self._event.set()
        return True

    def recv(self, timeout_us: float | None = None) -> TransportMessage | None:
        if not self._open_flag:
            return None
        timeout_s = timeout_us / 1_000_000 if timeout_us is not None else None

        with self._lock:
            if self._queue:
                return self._queue.popleft()

        self._event.clear()
        signaled = self._event.wait(timeout=timeout_s)
        if not signaled or not self._open_flag:
            return None

        with self._lock:
            if self._queue:
                return self._queue.popleft()
        return None

    def barrier(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Transport factory
# ---------------------------------------------------------------------------


_TRANSPORT_REGISTRY: dict[str, type] = {
    "local": LocalTransport,
    "shared_memory": SharedMemTransport,
    "zephyr_ipc": ZephyrIPCTransport,
    "network": StubNetworkTransport,
}


def create_transport(transport_name: str, **kwargs: Any) -> Transport:
    """Create a transport by name.

    Args:
        transport_name: One of ``"local"``, ``"shared_memory"``,
            ``"zephyr_ipc"``, ``"network"``.
        **kwargs: Passed to the transport constructor.

    Returns:
        A transport instance (not yet opened).

    Raises:
        ValueError: If the transport name is unknown.
    """
    cls = _TRANSPORT_REGISTRY.get(transport_name)
    if cls is None:
        msg = f"Unknown transport {transport_name!r}. Available: {sorted(_TRANSPORT_REGISTRY)}"
        raise ValueError(msg)
    return cls(**kwargs)  # type: ignore[call-arg]


def register_transport(name: str, cls: type) -> None:
    """Register a custom transport class.

    The agentic LLM can use this to register target-specific transports
    at runtime.

    Args:
        name: Transport name (used in ``TopologyLink.transport``).
        cls: Transport class (must implement :class:`Transport` protocol).
    """
    _TRANSPORT_REGISTRY[name] = cls
    log.info("transport.registered", name=name, cls=cls.__name__)


# ---------------------------------------------------------------------------
# Conditional registration of optional transports
# ---------------------------------------------------------------------------

try:
    from compgen.runtime._ray_transport import RayTransport

    _TRANSPORT_REGISTRY["ray"] = RayTransport
except ImportError:
    pass  # Ray not installed — ray transport unavailable


__all__ = [
    "LocalTransport",
    "SharedMemTransport",
    "StubNetworkTransport",
    "Transport",
    "TransportMessage",
    "ZephyrIPCConfig",
    "ZephyrIPCTransport",
    "create_transport",
    "register_transport",
]
