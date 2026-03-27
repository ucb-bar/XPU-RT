"""Python-C bridge for the CompGen native runtime.

Wraps ``libcompgen_runtime.so`` (the C execution engine and HAL layer) via
:mod:`ctypes`.  When the shared library is not available the bridge objects
still instantiate but every operation raises :class:`RuntimeError`, so higher
layers can gracefully fall back to pure-Python paths.

Three main classes:

* :class:`NativeBuffer` -- wraps ``compgen_buffer_*`` C functions.
* :class:`NativeDevice` -- wraps ``compgen_device_*`` HAL functions.
* :class:`NativeEngine` -- wraps ``cg_engine_*`` functions for task
  submission and lifecycle management.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Library loading helpers
# ---------------------------------------------------------------------------

_LIB_NAME = "compgen_runtime"
_LIB_FILENAME = f"lib{_LIB_NAME}.so"

_NOT_COMPILED_MSG = "C runtime not compiled -- libcompgen_runtime.so not found"

# Module-level cache so multiple classes share a single load attempt.
_cached_lib: ctypes.CDLL | None = None
_lib_probed: bool = False


def _try_load_library() -> ctypes.CDLL | None:
    """Attempt to load the native runtime shared library.

    The result is cached so subsequent calls are free.

    Search order:
        1. ``LD_LIBRARY_PATH`` / default linker paths (via :func:`ctypes.util.find_library`).
        2. Known project-relative path ``<repo>/build/lib/<filename>``.
        3. Direct ``CDLL(<filename>)`` as a last-ditch effort.

    Returns:
        The loaded :class:`ctypes.CDLL`, or ``None`` when unavailable.
    """
    global _cached_lib, _lib_probed  # noqa: PLW0603
    if _lib_probed:
        return _cached_lib

    lib = _probe_library()
    _cached_lib = lib
    _lib_probed = True
    return lib


def _probe_library() -> ctypes.CDLL | None:
    """Run the actual search (uncached)."""
    # Strategy 1 -- system search
    path = ctypes.util.find_library(_LIB_NAME)
    if path is not None:
        try:
            lib = ctypes.CDLL(path)
            log.info("native_runtime_loaded", path=path)
            return lib
        except OSError:
            pass

    # Strategy 2 -- project build dir (try directly, skip existence check)
    project_lib = Path(__file__).resolve().parents[4] / "build" / "lib" / _LIB_FILENAME
    try:
        lib = ctypes.CDLL(str(project_lib))
        log.info("native_runtime_loaded", path=str(project_lib))
        return lib
    except OSError:
        pass

    # Strategy 3 -- bare filename (relies on LD_LIBRARY_PATH at runtime)
    try:
        lib = ctypes.CDLL(_LIB_FILENAME)
        log.info("native_runtime_loaded", path=_LIB_FILENAME)
        return lib
    except OSError:
        pass

    log.debug("native_runtime_not_found")
    return None


def _reset_library_cache() -> None:
    """Reset the cached library state (for testing only)."""
    global _cached_lib, _lib_probed  # noqa: PLW0603
    _cached_lib = None
    _lib_probed = False


# ---------------------------------------------------------------------------
# NativeBuffer
# ---------------------------------------------------------------------------


class NativeBuffer:
    """Python wrapper around a C HAL buffer (``compgen_buffer_*``).

    Attributes:
        size: Allocation size in bytes.
        handle: Opaque ``ctypes.c_void_p`` returned by the C allocator,
            or ``None`` when running in fallback mode.
    """

    def __init__(self, size: int, handle: ctypes.c_void_p | None, lib: ctypes.CDLL | None) -> None:
        self.size: int = size
        self.handle: ctypes.c_void_p | None = handle
        self._lib: ctypes.CDLL | None = lib
        self._freed: bool = False

    # -- public API --------------------------------------------------------

    def write(self, data: bytes) -> None:
        """Write *data* into the buffer.

        Args:
            data: Raw bytes to copy into the device buffer.

        Raises:
            RuntimeError: If the C runtime is not available or the buffer
                has been freed.
            ValueError: If *data* exceeds the buffer size.
        """
        self._require_live()
        if len(data) > self.size:
            msg = f"data length ({len(data)}) exceeds buffer size ({self.size})"
            raise ValueError(msg)
        assert self._lib is not None  # guaranteed by _require_live
        fn = self._lib.compgen_buffer_write
        fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        fn.restype = ctypes.c_int
        rc = fn(self.handle, data, len(data))
        if rc != 0:
            msg = f"compgen_buffer_write failed with code {rc}"
            raise RuntimeError(msg)

    def read(self, nbytes: int | None = None) -> bytes:
        """Read bytes from the buffer.

        Args:
            nbytes: Number of bytes to read.  Defaults to the full buffer.

        Returns:
            Raw bytes read from the device buffer.

        Raises:
            RuntimeError: If the C runtime is not available or the buffer
                has been freed.
        """
        self._require_live()
        n = nbytes if nbytes is not None else self.size
        assert self._lib is not None
        out = ctypes.create_string_buffer(n)
        fn = self._lib.compgen_buffer_read
        fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        fn.restype = ctypes.c_int
        rc = fn(self.handle, out, n)
        if rc != 0:
            msg = f"compgen_buffer_read failed with code {rc}"
            raise RuntimeError(msg)
        return out.raw

    def free(self) -> None:
        """Release the underlying C buffer."""
        if self._freed:
            return
        if self._lib is not None and self.handle is not None:
            fn = self._lib.compgen_buffer_free
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = None
            fn(self.handle)
        self._freed = True
        self.handle = None

    def __del__(self) -> None:  # noqa: D105
        self.free()

    # -- internals ---------------------------------------------------------

    def _require_live(self) -> None:
        if self._lib is None:
            raise RuntimeError(_NOT_COMPILED_MSG)
        if self._freed or self.handle is None:
            msg = "buffer has been freed"
            raise RuntimeError(msg)

    def __repr__(self) -> str:  # noqa: D105
        state = "freed" if self._freed else "live"
        return f"NativeBuffer(size={self.size}, state={state})"


# ---------------------------------------------------------------------------
# NativeDevice
# ---------------------------------------------------------------------------


class NativeDevice:
    """Python wrapper for a C HAL device (``compgen_device_*``).

    Args:
        device_type: Device kind -- ``"cpu"``, ``"cuda"``, etc.
        device_index: Ordinal when multiple devices of the same type exist.
    """

    def __init__(self, device_type: str = "cpu", device_index: int = 0) -> None:
        self.device_type: str = device_type
        self.device_index: int = device_index
        self._lib: ctypes.CDLL | None = _try_load_library()
        self._handle: ctypes.c_void_p | None = None

        if self._lib is not None:
            self._open()

    # -- public API --------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether the native runtime is loaded and the device is open."""
        return self._lib is not None and self._handle is not None

    def alloc(self, size: int) -> NativeBuffer:
        """Allocate a device buffer of *size* bytes.

        Args:
            size: Number of bytes.

        Returns:
            A :class:`NativeBuffer` backed by native memory.

        Raises:
            RuntimeError: If the C runtime is not available.
        """
        self._require_available()
        assert self._lib is not None
        fn = self._lib.compgen_buffer_alloc
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        fn.restype = ctypes.c_void_p
        buf_handle = fn(self._handle, size)
        if buf_handle is None:
            msg = f"compgen_buffer_alloc returned NULL for size={size}"
            raise RuntimeError(msg)
        return NativeBuffer(size=size, handle=buf_handle, lib=self._lib)

    def dispatch(self, executable: Any, args: Sequence[NativeBuffer] | None = None) -> None:
        """Dispatch a compiled executable on this device.

        Args:
            executable: Opaque executable handle (e.g. a compiled kernel).
            args: Buffers to pass as kernel arguments.

        Raises:
            RuntimeError: If the C runtime is not available.
        """
        self._require_available()
        assert self._lib is not None
        # Build an array of buffer handles
        buf_handles: list[ctypes.c_void_p] = []
        if args:
            buf_handles = [b.handle for b in args if b.handle is not None]
        arr_type = ctypes.c_void_p * len(buf_handles)
        arr = arr_type(*buf_handles)
        fn = self._lib.compgen_device_dispatch
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        fn.restype = ctypes.c_int
        exe_ptr = executable if isinstance(executable, ctypes.c_void_p) else ctypes.c_void_p(executable)
        rc = fn(self._handle, exe_ptr, arr, len(buf_handles))
        if rc != 0:
            msg = f"compgen_device_dispatch failed with code {rc}"
            raise RuntimeError(msg)

    def sync(self) -> None:
        """Block until all pending work on this device completes.

        Raises:
            RuntimeError: If the C runtime is not available.
        """
        self._require_available()
        assert self._lib is not None
        fn = self._lib.compgen_device_sync
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_int
        rc = fn(self._handle)
        if rc != 0:
            msg = f"compgen_device_sync failed with code {rc}"
            raise RuntimeError(msg)

    def close(self) -> None:
        """Close the device handle."""
        if self._lib is not None and self._handle is not None:
            fn = self._lib.compgen_device_close
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = None
            fn(self._handle)
        self._handle = None

    def __del__(self) -> None:  # noqa: D105
        self.close()

    # -- internals ---------------------------------------------------------

    def _open(self) -> None:
        """Open the C HAL device."""
        assert self._lib is not None
        fn = self._lib.compgen_device_open
        fn.argtypes = [ctypes.c_char_p, ctypes.c_int]
        fn.restype = ctypes.c_void_p
        self._handle = fn(self.device_type.encode(), self.device_index)
        if self._handle is None:
            log.warning("native_device_open_failed", device_type=self.device_type, device_index=self.device_index)

    def _require_available(self) -> None:
        if self._lib is None:
            raise RuntimeError(_NOT_COMPILED_MSG)
        if self._handle is None:
            msg = f"device {self.device_type}:{self.device_index} not open"
            raise RuntimeError(msg)

    def __repr__(self) -> str:  # noqa: D105
        status = "available" if self.available else "unavailable"
        return f"NativeDevice(type={self.device_type!r}, index={self.device_index}, status={status})"


# ---------------------------------------------------------------------------
# NativeEngine
# ---------------------------------------------------------------------------


class NativeEngine:
    """Python wrapper for the C execution engine (``cg_engine_*``).

    The engine manages task submission and global scheduling across devices.
    When the native library is missing, the object still constructs but all
    operations raise :class:`RuntimeError`.
    """

    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = _try_load_library()
        self._handle: ctypes.c_void_p | None = None

        if self._lib is not None:
            self._init_engine()

    # -- public API --------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether the native runtime library was loaded and the engine initialized."""
        return self._lib is not None and self._handle is not None

    def submit(self, task_dag: Any) -> int:
        """Submit a task DAG for execution.

        Args:
            task_dag: An opaque task DAG descriptor.  Currently expected to
                be a ``ctypes.c_void_p`` or an integer address pointing to a
                C ``cg_task_dag_t``.

        Returns:
            A submission ID that can be used to query status.

        Raises:
            RuntimeError: If the C runtime is not available.
        """
        self._require_available()
        assert self._lib is not None
        fn = self._lib.cg_engine_submit
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        fn.restype = ctypes.c_int64
        dag_ptr = task_dag if isinstance(task_dag, ctypes.c_void_p) else ctypes.c_void_p(task_dag)
        sid = fn(self._handle, dag_ptr)
        if sid < 0:
            msg = f"cg_engine_submit failed with code {sid}"
            raise RuntimeError(msg)
        return int(sid)

    def wait_idle(self) -> None:
        """Block until all submitted work completes.

        Raises:
            RuntimeError: If the C runtime is not available.
        """
        self._require_available()
        assert self._lib is not None
        fn = self._lib.cg_engine_wait_idle
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_int
        rc = fn(self._handle)
        if rc != 0:
            msg = f"cg_engine_wait_idle failed with code {rc}"
            raise RuntimeError(msg)

    def shutdown(self) -> None:
        """Shut down the engine and release resources."""
        if self._lib is not None and self._handle is not None:
            fn = self._lib.cg_engine_destroy
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = None
            fn(self._handle)
        self._handle = None

    def __del__(self) -> None:  # noqa: D105
        self.shutdown()

    # -- internals ---------------------------------------------------------

    def _init_engine(self) -> None:
        assert self._lib is not None
        fn = self._lib.cg_engine_create
        fn.argtypes = []
        fn.restype = ctypes.c_void_p
        self._handle = fn()
        if self._handle is None:
            log.warning("native_engine_create_failed")

    def _require_available(self) -> None:
        if self._lib is None:
            raise RuntimeError(_NOT_COMPILED_MSG)
        if self._handle is None:
            msg = "engine not initialized"
            raise RuntimeError(msg)

    def __repr__(self) -> str:  # noqa: D105
        status = "available" if self.available else "unavailable"
        return f"NativeEngine(status={status})"


__all__ = [
    "NativeBuffer",
    "NativeDevice",
    "NativeEngine",
    "_reset_library_cache",
]
