"""Library loader + ctypes signature bindings for ``libcompgen_rt``.

All public primitives (``Device``, ``Semaphore``, ``CommandBuffer``,
``Queue``, ``Buffer``, ``Executable``, ``EventTensor``) import their
C function bindings from this module. Centralising the signature
setup here keeps the higher-level wrappers readable and ensures
every C entry point is bound exactly once.

Library discovery search order:
    1. ``LD_LIBRARY_PATH`` / default linker paths
       (``ctypes.util.find_library("compgen_rt")``).
    2. ``$COMPGEN_RT_LIBRARY`` environment variable (explicit path).
    3. The project build directory
       ``<repo>/runtime/native/libcompgen_rt/build/libcompgen_rt.so``
       (matches ``cmake -B build`` invocation from the repo root).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


# Mirror of cg_rt_status_t from compgen_rt.h. Kept in lockstep with
# the C side. Used so error messages carry the macro name rather
# than just the integer code — significantly cheaper to triage at
# the bridge level.
_CG_RT_STATUS_NAMES: dict[int, str] = {
    0: "CG_RT_OK",
    -1: "CG_RT_ERR_INVALID_ARGUMENT",
    -2: "CG_RT_ERR_OUT_OF_MEMORY",
    -3: "CG_RT_ERR_UNSUPPORTED",
    -4: "CG_RT_ERR_NOT_FOUND",
    -5: "CG_RT_ERR_TIMED_OUT",
    -6: "CG_RT_ERR_FAILED_PRECOND",
    -7: "CG_RT_ERR_ABORTED",
    -99: "CG_RT_ERR_UNKNOWN",
}


class CgRtError(RuntimeError):
    """Raised when a ``libcompgen_rt`` C call returns a non-zero status.

    Attributes:
        status: The ``cg_rt_status_t`` integer returned by the C call.
        what: The call that failed, for readable tracebacks.
        status_name: The macro name (e.g. ``"CG_RT_ERR_NOT_FOUND"``).
    """

    def __init__(self, status: int, what: str) -> None:
        self.status = int(status)
        self.what = what
        self.status_name = _CG_RT_STATUS_NAMES.get(self.status, f"unknown({status})")
        super().__init__(f"{what}: {self.status_name} (cg_rt_status={status})")


_LIB_BASENAME = "compgen_rt"
_LIB_FILENAMES = (f"lib{_LIB_BASENAME}.so", f"lib{_LIB_BASENAME}.dylib")

_lib: ctypes.CDLL | None = None
_lib_lock = threading.Lock()


def _candidate_paths() -> list[Path]:
    """Return the filesystem paths to probe in order.

    Order:
        1. ``$COMPGEN_RT_LIBRARY`` (explicit user override).
        2. The wheel's ``runtime/native/prebuilt/libcompgen_rt-cuda.so``
           (CUDA-built variant) and ``libcompgen_rt-cpu.so``
           (CPU-only). The CUDA variant is preferred when present —
           ``compgen.has_cuda_runtime()`` resolves to True precisely
           when this slot is populated.
        3. The source-tree build directory
           (``<repo>/runtime/native/libcompgen_rt/build/libcompgen_rt.so``)
           for in-repo development without an installed wheel.
    """
    paths: list[Path] = []

    env = os.environ.get("COMPGEN_RT_LIBRARY")
    if env:
        paths.append(Path(env))

    here = Path(__file__).resolve()

    # Wheel's prebuilt slot — what `pip install compgen[cuda]` ships.
    # This must take precedence over the source-tree build dir so an
    # installed wheel doesn't accidentally pick up a stale dev build
    # that happens to live at the repo path.
    prebuilt = here.parent / "prebuilt"
    for name in (
        "libcompgen_rt-cuda.so",
        "libcompgen_rt-cpu.so",
        "libcompgen_rt-cuda.dylib",
        "libcompgen_rt-cpu.dylib",
    ):
        paths.append(prebuilt / name)

    # Repo source-tree build directory — the canonical dev location.
    repo_root = here.parents[4]
    for name in _LIB_FILENAMES:
        paths.append(repo_root / "runtime" / "native" / "libcompgen_rt" / "build" / name)

    return paths


def _probe() -> ctypes.CDLL | None:
    # 1. System search.
    system_path = ctypes.util.find_library(_LIB_BASENAME)
    if system_path is not None:
        try:
            return ctypes.CDLL(system_path)
        except OSError as exc:
            log.debug("libcompgen_rt.system_load_failed", path=system_path, error=str(exc))

    # 2. Explicit env + 3. Build-dir fallback.
    for candidate in _candidate_paths():
        if candidate.is_file():
            try:
                return ctypes.CDLL(str(candidate))
            except OSError as exc:
                log.debug("libcompgen_rt.load_failed", path=str(candidate), error=str(exc))

    return None


def load_library() -> ctypes.CDLL:
    """Load ``libcompgen_rt`` once and cache the handle.

    Raises:
        RuntimeError: If the shared library cannot be located or loaded.
    """
    global _lib
    if _lib is not None:
        return _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        lib = _probe()
        if lib is None:
            raise RuntimeError(
                "libcompgen_rt not found. Build it with:\n"
                "    cmake -B runtime/native/libcompgen_rt/build -S runtime/native/libcompgen_rt\n"
                "    cmake --build runtime/native/libcompgen_rt/build\n"
                "or set COMPGEN_RT_LIBRARY to an explicit .so path."
            )
        _configure_signatures(lib)
        _lib = lib
        return _lib


def available() -> bool:
    """Return True when ``libcompgen_rt`` is importable (best-effort).

    Does not raise; callers use this for graceful feature-flag logic.
    """
    try:
        load_library()
        return True
    except RuntimeError:
        return False


# ---------------------------------------------------------------------------
# ctypes signature configuration
# ---------------------------------------------------------------------------

# Status code sentinel (matches compgen_rt.h).
CG_RT_OK: int = 0
CG_RT_TIMEOUT_POLL: int = 0
CG_RT_TIMEOUT_INFINITE: int = 0xFFFFFFFFFFFFFFFF


# Device traits mirror the C struct layout exactly.
class DeviceTraitsStruct(ctypes.Structure):
    _fields_ = (
        ("device_class", ctypes.c_int),
        ("vendor", ctypes.c_char * 32),
        ("name", ctypes.c_char * 64),
        ("has_native_timeline_semaphores", ctypes.c_uint8),
        ("has_global_atomics", ctypes.c_uint8),
        ("has_shared_memory_atomics", ctypes.c_uint8),
        ("supports_persistent_kernels", ctypes.c_uint8),
        ("supports_cooperative_launch", ctypes.c_uint8),
        ("supports_command_buffers", ctypes.c_uint8),
        ("supports_graph_capture", ctypes.c_uint8),
        ("supports_event_tensors", ctypes.c_uint8),
        ("is_bare_metal", ctypes.c_uint8),
        ("has_rtos_support", ctypes.c_uint8),
        ("max_device_memory_bytes", ctypes.c_uint64),
        ("supports_host_pinned", ctypes.c_uint8),
        ("supports_peer_access", ctypes.c_uint8),
        ("max_concurrent_queues", ctypes.c_uint32),
        ("max_workgroup_size", ctypes.c_uint32),
    )


class SemaphorePointStruct(ctypes.Structure):
    _fields_ = (
        ("semaphore", ctypes.c_void_p),
        ("value", ctypes.c_uint64),
    )


# CPU kernel function pointer type (matches cg_rt_cpu_kernel_fn).
# int fn(const void*, size_t, void**, const size_t*, size_t)
CpuKernelFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,  # push_constants
    ctypes.c_size_t,  # pc_size
    ctypes.POINTER(ctypes.c_void_p),  # bindings
    ctypes.POINTER(ctypes.c_size_t),  # binding_sizes
    ctypes.c_size_t,  # n_bindings
)


def _configure_signatures(lib: ctypes.CDLL) -> None:
    """Bind ``argtypes`` / ``restype`` for every call used by the
    Python wrappers.  Doing this once at load time is faster than
    re-setting at every call site and lets wrong-signature bugs crash
    early instead of at first invocation."""

    # instance / device lifecycle
    lib.cg_rt_status_string.argtypes = [ctypes.c_int32]
    lib.cg_rt_status_string.restype = ctypes.c_char_p

    lib.cg_rt_instance_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.cg_rt_instance_create.restype = ctypes.c_int32

    lib.cg_rt_instance_destroy.argtypes = [ctypes.c_void_p]
    lib.cg_rt_instance_destroy.restype = None

    lib.cg_rt_device_open.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
    lib.cg_rt_device_open.restype = ctypes.c_int32

    lib.cg_rt_device_close.argtypes = [ctypes.c_void_p]
    lib.cg_rt_device_close.restype = None

    lib.cg_rt_device_query_traits.argtypes = [ctypes.c_void_p, ctypes.POINTER(DeviceTraitsStruct)]
    lib.cg_rt_device_query_traits.restype = ctypes.c_int32

    # buffers
    lib.cg_rt_buffer_alloc.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.cg_rt_buffer_alloc.restype = ctypes.c_int32

    lib.cg_rt_buffer_destroy.argtypes = [ctypes.c_void_p]
    lib.cg_rt_buffer_destroy.restype = None

    lib.cg_rt_buffer_size.argtypes = [ctypes.c_void_p]
    lib.cg_rt_buffer_size.restype = ctypes.c_size_t

    lib.cg_rt_buffer_map.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.cg_rt_buffer_map.restype = ctypes.c_int32

    lib.cg_rt_buffer_unmap.argtypes = [ctypes.c_void_p]
    lib.cg_rt_buffer_unmap.restype = ctypes.c_int32

    # semaphores
    lib.cg_rt_semaphore_create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.cg_rt_semaphore_create.restype = ctypes.c_int32

    lib.cg_rt_semaphore_destroy.argtypes = [ctypes.c_void_p]
    lib.cg_rt_semaphore_destroy.restype = None

    lib.cg_rt_semaphore_signal.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.cg_rt_semaphore_signal.restype = ctypes.c_int32

    lib.cg_rt_semaphore_wait.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
    lib.cg_rt_semaphore_wait.restype = ctypes.c_int32

    lib.cg_rt_semaphore_query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.cg_rt_semaphore_query.restype = ctypes.c_int32

    lib.cg_rt_semaphore_fail.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.cg_rt_semaphore_fail.restype = None

    # executables + command buffers + queue
    lib.cg_rt_executable_create_cpu.argtypes = [
        ctypes.c_void_p,
        CpuKernelFn,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.cg_rt_executable_create_cpu.restype = ctypes.c_int32

    # Optional CUDA-specific NVRTC factory. Present only when the
    # library was compiled with ``-DCG_RT_WITH_CUDA``. Absence is
    # detected via ``hasattr`` on the CDLL so non-CUDA builds still
    # load cleanly.
    if hasattr(lib, "cg_rt_executable_create_cuda_ptx"):
        lib.cg_rt_executable_create_cuda_ptx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.cg_rt_executable_create_cuda_ptx.restype = ctypes.c_int32

    lib.cg_rt_executable_destroy.argtypes = [ctypes.c_void_p]
    lib.cg_rt_executable_destroy.restype = None

    lib.cg_rt_command_buffer_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.cg_rt_command_buffer_create.restype = ctypes.c_int32

    lib.cg_rt_command_buffer_destroy.argtypes = [ctypes.c_void_p]
    lib.cg_rt_command_buffer_destroy.restype = None

    lib.cg_rt_command_buffer_begin.argtypes = [ctypes.c_void_p]
    lib.cg_rt_command_buffer_begin.restype = ctypes.c_int32
    lib.cg_rt_command_buffer_end.argtypes = [ctypes.c_void_p]
    lib.cg_rt_command_buffer_end.restype = ctypes.c_int32

    lib.cg_rt_command_buffer_copy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.cg_rt_command_buffer_copy.restype = ctypes.c_int32

    lib.cg_rt_command_buffer_fill.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    lib.cg_rt_command_buffer_fill.restype = ctypes.c_int32

    lib.cg_rt_command_buffer_dispatch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
    ]
    lib.cg_rt_command_buffer_dispatch.restype = ctypes.c_int32

    lib.cg_rt_command_buffer_barrier.argtypes = [ctypes.c_void_p]
    lib.cg_rt_command_buffer_barrier.restype = ctypes.c_int32

    lib.cg_rt_queue_submit.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(SemaphorePointStruct),
        ctypes.c_size_t,
        ctypes.POINTER(SemaphorePointStruct),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.cg_rt_queue_submit.restype = ctypes.c_int32

    # event tensors
    lib.cg_rt_event_tensor_create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.cg_rt_event_tensor_create.restype = ctypes.c_int32

    lib.cg_rt_event_tensor_destroy.argtypes = [ctypes.c_void_p]
    lib.cg_rt_event_tensor_destroy.restype = None

    lib.cg_rt_event_tensor_num_cells.argtypes = [ctypes.c_void_p]
    lib.cg_rt_event_tensor_num_cells.restype = ctypes.c_size_t

    lib.cg_rt_event_tensor_notify.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64]
    lib.cg_rt_event_tensor_notify.restype = ctypes.c_int32

    lib.cg_rt_event_tensor_wait.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
    lib.cg_rt_event_tensor_wait.restype = ctypes.c_int32

    lib.cg_rt_event_tensor_query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.cg_rt_event_tensor_query.restype = ctypes.c_int32

    lib.cg_rt_event_tensor_reset.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lib.cg_rt_event_tensor_reset.restype = ctypes.c_int32


def check(status: int, what: str) -> None:
    """Raise :class:`CgRtError` on any non-zero status."""
    if status != CG_RT_OK:
        raise CgRtError(status, what)
