"""ctypes wrapper for the libcompgen_rt CUDA driver — Phase 4.

Exposes the four families of symbols the megakernel pipeline needs:

- :class:`CudaEventTensor`    — GPU-resident int64 counter tensors with
                                  the host-side allocator + load shim.
- :class:`CudaDynamicQueue`   — on-GPU ready queue allocator + seeder.
- :class:`CudaMegakernelLauncher` — the persistent-kernel launch wrapper
                                  (cooperative + optional cluster).
- :class:`CudaDeviceProbe`    — Phase-6 native HAL backend.

All four import lazily — instantiating any class on a CPU-only host
or on a wheel without the CUDA-built ``libcompgen_rt`` raises
:class:`CudaUnavailableError` with the install instructions the
remote agent should follow.

This module does NOT import torch or cuda-python. The Python probe
in :mod:`compgen.runtime.probe` falls back to those when this
module's symbols are missing; that path is the v0 setup. The native
HAL path here is the Phase-4 replacement.
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CudaUnavailableError(RuntimeError):
    """The CUDA-built libcompgen_rt isn't loadable on this host."""


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------


_CACHED_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Resolve + load the CUDA-flavoured libcompgen_rt.

    Search order:
      1. ``compgen/runtime/native/prebuilt/libcompgen_rt-cuda.so``
         (Phase-0 wheel-bundled location).
      2. ``compgen/runtime/native/prebuilt/libcompgen_rt.so``
         (CMake's default install name; CUDA may or may not be wired
         into this build — we'll detect at first symbol resolution).
      3. ``LD_LIBRARY_PATH`` lookup as a last resort.

    Each candidate is dlopen'd; the first that resolves wins. Cached
    across calls.
    """
    global _CACHED_LIB
    if _CACHED_LIB is not None:
        return _CACHED_LIB

    here = Path(__file__).resolve().parent
    prebuilt = here / "prebuilt"
    candidates = [
        prebuilt / "libcompgen_rt-cuda.so",
        prebuilt / "libcompgen_rt.so",
        Path("libcompgen_rt-cuda.so"),  # fallback to LD_LIBRARY_PATH
        Path("libcompgen_rt.so"),
    ]
    last_err: Exception | None = None
    for path in candidates:
        try:
            lib = ctypes.CDLL(str(path))
            # Probe a CUDA symbol; absence means this build was
            # CPU-only even though the .so is loadable.
            if not hasattr(lib, "cg_rt_cuda_etensor_alloc"):
                continue
            _CACHED_LIB = lib
            return lib
        except OSError as exc:
            last_err = exc
            continue
    raise CudaUnavailableError(
        "libcompgen_rt-cuda.so not loadable. Install the wheel built "
        "with `make build-cuda-rt` (Phase 0) on a CUDA-12 host. Last "
        f"loader error: {last_err!r}"
    )


# ---------------------------------------------------------------------------
# CUDA status code → Python exception
# ---------------------------------------------------------------------------


# Mirror of cg_rt_status_t codes in compgen_rt.h. Zero is success;
# errors are negative (CG_RT_ERR_* macros). Keep in sync with the
# public header.
_STATUS_OK = 0
_STATUS_NAMES = {
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


def _check(rc: int, where: str) -> None:
    if rc != _STATUS_OK:
        raise CudaUnavailableError(f"{where} returned status {_STATUS_NAMES.get(rc, str(rc))} ({rc})")


# ---------------------------------------------------------------------------
# Probe — Phase 6 native HAL backend
# ---------------------------------------------------------------------------


class _CudaProbeStruct(ctypes.Structure):
    """Mirror of cg_rt_cuda_probe_t in compgen_rt.h.

    Field order MUST match the C header exactly. The ctypes-side dict
    converter at the bottom flattens this into the same shape as
    :func:`compgen.runtime.probe.probe_via_torch` so downstream code
    is path-agnostic.
    """

    _fields_ = [
        ("device_name", ctypes.c_char * 128),
        ("compute_capability_major", ctypes.c_int),
        ("compute_capability_minor", ctypes.c_int),
        ("sm_count", ctypes.c_int),
        ("num_visible_devices", ctypes.c_int),
        ("max_threads_per_block", ctypes.c_int),
        ("max_threads_per_multiprocessor", ctypes.c_int),
        ("warp_size", ctypes.c_int),
        ("max_grid_dim_x", ctypes.c_int),
        ("max_grid_dim_y", ctypes.c_int),
        ("max_grid_dim_z", ctypes.c_int),
        ("max_device_memory_bytes", ctypes.c_longlong),
        ("l2_cache_bytes", ctypes.c_int),
        ("max_shared_memory_per_block_optin_bytes", ctypes.c_int),
        ("max_blocks_per_cluster", ctypes.c_int),
        ("cluster_launch", ctypes.c_int),
        ("cooperative_launch", ctypes.c_int),
        ("concurrent_kernels", ctypes.c_int),
        ("concurrent_managed_access", ctypes.c_int),
        ("supports_tma", ctypes.c_int),
        ("supports_clusters", ctypes.c_int),
        ("supports_fp8", ctypes.c_int),
        ("supports_fp4", ctypes.c_int),
        ("supports_ondevice_scheduler", ctypes.c_int),
        ("driver_version", ctypes.c_int),
        ("runtime_version", ctypes.c_int),
    ]


class CudaDeviceProbe:
    """Phase-6 native HAL probe.

    Use as the Phase-4-onward replacement for the torch-backed probe
    in :mod:`compgen.runtime.probe`. Returns the same dict shape so
    :meth:`compgen.runtime.traits.DeviceTraits.with_probe` consumes
    both interchangeably.
    """

    def __init__(self) -> None:
        self._lib = _load_lib()
        self._lib.cg_rt_cuda_probe_device.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_CudaProbeStruct),
        ]
        self._lib.cg_rt_cuda_probe_device.restype = ctypes.c_int

    def probe(self, device_index: int = 0) -> dict[str, Any]:
        out = _CudaProbeStruct()
        rc = self._lib.cg_rt_cuda_probe_device(int(device_index), ctypes.byref(out))
        _check(rc, "cg_rt_cuda_probe_device")
        result: dict[str, Any] = {
            "probe_source": "native_hal",
            "device_name": out.device_name.decode("utf-8", errors="replace"),
        }
        for name, _typ in _CudaProbeStruct._fields_[1:]:
            value = getattr(out, name)
            if name in (
                "supports_tma",
                "supports_clusters",
                "supports_fp8",
                "supports_fp4",
                "supports_ondevice_scheduler",
                "cluster_launch",
                "cooperative_launch",
                "concurrent_kernels",
                "concurrent_managed_access",
            ):
                result[name] = bool(value) if name != "cluster_launch" else int(value)
            else:
                result[name] = value
        return result


# ---------------------------------------------------------------------------
# Event tensors + dynamic queue (host shims; device primitives are
# inlined into Phase-5 emitted PTX).
# ---------------------------------------------------------------------------


class CudaEventTensor:
    """Wraps a GPU-resident int64 counter array.

    Use the host-side ``alloc`` / ``free`` / ``load`` here for
    Phase-4 bring-up tests; once Phase-5 lands the megakernel
    wrapper, these are managed by the launcher.
    """

    def __init__(self, num_cells: int, initial_wait_count: int = 1) -> None:
        self._lib = _load_lib()
        self._lib.cg_rt_cuda_etensor_alloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_longlong,
        ]
        self._lib.cg_rt_cuda_etensor_alloc.restype = ctypes.c_int
        self._lib.cg_rt_cuda_etensor_free.argtypes = [ctypes.c_void_p]
        self._lib.cg_rt_cuda_etensor_free.restype = None
        self._lib.cg_rt_cuda_etensor_load.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_longlong),
        ]
        self._lib.cg_rt_cuda_etensor_load.restype = ctypes.c_int

        self._ptr = ctypes.c_void_p()
        rc = self._lib.cg_rt_cuda_etensor_alloc(
            ctypes.byref(self._ptr),
            int(num_cells),
            int(initial_wait_count),
        )
        _check(rc, "cg_rt_cuda_etensor_alloc")
        self._num_cells = int(num_cells)

    @property
    def device_ptr(self) -> int:
        """Raw device pointer (as int) — pass to launcher kernel_args."""
        return int(self._ptr.value or 0)

    def load(self, idx: int) -> int:
        """Read one cell back to host. O(latency) — for tests, not
        hot path."""
        out = ctypes.c_longlong()
        rc = self._lib.cg_rt_cuda_etensor_load(self._ptr, int(idx), ctypes.byref(out))
        _check(rc, "cg_rt_cuda_etensor_load")
        return int(out.value)

    def __del__(self) -> None:
        try:
            if getattr(self, "_ptr", None) and self._ptr.value:
                self._lib.cg_rt_cuda_etensor_free(self._ptr)
                self._ptr = ctypes.c_void_p()
        except Exception:
            pass


class CudaDynamicQueue:
    """On-GPU circular ready queue for the dynamic scheduler."""

    def __init__(self, capacity: int) -> None:
        self._lib = _load_lib()
        self._lib.cg_rt_cuda_queue_alloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        self._lib.cg_rt_cuda_queue_alloc.restype = ctypes.c_int
        self._lib.cg_rt_cuda_queue_free.argtypes = [ctypes.c_void_p]
        self._lib.cg_rt_cuda_queue_free.restype = None
        self._lib.cg_rt_cuda_queue_seed_initial.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self._lib.cg_rt_cuda_queue_seed_initial.restype = ctypes.c_int

        self._ptr = ctypes.c_void_p()
        rc = self._lib.cg_rt_cuda_queue_alloc(ctypes.byref(self._ptr), int(capacity))
        _check(rc, "cg_rt_cuda_queue_alloc")
        self._capacity = int(capacity)

    @property
    def device_ptr(self) -> int:
        return int(self._ptr.value or 0)

    def seed_initial(self, initial_task_ids: list[int]) -> None:
        if not initial_task_ids:
            return
        arr = (ctypes.c_int * len(initial_task_ids))(*initial_task_ids)
        rc = self._lib.cg_rt_cuda_queue_seed_initial(self._ptr, arr, len(initial_task_ids))
        _check(rc, "cg_rt_cuda_queue_seed_initial")

    def __del__(self) -> None:
        try:
            if getattr(self, "_ptr", None) and self._ptr.value:
                self._lib.cg_rt_cuda_queue_free(self._ptr)
                self._ptr = ctypes.c_void_p()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Persistent megakernel launcher
# ---------------------------------------------------------------------------


class _LaunchConfigStruct(ctypes.Structure):
    """Mirror of cg_rt_cuda_megakernel_launch_t in compgen_rt.h."""

    _fields_ = [
        ("kernel_handle", ctypes.c_void_p),
        ("grid_dim_x", ctypes.c_int),
        ("grid_dim_y", ctypes.c_int),
        ("grid_dim_z", ctypes.c_int),
        ("block_dim_x", ctypes.c_int),
        ("block_dim_y", ctypes.c_int),
        ("block_dim_z", ctypes.c_int),
        ("cluster_dim_x", ctypes.c_int),
        ("cluster_dim_y", ctypes.c_int),
        ("cluster_dim_z", ctypes.c_int),
        ("shared_mem_bytes", ctypes.c_int),
    ]


class CudaMegakernelLauncher:
    """Wraps cg_rt_cuda_megakernel_launch.

    Phase-7's `CompiledModel.__call__` will route through this when
    the bundle has a megakernel manifest. Until then the launcher is
    standalone — Phase-5 emits the persistent kernel via NVRTC,
    yields a CUfunction handle, and Phase-7 hands both to here.
    """

    def __init__(self, device_handle: int, *, device_index: int = 0) -> None:
        self._lib = _load_lib()
        self._lib.cg_rt_cuda_megakernel_launch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_LaunchConfigStruct),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.cg_rt_cuda_megakernel_launch.restype = ctypes.c_int
        self._device_handle = ctypes.c_void_p(device_handle)
        self._device_index = int(device_index)

    def launch(
        self,
        *,
        kernel_handle: int,
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
        cluster_dim: tuple[int, int, int] | None = None,
        shared_mem_bytes: int = 0,
        kernel_args: list[int] | None = None,
    ) -> None:
        """Launch the persistent megakernel.

        Args:
            kernel_handle: CUfunction (as int).
            grid_dim: (x, y, z) grid shape — typically (sm_count, 1, 1).
            block_dim: thread block shape.
            cluster_dim: when set, opts into cluster launch + DSMEM.
                Set to ``None`` to skip; nonzero only on devices where
                :class:`DeviceTraits.metadata["supports_clusters"]` is True.
            shared_mem_bytes: dynamic shared memory the kernel needs.
            kernel_args: list of device pointers (as ints) passed as
                kernel arguments. Order must match the persistent
                kernel's parameter list.
        """
        cfg = _LaunchConfigStruct(
            kernel_handle=ctypes.c_void_p(int(kernel_handle)),
            grid_dim_x=int(grid_dim[0]),
            grid_dim_y=int(grid_dim[1]),
            grid_dim_z=int(grid_dim[2]),
            block_dim_x=int(block_dim[0]),
            block_dim_y=int(block_dim[1]),
            block_dim_z=int(block_dim[2]),
            cluster_dim_x=int(cluster_dim[0]) if cluster_dim else 0,
            cluster_dim_y=int(cluster_dim[1]) if cluster_dim else 0,
            cluster_dim_z=int(cluster_dim[2]) if cluster_dim else 0,
            shared_mem_bytes=int(shared_mem_bytes),
        )
        # cuLaunchKernelEx (which the C side invokes) expects
        # ``kernel_args`` as ``void **`` — an array of pointers, where
        # each entry points to the *storage* holding the argument
        # value, not the value itself. Naively packing
        # ``[ctypes.c_void_p(int(a)) ...]`` would put the argument
        # values into the pointer slots; the driver would then try to
        # dereference the values as host pointers and segfault.
        # Build the indirection explicitly: stash each arg in a
        # ctypes uint64 container (kept alive on ``arg_storage`` until
        # the launch returns), and the ``arg_ptr_array`` we pass holds
        # the address of each container.
        arg_storage: list[Any]  # holds the c_uint64 values
        if kernel_args:
            arg_storage = [ctypes.c_uint64(int(a)) for a in kernel_args]
            arg_ptr_array = (ctypes.c_void_p * len(arg_storage))(
                *(ctypes.cast(ctypes.pointer(s), ctypes.c_void_p) for s in arg_storage)
            )
        else:
            arg_storage = []
            arg_ptr_array = (ctypes.c_void_p * 0)()
        # Ensure the driver-API primary context for THIS launcher's
        # rank is retained + current before the C launcher invokes
        # cuLaunchKernelEx. For multi-GPU dispatch, every per-rank
        # launcher owns its device_index; without this set-current,
        # the launch would target whatever ctx is current (typically
        # rank 0's), which results in
        # ``cudaErrorInvalidContext`` / ``InvalidResourceHandle``
        # surfaced as ``CG_RT_ERR_UNKNOWN`` per #057's diagnosis.
        _ensure_cuda_driver_context(self._device_index)
        rc = self._lib.cg_rt_cuda_megakernel_launch(
            self._device_handle,
            ctypes.byref(cfg),
            arg_ptr_array,
        )
        # Touch arg_storage after the launch so the GC can't free the
        # backing c_uint64 containers before cuLaunchKernelEx reads
        # through the pointer array.
        del arg_storage
        _check(rc, "cg_rt_cuda_megakernel_launch")


class CudaModule:
    """NVRTC-compile a CUDA C++ source string into a loaded CUmodule
    and surface a kernel-function handle.

    Uses ``cuda.bindings`` (shipped inside the ``cuda-python`` wheel,
    pinned by the ``[cuda]`` extra). Imports lazily so the rest of
    this module stays usable without ``cuda-python``.

    Typical Phase-5 use::

        mod = CudaModule(
            cuda_source=emit_result.cuda_source,
            kernel_name=emit_result.kernel_name,
            arch="sm_90",          # safe default; sm_120 needs CUDA 13 toolkit
        )
        launcher = CudaMegakernelLauncher(device.handle)
        launcher.launch(kernel_handle=mod.kernel_handle, grid_dim=..., ...)

    Raises:
        CudaUnavailableError: ``cuda-python`` isn't importable.
        RuntimeError: NVRTC reports a compile error (full log surfaced).
    """

    def __init__(
        self,
        cuda_source: str,
        kernel_name: str,
        *,
        arch: str = "sm_90",
        extra_options: tuple[str, ...] = (),
        extra_include_paths: tuple[str, ...] = (),
        device_index: int = 0,
        use_cu13_nvrtc: bool = False,
    ) -> None:
        try:
            from cuda.bindings import driver as cu_driver  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise CudaUnavailableError(
                "cuda-python (>=12.6) is not importable. Install with "
                "`pip install 'compgen[cuda]>=0.2.0'` on a host that "
                "has the CUDA 12.6+ runtime."
            ) from exc

        self._cu_driver = cu_driver
        self._kernel_name = kernel_name

        # --- 1. NVRTC compile to PTX --------------------------------------
        # Two paths: the cuda.bindings.nvrtc shim (cu12 toolkit) used
        # everywhere else, OR a ctypes-direct call into the cu13 NVRTC
        # shipped under ``nvidia.cuda_nvrtc``. cu13 NVRTC knows
        # ``__CUDA_ARCH__ == 1000`` for Blackwell sm_100 / sm_120, which
        # is needed for cuBLASDx to engage the ``tcgen05.mma`` tensor-core
        # path (per bridge #089's diagnosis: the SM<1000> dispatcher
        # requires ``CUTE_ARCH_FFMA2_SM100_ENABLED`` which is gated by
        # ``__CUDA_ARCH__ == 1000`` — cu12 NVRTC's max is sm_90 and
        # silently falls back to SIMT, killing the perf gate).
        opts = [f"--gpu-architecture={arch}".encode(), b"--std=c++17"]
        opts.extend(o.encode("utf-8") for o in extra_options)
        for inc in extra_include_paths:
            opts.append(f"-I{inc}".encode())

        if use_cu13_nvrtc:
            # ctypes-direct cu13 NVRTC. Raises CudaUnavailableError if
            # nvidia-cuda-nvrtc-cu13 isn't installed; the caller can
            # retry with use_cu13_nvrtc=False to fall back.
            ptx = _compile_via_cu13_nvrtc(
                cuda_source=cuda_source,
                kernel_name=kernel_name,
                opts=opts,
            )
        else:
            from cuda.bindings import nvrtc  # type: ignore

            prog = _nvrtc_check(
                nvrtc.nvrtcCreateProgram(
                    cuda_source.encode("utf-8"),
                    f"{kernel_name}.cu".encode(),
                    0,
                    [],
                    [],
                )
            )
            compile_status = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
            if compile_status[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                log_size = _nvrtc_check(nvrtc.nvrtcGetProgramLogSize(prog))
                log_buf = b" " * log_size
                nvrtc.nvrtcGetProgramLog(prog, log_buf)
                raise RuntimeError(
                    f"NVRTC compile failed for {kernel_name}:\n{log_buf.decode('utf-8', errors='replace')}"
                )
            ptx_size = _nvrtc_check(nvrtc.nvrtcGetPTXSize(prog))
            ptx = b" " * ptx_size
            nvrtc.nvrtcGetPTX(prog, ptx)
            nvrtc.nvrtcDestroyProgram(prog)
        self._ptx = ptx

        # --- 2. Driver context bring-up + cuModuleLoadData ---------------
        # cuda-python's driver API needs an explicit primary context
        # retained + set-current on this thread before any cu* call
        # outside the init path. Without it the next cu* call fails
        # with CUDA_ERROR_INVALID_CONTEXT (201). Torch users have one
        # implicitly via torch.cuda.init(), but cuda-python-only callers
        # (the wheel's spec) don't, so we bring up our own.
        # Multi-GPU: the caller passes ``device_index`` so the module
        # is loaded into the right rank's context; otherwise the
        # default rank-0 ctx wins. ``cuModuleLoadData`` binds the
        # module to whatever ctx is current — getting this wrong
        # silently runs the kernel on the wrong device.
        _ensure_cuda_driver_context(device_index)
        module = _cu_check(cu_driver.cuModuleLoadData(ptx))
        self._module = module
        kernel = _cu_check(cu_driver.cuModuleGetFunction(module, kernel_name.encode("utf-8")))
        self._kernel = kernel

    @property
    def kernel_handle(self) -> int:
        """Integer-cast CUfunction the launcher passes to the C side."""
        return int(self._kernel)

    @property
    def ptx(self) -> bytes:
        """Compiled PTX (for cuobjdump / megakernel_inspect)."""
        return self._ptx

    def close(self) -> None:
        if self._module is not None:
            self._cu_driver.cuModuleUnload(self._module)
            self._module = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


_CUDA_DRIVER_INIT_DONE = False
_CUDA_PRIMARY_CTX: dict[int, Any] = {}


# ---------------------------------------------------------------------------
# cuBLASDx discovery — Phase 10b
# Wave 1.14 — moved to ``targets/gpu/nvidia/common/discovery.py``.
# Re-exported here for one round of backward compatibility; new
# callers should import from the new location.
# ---------------------------------------------------------------------------

from compgen.targets.gpu.nvidia.common.discovery import (  # noqa: E402, F401
    cublasdx_available,
    cutlass_available,
    discover_cublasdx_include,
    discover_cutlass_include,
    discover_libcudacxx_include,
    libcudacxx_available,
)


def _ensure_cuda_driver_context(device_index: int = 0) -> None:
    """Idempotently bring up the cuda-python driver-API primary context.

    cuda-python's driver bindings require an explicit primary-context
    retain + set-current sequence on each thread that does
    ``cu*`` work outside of ``cuInit``/``cuDeviceGet``. Skipping this
    yields ``CUDA_ERROR_INVALID_CONTEXT`` (201) on the next call.
    Torch users have a context implicitly via ``torch.cuda.init()``,
    but the wheel's spec is "user only depends on cuda-python", so we
    bring up our own and reuse it across :class:`CudaModule` /
    :class:`CudaMegakernelLauncher` calls.

    The retained primary context is shared with anyone else using the
    same device (including torch), so this doesn't fight torch when
    both are present — they cohabitate on one context.
    """
    global _CUDA_DRIVER_INIT_DONE  # noqa: PLW0603

    try:
        from cuda.bindings import driver as cu_driver  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise CudaUnavailableError(
            "cuda-python (>=12.6) is not importable. Install with `pip install 'compgen[cuda]>=0.2.0'`."
        ) from exc

    if not _CUDA_DRIVER_INIT_DONE:
        _cu_check(cu_driver.cuInit(0))
        _CUDA_DRIVER_INIT_DONE = True
    if device_index not in _CUDA_PRIMARY_CTX:
        dev = _cu_check(cu_driver.cuDeviceGet(device_index))
        ctx = _cu_check(cu_driver.cuDevicePrimaryCtxRetain(dev))
        _CUDA_PRIMARY_CTX[device_index] = ctx
    _cu_check(cu_driver.cuCtxSetCurrent(_CUDA_PRIMARY_CTX[device_index]))


def _nvrtc_check(result: tuple[Any, ...]) -> Any:
    """``cuda.bindings.nvrtc`` returns ``(status, value)`` tuples;
    raise on non-success and unwrap the value."""
    from cuda.bindings import nvrtc  # type: ignore

    status, *rest = result
    if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC call failed with status {status!r}")
    return rest[0] if len(rest) == 1 else tuple(rest)


# ---------------------------------------------------------------------------
# Cu13 NVRTC ctypes wrapper (Phase-10c+ tensor-core path)
# Wave 1.14 — moved to ``targets/gpu/nvidia/blackwell/cu13_nvrtc.py``
# (Blackwell-specific because cu13 NVRTC is what gates
# ``__CUDA_ARCH__ == 1000``). Re-exported here for backward compat;
# new callers should import from the new home.
# ---------------------------------------------------------------------------

from compgen.targets.gpu.nvidia.blackwell.cu13_nvrtc import (  # noqa: E402, F401
    _CU13_NVRTC_LIB,
    _compile_via_cu13_nvrtc,
    _load_cu13_nvrtc,
    _resolve_cu13_nvrtc_lib_path,
    cu13_nvrtc_available,
)


def _cu_check(result: tuple[Any, ...]) -> Any:
    """Same shape as ``_nvrtc_check`` but for the CUDA driver bindings."""
    from cuda.bindings import driver as cu_driver  # type: ignore

    status, *rest = result
    if status != cu_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed with status {status!r}")
    return rest[0] if len(rest) == 1 else tuple(rest)


class CudaCommGroup:
    """ctypes wrapper for the Phase-4b NCCL bridge.

    Wraps a single-process multi-device NCCL communicator (one rank
    per local CUDA device). The C side enables peer access pairwise
    so cross-GPU peer-mapped event-tensor atomics work — see
    :file:`runtime/native/libcompgen_rt/src/drivers/cuda/nccl_bridge.c`.

    The wheel's prebuilt ``libcompgen_rt-cuda.so`` is built without
    NCCL by default. Hosts that want the bridge must rebuild with
    ``-DCG_RT_WITH_NCCL=ON`` and ensure ``libnccl.so.2`` is in the
    loader path (typically via ``import torch`` ahead of time, which
    pulls in ``nvidia-nccl-cu13``'s ``libnccl.so.2``).

    Constructing :class:`CudaCommGroup` on a host whose libcompgen_rt
    lacks NCCL raises :class:`CudaUnavailableError` with a clear
    install message.

    Typical use::

        comm = CudaCommGroup(device_indices=[0, 1])
        # ... per-rank work; comm.allreduce_fp32_sum(...)
        comm.close()

    or as a context manager::

        with CudaCommGroup(device_indices=[0, 1]) as comm:
            ...
    """

    def __init__(self, device_indices: list[int] | tuple[int, ...]) -> None:
        self._lib = _load_lib()
        if not hasattr(self._lib, "cg_rt_cuda_comm_init_local"):
            raise CudaUnavailableError(
                "libcompgen_rt was built without NCCL. Rebuild with "
                "-DCG_RT_WITH_NCCL=ON and ensure libnccl.so.2 is "
                "available (e.g. via `import torch` to pull in "
                "nvidia-nccl-cu13)."
            )
        self._lib.cg_rt_cuda_comm_init_local.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.cg_rt_cuda_comm_init_local.restype = ctypes.c_int
        self._lib.cg_rt_cuda_comm_destroy.argtypes = [ctypes.c_void_p]
        self._lib.cg_rt_cuda_comm_destroy.restype = ctypes.c_int
        self._lib.cg_rt_cuda_comm_size.argtypes = [ctypes.c_void_p]
        self._lib.cg_rt_cuda_comm_size.restype = ctypes.c_int
        self._lib.cg_rt_cuda_comm_allreduce_fp32_sum.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self._lib.cg_rt_cuda_comm_allreduce_fp32_sum.restype = ctypes.c_int

        self._device_indices = list(device_indices)
        n = len(self._device_indices)
        idx_array = (ctypes.c_int * n)(*self._device_indices)
        self._handle: ctypes.c_void_p | None = ctypes.c_void_p()
        rc = self._lib.cg_rt_cuda_comm_init_local(n, idx_array, ctypes.byref(self._handle))
        _check(rc, "cg_rt_cuda_comm_init_local")

    @property
    def device_indices(self) -> list[int]:
        return list(self._device_indices)

    def size(self) -> int:
        if self._handle is None:
            return 0
        return int(self._lib.cg_rt_cuda_comm_size(self._handle))

    def allreduce_fp32_sum(
        self,
        inputs_per_rank: list[int],
        outputs_per_rank: list[int],
        count: int,
    ) -> None:
        """Sum-AllReduce of fp32 buffers across ranks.

        Args:
            inputs_per_rank: Device pointer (as int) for each rank's
                input buffer. Must have length :meth:`size`.
            outputs_per_rank: Device pointer (as int) for each rank's
                output buffer.
            count: Number of fp32 elements per buffer.
        """
        if self._handle is None:
            raise RuntimeError("CudaCommGroup is closed")
        n = self.size()
        if len(inputs_per_rank) != n or len(outputs_per_rank) != n:
            raise ValueError(
                f"per-rank pointer arrays must have length {n}, got "
                f"inputs={len(inputs_per_rank)}, outputs={len(outputs_per_rank)}"
            )
        in_arr = (ctypes.c_void_p * n)(*(ctypes.c_void_p(int(p)) for p in inputs_per_rank))
        out_arr = (ctypes.c_void_p * n)(*(ctypes.c_void_p(int(p)) for p in outputs_per_rank))
        rc = self._lib.cg_rt_cuda_comm_allreduce_fp32_sum(self._handle, in_arr, out_arr, ctypes.c_size_t(count))
        _check(rc, "cg_rt_cuda_comm_allreduce_fp32_sum")

    def close(self) -> None:
        if self._handle is not None:
            self._lib.cg_rt_cuda_comm_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> CudaCommGroup:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "CudaCommGroup",
    "CudaDeviceProbe",
    "CudaDynamicQueue",
    "CudaEventTensor",
    "CudaMegakernelLauncher",
    "CudaModule",
    "CudaUnavailableError",
]
