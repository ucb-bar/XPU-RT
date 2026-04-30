"""High-level Python wrappers for libcompgen_rt primitives.

Each wrapper owns a C handle and frees it in ``__del__`` / ``close``.
Ownership is single — a wrapper is the sole owner of its C handle
and must not be shared across processes. Within a process the C
primitives (semaphores, event tensors) support concurrent access per
their documented thread-safety guarantees.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from compgen.runtime.native.library import (
    CG_RT_TIMEOUT_INFINITE,
    CG_RT_TIMEOUT_POLL,
    CgRtError,
    CpuKernelFn,
    DeviceTraitsStruct,
    SemaphorePointStruct,
    check,
    load_library,
)


class DeviceClass(IntEnum):
    CPU = 1
    GPU = 2
    NPU = 3
    ACCEL = 4


class MemorySpace(IntEnum):
    HOST = 1
    DEVICE = 2
    UNIFIED = 3


class BufferUsage(IntEnum):
    NONE = 0
    TRANSFER = 1 << 0
    DISPATCH = 1 << 1
    INDIRECT = 1 << 2


class EventDType(IntEnum):
    I32 = 1
    I64 = 2


@dataclass(frozen=True)
class DeviceTraits:
    """Python-friendly mirror of ``cg_rt_device_traits_t``."""

    device_class: DeviceClass
    vendor: str
    name: str
    has_native_timeline_semaphores: bool
    has_global_atomics: bool
    has_shared_memory_atomics: bool
    supports_persistent_kernels: bool
    supports_cooperative_launch: bool
    supports_command_buffers: bool
    supports_graph_capture: bool
    supports_event_tensors: bool
    is_bare_metal: bool
    has_rtos_support: bool
    max_device_memory_bytes: int
    supports_host_pinned: bool
    supports_peer_access: bool
    max_concurrent_queues: int
    max_workgroup_size: int

    @classmethod
    def from_struct(cls, s: DeviceTraitsStruct) -> DeviceTraits:
        return cls(
            device_class=DeviceClass(int(s.device_class)),
            vendor=s.vendor.decode("utf-8", errors="replace"),
            name=s.name.decode("utf-8", errors="replace"),
            has_native_timeline_semaphores=bool(s.has_native_timeline_semaphores),
            has_global_atomics=bool(s.has_global_atomics),
            has_shared_memory_atomics=bool(s.has_shared_memory_atomics),
            supports_persistent_kernels=bool(s.supports_persistent_kernels),
            supports_cooperative_launch=bool(s.supports_cooperative_launch),
            supports_command_buffers=bool(s.supports_command_buffers),
            supports_graph_capture=bool(s.supports_graph_capture),
            supports_event_tensors=bool(s.supports_event_tensors),
            is_bare_metal=bool(s.is_bare_metal),
            has_rtos_support=bool(s.has_rtos_support),
            max_device_memory_bytes=int(s.max_device_memory_bytes),
            supports_host_pinned=bool(s.supports_host_pinned),
            supports_peer_access=bool(s.supports_peer_access),
            max_concurrent_queues=int(s.max_concurrent_queues),
            max_workgroup_size=int(s.max_workgroup_size),
        )


class Instance:
    """A libcompgen_rt driver instance. Use as a context manager or
    call :meth:`close` explicitly."""

    def __init__(self, driver_name: str = "cpu_sync") -> None:
        # Set _handle to None before any call that can raise, so a
        # failure here doesn't leave __del__ inspecting an attribute
        # that was never assigned.
        self._handle: ctypes.c_void_p | None = None
        self._lib = load_library()
        self._handle = ctypes.c_void_p()
        status = self._lib.cg_rt_instance_create(driver_name.encode("utf-8"), ctypes.byref(self._handle))
        check(status, f"cg_rt_instance_create({driver_name!r})")
        self._driver = driver_name

    @property
    def handle(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("instance has been destroyed")
        return self._handle

    def open_device(self, index: int = 0) -> Device:
        return Device(self, index)

    def close(self) -> None:
        if self._handle is not None:
            self._lib.cg_rt_instance_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> Instance:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class Device:
    """An opened compute device on a :class:`Instance`."""

    @classmethod
    def create(cls, target: str) -> Device:
        """Convenience: open a device by ``"<driver>:<index>"`` string.

        Examples::

            Device.create("cuda:0")     # CUDA driver, device 0
            Device.create("cpu_sync")   # CPU driver, default index 0

        The constructed :class:`Instance` is stored on the returned
        Device so it stays alive for the device's lifetime; tear it
        down by calling :meth:`Device.close` (or letting the Device
        go out of scope) — the Instance is dropped after the Device
        is closed, in the correct order for libcompgen_rt's C teardown.
        """
        if ":" in target:
            driver, idx_part = target.rsplit(":", 1)
            index = int(idx_part)
        else:
            driver, index = target, 0
        instance = Instance(driver)
        return cls(instance, index)

    def __init__(self, instance: Instance, index: int = 0) -> None:
        # Set _handle to None before anything that can raise so __del__
        # is safe to run on a partially-constructed instance.
        self._handle: ctypes.c_void_p | None = None
        self._lib = load_library()
        self._instance = instance
        self._handle = ctypes.c_void_p()
        check(
            self._lib.cg_rt_device_open(instance.handle, int(index), ctypes.byref(self._handle)),
            "cg_rt_device_open",
        )
        traits_struct = DeviceTraitsStruct()
        check(
            self._lib.cg_rt_device_query_traits(self._handle, ctypes.byref(traits_struct)),
            "cg_rt_device_query_traits",
        )
        self.traits = DeviceTraits.from_struct(traits_struct)

    @property
    def handle(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("device is closed")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._lib.cg_rt_device_close(self._handle)
            self._handle = None

    def __enter__(self) -> Device:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class Buffer:
    """Device buffer. On cpu_sync this is a malloc'd host array."""

    def __init__(
        self,
        device: Device,
        size: int,
        *,
        memory_space: MemorySpace = MemorySpace.HOST,
        usage: int = BufferUsage.TRANSFER | BufferUsage.DISPATCH,
    ) -> None:
        self._lib = load_library()
        self._device = device
        self._handle: ctypes.c_void_p | None = ctypes.c_void_p()
        check(
            self._lib.cg_rt_buffer_alloc(
                device.handle,
                int(size),
                int(memory_space),
                int(usage),
                ctypes.byref(self._handle),
            ),
            "cg_rt_buffer_alloc",
        )
        self._size = int(size)

    @property
    def handle(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("buffer is destroyed")
        return self._handle

    @property
    def size(self) -> int:
        return self._size

    def write(self, data: bytes, offset: int = 0) -> None:
        """Copy ``data`` into the buffer at ``offset``."""
        ptr = ctypes.c_void_p()
        check(
            self._lib.cg_rt_buffer_map(self.handle, offset, len(data), ctypes.byref(ptr)),
            "cg_rt_buffer_map(write)",
        )
        try:
            ctypes.memmove(ptr, data, len(data))
        finally:
            check(self._lib.cg_rt_buffer_unmap(self.handle), "cg_rt_buffer_unmap(write)")

    def read(self, size: int | None = None, offset: int = 0) -> bytes:
        """Copy ``size`` bytes out of the buffer, starting at ``offset``."""
        n = int(size) if size is not None else self._size - offset
        ptr = ctypes.c_void_p()
        check(
            self._lib.cg_rt_buffer_map(self.handle, offset, n, ctypes.byref(ptr)),
            "cg_rt_buffer_map(read)",
        )
        try:
            out = ctypes.string_at(ptr, n)
        finally:
            check(self._lib.cg_rt_buffer_unmap(self.handle), "cg_rt_buffer_unmap(read)")
        return out

    def close(self) -> None:
        if self._handle is not None:
            self._lib.cg_rt_buffer_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()


class Semaphore:
    """Timeline semaphore — monotonic uint64 payload."""

    def __init__(self, device: Device, initial_value: int = 0) -> None:
        self._lib = load_library()
        self._device = device
        self._handle: ctypes.c_void_p | None = ctypes.c_void_p()
        check(
            self._lib.cg_rt_semaphore_create(device.handle, int(initial_value), ctypes.byref(self._handle)),
            "cg_rt_semaphore_create",
        )

    @property
    def handle(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("semaphore is destroyed")
        return self._handle

    def signal(self, value: int) -> None:
        check(self._lib.cg_rt_semaphore_signal(self.handle, int(value)), "cg_rt_semaphore_signal")

    def query(self) -> int:
        out = ctypes.c_uint64(0)
        check(self._lib.cg_rt_semaphore_query(self.handle, ctypes.byref(out)), "cg_rt_semaphore_query")
        return int(out.value)

    def wait(self, value: int, timeout_s: float | None = None) -> None:
        """Block until the semaphore payload reaches ``value``.

        Args:
            value: Target timeline value.
            timeout_s: Timeout in seconds. ``None`` → wait forever.
                Use ``0`` for a non-blocking poll.
        """
        if timeout_s is None:
            timeout_ns = CG_RT_TIMEOUT_INFINITE
        elif timeout_s == 0:
            timeout_ns = CG_RT_TIMEOUT_POLL
        else:
            timeout_ns = int(timeout_s * 1e9)
        status = self._lib.cg_rt_semaphore_wait(self.handle, int(value), timeout_ns)
        if status != 0:
            raise CgRtError(status, "cg_rt_semaphore_wait")

    def fail(self, status: int = -7) -> None:  # default = CG_RT_ERR_ABORTED
        self._lib.cg_rt_semaphore_fail(self.handle, int(status))

    def close(self) -> None:
        if self._handle is not None:
            self._lib.cg_rt_semaphore_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()


class Executable:
    """Wraps a Python callable into a CPU kernel entry point.

    The callback receives:
        - ``push_constants``: ``bytes`` copy of the push-constant block
        - ``bindings``: list of ``memoryview`` objects, one per bound buffer
    """

    def __init__(
        self,
        device: Device,
        callback: Any,
    ) -> None:
        self._lib = load_library()
        self._device = device
        self._callback = callback

        # Keep a reference to the CFUNCTYPE trampoline so it isn't
        # garbage-collected while the C side holds the pointer.
        def trampoline(pc: int, pc_size: int, bindings_ptr: Any, binding_sizes: Any, n: int) -> int:
            # Reconstitute push constants as bytes.
            pc_bytes = b""
            if pc and pc_size:
                pc_bytes = ctypes.string_at(pc, pc_size)
            # Reconstitute bindings as a list of memoryviews.
            views: list[memoryview] = []
            for i in range(int(n)):
                p = bindings_ptr[i]
                sz = int(binding_sizes[i])
                if p and sz:
                    buf = (ctypes.c_ubyte * sz).from_address(p)
                    views.append(memoryview(buf).cast("B"))
                else:
                    views.append(memoryview(b""))
            try:
                rc = self._callback(pc_bytes, views)
                return 0 if rc is None else int(rc)
            except Exception:  # noqa: BLE001
                return 1

        self._trampoline = CpuKernelFn(trampoline)
        self._handle: ctypes.c_void_p | None = ctypes.c_void_p()
        check(
            self._lib.cg_rt_executable_create_cpu(device.handle, self._trampoline, ctypes.byref(self._handle)),
            "cg_rt_executable_create_cpu",
        )

    @property
    def handle(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("executable is destroyed")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._lib.cg_rt_executable_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()


class CudaExecutable:
    """NVRTC-compiled CUDA kernel. Constructed with a CUDA C source
    string + entry-point name; stays alive until ``close()`` so the
    underlying ``CUmodule`` is unloaded.

    The dispatch push-constant block must be at least 24 bytes and
    encode the launch descriptor as six little-endian uint32s:
    ``(grid_x, grid_y, grid_z, block_x, block_y, block_z)``. See the
    public C header ``cg_rt_executable_create_cuda_ptx`` doc comment.
    """

    def __init__(self, device: Device, cuda_c_source: str, kernel_name: str) -> None:
        self._lib = load_library()
        if not hasattr(self._lib, "cg_rt_executable_create_cuda_ptx"):
            raise RuntimeError(
                "libcompgen_rt was built without CUDA support. Rebuild with "
                "CUDA_TOOLKIT installed (CMake will enable the driver automatically)."
            )
        self._device = device
        self._handle: ctypes.c_void_p | None = ctypes.c_void_p()
        check(
            self._lib.cg_rt_executable_create_cuda_ptx(
                device.handle,
                cuda_c_source.encode("utf-8"),
                kernel_name.encode("utf-8"),
                ctypes.byref(self._handle),
            ),
            "cg_rt_executable_create_cuda_ptx",
        )

    @property
    def handle(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("CUDA executable is destroyed")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._lib.cg_rt_executable_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()


def cuda_available() -> bool:
    """True when libcompgen_rt was built with CUDA **and** at least
    one CUDA device opens cleanly. Safe to call from tests to decide
    whether to skip."""
    try:
        lib = load_library()
    except RuntimeError:
        return False
    if not hasattr(lib, "cg_rt_executable_create_cuda_ptx"):
        return False
    try:
        inst = Instance("cuda")
        dev = inst.open_device(0)
        dev.close()
        inst.close()
        return True
    except (CgRtError, RuntimeError):
        return False


class CommandBuffer:
    """Recorded command buffer. Call :meth:`begin`, record ops, then
    :meth:`end` before submitting via :meth:`Queue.submit`."""

    def __init__(self, device: Device) -> None:
        self._lib = load_library()
        self._device = device
        self._handle: ctypes.c_void_p | None = ctypes.c_void_p()
        check(
            self._lib.cg_rt_command_buffer_create(device.handle, ctypes.byref(self._handle)),
            "cg_rt_command_buffer_create",
        )

    @property
    def handle(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("command buffer is destroyed")
        return self._handle

    def begin(self) -> CommandBuffer:
        check(self._lib.cg_rt_command_buffer_begin(self.handle), "cg_rt_command_buffer_begin")
        return self

    def end(self) -> CommandBuffer:
        check(self._lib.cg_rt_command_buffer_end(self.handle), "cg_rt_command_buffer_end")
        return self

    def copy(
        self,
        src: Buffer,
        dst: Buffer,
        size: int,
        *,
        src_offset: int = 0,
        dst_offset: int = 0,
    ) -> CommandBuffer:
        check(
            self._lib.cg_rt_command_buffer_copy(
                self.handle, src.handle, int(src_offset), dst.handle, int(dst_offset), int(size)
            ),
            "cg_rt_command_buffer_copy",
        )
        return self

    def fill(
        self,
        dst: Buffer,
        size: int,
        pattern: int,
        *,
        dst_offset: int = 0,
    ) -> CommandBuffer:
        check(
            self._lib.cg_rt_command_buffer_fill(
                self.handle, dst.handle, int(dst_offset), int(size), int(pattern) & 0xFFFFFFFF
            ),
            "cg_rt_command_buffer_fill",
        )
        return self

    def dispatch(
        self,
        executable: Executable | CudaExecutable,
        bindings: list[Buffer],
        push_constants: bytes = b"",
    ) -> CommandBuffer:
        pc_ptr = None
        pc_size = 0
        pc_buf = None
        if push_constants:
            pc_buf = ctypes.create_string_buffer(push_constants, len(push_constants))
            pc_ptr = ctypes.cast(pc_buf, ctypes.c_void_p)
            pc_size = len(push_constants)

        n = len(bindings)
        if n > 0:
            arr_t = ctypes.c_void_p * n
            arr = arr_t(*[b.handle for b in bindings])
        else:
            arr = None

        check(
            self._lib.cg_rt_command_buffer_dispatch(
                self.handle,
                executable.handle,
                pc_ptr if pc_ptr is not None else ctypes.c_void_p(0),
                pc_size,
                arr,
                n,
            ),
            "cg_rt_command_buffer_dispatch",
        )
        # Keep pc_buf alive beyond the call (dispatch copies internally
        # but we hold a ref to be safe against re-ordering).
        _ = pc_buf
        return self

    def barrier(self) -> CommandBuffer:
        check(self._lib.cg_rt_command_buffer_barrier(self.handle), "cg_rt_command_buffer_barrier")
        return self

    def close(self) -> None:
        if self._handle is not None:
            self._lib.cg_rt_command_buffer_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()


def submit(
    device: Device,
    command_buffer: CommandBuffer,
    *,
    queue_index: int = 0,
    wait: list[tuple[Semaphore, int]] | None = None,
    signal: list[tuple[Semaphore, int]] | None = None,
) -> None:
    """Submit ``command_buffer`` on ``device``'s ``queue_index``.

    ``wait`` and ``signal`` are lists of ``(semaphore, value)`` pairs.
    """
    lib = load_library()
    wait_pairs = wait or []
    signal_pairs = signal or []

    def to_array(pairs: list[tuple[Semaphore, int]]) -> tuple[Any, int]:
        if not pairs:
            return None, 0
        arr_t = SemaphorePointStruct * len(pairs)
        arr = arr_t()
        for i, (sem, val) in enumerate(pairs):
            arr[i].semaphore = sem.handle
            arr[i].value = int(val)
        return arr, len(pairs)

    wait_arr, n_wait = to_array(wait_pairs)
    signal_arr, n_signal = to_array(signal_pairs)

    check(
        lib.cg_rt_queue_submit(
            device.handle,
            int(queue_index),
            wait_arr,
            n_wait,
            signal_arr,
            n_signal,
            command_buffer.handle,
        ),
        "cg_rt_queue_submit",
    )


class EventTensor:
    """Atomic counter array — paper's megakernel primitive."""

    def __init__(
        self,
        device: Device,
        shape: tuple[int, ...],
        *,
        dtype: EventDType = EventDType.I64,
        initial_value: int = 0,
    ) -> None:
        self._lib = load_library()
        self._device = device
        self._shape = tuple(int(s) for s in shape)
        rank = len(self._shape)
        arr_t = ctypes.c_int64 * rank
        arr = arr_t(*self._shape)
        self._handle: ctypes.c_void_p | None = ctypes.c_void_p()
        check(
            self._lib.cg_rt_event_tensor_create(
                device.handle,
                rank,
                arr,
                int(dtype),
                int(initial_value),
                ctypes.byref(self._handle),
            ),
            "cg_rt_event_tensor_create",
        )

    @property
    def handle(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("event tensor is destroyed")
        return self._handle

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def num_cells(self) -> int:
        return int(self._lib.cg_rt_event_tensor_num_cells(self.handle))

    def _linearize(self, idx: tuple[int, ...] | int) -> int:
        if isinstance(idx, int):
            return idx
        if len(idx) != len(self._shape):
            raise ValueError(f"index rank {len(idx)} != tensor rank {len(self._shape)}")
        linear = 0
        for dim, coord in zip(self._shape, idx, strict=True):
            if not (0 <= coord < dim):
                raise IndexError(f"index {idx} out of bounds for shape {self._shape}")
            linear = linear * dim + coord
        return linear

    def notify(self, idx: tuple[int, ...] | int, decrement: int = 1) -> None:
        check(
            self._lib.cg_rt_event_tensor_notify(self.handle, self._linearize(idx), int(decrement)),
            "cg_rt_event_tensor_notify",
        )

    def wait(self, idx: tuple[int, ...] | int, timeout_s: float | None = None) -> None:
        if timeout_s is None:
            timeout_ns = CG_RT_TIMEOUT_INFINITE
        elif timeout_s == 0:
            timeout_ns = CG_RT_TIMEOUT_POLL
        else:
            timeout_ns = int(timeout_s * 1e9)
        status = self._lib.cg_rt_event_tensor_wait(self.handle, self._linearize(idx), timeout_ns)
        if status != 0:
            raise CgRtError(status, "cg_rt_event_tensor_wait")

    def query(self, idx: tuple[int, ...] | int) -> int:
        out = ctypes.c_int64(0)
        check(
            self._lib.cg_rt_event_tensor_query(self.handle, self._linearize(idx), ctypes.byref(out)),
            "cg_rt_event_tensor_query",
        )
        return int(out.value)

    def reset(self, value: int = 0) -> None:
        check(self._lib.cg_rt_event_tensor_reset(self.handle, int(value)), "cg_rt_event_tensor_reset")

    def close(self) -> None:
        if self._handle is not None:
            self._lib.cg_rt_event_tensor_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()


__all__ = [
    "Buffer",
    "BufferUsage",
    "CommandBuffer",
    "CudaExecutable",
    "Device",
    "DeviceClass",
    "DeviceTraits",
    "EventDType",
    "EventTensor",
    "Executable",
    "Instance",
    "MemorySpace",
    "Semaphore",
    "cuda_available",
    "submit",
]
