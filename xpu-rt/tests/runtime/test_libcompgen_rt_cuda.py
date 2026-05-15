"""CUDA driver — Python integration.

Skipped when the library was built without CUDA support *or* when no
CUDA device opens. Running locally requires the CUDA toolkit be
available at configure time (``find_package(CUDAToolkit)`` succeeds)
and at least one NVIDIA device visible to ``cuInit``.
"""

from __future__ import annotations

import struct

import pytest
from compgen.runtime.native.library import available

if not available():
    pytest.skip("libcompgen_rt shared library not built", allow_module_level=True)

from compgen.runtime.native.device import (  # noqa: E402
    Buffer,
    CommandBuffer,
    CudaExecutable,
    DeviceClass,
    Instance,
    Semaphore,
    cuda_available,
    submit,
)

if not cuda_available():
    pytest.skip("No CUDA device available (or library built without CUDA)", allow_module_level=True)


@pytest.fixture(scope="module")
def device():
    inst = Instance("cuda")
    dev = inst.open_device(0)
    yield dev
    dev.close()
    inst.close()


def test_cuda_traits(device) -> None:
    t = device.traits
    assert t.device_class == DeviceClass.GPU
    assert t.vendor == "nvidia"
    assert t.max_device_memory_bytes > 0
    assert t.supports_command_buffers
    assert t.supports_graph_capture
    assert t.max_concurrent_queues >= 2


def test_cuda_buffer_managed_memory(device) -> None:
    """Managed-memory buffer round-trips through the Python wrapper."""
    buf = Buffer(device, size=256)
    try:
        # Initial zero-init contract.
        assert buf.read() == b"\x00" * 256

        data = bytes(range(256))
        buf.write(data)
        assert buf.read() == data
    finally:
        buf.close()


def test_cuda_fill_via_cu_memset(device) -> None:
    buf = Buffer(device, size=64)
    try:
        cb = CommandBuffer(device)
        cb.begin().fill(buf, size=64, pattern=0xCAFED00D).end()

        sig = Semaphore(device, 0)
        try:
            submit(device, cb, signal=[(sig, 1)])
            sig.wait(1, timeout_s=2.0)
            expected = struct.pack("<I", 0xCAFED00D) * 16
            assert buf.read() == expected
        finally:
            sig.close()
            cb.close()
    finally:
        buf.close()


def test_cuda_nvrtc_dispatch(device) -> None:
    """NVRTC-compiled add-one kernel runs via cuLaunchKernel.

    This is the flagship end-to-end test: compile CUDA C from Python,
    load the PTX module into the driver, dispatch through the HAL,
    and validate the numerical result on managed memory.
    """
    import numpy as np

    # Kernel scalars are monomorphised into the source so the kernel
    # takes only the buffer binding. Dispatch currently supports a
    # single bindings vector; scalar kernel args arrive as additional
    # bindings in a future push-constants-based ABI.
    N = 64
    cuda_source = f"""
    extern "C" __global__ void add_const(float *data) {{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < {N}) data[i] += {2.5:.6g}f;
    }}
    """

    exe = CudaExecutable(device, cuda_source, "add_const")
    buf = Buffer(device, size=N * 4)
    try:
        initial = np.arange(N, dtype=np.float32)
        buf.write(initial.tobytes())

        # block=(32,1,1), grid=(ceil(N/32),1,1).
        block_x = 32
        grid_x = (N + block_x - 1) // block_x
        descriptor = struct.pack("<6I", grid_x, 1, 1, block_x, 1, 1)

        cb = CommandBuffer(device)
        cb.begin().dispatch(exe, [buf], push_constants=descriptor).end()

        sig = Semaphore(device, 0)
        try:
            submit(device, cb, signal=[(sig, 1)])
            sig.wait(1, timeout_s=5.0)

            result = np.frombuffer(buf.read(), dtype=np.float32)
            expected = initial + 2.5
            np.testing.assert_allclose(result, expected)
        finally:
            sig.close()
            cb.close()
    finally:
        exe.close()
        buf.close()


def test_cuda_cross_queue_chain(device) -> None:
    """Producer fills, consumer copies; semaphore handoff across
    CUDA streams. Validates that different queue indices map to
    different cuStreams and the wait-before-copy handoff works."""
    mid = Buffer(device, size=32)
    out = Buffer(device, size=32)
    try:
        cb_fill = CommandBuffer(device)
        cb_fill.begin().fill(mid, size=32, pattern=0x12344321).end()
        cb_copy = CommandBuffer(device)
        cb_copy.begin().copy(mid, out, 32).end()

        fill_done = Semaphore(device, 0)
        copy_done = Semaphore(device, 0)
        try:
            submit(device, cb_fill, queue_index=0, signal=[(fill_done, 1)])
            submit(
                device,
                cb_copy,
                queue_index=1,
                wait=[(fill_done, 1)],
                signal=[(copy_done, 1)],
            )
            copy_done.wait(1, timeout_s=5.0)
            expected = struct.pack("<I", 0x12344321) * 8
            assert out.read() == expected
        finally:
            fill_done.close()
            copy_done.close()
            cb_fill.close()
            cb_copy.close()
    finally:
        mid.close()
        out.close()
