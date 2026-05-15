"""Integration tests driving libcompgen_rt from Python.

Skipped when the native library is not built. To build it::

    /usr/bin/cmake -B runtime/native/libcompgen_rt/build -S runtime/native/libcompgen_rt
    /usr/bin/cmake --build runtime/native/libcompgen_rt/build

These tests are the Python mirror of the C unit tests — they exercise
every public primitive through the ctypes bindings, so a ``.so`` that
builds but does not link correctly (mismatched signatures, missing
exports) fails here instead of at the first Python-side use.
"""

from __future__ import annotations

import struct
import threading
import time

import pytest
from compgen.runtime.native.library import CgRtError, available

if not available():
    pytest.skip("libcompgen_rt shared library not built", allow_module_level=True)

from compgen.runtime.native.device import (  # noqa: E402
    Buffer,
    CommandBuffer,
    DeviceClass,
    EventDType,
    EventTensor,
    Executable,
    Instance,
    Semaphore,
    submit,
)

# Provide a session-scoped (instance, device) fixture so multi-test
# runs don't re-open the driver dozens of times.


@pytest.fixture(scope="module")
def device() -> object:
    inst = Instance("cpu_sync")
    dev = inst.open_device(0)
    yield dev
    dev.close()
    inst.close()


# --------------------------------------------------------------------
# Traits
# --------------------------------------------------------------------


def test_traits_report_cpu_class(device) -> None:
    t = device.traits
    assert t.device_class == DeviceClass.CPU
    assert t.vendor == "host"
    assert t.has_native_timeline_semaphores
    assert t.has_global_atomics
    assert t.supports_event_tensors
    assert t.supports_command_buffers
    assert t.max_concurrent_queues >= 2


# --------------------------------------------------------------------
# Buffer
# --------------------------------------------------------------------


def test_buffer_roundtrip(device) -> None:
    buf = Buffer(device, size=64)
    try:
        payload = b"\x01" * 32 + b"\x02" * 32
        buf.write(payload)
        assert buf.read() == payload
        assert buf.size == 64
    finally:
        buf.close()


def test_buffer_partial_read(device) -> None:
    buf = Buffer(device, size=16)
    try:
        buf.write(b"0123456789abcdef")
        assert buf.read(size=4, offset=2) == b"2345"
    finally:
        buf.close()


# --------------------------------------------------------------------
# Semaphore
# --------------------------------------------------------------------


def test_semaphore_query_and_monotonic_signal(device) -> None:
    sem = Semaphore(device, initial_value=2)
    try:
        assert sem.query() == 2
        sem.signal(5)
        assert sem.query() == 5
        sem.signal(3)  # no-op — monotonic
        assert sem.query() == 5
    finally:
        sem.close()


def test_semaphore_wait_immediate(device) -> None:
    sem = Semaphore(device, initial_value=10)
    try:
        sem.wait(5, timeout_s=0)  # poll; already satisfied
    finally:
        sem.close()


def test_semaphore_wait_timeout(device) -> None:
    sem = Semaphore(device, initial_value=0)
    try:
        with pytest.raises(CgRtError) as excinfo:
            sem.wait(1, timeout_s=0.01)
        # CG_RT_ERR_TIMED_OUT = -5
        assert excinfo.value.status == -5
    finally:
        sem.close()


def test_semaphore_blocks_until_background_signal(device) -> None:
    sem = Semaphore(device, initial_value=0)
    try:

        def _signaller() -> None:
            time.sleep(0.02)
            sem.signal(1)

        t = threading.Thread(target=_signaller)
        t.start()
        # Generous timeout — 2s for a 20ms delay.
        sem.wait(1, timeout_s=2.0)
        t.join()
        assert sem.query() == 1
    finally:
        sem.close()


# --------------------------------------------------------------------
# Command buffer + queue submit
# --------------------------------------------------------------------


def test_queue_submit_copy(device) -> None:
    src = Buffer(device, size=16)
    dst = Buffer(device, size=16)
    try:
        src.write(b"\xab" * 16)
        cb = CommandBuffer(device)
        try:
            cb.begin().copy(src, dst, 16).end()
            sig = Semaphore(device, initial_value=0)
            try:
                submit(device, cb, signal=[(sig, 1)])
                sig.wait(1)
                assert dst.read() == b"\xab" * 16
            finally:
                sig.close()
        finally:
            cb.close()
    finally:
        src.close()
        dst.close()


def test_queue_submit_fill(device) -> None:
    buf = Buffer(device, size=32)
    try:
        cb = CommandBuffer(device)
        try:
            cb.begin().fill(buf, size=32, pattern=0x11223344).end()
            sig = Semaphore(device, initial_value=0)
            try:
                submit(device, cb, signal=[(sig, 1)])
                sig.wait(1)
                read_back = buf.read()
                # Little-endian 4-byte pattern repeated 8x.
                expected = struct.pack("<I", 0x11223344) * 8
                assert read_back == expected
            finally:
                sig.close()
        finally:
            cb.close()
    finally:
        buf.close()


def test_queue_submit_dispatch_cpu_kernel(device) -> None:
    """Run a Python callback through the CPU executable trampoline.

    The callback adds two float32 arrays into a third. This is the
    same shape the C test_command_buffer::cb_dispatch_executes case
    uses, but driven from Python.
    """
    n = 16
    ba = Buffer(device, size=n * 4)
    bb = Buffer(device, size=n * 4)
    bc = Buffer(device, size=n * 4)
    try:
        import numpy as np

        a = np.arange(n, dtype=np.float32)
        b = np.full(n, 100.0, dtype=np.float32)
        ba.write(a.tobytes())
        bb.write(b.tobytes())

        def add_kernel(pc: bytes, bindings: list[memoryview]) -> int:
            (count,) = struct.unpack("<I", pc)
            if len(bindings) != 3 or count != n:
                return 1
            a_view = np.frombuffer(bindings[0], dtype=np.float32, count=count)
            b_view = np.frombuffer(bindings[1], dtype=np.float32, count=count)
            c_view = np.frombuffer(bindings[2], dtype=np.float32, count=count)
            c_view[:] = a_view + b_view
            return 0

        exe = Executable(device, add_kernel)
        cb = CommandBuffer(device)
        try:
            cb.begin()
            cb.dispatch(exe, [ba, bb, bc], push_constants=struct.pack("<I", n))
            cb.end()

            sig = Semaphore(device, initial_value=0)
            try:
                submit(device, cb, signal=[(sig, 1)])
                sig.wait(1)
                result = np.frombuffer(bc.read(), dtype=np.float32)
                np.testing.assert_allclose(result, a + b)
            finally:
                sig.close()
        finally:
            cb.close()
            exe.close()
    finally:
        ba.close()
        bb.close()
        bc.close()


def test_cross_queue_dependency(device) -> None:
    """Queue 0 fills a buffer; queue 1 waits on the signal, then copies it."""
    mid = Buffer(device, size=16)
    out = Buffer(device, size=16)
    try:
        cb0 = CommandBuffer(device)
        cb0.begin().fill(mid, size=16, pattern=0xDEADBEEF).end()
        cb1 = CommandBuffer(device)
        cb1.begin().copy(mid, out, 16).end()

        fill_done = Semaphore(device, 0)
        copy_done = Semaphore(device, 0)
        try:
            submit(device, cb0, queue_index=0, signal=[(fill_done, 1)])
            submit(
                device,
                cb1,
                queue_index=1,
                wait=[(fill_done, 1)],
                signal=[(copy_done, 1)],
            )
            copy_done.wait(1)
            expected = struct.pack("<I", 0xDEADBEEF) * 4
            assert out.read() == expected
        finally:
            fill_done.close()
            copy_done.close()
            cb0.close()
            cb1.close()
    finally:
        mid.close()
        out.close()


# --------------------------------------------------------------------
# Event tensor
# --------------------------------------------------------------------


def test_event_tensor_notify_drops_counter(device) -> None:
    et = EventTensor(device, shape=(4,), dtype=EventDType.I32, initial_value=3)
    try:
        et.notify(1, 1)
        et.notify(1, 1)
        assert et.query(1) == 1
        et.notify(1, 1)
        # Wait should now return immediately.
        et.wait(1, timeout_s=0)
        assert et.query(1) == 0
        # Unrelated cells untouched.
        assert et.query(0) == 3
    finally:
        et.close()


def test_event_tensor_wait_blocks_until_notify(device) -> None:
    et = EventTensor(device, shape=(2, 2), dtype=EventDType.I64, initial_value=4)
    try:

        def _notifier() -> None:
            for _ in range(4):
                time.sleep(0.005)
                et.notify((0, 1), 1)

        t = threading.Thread(target=_notifier)
        t.start()
        et.wait((0, 1), timeout_s=2.0)
        t.join()
        assert et.query((0, 1)) == 0
        # Other cells untouched.
        assert et.query((1, 1)) == 4
    finally:
        et.close()


def test_event_tensor_reset(device) -> None:
    et = EventTensor(device, shape=(3,), initial_value=1)
    try:
        for i in range(3):
            et.notify(i, 1)
        for i in range(3):
            assert et.query(i) == 0
        et.reset(5)
        for i in range(3):
            assert et.query(i) == 5
    finally:
        et.close()
