"""cpu_task driver — async queue_submit + cross-queue parallelism.

These tests only exercise behaviour that distinguishes ``cpu_task`` from
``cpu_sync``:
    - submit returns before execution
    - producer/consumer semaphore chains survive reversed submit order
    - workers on different queues run concurrently

The shared primitive surface (buffers, command buffers, event tensors)
is already covered by ``tests/runtime/test_libcompgen_rt.py``. The
traits smoke test here is the only duplicate — it's a quick sanity
check that the driver registered.
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
    Instance,
    Semaphore,
    submit,
)


@pytest.fixture(scope="module")
def device():
    inst = Instance("cpu_task")
    dev = inst.open_device(0)
    yield dev
    dev.close()
    inst.close()


def test_cpu_task_traits(device) -> None:
    t = device.traits
    assert t.device_class == DeviceClass.CPU
    assert t.name == "cpu_task"
    assert t.max_concurrent_queues >= 2


def test_submit_is_async(device) -> None:
    """Submit behind a gate semaphore; the kernel must NOT run
    until we release the gate. This verifies the driver is genuinely
    async: on cpu_sync the submit would block on the gate wait and
    we'd never reach the post-submit line."""
    buf = Buffer(device, size=16)
    try:
        cb = CommandBuffer(device)
        cb.begin().fill(buf, size=16, pattern=0xA5A5A5A5).end()

        gate = Semaphore(device, 0)
        done = Semaphore(device, 0)
        try:
            submit(device, cb, wait=[(gate, 1)], signal=[(done, 1)])
            # done is not yet signalled because the worker is blocked on gate.
            with pytest.raises(CgRtError) as excinfo:
                done.wait(1, timeout_s=0)
            assert excinfo.value.status == -5  # TIMED_OUT

            gate.signal(1)
            done.wait(1, timeout_s=2.0)

            # Worker completed; buffer holds the fill.
            expected = struct.pack("<I", 0xA5A5A5A5) * 4
            assert buf.read() == expected
        finally:
            gate.close()
            done.close()
            cb.close()
    finally:
        buf.close()


def test_reverse_submit_order(device) -> None:
    """Submit the *consumer* first, then the producer. The consumer
    worker must block on the producer's signal, not spin or error."""
    mid = Buffer(device, size=16)
    out = Buffer(device, size=16)
    try:
        cb_fill = CommandBuffer(device)
        cb_fill.begin().fill(mid, size=16, pattern=0xDEADBEEF).end()
        cb_copy = CommandBuffer(device)
        cb_copy.begin().copy(mid, out, 16).end()

        fill_done = Semaphore(device, 0)
        copy_done = Semaphore(device, 0)
        try:
            # Consumer first — blocks its worker until the producer fires.
            submit(
                device,
                cb_copy,
                queue_index=1,
                wait=[(fill_done, 1)],
                signal=[(copy_done, 1)],
            )
            # Producer — signals fill_done, worker on queue 1 wakes up,
            # copies mid -> out, signals copy_done.
            submit(device, cb_fill, queue_index=0, signal=[(fill_done, 1)])
            copy_done.wait(1, timeout_s=2.0)

            expected = struct.pack("<I", 0xDEADBEEF) * 4
            assert out.read() == expected
        finally:
            fill_done.close()
            copy_done.close()
            cb_fill.close()
            cb_copy.close()
    finally:
        mid.close()
        out.close()


def test_cross_queue_parallelism(device) -> None:
    """Two slow kernels on different queues should complete in ~max
    rather than ~sum of their runtimes, confirming real cross-queue
    parallelism. We check concurrency by comparing the observed
    elapsed time to the sum of the per-kernel sleeps; a sequential
    driver would take at least the sum."""
    # 40ms of simulated work each, so serialized would take ~80ms.
    # We assert < 60ms to leave headroom for thread startup while
    # still catching a serialisation regression.
    KERNEL_SLEEP_S = 0.04
    BUDGET_S = 0.06

    cbs = []
    sigs = []
    try:
        sig_a = Semaphore(device, 0)
        sig_b = Semaphore(device, 0)
        sigs = [sig_a, sig_b]

        def _enqueue(queue_idx: int, sig: Semaphore) -> CommandBuffer:
            buf = Buffer(device, size=4)
            # Use Python-side fill via the kernel — the simplest way
            # to inject a sleep into the worker path. We dispatch a
            # kernel that just sleeps.
            from compgen.runtime.native.device import Executable

            def sleeper(pc: bytes, bindings: list[memoryview]) -> int:
                time.sleep(KERNEL_SLEEP_S)
                return 0

            exe = Executable(device, sleeper)
            cb = CommandBuffer(device)
            cb.begin().dispatch(exe, [buf]).end()
            cbs.append((cb, buf, exe))
            submit(device, cb, queue_index=queue_idx, signal=[(sig, 1)])
            return cb

        start = time.perf_counter()
        _enqueue(0, sig_a)
        _enqueue(1, sig_b)
        sig_a.wait(1, timeout_s=2.0)
        sig_b.wait(1, timeout_s=2.0)
        elapsed = time.perf_counter() - start

        assert elapsed < BUDGET_S, (
            f"parallel submits took {elapsed:.3f}s, expected < {BUDGET_S}s (driver may be serialising across queues)"
        )
    finally:
        for sig in sigs:
            sig.close()
        for cb, buf, exe in cbs:
            cb.close()
            exe.close()
            buf.close()


def test_many_submits_preserve_order(device) -> None:
    """Submit 50 appends to the same queue; observed tick order
    must match submit order (same-queue FIFO invariant)."""
    N = 50
    results: list[int] = []
    results_lock = threading.Lock()

    from compgen.runtime.native.device import Executable

    ticks = {"counter": 0}

    def make_kernel(expected_idx: int):
        def kernel(pc: bytes, bindings: list[memoryview]) -> int:
            with results_lock:
                results.append(expected_idx)
                ticks["counter"] += 1
            return 0

        return kernel

    buf = Buffer(device, size=4)
    cbs = []
    exes = []
    final_sig = Semaphore(device, 0)
    try:
        for i in range(N):
            exe = Executable(device, make_kernel(i))
            cb = CommandBuffer(device)
            cb.begin().dispatch(exe, [buf]).end()
            cbs.append(cb)
            exes.append(exe)
            # Only the last one signals, so we know when to wake up.
            signal = [(final_sig, 1)] if i == N - 1 else None
            submit(device, cb, queue_index=0, signal=signal)
        final_sig.wait(1, timeout_s=5.0)
        assert results == list(range(N)), f"unexpected order: {results}"
    finally:
        final_sig.close()
        for cb in cbs:
            cb.close()
        for exe in exes:
            exe.close()
        buf.close()
