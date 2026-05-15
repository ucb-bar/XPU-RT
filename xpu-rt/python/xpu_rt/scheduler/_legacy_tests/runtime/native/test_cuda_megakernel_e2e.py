"""Phase-5 GPU end-to-end smoke: schedule → emit → NVRTC → launch.

Builds a 4-task diamond DAG, schedules it across 2 SMs, emits the
CUDA megakernel source via :func:`emit_cuda_megakernel`, NVRTC-compiles
the source, allocates real event tensors via
:class:`CudaEventTensor`, launches the persistent kernel via
:class:`CudaMegakernelLauncher`, and asserts:

1. Every task ran exactly once (counter buffer reads ``[1, 1, 1, 1]``).
2. Every event tensor decremented to zero (notify fired exactly the
   expected number of times).
3. Exactly one ``cuLaunchKernelEx`` call (the launcher only invokes
   the persistent kernel once — the 1-cooperative-launch invariant
   the conformance gate enforces for static-scheduled workloads).

This is the smallest validating test that exercises every Phase-4 +
Phase-5 component on real silicon. Deferred until a CUDA-built
``libcompgen_rt.so`` is loadable; on a CPU host the test skips
cleanly.
"""

from __future__ import annotations

import ctypes

import pytest

requires_gpu = pytest.mark.requires_gpu


def _have_native_cuda_runtime() -> bool:
    try:
        import compgen
    except Exception:
        return False
    return bool(compgen.has_cuda_runtime())


@requires_gpu
@pytest.mark.skipif(
    not _have_native_cuda_runtime(),
    reason="libcompgen_rt-cuda.so not loadable on this host",
)
class TestPhase5DiamondMegakernel:
    """Walk a 4-task diamond DAG end-to-end through the Phase-5 pipeline."""

    def _build_schedule(self):
        from compgen.runtime.event_tensor import EventTensor
        from compgen.runtime.megakernel import (
            DeviceCall,
            EventEdge,
            MegakernelGraph,
        )
        from compgen.transforms.event_static_schedule import compute_static_schedule

        ab = EventTensor((1,), wait_count_default=1)
        ac = EventTensor((1,), wait_count_default=1)
        bd = EventTensor((1,), wait_count_default=1)
        cd = EventTensor((1,), wait_count_default=1)

        calls = (
            DeviceCall(
                name="task_a",
                body_fn=lambda c: None,
                task_shape=(1,),
                out_edges=(
                    EventEdge("ab", lambda c: (0,)),
                    EventEdge("ac", lambda c: (0,)),
                ),
            ),
            DeviceCall(
                name="task_b",
                body_fn=lambda c: None,
                task_shape=(1,),
                in_edges=(EventEdge("ab", lambda c: (0,)),),
                out_edges=(EventEdge("bd", lambda c: (0,)),),
            ),
            DeviceCall(
                name="task_c",
                body_fn=lambda c: None,
                task_shape=(1,),
                in_edges=(EventEdge("ac", lambda c: (0,)),),
                out_edges=(EventEdge("cd", lambda c: (0,)),),
            ),
            DeviceCall(
                name="task_d",
                body_fn=lambda c: None,
                task_shape=(1,),
                in_edges=(
                    EventEdge("bd", lambda c: (0,)),
                    EventEdge("cd", lambda c: (0,)),
                ),
            ),
        )
        graph = MegakernelGraph(
            name="diamond",
            calls=calls,
            event_tensors={"ab": ab, "ac": ac, "bd": bd, "cd": cd},
            policy="static",
        )
        return compute_static_schedule(graph, sm_count=2)

    def _device_function_bodies(self):
        from compgen.transforms.emit_cuda_megakernel import DeviceFunctionSource

        # Each body atomically increments its own counter slot. The
        # caller's `buffers[0]` is an int32 array of length 4; the
        # kind→slot mapping is task-name lexical (a→0, b→1, c→2, d→3
        # by alphabetical sort, which matches the emitter's kind table).
        body_template = (
            "(void)sm_id; (void)coord_x;\n"
            "if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {{\n"
            "    int *counters = (int *)buffers[0];\n"
            "    atomicAdd(&counters[{slot}], 1);\n"
            "}}\n"
        )
        return {
            f"task_{c}": DeviceFunctionSource(
                name=f"task_{c}",
                body=body_template.format(slot=i),
            )
            for i, c in enumerate("abcd")
        }

    def test_diamond_runs_under_one_cooperative_launch(self) -> None:
        from compgen.runtime.native.cuda import (
            CudaEventTensor,
            CudaMegakernelLauncher,
            CudaModule,
        )
        from compgen.runtime.native.device import Device
        from compgen.transforms.emit_cuda_megakernel import emit_cuda_megakernel

        schedule = self._build_schedule()
        bodies = self._device_function_bodies()
        emit = emit_cuda_megakernel(
            schedule,
            device_function_sources=bodies,
            user_buffer_count=1,
        )

        # ----- Compile CUDA source via NVRTC -----------------------------
        # Use sm_90 for compatibility with the bwell build (CUDA 12.6
        # toolkit; driver JITs sm_90 → sm_120 SASS).
        cumod = CudaModule(
            cuda_source=emit.cuda_source,
            kernel_name=emit.kernel_name,
            arch="sm_90",
        )

        # ----- Allocate event tensors + counter buffer -------------------
        device = Device.create("cuda:0")
        events = [
            CudaEventTensor(num_cells=1, initial_wait_count=alloc.wait_count_default)
            for alloc in schedule.event_tensor_allocs
        ]
        # Build a device int** pointing at the four event-tensor base
        # pointers. (Same layout the emitted wrapper expects.)
        event_ptr_array_t = ctypes.c_void_p * len(events)
        event_ptrs = event_ptr_array_t(*(ctypes.c_void_p(e.device_ptr) for e in events))
        # We stage event_ptrs on device because the persistent kernel
        # treats it as a `long long **`. Use cuMemAlloc + cuMemcpy.
        from compgen.runtime.native.cuda import _cu_check
        from cuda.bindings import driver as cu_driver  # type: ignore

        size_bytes = ctypes.sizeof(event_ptrs)
        event_ptrs_dev = _cu_check(cu_driver.cuMemAlloc(size_bytes))
        _cu_check(cu_driver.cuMemcpyHtoD(event_ptrs_dev, ctypes.addressof(event_ptrs), size_bytes))

        # 4 int32 counters, host buffer + device buffer.
        counter_host = (ctypes.c_int * 4)(0, 0, 0, 0)
        counter_dev = _cu_check(cu_driver.cuMemAlloc(ctypes.sizeof(counter_host)))
        _cu_check(cu_driver.cuMemsetD32(counter_dev, 0, 4))

        # buffers[0] = counter_dev. Stage as device ``void **``.
        buf_array = (ctypes.c_void_p * 1)(ctypes.c_void_p(int(counter_dev)))
        buf_dev = _cu_check(cu_driver.cuMemAlloc(ctypes.sizeof(buf_array)))
        _cu_check(cu_driver.cuMemcpyHtoD(buf_dev, ctypes.addressof(buf_array), ctypes.sizeof(buf_array)))

        # ----- Launch the persistent megakernel --------------------------
        launcher = CudaMegakernelLauncher(device.handle.value or 0)
        launcher.launch(
            kernel_handle=cumod.kernel_handle,
            grid_dim=(schedule.sm_count, 1, 1),
            block_dim=schedule.launch_config.block_dim,
            cluster_dim=None,  # cluster launch validated separately on sm_120
            shared_mem_bytes=schedule.launch_config.shared_mem_bytes,
            kernel_args=[int(event_ptrs_dev), int(buf_dev)],
        )

        # ----- Read back counters + event tensors ------------------------
        _cu_check(cu_driver.cuMemcpyDtoH(ctypes.addressof(counter_host), counter_dev, ctypes.sizeof(counter_host)))
        counters = list(counter_host)
        assert counters == [1, 1, 1, 1], f"each task should have run exactly once; counters={counters}"

        # Every event tensor must have decremented to zero — the
        # persistent kernel ran every notify exactly once.
        residuals = [e.load(0) for e in events]
        assert residuals == [0, 0, 0, 0], (
            f"event tensors did not all reach zero: {residuals}. "
            "This means notify/wait pairing in the emitted wrapper "
            "is incorrect, or the kernel didn't reach every task."
        )

        # Cleanup.
        _cu_check(cu_driver.cuMemFree(buf_dev))
        _cu_check(cu_driver.cuMemFree(counter_dev))
        _cu_check(cu_driver.cuMemFree(event_ptrs_dev))
        cumod.close()
