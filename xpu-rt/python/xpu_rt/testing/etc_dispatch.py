"""Phase-7 ETC dispatch: compile + run + measure for a single workload.

Glues together everything Phases 2-5 produced into a single function
the conformance harness can call:

    compile_and_run_etc_workload(
        workload_name="diamond_dag",
        dtype="fp32",
        device_index=0,
        num_correctness_inputs=16,
        num_benchmark_iters=50,
        output_path=Path("/tmp/conf_diamond"),
    )
    →  (correctness, timing, launch_profile, bundle_dir)

Steps:

1. Build the workload (model + sample_inputs + MegakernelGraph
   factory + CUDA C++ device-function bodies).
2. Run :func:`compute_static_schedule` to get the per-SM queues +
   event-tensor allocation specs.
3. Run :func:`emit_cuda_megakernel` to produce the persistent kernel
   source + manifest.
4. Stage them into ``bundle/megakernel/source.cu`` +
   ``manifest.yaml``.
5. NVRTC-compile via :class:`CudaModule`.
6. Allocate the user-buffer set + event tensors on device.
7. Run ``num_correctness_inputs`` random inputs through both the
   eager model and the megakernel; compute max abs/rel err.
8. Time ``num_benchmark_iters`` iterations of each. Compute the
   speedup ratio.
9. Read launch-profile data: ``num_launches=1`` (the launcher does
   exactly one cooperative ``cuLaunchKernelEx``), atomics counted by
   regex over the emitted source (every ``atomicAdd_system`` /
   ``atomicExch_system`` is a notify/update/trigger; every
   ``__nanosleep`` inside a ``while`` is a wait spin).

Returns the four data dicts ready for :func:`_evaluate_gate`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import torch

from compgen.testing.workloads import WORKLOAD_FACTORIES
from compgen.transforms.emit_cuda_megakernel import (
    emit_cuda_megakernel,
)
from compgen.transforms.event_static_schedule import (
    StaticSchedule,
    compute_static_schedule,
)

log = structlog.get_logger(__name__)


class EtcDispatchError(RuntimeError):
    """Raised when the ETC dispatch path fails for a known reason
    (workload unsupported, compile error, hardware mismatch).

    Distinct from RuntimeError so the harness's gate evaluation can
    classify "ran but didn't pass the bar" vs "couldn't run at all".
    """


def compile_and_run_etc_workload(
    *,
    workload_name: str,
    dtype: str,
    device_index: int,
    num_correctness_inputs: int,
    num_benchmark_iters: int,
    output_path: Path,
) -> tuple[dict[str, float], dict[str, float], dict[str, int], Path]:
    """Run one workload end-to-end via ETC dispatch.

    Args:
        workload_name: Key in :data:`WORKLOAD_FACTORIES`.
        dtype: Forwarded to the workload factory.
        device_index: ``cuda:<index>`` to run on.
        num_correctness_inputs: Random inputs to compare eager vs
            megakernel.
        num_benchmark_iters: Timed iterations after warmup.
        output_path: Bundle staging dir.

    Returns:
        ``(correctness, timing, launch_profile, bundle_dir)``.

    Raises:
        EtcDispatchError: workload not registered, compile failure,
            or non-recoverable runtime failure.
    """
    if workload_name not in WORKLOAD_FACTORIES:
        raise EtcDispatchError(
            f"workload {workload_name!r} not registered in "
            "compgen.testing.workloads.WORKLOAD_FACTORIES. "
            f"Available: {sorted(WORKLOAD_FACTORIES)}"
        )

    # Multi-rank workloads (gemm_rs, ag_gemm) take a different
    # dispatch path: allocate per-rank state, run each rank's
    # megakernel, then drive the cross-rank NCCL collective. The
    # single-rank path below is unchanged.
    if workload_name in ("gemm_rs", "ag_gemm"):
        return _compile_and_run_multi_gpu(
            workload_name=workload_name,
            dtype=dtype,
            num_correctness_inputs=num_correctness_inputs,
            num_benchmark_iters=num_benchmark_iters,
            output_path=output_path,
        )

    workload = WORKLOAD_FACTORIES[workload_name](dtype=dtype, num_gpus=1)

    # ---- 1. Build event graph + schedule -----------------------------
    graph = workload.build_megakernel_graph(workload.model, workload.sample_inputs)
    sm_count = _resolve_sm_count(device_index)
    schedule = compute_static_schedule(
        graph,
        sm_count=sm_count,
        # Tile-level cost hints not yet derived from a roofline model;
        # equal weights are fine for the diamond_dag scheduling
        # fairness gate (every task is ~equally expensive at this
        # tile size).
        cost_hints_us=None,
        # 32x32 = 1024 threads per block. The diamond's GEMM bodies
        # use threadIdx.x / threadIdx.y for shared-memory tile
        # coordinates; the elementwise bodies use a flattened 1D
        # stride loop so they're agnostic to the block shape.
        block_dim=(32, 32, 1),
    )

    # ---- 2. Emit CUDA megakernel + stage bundle ---------------------
    emit = emit_cuda_megakernel(
        schedule,
        device_function_sources=workload.device_function_sources,
        user_buffer_count=len(workload.user_buffer_layout),
    )
    bundle_dir = output_path / f"bundle_{workload_name}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    emit.write_to_bundle(bundle_dir / "megakernel")

    # ---- 3. NVRTC + cooperative launch infrastructure ---------------
    from compgen.runtime.native.cuda import (
        CudaMegakernelLauncher,
        CudaModule,
    )
    from compgen.runtime.native.device import Device

    cumod = CudaModule(
        cuda_source=emit.cuda_source,
        kernel_name=emit.kernel_name,
        arch="sm_90",  # JITs to sm_120 on Blackwell workstation
    )
    device = Device.create(f"cuda:{device_index}")
    launcher = CudaMegakernelLauncher(device.handle.value or 0)

    # ---- 4. Stage device-side state once (event tensors + ptr arrays)
    sample_x = workload.sample_inputs[0]
    state = _allocate_etc_state(
        workload=workload,
        schedule=schedule,
        sample_x=sample_x,
        device_index=device_index,
    )

    try:
        # ---- 5. Run correctness sweep -------------------------------
        correctness = _run_correctness(
            workload=workload,
            workload_name=workload_name,
            schedule=schedule,
            cumod=cumod,
            launcher=launcher,
            emit=emit,
            device_index=device_index,
            num_inputs=num_correctness_inputs,
            state=state,
        )

        # ---- 6. Benchmark ETC vs eager ------------------------------
        timing = _run_benchmark(
            workload=workload,
            schedule=schedule,
            cumod=cumod,
            launcher=launcher,
            emit=emit,
            device_index=device_index,
            num_iters=num_benchmark_iters,
            state=state,
        )
    finally:
        _release_state(state)

    # ---- 7. Launch profile ------------------------------------------
    launch_profile = _profile_launch(emit.cuda_source, schedule)

    # Cleanup
    cumod.close()
    device.close()

    log.info(
        "etc_dispatch.done",
        workload=workload_name,
        max_abs_err=correctness.get("max_abs_err"),
        max_rel_err=correctness.get("max_rel_err"),
        speedup_vs_eager=timing.get("speedup_vs_eager"),
        num_launches=launch_profile.get("num_launches"),
        notify_atomics=launch_profile.get("notify_atomics"),
    )
    return correctness, timing, launch_profile, bundle_dir


def _resolve_sm_count(device_index: int) -> int:
    """Probe the device for its SM count. Falls back to 132 (B200)
    if the probe fails — the harness's perf gate is per-device, so a
    misjudged SM count just means the schedule is suboptimal, not
    incorrect."""
    try:
        from compgen.runtime.probe import probe_cuda_device

        probe = probe_cuda_device(device_index)
        sm = probe.get("sm_count") or probe.get("multi_processor_count") or 132
        return int(sm)
    except Exception:
        return 132


def _run_correctness(
    *,
    workload: Any,
    workload_name: str,
    schedule: StaticSchedule,
    cumod: Any,
    launcher: Any,
    emit: Any,
    device_index: int,
    num_inputs: int,
    state: _DispatchState,
) -> dict[str, float]:
    """Run ``num_inputs`` random inputs through both paths.

    Eager: run on GPU (same device as ETC) for a fair comparison.
    ETC: reset event tensors, launch, read back yout.

    Uses numpy-allclose semantics for the per-element pass check
    (``|a-b| <= atol + rtol*|b|``). Reports ``num_failing_elements``
    alongside the diagnostic ``max_abs_err`` / ``max_rel_err`` so the
    harness gate stops false-positiving on tiny outputs near zero
    where fp32 ULP noise blows up the naive relative error.
    """
    from compgen.testing.etc_conformance import gate_for

    gate = gate_for(workload_name)
    atol = gate.correctness_atol
    rtol = gate.correctness_rtol

    max_abs = 0.0
    max_rel = 0.0
    num_failing = 0
    sample_x = workload.sample_inputs[0]
    device = torch.device(f"cuda:{device_index}")
    eager_model = workload.model.to(device).eval()

    for _ in range(num_inputs):
        x = torch.randn_like(sample_x)
        x_dev = x.to(device, dtype=torch.float32)
        with torch.no_grad():
            y_eager = eager_model(x_dev).detach().to(torch.float32).cpu()

        y_etc = _launch_and_readback(
            workload=workload,
            schedule=schedule,
            launcher=launcher,
            cumod=cumod,
            emit=emit,
            x=x,
            state=state,
        )

        diff = (y_etc - y_eager).abs()
        # allclose semantics: failing iff |a-b| > atol + rtol*|b|.
        tolerance = atol + rtol * y_eager.abs()
        num_failing += int((diff > tolerance).sum().item())

        max_abs = max(max_abs, float(diff.max().item()))
        denom = y_eager.abs().clamp_min(1e-6)
        max_rel = max(max_rel, float((diff / denom).max().item()))

    return {
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "num_failing_elements": float(num_failing),
        "num_inputs": float(num_inputs),
    }


def _run_benchmark(
    *,
    workload: Any,
    schedule: StaticSchedule,
    cumod: Any,
    launcher: Any,
    emit: Any,
    device_index: int,
    num_iters: int,
    state: _DispatchState,
) -> dict[str, float]:
    """Time both paths and compute the speedup ratio.

    Both paths run on the same CUDA device — comparing GPU eager vs
    GPU ETC. Each timing block is bracketed by ``torch.cuda.synchronize``
    so eager's async launch queue doesn't underestimate its cost.
    """
    sample_x = workload.sample_inputs[0]
    device = torch.device(f"cuda:{device_index}")
    eager_model = workload.model.to(device).eval()
    sample_x_dev = sample_x.to(device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            eager_model(sample_x_dev)
        _launch_and_readback(
            workload=workload,
            schedule=schedule,
            launcher=launcher,
            cumod=cumod,
            emit=emit,
            x=sample_x,
            state=state,
        )
    torch.cuda.synchronize(device)

    # Eager — sync before + after so we measure actual completion.
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            eager_model(sample_x_dev)
    torch.cuda.synchronize(device)
    eager_us = (time.perf_counter() - t0) * 1e6 / num_iters

    # ETC — the launcher already calls cudaDeviceSynchronize per
    # launch, so each iteration completes before the next starts.
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _launch_and_readback(
            workload=workload,
            schedule=schedule,
            launcher=launcher,
            cumod=cumod,
            emit=emit,
            x=sample_x,
            state=state,
        )
    etc_us = (time.perf_counter() - t0) * 1e6 / num_iters

    speedup = eager_us / max(etc_us, 1e-6)
    return {
        "etc_us": etc_us,
        "eager_us": eager_us,
        "speedup_vs_eager": speedup,
    }


def _profile_launch(cuda_source: str, schedule: StaticSchedule) -> dict[str, int]:
    """Count notify/wait sites in the emitted source.

    The 1-launch invariant comes from the launcher itself (it does
    exactly one ``cuLaunchKernelEx`` per call), so ``num_launches=1``
    is hardcoded for the static scheduler.

    Atomics + wait sites are counted by regex — the emitter inlines
    primitives, so each occurrence of ``atomicAdd_system`` /
    ``atomicExch_system`` and each ``while ( ... > 0 ) { __nanosleep``
    is a real device-side instruction.
    """
    # Notifies: ``atomicAdd_system((unsigned long long *)&E[idx], (unsigned long long)(-...))``
    # plus the cells emitted in CG_OUT_CELLS.
    notify_atomics = sum(1 for q in schedule.sm_queues for t in q.tasks for _ in t.out_cells)
    wait_sites = sum(1 for q in schedule.sm_queues for t in q.tasks for _ in t.in_cells)
    # Sanity: the source itself contains the primitive bodies.
    assert "atomicAdd_system" in cuda_source
    return {
        "num_launches": 1,
        "notify_atomics": notify_atomics,
        "wait_sites": wait_sites,
        "cluster_launch": 1 if schedule.launch_config.cluster_dim is not None else 0,
        "fp8_mmas": int(bool(re.search(r"wgmma|mma\.fp8", cuda_source))),
    }


# ---------------------------------------------------------------------------
# Device-state helpers
# ---------------------------------------------------------------------------


@dataclass
class _DispatchState:
    """Per-workload state staged once and reused across launches.

    ETC dispatch's per-launch hot path used to allocate fresh event
    tensors + cuMemAlloc/Free the device pointer arrays each time.
    On a 14-µs reference workload that overhead dominated and pinned
    the speedup ratio at ~0.12×. By caching everything that doesn't
    depend on the input, we collapse per-launch host work to "memcpy
    x + cuMemcpyHtoD reset to event-tensor cells + launch".
    """

    user_buffers: dict[str, torch.Tensor]
    event_tensors: list[Any]
    et_ptrs_dev: int  # CUdeviceptr (as int)
    buf_dev: int  # CUdeviceptr (as int)
    et_reset_host_ptr: int  # host pointer to cached int64 reset buffer
    et_reset_host_size: int  # bytes
    _backref: list[Any] = field(default_factory=list)  # keep ctypes objects alive


def _allocate_etc_state(
    *,
    workload: Any,
    schedule: StaticSchedule,
    sample_x: torch.Tensor,
    device_index: int,
    rank: int = 0,
) -> _DispatchState:
    """Allocate everything that doesn't change between launches.

    Returns a :class:`_DispatchState` whose ``et_ptrs_dev`` /
    ``buf_dev`` device pointers are valid for the lifetime of the
    dispatch. The caller is expected to call :func:`_release_state`
    to free them when done.
    """
    import ctypes

    from cuda.bindings import driver as cu_driver  # type: ignore

    from compgen.runtime.native.cuda import (
        CudaEventTensor,
        _cu_check,
        _ensure_cuda_driver_context,
    )

    _ensure_cuda_driver_context()

    # --- 1. Event tensors ----------------------------------------------
    event_tensors = [
        CudaEventTensor(
            num_cells=int(_prod(alloc.shape)),
            initial_wait_count=alloc.wait_count_default,
        )
        for alloc in schedule.event_tensor_allocs
    ]

    # --- 2. User buffers (torch tensors) -------------------------------
    # Each workload module knows the shapes of its own buffers.
    # ``_workload_buffers(workload, sample_x, device)`` is a thin
    # dispatcher keyed on the workload's class name; new workloads
    # add a branch. The Workload dataclass's
    # ``user_buffer_layout`` orders these into the device pointer
    # array consumed by the megakernel.
    device = torch.device(f"cuda:{device_index}")
    user_buffers = _workload_buffers(workload, sample_x, device, rank=rank)
    # Sanity: every name in user_buffer_layout must be present.
    missing = [n for n in workload.user_buffer_layout if n not in user_buffers]
    if missing:
        raise EtcDispatchError(
            f"workload {type(workload).__name__} buffer layout mismatch: "
            f"layout names {missing} have no allocated tensor"
        )

    # --- 3. event_tensors device pointer array (long long **) ---------
    et_ptr_array_t = ctypes.c_void_p * len(event_tensors)
    et_ptrs = et_ptr_array_t(*(ctypes.c_void_p(e.device_ptr) for e in event_tensors))
    et_ptrs_dev = _cu_check(cu_driver.cuMemAlloc(ctypes.sizeof(et_ptrs)))
    _cu_check(cu_driver.cuMemcpyHtoD(et_ptrs_dev, ctypes.addressof(et_ptrs), ctypes.sizeof(et_ptrs)))

    # --- 4. user buffers device pointer array (void **) ---------------
    buffer_ptrs_t = ctypes.c_void_p * len(workload.user_buffer_layout)
    buffer_ptrs = buffer_ptrs_t(
        *(ctypes.c_void_p(int(user_buffers[name].data_ptr())) for name in workload.user_buffer_layout)
    )
    buf_dev = _cu_check(cu_driver.cuMemAlloc(ctypes.sizeof(buffer_ptrs)))
    _cu_check(cu_driver.cuMemcpyHtoD(buf_dev, ctypes.addressof(buffer_ptrs), ctypes.sizeof(buffer_ptrs)))

    # --- 5. Cached event-tensor reset buffer (int64 wait_counts) ------
    # Each cell is an int64; pre-fill with each spec's
    # ``wait_count_default`` so a single cuMemcpyHtoD per launch
    # restores all event tensors to their initial state.
    total_cells = sum(_prod(a.shape) for a in schedule.event_tensor_allocs)
    et_reset_host = (ctypes.c_int64 * total_cells)()
    cursor = 0
    for a in schedule.event_tensor_allocs:
        cells = _prod(a.shape)
        for j in range(cells):
            et_reset_host[cursor + j] = int(a.wait_count_default)
        cursor += cells

    return _DispatchState(
        user_buffers=user_buffers,
        event_tensors=event_tensors,
        et_ptrs_dev=int(et_ptrs_dev),
        buf_dev=int(buf_dev),
        et_reset_host_ptr=ctypes.addressof(et_reset_host),
        et_reset_host_size=ctypes.sizeof(et_reset_host),
        _backref=[et_ptrs, buffer_ptrs, et_reset_host],
    )


def _release_state(state: _DispatchState) -> None:
    """Free the device-side staging buffers from
    :func:`_allocate_etc_state`. Event tensors release themselves
    via :class:`CudaEventTensor`'s ``__del__``."""
    from cuda.bindings import driver as cu_driver  # type: ignore

    from compgen.runtime.native.cuda import _cu_check

    if state.et_ptrs_dev:
        _cu_check(cu_driver.cuMemFree(state.et_ptrs_dev))
    if state.buf_dev:
        _cu_check(cu_driver.cuMemFree(state.buf_dev))


def _launch_and_readback(
    *,
    workload: Any,
    schedule: StaticSchedule,
    launcher: Any,
    cumod: Any,
    emit: Any,
    x: torch.Tensor,
    state: _DispatchState,
) -> torch.Tensor:
    """Copy x → device, reset event-tensor cells, launch, return yout.

    Hot path: a single H2D copy for x, a single H2D copy for the
    cached event-tensor reset buffer, the cooperative launch, and
    one D2H readback for ``yout``. No allocs.
    """
    from cuda.bindings import driver as cu_driver  # type: ignore

    from compgen.runtime.native.cuda import _cu_check

    # Flatten ND inputs to match the (batch_flat, in_dim) buffer the
    # matcher emits for the tile graph. Per bridge #118: with ND
    # inputs (e.g. ``(1, 64, 64)``) the buffer is alloc'd at
    # (batch_flat, in_dim) but the user tensor still carries leading
    # dims; ``copy_`` raises a broadcast error if we don't reshape
    # first.
    x_buf = state.user_buffers["x"]
    src = x.to(x_buf.device, dtype=torch.float32).reshape(x_buf.shape)
    x_buf.copy_(src)

    # Reset every event-tensor cell to its wait_count_default in one
    # H2D copy across the contiguous block of cells. Tensors were
    # allocated back-to-back in :func:`_allocate_etc_state`'s loop,
    # but their device pointers aren't necessarily contiguous; do
    # one H2D per tensor for correctness.
    cursor_bytes = 0
    int64_size = 8
    for et, alloc in zip(state.event_tensors, schedule.event_tensor_allocs, strict=True):
        cells = _prod(alloc.shape)
        nbytes = cells * int64_size
        _cu_check(
            cu_driver.cuMemcpyHtoD(
                et.device_ptr,
                state.et_reset_host_ptr + cursor_bytes,
                nbytes,
            )
        )
        cursor_bytes += nbytes

    launcher.launch(
        kernel_handle=cumod.kernel_handle,
        grid_dim=(schedule.sm_count, 1, 1),
        block_dim=schedule.launch_config.block_dim,
        cluster_dim=None,
        shared_mem_bytes=schedule.launch_config.shared_mem_bytes,
        kernel_args=[state.et_ptrs_dev, state.buf_dev],
    )
    # The output tensor's name varies by workload; pick the last
    # entry in user_buffer_layout, which is by convention the
    # final output (``yout`` for diamond, ``y_out`` for decoder).
    output_name = workload.user_buffer_layout[-1]
    output = state.user_buffers[output_name].detach().cpu()
    # Restore the ND leading dims that ``_workload_buffers`` flattened
    # so the readback shape matches the user's eager-output shape.
    # Per bridge #118 — with ``x.shape == (1, 64, 64)``, the buffer is
    # ``(64, 64)`` but the user expects ``(1, 64, 64)`` back.
    if x.ndim > 2 and output.ndim == 2:
        expected_shape = tuple(int(d) for d in x.shape[:-1]) + (output.shape[-1],)
        output = output.reshape(expected_shape)
    return output


def _workload_buffers(
    workload: Any,
    sample_x: torch.Tensor,
    device: torch.device,
    *,
    rank: int = 0,
) -> dict[str, torch.Tensor]:
    """Build the per-workload (per-rank) buffer dict.

    Each branch matches its workload module by the layout field it
    declares. Multi-rank workloads use ``rank`` to slice their
    sharded weights/inputs; single-rank workloads ignore it.

    Adding a new workload: add a branch keyed on the unique buffer
    names. The static dispatch keeps per-workload tensor allocation
    honest (no dynamic shape inference; you must declare what your
    bodies expect).
    """
    layout = workload.user_buffer_layout
    model = workload.model

    # Diamond: x → linear_a, linear_b → add → relu → yout.
    if layout == ("x", "wa", "wb", "ya", "yb", "yadd", "yout"):
        a_weight = model.a.weight.detach().to(device).contiguous().to(torch.float32)
        b_weight = model.b.weight.detach().to(device).contiguous().to(torch.float32)
        out_dim, in_dim = a_weight.shape
        # Flatten leading dims into the batch axis — matcher does the
        # same when emitting the tile graph (per bridge #109/#118), so
        # buffer allocation must mirror the flattened (batch_flat, in)
        # contract or the kernel reads stale memory and the readback
        # buffer is too small for the eager comparison's ND shape.
        batch = 1
        for d in sample_x.shape[:-1]:
            batch *= int(d)
        x_dev = torch.empty(batch, in_dim, device=device, dtype=torch.float32)
        ya = torch.empty(batch, out_dim, device=device, dtype=torch.float32)
        yb = torch.empty_like(ya)
        yadd = torch.empty_like(ya)
        yout = torch.empty_like(ya)
        return {
            "x": x_dev,
            "wa": a_weight,
            "wb": b_weight,
            "ya": ya,
            "yb": yb,
            "yadd": yadd,
            "yout": yout,
        }

    # Decoder layer (FFN portion): x → up → relu → down → y_out.
    if layout == ("x", "w_up", "w_down", "y_up", "y_relu", "y_out"):
        w_up = model.up.weight.detach().to(device).contiguous().to(torch.float32)
        w_down = model.down.weight.detach().to(device).contiguous().to(torch.float32)
        d_ff, d_model = w_up.shape
        batch = 1
        for d in sample_x.shape[:-1]:
            batch *= int(d)
        x_dev = torch.empty(batch, d_model, device=device, dtype=torch.float32)
        y_up = torch.empty(batch, d_ff, device=device, dtype=torch.float32)
        y_relu = torch.empty_like(y_up)
        y_out = torch.empty(batch, d_model, device=device, dtype=torch.float32)
        return {
            "x": x_dev,
            "w_up": w_up,
            "w_down": w_down,
            "y_up": y_up,
            "y_relu": y_relu,
            "y_out": y_out,
        }

    # Decoder layer FFN with epilogue fusion (Wave 2.5): no y_up
    # intermediate — linear_up_relu writes y_relu directly.
    if layout == ("x", "w_up", "w_down", "y_relu", "y_out"):
        w_up = model.up.weight.detach().to(device).contiguous().to(torch.float32)
        w_down = model.down.weight.detach().to(device).contiguous().to(torch.float32)
        d_ff, d_model = w_up.shape
        batch = 1
        for d in sample_x.shape[:-1]:
            batch *= int(d)
        x_dev = torch.empty(batch, d_model, device=device, dtype=torch.float32)
        y_relu = torch.empty(batch, d_ff, device=device, dtype=torch.float32)
        y_out = torch.empty(batch, d_model, device=device, dtype=torch.float32)
        return {
            "x": x_dev,
            "w_up": w_up,
            "w_down": w_down,
            "y_relu": y_relu,
            "y_out": y_out,
        }

    # gemm_reduce_scatter: per-rank shards of x (col-shard along K) +
    # W (row-shard along K). ``rank`` selects the K-band each rank owns.
    if layout == ("x_shard", "w_shard", "y_partial"):
        full_w = model.linear.weight.detach().contiguous().to(torch.float32)
        # nn.Linear.weight is (N, K_total); column-shard the K axis
        # so rank r owns columns ``r*K_local:(r+1)*K_local``.
        n, k_total = full_w.shape
        num_ranks = workload.num_ranks
        if k_total % num_ranks != 0:
            raise EtcDispatchError(f"gemm_rs needs K ({k_total}) divisible by num_ranks ({num_ranks})")
        k_local = k_total // num_ranks
        w_shard = full_w[:, rank * k_local : (rank + 1) * k_local].contiguous().to(device)
        # x is column-sharded along K: each rank gets ``(B, k_local)``.
        batch = sample_x.shape[0]
        x_shard = torch.empty(batch, k_local, device=device, dtype=torch.float32)
        y_partial = torch.empty(batch, n, device=device, dtype=torch.float32)
        return {
            "x_shard": x_shard,
            "w_shard": w_shard,
            "y_partial": y_partial,
        }

    raise EtcDispatchError(
        f"no buffer-allocation rule for workload layout {layout!r}; "
        "add a branch in compgen.testing.etc_dispatch._workload_buffers"
    )


def _prod(values: tuple[int, ...]) -> int:
    out = 1
    for v in values:
        out *= max(int(v), 1)
    return out


# ---------------------------------------------------------------------------
# Multi-GPU dispatch (Phase 4b round 2)
# ---------------------------------------------------------------------------


def _compile_and_run_multi_gpu(
    *,
    workload_name: str,
    dtype: str,
    num_correctness_inputs: int,
    num_benchmark_iters: int,
    output_path: Path,
) -> tuple[dict[str, float], dict[str, float], dict[str, int], Path]:
    """Per-rank megakernel + NCCL collective dispatcher.

    v1 implementation: each rank runs a local single-GPU megakernel
    that produces a per-rank partial output, then the harness
    drives ``CudaCommGroup.allreduce_fp32_sum`` on the partials and
    each rank slices its row band. v2 will replace the AllReduce
    with cross-rank Event Tensor edges so the entire forward is one
    cooperative launch across both ranks.
    """
    from compgen.runtime.native.cuda import (
        CudaCommGroup,
        CudaMegakernelLauncher,
        CudaModule,
    )
    from compgen.runtime.native.device import Device

    workload = WORKLOAD_FACTORIES[workload_name](dtype=dtype, num_gpus=2)
    num_ranks = workload.num_ranks
    if num_ranks != 2:
        raise EtcDispatchError(f"v1 multi-GPU dispatch supports num_ranks=2 only; got {num_ranks}")

    bundle_dir = output_path / f"bundle_{workload_name}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Per-rank megakernel emit. Each rank runs the SAME source —
    # bodies are shape-aware via the workload's K_LOCAL constant.
    graph = workload.build_megakernel_graph(workload.model, workload.sample_inputs)
    sm_count = _resolve_sm_count(0)
    schedule = compute_static_schedule(
        graph,
        sm_count=sm_count,
        cost_hints_us=None,
        block_dim=(32, 32, 1),
    )
    emit = emit_cuda_megakernel(
        schedule,
        device_function_sources=workload.device_function_sources,
        user_buffer_count=len(workload.user_buffer_layout),
    )
    emit.write_to_bundle(bundle_dir / "megakernel")

    # Per-rank infrastructure: each rank gets its own
    # CudaModule (NVRTC compile per-rank-context), Device, launcher,
    # and dispatch state. NVRTC compile is duplicated across ranks
    # but cheap relative to the megakernel work.
    # Critical ordering: ``_ensure_cuda_driver_context(rank)`` MUST
    # fire before any rank-local cuda call. Without the explicit
    # set-current, the first iteration's ctx0 stays current for
    # rank 1's setup → ``cg_rt_device_open(inst, 1)`` may fail
    # depending on the driver's enumeration state, and CudaModule
    # would load rank 1's PTX into rank 0's context.
    from compgen.runtime.native.cuda import _ensure_cuda_driver_context

    per_rank: list[dict[str, Any]] = []
    for rank in range(num_ranks):
        _ensure_cuda_driver_context(rank)
        rank_state: dict[str, Any] = {"rank": rank, "device_index": rank}
        cumod = CudaModule(
            cuda_source=emit.cuda_source,
            kernel_name=emit.kernel_name,
            arch="sm_90",
            device_index=rank,
        )
        device = Device.create(f"cuda:{rank}")
        launcher = CudaMegakernelLauncher(device.handle.value or 0, device_index=rank)
        rank_state["cumod"] = cumod
        rank_state["device"] = device
        rank_state["launcher"] = launcher
        sample_x = workload.sample_inputs[0]
        rank_state["state"] = _allocate_etc_state(
            workload=workload,
            schedule=schedule,
            sample_x=sample_x,
            device_index=rank,
            rank=rank,
        )
        per_rank.append(rank_state)

    # Initialise NCCL comm spanning both ranks.
    comm = CudaCommGroup(device_indices=list(range(num_ranks)))

    try:
        correctness = _run_correctness_multi(
            workload=workload,
            workload_name=workload_name,
            schedule=schedule,
            comm=comm,
            per_rank=per_rank,
            num_inputs=num_correctness_inputs,
        )
        timing = _run_benchmark_multi(
            workload=workload,
            schedule=schedule,
            comm=comm,
            per_rank=per_rank,
            num_iters=num_benchmark_iters,
        )
    finally:
        comm.close()
        for rank_state in per_rank:
            _release_state(rank_state["state"])
            rank_state["cumod"].close()
            rank_state["device"].close()

    launch_profile = _profile_launch(emit.cuda_source, schedule)
    # Total launches across ranks. v1 = 1 megakernel launch per rank.
    launch_profile["num_launches"] = num_ranks

    log.info(
        "etc_dispatch.multi_gpu.done",
        workload=workload_name,
        num_ranks=num_ranks,
        max_abs_err=correctness.get("max_abs_err"),
        speedup_vs_eager=timing.get("speedup_vs_eager"),
    )
    return correctness, timing, launch_profile, bundle_dir


def _run_correctness_multi(
    *,
    workload: Any,
    workload_name: str,
    schedule: StaticSchedule,
    comm: Any,
    per_rank: list[dict[str, Any]],
    num_inputs: int,
) -> dict[str, float]:
    """Multi-rank correctness sweep.

    Eager reference: full ``y = x @ W`` on rank 0's GPU.
    ETC: each rank runs its local megakernel on its shard, then
    ``comm.allreduce_fp32_sum`` sums the (B, N) partials across
    ranks; each rank's row band is its slice of the result.
    """
    from compgen.testing.etc_conformance import gate_for

    gate = gate_for(workload_name)
    atol = gate.correctness_atol
    rtol = gate.correctness_rtol

    max_abs = 0.0
    max_rel = 0.0
    num_failing = 0

    sample_x_full = workload.sample_inputs[0]
    eager_device = torch.device("cuda:0")
    eager_model = workload.model.to(eager_device).eval()

    num_ranks = workload.num_ranks
    batch = sample_x_full.shape[0]
    if batch % num_ranks != 0:
        raise EtcDispatchError(f"gemm_rs needs B ({batch}) divisible by num_ranks ({num_ranks})")
    rows_per_rank = batch // num_ranks

    for _ in range(num_inputs):
        x_full = torch.randn_like(sample_x_full)
        with torch.no_grad():
            y_eager_full = eager_model(x_full.to(eager_device, dtype=torch.float32)).detach().cpu()

        # Each rank runs its local megakernel on its column-shard.
        k_total = x_full.shape[1]
        k_local = k_total // num_ranks
        for rank_state in per_rank:
            rank = rank_state["rank"]
            x_shard = x_full[:, rank * k_local : (rank + 1) * k_local].to(
                rank_state["state"].user_buffers["x_shard"].device,
                dtype=torch.float32,
            )
            _launch_one_rank(
                workload=workload,
                schedule=schedule,
                rank_state=rank_state,
                x_shard=x_shard,
            )

        # AllReduce-sum the per-rank partials. Each rank's
        # ``y_partial`` is (B, N); after AllReduce all ranks have the
        # sum, which is the full y.
        y_partial_ptrs = [int(rs["state"].user_buffers["y_partial"].data_ptr()) for rs in per_rank]
        n = per_rank[0]["state"].user_buffers["y_partial"].shape[-1]
        comm.allreduce_fp32_sum(y_partial_ptrs, y_partial_ptrs, batch * n)

        # Each rank takes its row band from the AllReduce'd partials.
        # ReduceScatter would do this in one step; AllReduce + slice is
        # simpler for v1.
        y_etc_full = torch.zeros(batch, n, dtype=torch.float32)
        for rank_state in per_rank:
            rank = rank_state["rank"]
            y_partial = rank_state["state"].user_buffers["y_partial"].cpu()
            y_etc_full[rank * rows_per_rank : (rank + 1) * rows_per_rank] = y_partial[
                rank * rows_per_rank : (rank + 1) * rows_per_rank
            ]

        diff = (y_etc_full - y_eager_full).abs()
        tolerance = atol + rtol * y_eager_full.abs()
        num_failing += int((diff > tolerance).sum().item())
        max_abs = max(max_abs, float(diff.max().item()))
        denom = y_eager_full.abs().clamp_min(1e-6)
        max_rel = max(max_rel, float((diff / denom).max().item()))

    return {
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "num_failing_elements": float(num_failing),
        "num_inputs": float(num_inputs),
    }


def _run_benchmark_multi(
    *,
    workload: Any,
    schedule: StaticSchedule,
    comm: Any,
    per_rank: list[dict[str, Any]],
    num_iters: int,
) -> dict[str, float]:
    """Multi-rank perf bench. Both paths run on the same set of GPUs.

    Eager: ``y = x @ W`` via cuBLAS on rank 0 (single-GPU full GEMM).
    ETC: per-rank local GEMM + AllReduce + slice. v1 perf is
    bandwidth-bound by the AllReduce on PCIe gen5; v2 with
    in-megakernel peer atomics will close most of that gap.
    """
    sample_x_full = workload.sample_inputs[0]
    eager_device = torch.device("cuda:0")
    eager_model = workload.model.to(eager_device).eval()
    sample_x_dev = sample_x_full.to(eager_device, dtype=torch.float32)

    num_ranks = workload.num_ranks
    batch = sample_x_full.shape[0]
    n = per_rank[0]["state"].user_buffers["y_partial"].shape[-1]

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            eager_model(sample_x_dev)
        for rank_state in per_rank:
            x_shard = sample_x_full[
                :,
                rank_state["rank"] * (sample_x_full.shape[1] // num_ranks) : (rank_state["rank"] + 1)
                * (sample_x_full.shape[1] // num_ranks),
            ].to(rank_state["state"].user_buffers["x_shard"].device, dtype=torch.float32)
            _launch_one_rank(workload=workload, schedule=schedule, rank_state=rank_state, x_shard=x_shard)
        y_partial_ptrs = [int(rs["state"].user_buffers["y_partial"].data_ptr()) for rs in per_rank]
        comm.allreduce_fp32_sum(y_partial_ptrs, y_partial_ptrs, batch * n)
    for r in range(num_ranks):
        torch.cuda.synchronize(r)

    # Eager
    torch.cuda.synchronize(eager_device)
    t0 = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            eager_model(sample_x_dev)
    torch.cuda.synchronize(eager_device)
    eager_us = (time.perf_counter() - t0) * 1e6 / num_iters

    # ETC: per-rank megakernel + AllReduce.
    t0 = time.perf_counter()
    k_local = sample_x_full.shape[1] // num_ranks
    for _ in range(num_iters):
        for rank_state in per_rank:
            rank = rank_state["rank"]
            x_shard = sample_x_full[:, rank * k_local : (rank + 1) * k_local].to(
                rank_state["state"].user_buffers["x_shard"].device, dtype=torch.float32
            )
            _launch_one_rank(workload=workload, schedule=schedule, rank_state=rank_state, x_shard=x_shard)
        y_partial_ptrs = [int(rs["state"].user_buffers["y_partial"].data_ptr()) for rs in per_rank]
        comm.allreduce_fp32_sum(y_partial_ptrs, y_partial_ptrs, batch * n)
    etc_us = (time.perf_counter() - t0) * 1e6 / num_iters

    speedup = eager_us / max(etc_us, 1e-6)
    return {
        "etc_us": etc_us,
        "eager_us": eager_us,
        "speedup_vs_eager": speedup,
    }


def _launch_one_rank(
    *,
    workload: Any,
    schedule: StaticSchedule,
    rank_state: dict[str, Any],
    x_shard: torch.Tensor,
) -> None:
    """Single-rank launch: copy x_shard, reset event tensors, launch
    the rank's megakernel."""
    from cuda.bindings import driver as cu_driver  # type: ignore

    from compgen.runtime.native.cuda import _cu_check, _ensure_cuda_driver_context

    rank = rank_state["rank"]
    state = rank_state["state"]
    _ensure_cuda_driver_context(rank)

    state.user_buffers["x_shard"].copy_(x_shard)

    int64_size = 8
    cursor_bytes = 0
    for et, alloc in zip(state.event_tensors, schedule.event_tensor_allocs, strict=True):
        cells = _prod(alloc.shape)
        nbytes = cells * int64_size
        _cu_check(
            cu_driver.cuMemcpyHtoD(
                et.device_ptr,
                state.et_reset_host_ptr + cursor_bytes,
                nbytes,
            )
        )
        cursor_bytes += nbytes

    rank_state["launcher"].launch(
        kernel_handle=rank_state["cumod"].kernel_handle,
        grid_dim=(schedule.sm_count, 1, 1),
        block_dim=schedule.launch_config.block_dim,
        cluster_dim=None,
        shared_mem_bytes=schedule.launch_config.shared_mem_bytes,
        kernel_args=[state.et_ptrs_dev, state.buf_dev],
    )


__all__ = [
    "EtcDispatchError",
    "compile_and_run_etc_workload",
]
