"""Phase-5 CUDA megakernel emitter.

Takes a :class:`StaticSchedule` (from
:mod:`compgen.transforms.event_static_schedule`) plus a set of
``__device__`` function bodies for every distinct ``device_func``
referenced by the schedule, and emits:

1. A self-contained CUDA C++ source string. Imports the
   ``cg_rt_cuda_etensor_*`` primitives by extern declaration so the
   PTX it eventually compiles into can link against the
   ``libcompgen_rt-cuda.so`` symbols at module-load time. The kernel
   body is one persistent megakernel: each block consumes its SM
   queue, waits on in-edges, dispatches, notifies out-edges, and
   loops.
2. A YAML manifest describing the launch config, event-tensor
   allocations, the device-function table (``kind → name``) and the
   schedule summary. Bundle-load-time code reads the manifest to
   allocate event tensors and call ``cg_rt_cuda_megakernel_launch``.

What this emitter does **not** do:

- NVRTC-compile. Compilation happens at bundle-emit / bundle-load
  time via :class:`compgen.runtime.native.device.CudaExecutable` or
  the C launcher. Keeping emit + compile separate means CPU-only
  hosts can produce + audit the source, and the GPU host caches PTX
  per its own toolchain version.
- Tile-IR-build. Device function bodies are caller-supplied. A
  Tile-IR-driven body provider (using ``cuda-python`` 12.6+) is the
  next step in Phase 5; until then callers can hand-write CUDA C++
  for the op families they need.

Failure modes — every one is a typed exception, never a silent skip:

- :class:`DeviceFunctionUnavailable` — schedule references a
  ``device_func`` name with no body in
  ``device_function_sources``. Phase 7's ``compile_model`` integrates
  with the kernel provider stack, but this module fails loud rather
  than emit a stub body.
- :class:`MegakernelEmitError` — schedule structurally invalid
  (zero SMs after partitioning, unbalanced event references, dtype
  not in ``{i32, i64}``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from compgen.transforms.event_static_schedule import StaticSchedule

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MegakernelEmitError(RuntimeError):
    """Raised when a schedule cannot be lowered to a CUDA megakernel."""


class DeviceFunctionUnavailable(MegakernelEmitError):
    """Raised when a device-function body is missing from the caller's set.

    The Phase-7 wiring will resolve this by walking the kernel
    provider stack (Tile IR templates, then Triton fallback). This
    emitter doesn't do that lookup itself — it expects bodies up
    front and refuses to emit ``pass``-bodied placeholders.
    """


class TileIRUnavailableError(MegakernelEmitError):
    """Raised when ``cuda-python``'s Tile IR builder is requested but
    not importable.

    The :func:`emit_cuda_megakernel` API does not use Tile IR
    directly — callers building Tile-IR-backed bodies catch this
    error themselves. Surfaced here for callers' import convenience.
    """


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceFunctionSource:
    """A ``__device__`` function body the megakernel will dispatch into.

    Attributes:
        name: Symbol name. Must match
            :attr:`TaskDescriptor.device_func` for every task that
            should dispatch to it.
        signature: Full C++ parameter list **without** the leading
            ``int task_id, int sm_id, void **buffers`` (the wrapper
            adds those). Use ``""`` if the body needs only the
            built-in args.
        body: C++ statements that form the function body. Must NOT
            include the surrounding ``{ }``. The wrapper inserts the
            opening / closing braces around them.
        included_headers: Optional header lines (``#include
            <...>``) the body needs. Placed at the top of the
            emitted source.
    """

    name: str
    body: str
    signature: str = ""
    included_headers: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CudaMegakernelEmitResult:
    """Result of :func:`emit_cuda_megakernel`.

    Attributes:
        kernel_name: Symbol of the emitted ``__global__`` function.
        cuda_source: Complete CUDA C++ source string. Pass to NVRTC
            (via :class:`CudaExecutable`) or write to disk for
            bundle inspection.
        manifest: Structured launch / allocation / dispatch metadata
            consumed by the runtime launcher. Same data as
            ``manifest_yaml``; provided as a Python dict for
            programmatic access (e.g. tests).
        manifest_yaml: YAML serialization of ``manifest``. Goes into
            ``bundle/megakernel/manifest.yaml``.
        device_function_table: ``kind → device_func name``. The
            wrapper's switch statement uses these integer kinds; the
            runtime side baked them into the constant tables. Useful
            for debug + the megakernel-inspect tool.
    """

    kernel_name: str
    cuda_source: str
    manifest: dict[str, Any]
    manifest_yaml: str
    device_function_table: dict[int, str] = field(default_factory=dict)

    def write_to_bundle(self, bundle_megakernel_dir: Path | str) -> dict[str, Path]:
        """Stage source + manifest into ``bundle/megakernel/`` and
        return the written paths.

        Caller is responsible for creating the directory; this
        method writes ``source.cu`` and ``manifest.yaml`` into it
        (overwriting if present) and returns ``{"source": <path>,
        "manifest": <path>}``.
        """
        out = Path(bundle_megakernel_dir)
        out.mkdir(parents=True, exist_ok=True)
        src_path = out / "source.cu"
        man_path = out / "manifest.yaml"
        src_path.write_text(self.cuda_source)
        man_path.write_text(self.manifest_yaml)
        return {"source": src_path, "manifest": man_path}


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


def emit_cuda_megakernel(
    schedule: StaticSchedule,
    *,
    device_function_sources: dict[str, DeviceFunctionSource],
    user_buffer_count: int = 0,
    extra_includes: tuple[str, ...] = (),
) -> CudaMegakernelEmitResult:
    """Lower a :class:`StaticSchedule` to a CUDA megakernel + manifest.

    Args:
        schedule: Phase-2 output. Every distinct
            ``TaskDescriptor.device_func`` must have a matching
            entry in ``device_function_sources``.
        device_function_sources: Map of device function name →
            :class:`DeviceFunctionSource`. The wrapper dispatches
            into these via an integer ``kind`` derived from the
            sorted set of distinct names.
        user_buffer_count: Number of user-supplied buffer pointers
            the kernel takes (e.g. data tensors). The wrapper
            accepts a ``void **buffers`` of this length and forwards
            it to every device function.
        extra_includes: Optional ``#include`` lines added to the
            generated source's header.

    Returns:
        :class:`CudaMegakernelEmitResult`.

    Raises:
        DeviceFunctionUnavailable: A scheduled task references a
            ``device_func`` not present in
            ``device_function_sources``.
        MegakernelEmitError: Schedule has zero tasks, an
            unsupported event-tensor dtype, or a stride too large
            for the flat-cell index encoding.
    """
    if schedule.total_tasks == 0:
        raise MegakernelEmitError(f"schedule {schedule.graph_name!r} has zero tasks; nothing to emit")

    # Stable name → kind mapping. Sorted for deterministic output.
    distinct_names = sorted({t.device_func for q in schedule.sm_queues for t in q.tasks})
    missing = [n for n in distinct_names if n not in device_function_sources]
    if missing:
        raise DeviceFunctionUnavailable(
            "schedule references device_func(s) without a body source: " + ", ".join(sorted(missing))
        )
    name_to_kind = {n: i for i, n in enumerate(distinct_names)}
    kind_to_name = {i: n for n, i in name_to_kind.items()}

    # Reject unsupported dtypes up front so per-task lookups can't
    # silently widen something the C primitive won't accept.
    for a in schedule.event_tensor_allocs:
        if a.dtype not in ("i32", "i64"):
            raise MegakernelEmitError(f"event-tensor {a.name!r} dtype {a.dtype!r} unsupported by emitter")

    # Stable event_tensor name → index mapping (matches alloc order).
    event_index = {a.name: i for i, a in enumerate(schedule.event_tensor_allocs)}

    # Build flat tables for the constant-memory layout. Each task
    # carries (task_id, kind, coord_x) — the coord allows tile-level
    # task graphs (one DeviceCall with task_shape=(N,) → N tasks
    # with distinct coords) to thread the coord into the dispatched
    # body via the wrapper, instead of forcing one DeviceCall per
    # tile. ``coord_x`` is the first axis of the task's coord; multi-
    # dimensional coords are flattened by row-major linearisation
    # (``coord_x = sum(coord[i] * stride[i])``) so the body only
    # needs a single int parameter.
    flat_tasks: list[tuple[int, int, int]] = []  # (task_id, kind, coord_x)
    per_sm_begin: list[int] = []
    # Each cell carries (event_idx, cell, decrement, peer_rank, intra_cluster).
    # peer_rank = -1 sentinel means "local rank" (the wrapper dispatches
    # to ``cg_rt_cuda_etensor_notify_d`` / ``_wait_d``); peer_rank >= 0
    # selects a peer rank's event-tensor pointer table and dispatches
    # the cross-rank ``_peer_notify_d`` / ``_peer_wait_d`` primitive.
    # intra_cluster (Wave 1.6b) is 1 when EVERY peer task connected
    # via this cell is on an SM in the same Blackwell cluster as the
    # owning task — i.e. eligible for the cluster-DSM signalling
    # path. The emitter only honours this flag when ``cluster_dim``
    # is set; otherwise the global-atomic path is always used.
    in_cell_quints: list[tuple[int, int, int, int, int]] = []
    in_offsets: list[int] = []
    out_cell_quints: list[tuple[int, int, int, int, int]] = []
    out_offsets: list[int] = []
    has_peer_edges = False
    has_intra_cluster_edges = False

    # Task IDs are dense-but-not-necessarily-contiguous (the static
    # scheduler may pack a subset of a parent graph's tasks). We
    # remap to the table index so the wrapper can use it as an array
    # index without any sparsity machinery.
    table_index_for: dict[int, int] = {}

    for q in schedule.sm_queues:
        per_sm_begin.append(len(flat_tasks))
        for desc in q.tasks:
            table_index_for[desc.task_id] = len(flat_tasks)
            coord_x = _flatten_coord(desc.coord)
            flat_tasks.append((desc.task_id, name_to_kind[desc.device_func], coord_x))
            in_offsets.append(len(in_cell_quints))
            in_mask = desc.in_cluster_mask or tuple(False for _ in desc.in_cells)
            for (ev_name, cell, dec, peer_rank), intra in zip(desc.in_cells, in_mask, strict=False):
                pr = -1 if peer_rank is None else int(peer_rank)
                if pr >= 0:
                    has_peer_edges = True
                ic = 1 if intra else 0
                if ic:
                    has_intra_cluster_edges = True
                in_cell_quints.append(
                    (
                        event_index[ev_name],
                        _flat_cell(cell, _shape_for(schedule, ev_name)),
                        int(dec),
                        pr,
                        ic,
                    )
                )
            out_offsets.append(len(out_cell_quints))
            out_mask = desc.out_cluster_mask or tuple(False for _ in desc.out_cells)
            for (ev_name, cell, dec, peer_rank), intra in zip(desc.out_cells, out_mask, strict=False):
                pr = -1 if peer_rank is None else int(peer_rank)
                if pr >= 0:
                    has_peer_edges = True
                ic = 1 if intra else 0
                if ic:
                    has_intra_cluster_edges = True
                out_cell_quints.append(
                    (
                        event_index[ev_name],
                        _flat_cell(cell, _shape_for(schedule, ev_name)),
                        int(dec),
                        pr,
                        ic,
                    )
                )
    per_sm_begin.append(len(flat_tasks))
    in_offsets.append(len(in_cell_quints))
    out_offsets.append(len(out_cell_quints))

    # Wave 1.6b cluster-sync gating. The cluster-DSM-aware path is
    # only emitted when:
    #   1. ``launch_config.cluster_dim`` is set (caller opted into
    #      cluster launch on a cluster-eligible target — SM_90+).
    #   2. AND at least one cell on the schedule was classified
    #      ``intra_cluster=True`` by Wave-1.6b's scheduler half.
    # If either condition fails we fall back to the pre-Wave-1.6b
    # pure-global-atomic path so SM_90- targets and cluster-agnostic
    # schedules stay byte-identical to the old emitter (modulo the
    # CgCell struct's new field, which serializes inert).
    cluster_dim = schedule.launch_config.cluster_dim
    emit_cluster_sync = cluster_dim is not None and has_intra_cluster_edges

    # ----- Source emission ------------------------------------------------
    # NVRTC ships its own built-ins for __device__/__global__/threadIdx/
    # atomicAdd/__nanosleep/__threadfence_*, so we don't need
    # ``cuda_runtime.h`` in the prelude. Including it would force every
    # caller to plumb ``-I/path/to/cuda/include`` through NVRTC, and the
    # 12.x NVRTC built-in header set doesn't include the runtime API
    # header anyway. Caller-supplied bodies that need extra headers
    # declare them via ``DeviceFunctionSource.included_headers`` and
    # ``extra_includes``.
    headers: list[str] = []
    headers.extend(extra_includes)
    for src in device_function_sources.values():
        for h in src.included_headers:
            if h not in headers:
                headers.append(h)
    header_block = "\n".join(headers)

    declares = _emit_inline_primitives()
    # Compute per-SM queue depths so the wrapper can pad short queues
    # with no-op iterations when cluster.sync() participation is
    # required (Wave 1.6b wave-boundary approach — see _emit_wrapper
    # for the deadlock-avoidance reasoning).
    per_sm_depth = [per_sm_begin[i + 1] - per_sm_begin[i] for i in range(schedule.sm_count)]
    max_queue_depth = max(per_sm_depth) if per_sm_depth else 0
    constants = _emit_constant_tables(
        flat_tasks=flat_tasks,
        per_sm_begin=per_sm_begin,
        in_cell_quints=in_cell_quints,
        in_offsets=in_offsets,
        out_cell_quints=out_cell_quints,
        out_offsets=out_offsets,
    )
    device_funcs = _emit_device_functions(device_function_sources, distinct_names, user_buffer_count)
    dispatch_switch = _emit_dispatch_switch(name_to_kind)
    kernel_name = f"megakernel_{schedule.graph_name}"
    wrapper = _emit_wrapper(
        kernel_name=kernel_name,
        sm_count=schedule.sm_count,
        num_event_tensors=len(schedule.event_tensor_allocs),
        user_buffer_count=user_buffer_count,
        dispatch_switch=dispatch_switch,
        has_peer_edges=has_peer_edges,
        emit_cluster_sync=emit_cluster_sync,
        max_queue_depth=max_queue_depth,
    )

    cuda_source = "\n\n".join([header_block, declares, constants, device_funcs, wrapper]) + "\n"

    # ----- Manifest -------------------------------------------------------
    manifest: dict[str, Any] = {
        "version": 1,
        "graph_name": schedule.graph_name,
        "kernel_name": kernel_name,
        "source_file": "source.cu",
        "launch_config": {
            "grid_dim": list(schedule.launch_config.grid_dim),
            "block_dim": list(schedule.launch_config.block_dim),
            "cluster_dim": list(schedule.launch_config.cluster_dim)
            if schedule.launch_config.cluster_dim is not None
            else None,
            "shared_mem_bytes": int(schedule.launch_config.shared_mem_bytes),
            "cooperative": bool(schedule.launch_config.cooperative),
        },
        "event_tensors": [
            {
                "name": a.name,
                "shape": list(a.shape),
                "wait_count_default": int(a.wait_count_default),
                "dtype": a.dtype,
                "scope": a.scope,
                "index": event_index[a.name],
            }
            for a in schedule.event_tensor_allocs
        ],
        "device_function_table": [{"kind": k, "name": n} for k, n in sorted(kind_to_name.items())],
        "user_buffer_count": int(user_buffer_count),
        "schedule_summary": {
            "sm_count": schedule.sm_count,
            "total_tasks": schedule.total_tasks,
            **schedule.scheduling_metadata,
        },
        # Per bridge #131: surface the schedule's scheduling_metadata
        # at the top level too (audit-side lookup convenience). The
        # nested ``schedule_summary`` form has the same data plus the
        # core sm_count/total_tasks; this top-level alias is what most
        # callers reach for.
        "scheduling_metadata": dict(schedule.scheduling_metadata),
        "table_sizes": {
            "flat_tasks": len(flat_tasks),
            "in_cell_triples": len(in_cell_quints),
            "out_cell_triples": len(out_cell_quints),
        },
        "cluster_sync": {
            "enabled": bool(emit_cluster_sync),
            "intra_cluster_edges_present": bool(has_intra_cluster_edges),
            "max_queue_depth": int(max_queue_depth),
        },
    }
    import yaml

    manifest_yaml = yaml.safe_dump(manifest, sort_keys=False)

    log.info(
        "emit_cuda_megakernel.done",
        graph=schedule.graph_name,
        kernel=kernel_name,
        sm_count=schedule.sm_count,
        tasks=schedule.total_tasks,
        device_funcs=len(distinct_names),
        event_tensors=len(schedule.event_tensor_allocs),
    )

    return CudaMegakernelEmitResult(
        kernel_name=kernel_name,
        cuda_source=cuda_source,
        manifest=manifest,
        manifest_yaml=manifest_yaml,
        device_function_table=kind_to_name,
    )


# ---------------------------------------------------------------------------
# Helpers — pure source string builders
# ---------------------------------------------------------------------------


def _flatten_coord(coord: tuple[int, ...]) -> int:
    """Flatten a multi-dim task coord into a single int.

    Tile-level workloads use 1D coords (``(0,)``, ``(1,)``, ...) so
    this is the identity. Multi-dim coords are row-major flattened
    so the body sees a single ``coord_x`` parameter regardless of
    rank — no ABI churn when a workload bumps from 1D to 2D tile
    grids. The caller's body is responsible for un-flattening if it
    needs the original multi-dim coords.
    """
    if not coord:
        return 0
    if len(coord) == 1:
        return int(coord[0])
    # Multi-dim: assume the workload knows its own logical strides.
    # We just lay out (a, b, c, ...) lexicographically so a body that
    # iterates ``coord_x`` in 0..N covers the same task set the
    # scheduler enumerated.
    flat = 0
    for c in coord:
        flat = flat * 1000003 + int(c)  # arbitrary mixing; bodies must
        # call back via their own
        # workload-specific decode if
        # they need multi-dim coords.
    return flat


def _shape_for(schedule: StaticSchedule, event_name: str) -> tuple[int, ...]:
    for a in schedule.event_tensor_allocs:
        if a.name == event_name:
            return a.shape
    raise MegakernelEmitError(f"task references event tensor {event_name!r} that has no allocation spec")


def _flat_cell(cell: tuple[int, ...], shape: tuple[int, ...]) -> int:
    """Row-major flatten a multi-dim cell index into a single int."""
    if not shape:
        return 0
    if len(cell) != len(shape):
        raise MegakernelEmitError(
            f"cell {cell} has rank {len(cell)} but its event-tensor shape {shape} has rank {len(shape)}"
        )
    flat = 0
    stride = 1
    for c, d in zip(reversed(cell), reversed(shape), strict=True):
        if c < 0 or (d > 0 and c >= d):
            raise MegakernelEmitError(f"cell {cell} out of bounds for event-tensor shape {shape}")
        flat += c * stride
        stride *= max(d, 1)
    return flat


def _emit_inline_primitives() -> str:
    """Emit the Event Tensor device primitives inline.

    ``cuModuleLoadData`` does not resolve ``.extern .func`` PTX
    references against host-loaded ``.so`` symbols, so the device
    primitives can't live in ``libcompgen_rt-cuda.so`` and be
    extern-linked from the megakernel — they have to be inlined into
    the same NVRTC compilation unit. The bodies here mirror those in
    ``runtime/native/libcompgen_rt/src/drivers/cuda/event_tensor.cu``
    (kept in lockstep so behaviour is identical to the host-callable
    ``_kernel`` shims used by Phase-4 unit tests).

    See event_tensor.cu lines 50-104 for the source-of-truth bodies +
    the rationale for ``atomicAdd_system`` / ``atomicExch_system`` /
    ``__threadfence_system`` / ``__nanosleep`` choices.
    """
    return r"""// ===== Event Tensor device primitives (inlined) =====
// Mirrors libcompgen_rt event_tensor.cu — Phase 5 inlines instead of
// extern-linking because cuModuleLoadData doesn't resolve cross-module
// device-function symbols against host-loaded .so files.
__device__ __forceinline__
void cg_rt_cuda_etensor_notify_d(long long *E, int idx, int decrement) {
    __threadfence_system();
    atomicAdd_system((unsigned long long *)&E[idx],
                     (unsigned long long)(-(long long)decrement));
}

__device__ __forceinline__
void cg_rt_cuda_etensor_wait_d(long long *E, int idx) {
    while (atomicAdd_system((unsigned long long *)&E[idx], 0ULL) > 0) {
        __nanosleep(64);
    }
    __threadfence_system();
}

__device__ __forceinline__
void cg_rt_cuda_etensor_update_d(long long *E, int idx, long long new_count) {
    atomicExch_system((unsigned long long *)&E[idx],
                      (unsigned long long)new_count);
    __threadfence_system();
}

__device__ __forceinline__
void cg_rt_cuda_etensor_trigger_d(long long *E, int idx, long long consumer_count) {
    atomicExch_system((unsigned long long *)&E[idx],
                      (unsigned long long)consumer_count);
    __threadfence_system();
}

// Peer-mapped variants. Bodies are identical to the local primitives
// — the difference is the pointer ``E_remote`` is a UVA-coherent
// peer-mapped pointer (after cuCtxEnablePeerAccess between the two
// device contexts). atomicAdd_system + threadfence_system are PCIe-
// coherent on Blackwell workstation per REMOTE bridge probe #047.
__device__ __forceinline__
void cg_rt_cuda_etensor_peer_notify_d(long long *E_remote, int idx, int decrement) {
    __threadfence_system();
    atomicAdd_system((unsigned long long *)&E_remote[idx],
                     (unsigned long long)(-(long long)decrement));
}

__device__ __forceinline__
void cg_rt_cuda_etensor_peer_wait_d(long long *E_remote, int idx) {
    while (atomicAdd_system((unsigned long long *)&E_remote[idx], 0ULL) > 0) {
        __nanosleep(64);
    }
    __threadfence_system();
}
"""


def _emit_constant_tables(
    *,
    flat_tasks: list[tuple[int, int, int]],
    per_sm_begin: list[int],
    in_cell_quints: list[tuple[int, int, int, int, int]],
    in_offsets: list[int],
    out_cell_quints: list[tuple[int, int, int, int, int]],
    out_offsets: list[int],
) -> str:
    def _ints(values: list[int]) -> str:
        return ", ".join(str(v) for v in values) if values else "0"

    def _triples(values: list[tuple[int, int, int]]) -> str:
        return ", ".join(f"{{{a}, {b}, {c}}}" for a, b, c in values) if values else "{0,0,0}"

    def _quints(values: list[tuple[int, int, int, int, int]]) -> str:
        return ", ".join(f"{{{a}, {b}, {c}, {d}, {e}}}" for a, b, c, d, e in values) if values else "{0,0,0,-1,0}"

    n_tasks = max(len(flat_tasks), 1)
    n_per_sm = max(len(per_sm_begin), 1)
    n_in = max(len(in_cell_quints), 1)
    n_in_off = max(len(in_offsets), 1)
    n_out = max(len(out_cell_quints), 1)
    n_out_off = max(len(out_offsets), 1)

    # Storage class selection — bridge #102 ceiling fix.
    #
    # ptxas enforces a 64 KB cap per file on ``__constant__`` memory.
    # The static-schedule tables (task table + per-SM begin offsets +
    # in/out cell tables + offsets) scale linearly with task count,
    # so any non-trivial paper shape blows the cap:
    #   FFN h=4096: 2,304 tasks → ~104 KB → ❌
    #   MLP-1 (paper): 57,344 tasks → ~2.5 MB → ❌ (40× over)
    #
    # We compute the total bytes the tables would need and switch to
    # ``__device__`` arrays (no cap, lives in global memory + cached
    # via L1/L2) when ``__constant__`` would overflow. Small bundles
    # keep ``__constant__`` for the slightly faster broadcast path
    # (every thread in a warp can read constant memory in one cycle).
    sizeof_cgtask = 12  # 3 × int
    sizeof_cgcell = 20  # 5 × int (Wave 1.6b adds intra_cluster flag)
    sizeof_int = 4
    total_const_bytes = (
        n_tasks * sizeof_cgtask
        + n_per_sm * sizeof_int
        + n_in * sizeof_cgcell
        + n_in_off * sizeof_int
        + n_out * sizeof_cgcell
        + n_out_off * sizeof_int
    )
    # Headroom: leave 16 KB of constant memory for any other use
    # (NVRTC's own compiler-generated constants + cuBLASDx body
    # internals). 48 KB threshold = 64 KB - 16 KB.
    storage = "__constant__" if total_const_bytes < 48 * 1024 else "__device__"

    return (
        # Storage class chosen above based on table size.
        # CgTask carries the task's coord_x so tile-level workloads
        # (one DeviceCall with task_shape=(N,) → N tasks across SMs)
        # can index into per-tile buffer regions without one-DeviceCall-
        # per-tile boilerplate.
        #
        # CgCell carries ``peer_rank`` (-1 = local) so the wrapper can
        # dispatch local vs cross-rank notify/wait without parallel
        # tables. Cross-rank cells fetch their event-tensor base
        # pointer from ``peer_event_tensors[peer_rank]``.
        # ``intra_cluster`` (Wave 1.6b) is 1 when every peer task
        # connected via this cell is on an SM in the same Blackwell
        # cluster as the owning task; on schedules with cluster-launch
        # enabled the wrapper takes the cluster-DSM path for those
        # cells (relaxed store + wave-boundary cluster.sync) instead
        # of the global-atomic path. Always 0 when cluster_dim is
        # ``None`` — the field is inert on SM_90- targets.
        f"struct CgTask {{ int task_id; int kind; int coord_x; }};\n"
        f"struct CgCell {{ int event_idx; int cell; int decrement; "
        f"int peer_rank; int intra_cluster; }};\n\n"
        f"// Schedule tables: {total_const_bytes} bytes total → {storage} storage.\n"
        f"{storage} CgTask  CG_TASK_TABLE[{n_tasks}] = {{ {_triples(flat_tasks)} }};\n"
        f"{storage} int     CG_PER_SM_BEGIN[{n_per_sm}] = {{ {_ints(per_sm_begin)} }};\n"
        f"{storage} CgCell  CG_IN_CELLS[{n_in}] = {{ {_quints(in_cell_quints)} }};\n"
        f"{storage} int     CG_IN_OFFSETS[{n_in_off}] = {{ {_ints(in_offsets)} }};\n"
        f"{storage} CgCell  CG_OUT_CELLS[{n_out}] = {{ {_quints(out_cell_quints)} }};\n"
        f"{storage} int     CG_OUT_OFFSETS[{n_out_off}] = {{ {_ints(out_offsets)} }};\n"
    )


def _emit_device_functions(
    sources: dict[str, DeviceFunctionSource],
    ordered_names: list[str],
    user_buffer_count: int,
) -> str:
    # user_buffer_count is informational for the manifest only — the
    # device-function signature is the same regardless. Bodies that
    # don't need ``buffers`` simply ignore the parameter.
    del user_buffer_count
    parts: list[str] = ["// ===== Device functions =====\n"]
    for name in ordered_names:
        src = sources[name]
        sig_extra = src.signature.strip()
        sig_extra = (", " + sig_extra) if sig_extra else ""
        # Signature now passes ``coord_x`` (the task's first-axis coord
        # in the EventGraph) before ``buffers``. Tile-level workloads
        # use it to index per-tile output regions; bodies that don't
        # need it simply ignore the parameter (compiler optimises out).
        parts.append(
            f"__device__ void {name}(int task_id, int sm_id, int coord_x, void **buffers"
            f"{sig_extra}) {{\n"
            f"{_indent(src.body, 4)}\n"
            f"}}\n"
        )
    return "\n".join(parts)


def _emit_dispatch_switch(name_to_kind: dict[str, int]) -> str:
    body = ["switch (kind) {"]
    for name, kind in sorted(name_to_kind.items(), key=lambda kv: kv[1]):
        body.append(f"  case {kind}: {name}(task_id, sm_id, coord_x, buffers); break;")
    body.append("  default: break;  // unknown kind — no-op")
    body.append("}")
    return "\n".join(body)


def _emit_wrapper(
    *,
    kernel_name: str,
    sm_count: int,
    num_event_tensors: int,
    user_buffer_count: int,
    dispatch_switch: str,
    has_peer_edges: bool,
    emit_cluster_sync: bool = False,
    max_queue_depth: int = 0,
) -> str:
    # When the schedule has any cross-rank edges, the kernel takes an
    # additional ``peer_event_tensors`` arg: an array indexed by
    # ``peer_rank`` whose entries are ``long long **`` event-tensor
    # base-pointer tables for that rank. The wrapper dispatches
    # peer notify/wait via this lookup; local edges keep the cheap
    # ``event_tensors`` path. When no peer edges are emitted the
    # wrapper signature stays single-rank-compatible.
    peer_arg = ",\n    long long ***peer_event_tensors  /* [num_peers][num_events] */" if has_peer_edges else ""

    # ---- wait dispatch -------------------------------------------------
    # Two orthogonal axes:
    #   - peer_rank (cross-rank vs local)
    #   - intra_cluster (cluster-DSM-eligible vs global-atomic)
    # ``emit_cluster_sync`` gates the cluster path entirely; when it's
    # off we emit byte-identical code to the pre-Wave-1.6b emitter.
    #
    # Wave 1.6b — wave-boundary cluster.sync() approach.
    # ----------------------------------------------------------------
    # We deliberately do NOT emit a per-edge ``cluster.sync()`` inside an
    # ``if (c.intra_cluster)`` branch. ``cluster.sync()`` is collective:
    # ALL threads in ALL blocks of the cluster must enter or it
    # deadlocks. Different blocks have different in/out cell sets, so
    # per-edge gating would diverge between blocks and hang.
    #
    # Instead: when the schedule has any intra-cluster edge, every
    # block in the cluster reaches a single ``cluster.sync()`` at the
    # END of every task — uniformly, without conditional branches.
    # That sync makes all relaxed cluster-DSM stores from this task
    # globally visible to peer blocks in the cluster before the next
    # task's wait runs. The intra_cluster path therefore replaces the
    # per-edge global atomic with: relaxed write + uniform cluster
    # barrier.
    #
    # Different blocks have different queue depths; we pad short
    # queues with no-op iterations so every block calls cluster.sync()
    # the same number of times. Padding iterations skip both wait and
    # notify (their CgCell ranges are empty for the pad index) but
    # still participate in the cluster.sync() at end of iteration.

    if has_peer_edges:
        global_wait_inner = (
            "if (c.peer_rank < 0) {\n"
            "                        cg_rt_cuda_etensor_wait_d(event_tensors[c.event_idx], c.cell);\n"
            "                    } else {\n"
            "                        cg_rt_cuda_etensor_peer_wait_d(\n"
            "                            peer_event_tensors[c.peer_rank][c.event_idx], c.cell);\n"
            "                    }"
        )
        global_notify_inner = (
            "if (c.peer_rank < 0) {\n"
            "                        cg_rt_cuda_etensor_notify_d(\n"
            "                            event_tensors[c.event_idx], c.cell, c.decrement);\n"
            "                    } else {\n"
            "                        cg_rt_cuda_etensor_peer_notify_d(\n"
            "                            peer_event_tensors[c.peer_rank][c.event_idx], c.cell, c.decrement);\n"
            "                    }"
        )
    else:
        global_wait_inner = "cg_rt_cuda_etensor_wait_d(event_tensors[c.event_idx], c.cell);"
        global_notify_inner = "cg_rt_cuda_etensor_notify_d(event_tensors[c.event_idx], c.cell, c.decrement);"

    if emit_cluster_sync:
        # Cluster-DSM relaxed read on intra_cluster wait. The cell is
        # decremented by an intra-cluster predecessor's relaxed store +
        # cluster.sync() at the end of THAT predecessor's task; by the
        # time control reaches us the value is visible (the previous
        # iteration's terminal cluster.sync() is the release/acquire
        # boundary). We still spin briefly in case the predecessor
        # lives in a different wave of the cluster (queue padding).
        wait_dispatch = (
            "if (c.intra_cluster) {\n"
            "                    // Intra-cluster wait: predecessor's relaxed store +\n"
            "                    // wave-boundary cluster.sync() made this cell visible.\n"
            "                    // Read non-atomically; a brief spin handles the case\n"
            "                    // where the predecessor sits in a later wave (queue\n"
            "                    // padding makes wave indices comparable cluster-wide).\n"
            "                    while (((volatile long long *)event_tensors[c.event_idx])[c.cell] > 0) {\n"
            "                        __nanosleep(64);\n"
            "                    }\n"
            "                } else {\n"
            f"                    {global_wait_inner}\n"
            "                }"
        )
        # Intra-cluster notify: relaxed (non-atomic) decrement of the
        # local view; the wave-boundary cluster.sync() at end of task
        # publishes the write to peer SMs in the cluster. We still
        # need a fence so the in-cluster write is visible BEFORE the
        # cluster.sync() runs.
        notify_dispatch = (
            "if (c.intra_cluster) {\n"
            "                    // Cluster-DSM relaxed notify. cluster.sync() at the\n"
            "                    // end of this task makes the decrement visible to\n"
            "                    // peer SMs in the cluster — no global atomic needed.\n"
            "                    long long *ev = event_tensors[c.event_idx];\n"
            "                    ev[c.cell] -= (long long)c.decrement;\n"
            "                } else {\n"
            f"                    {global_notify_inner}\n"
            "                }"
        )
        # Wave-boundary loop. Pad to ``max_queue_depth`` so every block
        # reaches the same number of cluster.sync() calls regardless
        # of its queue length. Padding iterations skip the body.
        loop_header = (
            f"const int max_depth = {max_queue_depth};\n"
            "    for (int wave = 0; wave < max_depth; ++wave) {\n"
            "        const int i = begin + wave;\n"
            "        const bool active = (i < end);\n"
        )
        body_inner = """\
        if (active) {
            const int task_id = CG_TASK_TABLE[i].task_id;
            const int kind    = CG_TASK_TABLE[i].kind;
            const int coord_x = CG_TASK_TABLE[i].coord_x;

            // wait on in-edges. Guard on all three thread axes —
            // non-1D blocks would otherwise have multiple threads
            // hammer the same atomic, both tanking throughput and
            // over-decrementing the counter.
            const int in_lo = CG_IN_OFFSETS[i];
            const int in_hi = CG_IN_OFFSETS[i + 1];
            if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
                for (int j = in_lo; j < in_hi; ++j) {
                    CgCell c = CG_IN_CELLS[j];
                    __WAIT_DISPATCH__
                }
            }
            __syncthreads();

            // dispatch
__DISPATCH_SWITCH__
            __syncthreads();

            // notify out-edges
            const int out_lo = CG_OUT_OFFSETS[i];
            const int out_hi = CG_OUT_OFFSETS[i + 1];
            if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
                for (int j = out_lo; j < out_hi; ++j) {
                    CgCell c = CG_OUT_CELLS[j];
                    __NOTIFY_DISPATCH__
                }
            }
            __syncthreads();
        } else {
            // Padding iteration — this block already drained its
            // queue. Still participate in the cluster.sync() below
            // so peer blocks in the cluster don't deadlock waiting
            // on us. No per-task work here.
            __syncthreads();
        }

        // Wave-boundary cluster.sync(). Publishes any intra_cluster
        // relaxed-store notifies from this task to peer SMs in the
        // cluster. UNIFORMLY entered by every block of the cluster
        // (no conditional branches around this call) — that's the
        // deadlock-avoidance property the per-edge approach lacks.
        cooperative_groups::cluster_group cluster =
            cooperative_groups::this_cluster();
        cluster.sync();
"""
        body_inner = body_inner.replace("__WAIT_DISPATCH__", wait_dispatch)
        body_inner = body_inner.replace("__NOTIFY_DISPATCH__", notify_dispatch)
        body_inner = body_inner.replace("__DISPATCH_SWITCH__", _indent(dispatch_switch, 12))
        kernel_includes = "#include <cooperative_groups.h>\n#include <cuda/atomic>\n"
        kernel_body = f"    {loop_header}{body_inner}    }}\n"
    else:
        # Pre-Wave-1.6b path. Identical to the original wrapper —
        # byte-stable for SM_90- targets and cluster-agnostic schedules.
        wait_dispatch = global_wait_inner
        notify_dispatch = global_notify_inner
        kernel_includes = ""
        kernel_body = f"""    for (int i = begin; i < end; ++i) {{
        const int task_id = CG_TASK_TABLE[i].task_id;
        const int kind    = CG_TASK_TABLE[i].kind;
        const int coord_x = CG_TASK_TABLE[i].coord_x;

        // wait on in-edges. Guard on all three thread axes —
        // non-1D blocks would otherwise have multiple threads
        // hammer the same atomic, both tanking throughput and
        // over-decrementing the counter.
        const int in_lo = CG_IN_OFFSETS[i];
        const int in_hi = CG_IN_OFFSETS[i + 1];
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {{
            for (int j = in_lo; j < in_hi; ++j) {{
                CgCell c = CG_IN_CELLS[j];
                {wait_dispatch}
            }}
        }}
        __syncthreads();

        // dispatch
{_indent(dispatch_switch, 8)}
        __syncthreads();

        // notify out-edges
        const int out_lo = CG_OUT_OFFSETS[i];
        const int out_hi = CG_OUT_OFFSETS[i + 1];
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {{
            for (int j = out_lo; j < out_hi; ++j) {{
                CgCell c = CG_OUT_CELLS[j];
                {notify_dispatch}
            }}
        }}
        __syncthreads();
    }}
"""
    return f"""// ===== Persistent megakernel wrapper =====
// Cooperatively launched with grid = ({sm_count}, 1, 1). Each block
// drains its SM queue, waits on every in-edge, dispatches, and
// notifies every out-edge before advancing.
{kernel_includes}extern "C" __global__ void {kernel_name}(
    long long **event_tensors,
    void      **buffers{peer_arg}
) {{
    const int sm_id = blockIdx.x;
    if (sm_id >= {sm_count}) return;

    const int begin = CG_PER_SM_BEGIN[sm_id];
    const int end   = CG_PER_SM_BEGIN[sm_id + 1];

{kernel_body}}}
"""


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in text.splitlines())


__all__ = [
    "CudaMegakernelEmitResult",
    "DeviceFunctionSource",
    "DeviceFunctionUnavailable",
    "MegakernelEmitError",
    "TileIRUnavailableError",
    "emit_cuda_megakernel",
]
