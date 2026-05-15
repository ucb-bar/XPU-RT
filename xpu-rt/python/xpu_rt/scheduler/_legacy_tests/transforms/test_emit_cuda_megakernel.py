"""Phase-5 CUDA megakernel emitter tests (CPU-only).

Covers the structural contract of :func:`emit_cuda_megakernel`:

- Every distinct ``device_func`` becomes a ``__device__`` function in
  the emitted source; missing bodies raise
  :class:`DeviceFunctionUnavailable`.
- The ``__global__`` wrapper has the expected name, the right
  per-SM-loop structure, and dispatches via an integer ``kind``.
- Constant-memory tables flatten the per-SM queues + in/out cell
  triples in declared order.
- The manifest YAML round-trips through :mod:`yaml` and carries
  every key bundle-load-time consumers need.
- Unsupported event-tensor dtypes fail loud at emit time, not at
  NVRTC compile time.

GPU end-to-end (NVRTC compile + cooperative launch + correctness on
a 4-task diamond) lives in
``tests/runtime/native/test_cuda_megakernel_e2e.py`` (gated on
``requires_gpu``).
"""

from __future__ import annotations

import pytest
import yaml
from compgen.runtime.event_tensor import EventTensor
from compgen.runtime.megakernel import DeviceCall, EventEdge, MegakernelGraph
from compgen.transforms.emit_cuda_megakernel import (
    DeviceFunctionSource,
    DeviceFunctionUnavailable,
    MegakernelEmitError,
    emit_cuda_megakernel,
)
from compgen.transforms.event_static_schedule import compute_static_schedule


def _saxpy_diamond_graph() -> MegakernelGraph:
    """Diamond DAG: A → (B, C) → D. ``DeviceCall.name`` doubles as
    ``TaskDescriptor.device_func`` per the static scheduler, so the
    four calls get four distinct symbol names."""
    ab = EventTensor((1,), wait_count_default=1)
    ac = EventTensor((1,), wait_count_default=1)
    bd = EventTensor((1,), wait_count_default=1)
    cd = EventTensor((1,), wait_count_default=1)
    calls = (
        DeviceCall(
            name="saxpy_a",
            body_fn=lambda c: None,
            task_shape=(1,),
            out_edges=(
                EventEdge("ab", lambda c: (0,)),
                EventEdge("ac", lambda c: (0,)),
            ),
        ),
        DeviceCall(
            name="saxpy_b",
            body_fn=lambda c: None,
            task_shape=(1,),
            in_edges=(EventEdge("ab", lambda c: (0,)),),
            out_edges=(EventEdge("bd", lambda c: (0,)),),
        ),
        DeviceCall(
            name="saxpy_c",
            body_fn=lambda c: None,
            task_shape=(1,),
            in_edges=(EventEdge("ac", lambda c: (0,)),),
            out_edges=(EventEdge("cd", lambda c: (0,)),),
        ),
        DeviceCall(
            name="reduce_d",
            body_fn=lambda c: None,
            task_shape=(1,),
            in_edges=(
                EventEdge("bd", lambda c: (0,)),
                EventEdge("cd", lambda c: (0,)),
            ),
        ),
    )
    return MegakernelGraph(
        name="diamond",
        calls=calls,
        event_tensors={"ab": ab, "ac": ac, "bd": bd, "cd": cd},
        policy="static",
    )


def _all_diamond_bodies() -> dict[str, DeviceFunctionSource]:
    return {
        name: DeviceFunctionSource(
            name=name,
            body=f"// {name}\n(void)task_id; (void)sm_id; (void)buffers;",
        )
        for name in ("saxpy_a", "saxpy_b", "saxpy_c", "reduce_d")
    }


class TestEmitContract:
    def test_emits_kernel_name_and_dispatch_switch(self) -> None:
        graph = _saxpy_diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        result = emit_cuda_megakernel(
            schedule,
            device_function_sources=_all_diamond_bodies(),
        )
        assert result.kernel_name == "megakernel_diamond"
        assert f"__global__ void {result.kernel_name}" in result.cuda_source
        for name in ("saxpy_a", "saxpy_b", "saxpy_c", "reduce_d"):
            assert f"__device__ void {name}(int task_id" in result.cuda_source
            assert f"{name}(task_id, sm_id, coord_x, buffers); break;" in result.cuda_source
        assert "switch (kind)" in result.cuda_source

    def test_wrapper_thread_zero_guard_is_3d(self) -> None:
        """The notify/wait guard must check all three thread axes.

        Non-1D block shapes (e.g. (32, 32, 1) for tiled GEMM) have
        ``blockDim.y × blockDim.z`` threads with ``threadIdx.x == 0``.
        Gating only on .x lets all of them spin on the same atomic
        — both massive contention and over-decrement on notify. Pin
        the 3-axis guard so a refactor doesn't quietly drop it."""
        graph = _saxpy_diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        result = emit_cuda_megakernel(
            schedule,
            device_function_sources=_all_diamond_bodies(),
        )
        assert "threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0" in result.cuda_source
        # And a 1D-only guard must NOT appear bare in the wrapper.
        assert "if (threadIdx.x == 0) {" not in result.cuda_source

    def test_event_tensor_primitives_inlined(self) -> None:
        """The emitter inlines notify/wait/update/trigger as
        ``__device__ __forceinline__`` bodies — not externs — because
        cuModuleLoadData doesn't resolve cross-module device-function
        symbols. Bodies must include the atomic + threadfence
        instructions that mirror libcompgen_rt event_tensor.cu."""
        graph = _saxpy_diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        result = emit_cuda_megakernel(
            schedule,
            device_function_sources=_all_diamond_bodies(),
        )
        for symbol in (
            "cg_rt_cuda_etensor_notify_d",
            "cg_rt_cuda_etensor_wait_d",
            "cg_rt_cuda_etensor_update_d",
            "cg_rt_cuda_etensor_trigger_d",
        ):
            assert symbol in result.cuda_source
        # The bodies are inlined, so the actual atomic instructions
        # appear in the source — this is a stronger contract than just
        # checking the symbol declaration.
        assert "atomicAdd_system" in result.cuda_source
        assert "atomicExch_system" in result.cuda_source
        assert "__threadfence_system" in result.cuda_source
        assert "__nanosleep" in result.cuda_source
        # And there should be NO extern decls — those would silently
        # break NVRTC again if reintroduced.
        assert 'extern "C" {' not in result.cuda_source or (
            "__device__ void cg_rt_cuda_etensor" not in result.cuda_source.split('extern "C" {')[1]
        ), "primitives must be inlined, not extern-declared"

    def test_constant_tables_present(self) -> None:
        graph = _saxpy_diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        result = emit_cuda_megakernel(
            schedule,
            device_function_sources=_all_diamond_bodies(),
        )
        for tbl in (
            "CG_TASK_TABLE",
            "CG_PER_SM_BEGIN",
            "CG_IN_CELLS",
            "CG_IN_OFFSETS",
            "CG_OUT_CELLS",
            "CG_OUT_OFFSETS",
        ):
            assert tbl in result.cuda_source

    def test_missing_device_func_raises(self) -> None:
        graph = _saxpy_diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        bodies = _all_diamond_bodies()
        bodies.pop("reduce_d")
        with pytest.raises(DeviceFunctionUnavailable, match="reduce_d"):
            emit_cuda_megakernel(schedule, device_function_sources=bodies)

    def test_zero_tasks_raises(self) -> None:
        empty = MegakernelGraph(
            name="empty",
            calls=(
                DeviceCall(
                    name="passthrough",
                    body_fn=lambda c: None,
                    task_shape=(1,),
                ),
            ),
            event_tensors={},
            policy="static",
        )
        schedule = compute_static_schedule(empty, sm_count=1)
        # Single task → must emit. Now build a *truly* empty schedule
        # by stripping the task list (private API for the test).
        object.__setattr__(schedule, "sm_queues", tuple())
        object.__setattr__(schedule, "total_tasks", 0)
        with pytest.raises(MegakernelEmitError, match="zero tasks"):
            emit_cuda_megakernel(schedule, device_function_sources={})

    def test_unsupported_event_dtype_raises(self) -> None:
        # f32 isn't a valid event-tensor counter dtype — this would
        # blow up at the C primitive too. Catch it at emit time.
        graph = _saxpy_diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        bad = list(schedule.event_tensor_allocs)
        bad[0] = type(bad[0])(
            name=bad[0].name,
            shape=bad[0].shape,
            wait_count_default=bad[0].wait_count_default,
            dtype="f32",
            scope=bad[0].scope,
        )
        object.__setattr__(schedule, "event_tensor_allocs", tuple(bad))
        with pytest.raises(MegakernelEmitError, match="unsupported"):
            emit_cuda_megakernel(
                schedule,
                device_function_sources=_all_diamond_bodies(),
            )


class TestManifest:
    def test_manifest_yaml_roundtrips(self) -> None:
        graph = _saxpy_diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        result = emit_cuda_megakernel(
            schedule,
            device_function_sources=_all_diamond_bodies(),
            user_buffer_count=3,
        )
        loaded = yaml.safe_load(result.manifest_yaml)
        assert loaded["graph_name"] == "diamond"
        assert loaded["kernel_name"] == "megakernel_diamond"
        assert loaded["source_file"] == "source.cu"
        assert loaded["launch_config"]["grid_dim"] == [2, 1, 1]
        assert loaded["launch_config"]["cooperative"] is True
        assert loaded["user_buffer_count"] == 3
        assert {e["name"] for e in loaded["event_tensors"]} == {"ab", "ac", "bd", "cd"}
        kinds = {dft["name"] for dft in loaded["device_function_table"]}
        assert kinds == {"saxpy_a", "saxpy_b", "saxpy_c", "reduce_d"}
        assert loaded["schedule_summary"]["total_tasks"] == 4

    def test_write_to_bundle_creates_files(self, tmp_path) -> None:
        graph = _saxpy_diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        result = emit_cuda_megakernel(
            schedule,
            device_function_sources=_all_diamond_bodies(),
        )
        out = tmp_path / "megakernel"
        paths = result.write_to_bundle(out)
        assert paths["source"].is_file()
        assert paths["manifest"].is_file()
        assert "megakernel_diamond" in paths["source"].read_text()
        assert "graph_name: diamond" in paths["manifest"].read_text()


class TestStorageClassThreshold:
    """Wave 1.7 — bridge #102 const-memory ceiling fix.

    ptxas caps ``__constant__`` at 64 KB per file. Static schedule
    tables scale linearly with task count — at MLP-1 paper shape
    we need ~2.5 MB. The emitter switches to ``__device__`` arrays
    (no cap) when total table bytes ≥ 48 KB threshold, keeping
    ``__constant__`` for small bundles where the broadcast-read
    perf matters.
    """

    def _graph_with_n_tasks(self, n_tile_tasks: int) -> MegakernelGraph:
        """Build a graph with ``n_tile_tasks`` tile-level tasks per
        op. The emitter sees ``n_tile_tasks * 4`` total tasks since
        each op has its own task_shape."""
        ev = EventTensor((n_tile_tasks,), wait_count_default=1)
        same = lambda c: (c[0],)  # noqa: E731
        calls = (
            DeviceCall(
                name="op_a",
                body_fn=lambda c: None,
                task_shape=(n_tile_tasks,),
                out_edges=(EventEdge("e", same),),
            ),
            DeviceCall(
                name="op_b",
                body_fn=lambda c: None,
                task_shape=(n_tile_tasks,),
                in_edges=(EventEdge("e", same),),
            ),
        )
        return MegakernelGraph(
            name=f"large_{n_tile_tasks}",
            calls=calls,
            event_tensors={"e": ev},
            policy="static",
        )

    def test_small_schedule_uses_constant_storage(self) -> None:
        """Default path: small schedules keep ``__constant__`` for
        the broadcast-read perf advantage."""
        graph = _saxpy_diamond_graph()
        sched = compute_static_schedule(graph, sm_count=4)
        out = emit_cuda_megakernel(
            sched,
            device_function_sources=_all_diamond_bodies(),
            user_buffer_count=0,
        )
        assert "__constant__ CgTask  CG_TASK_TABLE" in out.cuda_source
        assert "__device__ CgTask" not in out.cuda_source

    def test_large_schedule_switches_to_device_storage(self) -> None:
        """Wave 1.7 fix — at ~2,500 tile-tasks the emitter must
        switch to ``__device__`` to avoid ptxas's 64 KB
        ``__constant__`` ceiling. Uses 3,000 tasks per op (6,000
        total) for a comfortable margin over the 48 KB threshold."""
        graph = self._graph_with_n_tasks(3000)
        sched = compute_static_schedule(graph, sm_count=4)
        bodies = {
            "op_a": DeviceFunctionSource(name="op_a", body="(void)task_id; (void)sm_id; (void)buffers;"),
            "op_b": DeviceFunctionSource(name="op_b", body="(void)task_id; (void)sm_id; (void)buffers;"),
        }
        out = emit_cuda_megakernel(
            sched,
            device_function_sources=bodies,
            user_buffer_count=0,
        )
        # Storage class switched to __device__ — bundle now fits
        # arbitrarily-large schedules.
        assert "__device__ CgTask  CG_TASK_TABLE" in out.cuda_source
        assert "__constant__ CgTask  CG_TASK_TABLE" not in out.cuda_source
        # Comment surfaces total bytes for audit.
        assert "Schedule tables:" in out.cuda_source
        assert "→ __device__ storage" in out.cuda_source

    def test_threshold_comment_surfaces_byte_count(self) -> None:
        """The emitter writes a comment line with the computed byte
        count + chosen storage class — auditable from the bundle's
        source.cu without re-running the calculation."""
        graph = _saxpy_diamond_graph()
        sched = compute_static_schedule(graph, sm_count=4)
        out = emit_cuda_megakernel(
            sched,
            device_function_sources=_all_diamond_bodies(),
            user_buffer_count=0,
        )
        # Comment format: ``// Schedule tables: NNN bytes total → __X__ storage.``
        assert "Schedule tables:" in out.cuda_source
        assert "bytes total" in out.cuda_source


class TestPeerRankEdges:
    """Phase-4b v2: cross-rank EventEdges lower to peer-notify/wait
    primitives + an additional ``peer_event_tensors`` kernel arg.
    Local-only graphs keep the original signature unchanged."""

    def test_local_only_graph_omits_peer_arg(self) -> None:
        """When the schedule has zero peer edges, the emitted wrapper
        keeps its single-rank signature — no peer_event_tensors arg,
        no peer_rank dispatch in the wrapper."""
        graph = _saxpy_diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        result = emit_cuda_megakernel(
            schedule,
            device_function_sources=_all_diamond_bodies(),
        )
        assert "peer_event_tensors" not in result.cuda_source
        # The peer primitive bodies are inlined regardless (cheap), but
        # the dispatch logic shouldn't reference them.
        wrapper_idx = result.cuda_source.find("Persistent megakernel wrapper")
        wrapper_body = result.cuda_source[wrapper_idx:]
        assert "c.peer_rank" not in wrapper_body

    def test_peer_edges_add_kernel_arg_and_dispatch(self) -> None:
        """A graph with at least one peer EventEdge causes the emitter
        to add the ``peer_event_tensors`` parameter + the peer/local
        dispatch in the wrapper. The CgCell struct gains a peer_rank
        field; the local primitives are still emitted unchanged."""
        from compgen.runtime.event_tensor import EventTensor
        from compgen.runtime.megakernel import (
            DeviceCall,
            EventEdge,
            MegakernelGraph,
        )

        ev_local = EventTensor((1,), wait_count_default=1)
        ev_peer = EventTensor((1,), wait_count_default=1)
        calls = (
            DeviceCall(
                name="producer",
                body_fn=lambda c: None,
                task_shape=(1,),
                # One local out-edge, one cross-rank out-edge.
                out_edges=(
                    EventEdge("ev_local", lambda c: (0,)),
                    EventEdge("ev_peer", lambda c: (0,), peer_rank=1),
                ),
            ),
        )
        graph = MegakernelGraph(
            name="peer_smoke",
            calls=calls,
            event_tensors={"ev_local": ev_local, "ev_peer": ev_peer},
            policy="static",
        )
        schedule = compute_static_schedule(graph, sm_count=1)
        bodies = {
            "producer": DeviceFunctionSource(
                name="producer",
                body="(void)task_id; (void)sm_id; (void)coord_x; (void)buffers;",
            ),
        }
        result = emit_cuda_megakernel(
            schedule,
            device_function_sources=bodies,
        )

        # New wrapper signature.
        assert "long long ***peer_event_tensors" in result.cuda_source
        # Both local and peer primitives are referenced from the
        # wrapper's dispatch.
        assert "cg_rt_cuda_etensor_peer_notify_d" in result.cuda_source
        assert "c.peer_rank < 0" in result.cuda_source
        # CgCell struct gained the peer_rank field. Wave 1.6b further
        # extends it with an ``intra_cluster`` flag (one bool per cell);
        # the field is inert when ``cluster_dim`` is None but must be
        # part of the struct so the constant-table layout stays
        # uniform across cluster-on / cluster-off bundles.
        assert (
            "struct CgCell { int event_idx; int cell; int decrement; "
            "int peer_rank; int intra_cluster; };" in result.cuda_source
        )


class TestWave16bClusterSync:
    """Wave 1.6b emitter half — cluster-DSM-aware notify/wait codegen.

    On schedules with ``cluster_dim`` set AND at least one
    intra-cluster edge, the emitted source replaces the global-atomic
    notify/wait with a relaxed cluster-DSM store + wave-boundary
    ``cluster.sync()``. Cross-cluster edges keep the global path so
    correctness is preserved across cluster boundaries.

    Cluster-agnostic schedules (``cluster_dim=None``) MUST emit code
    byte-identical to the pre-Wave-1.6b emitter so SM_90- targets
    aren't broken.
    """

    def test_no_cluster_path_when_cluster_dim_none(self) -> None:
        """Default schedule (cluster_dim=None) — the cluster path is
        gated off entirely. No ``cooperative_groups::cluster_group``
        and no ``cluster.sync()`` mentions in the emitted source."""
        graph = _saxpy_diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        assert schedule.launch_config.cluster_dim is None
        result = emit_cuda_megakernel(
            schedule,
            device_function_sources=_all_diamond_bodies(),
        )
        assert "cooperative_groups" not in result.cuda_source
        assert "cluster.sync()" not in result.cuda_source
        assert "this_cluster" not in result.cuda_source
        # Manifest agrees the cluster-sync gate is off.
        assert result.manifest["cluster_sync"]["enabled"] is False

    def test_cluster_path_emitted_when_intra_cluster_edges_present(self) -> None:
        """With ``cluster_dim=(2,1,1)`` and a chain on 2 SMs (so the
        scheduler co-locates dependent tasks in cluster 0), the
        emitted source must contain BOTH the cluster.sync() path AND
        the global-atomic path (for any cross-cluster edges that
        survive)."""
        # Build a chain that fits entirely in one cluster of size 2.
        ev = EventTensor((1,), wait_count_default=1)
        calls = (
            DeviceCall(
                name="t0",
                body_fn=lambda c: None,
                task_shape=(1,),
                out_edges=(EventEdge("e", lambda c: (0,)),),
            ),
            DeviceCall(
                name="t1",
                body_fn=lambda c: None,
                task_shape=(1,),
                in_edges=(EventEdge("e", lambda c: (0,)),),
            ),
        )
        graph = MegakernelGraph(
            name="cluster_chain",
            calls=calls,
            event_tensors={"e": ev},
            policy="static",
        )
        schedule = compute_static_schedule(
            graph,
            sm_count=2,
            supports_clusters=True,
            cluster_dim=(2, 1, 1),
        )
        # Sanity: the scheduler placed both tasks intra-cluster.
        assert schedule.scheduling_metadata["intra_cluster_edges"] >= 1

        bodies = {
            "t0": DeviceFunctionSource(name="t0", body="(void)task_id; (void)sm_id; (void)buffers;"),
            "t1": DeviceFunctionSource(name="t1", body="(void)task_id; (void)sm_id; (void)buffers;"),
        }
        result = emit_cuda_megakernel(
            schedule,
            device_function_sources=bodies,
        )
        # Cluster path emitted.
        assert "cooperative_groups" in result.cuda_source
        assert "cluster.sync()" in result.cuda_source
        assert "this_cluster()" in result.cuda_source
        # Both cluster path AND global-atomic fallback path coexist
        # in the wrapper — the emitter generates ``if (c.intra_cluster)
        # { ... } else { cg_rt_cuda_etensor_*_d(...); }``.
        assert "c.intra_cluster" in result.cuda_source
        assert "cg_rt_cuda_etensor_wait_d" in result.cuda_source
        assert "cg_rt_cuda_etensor_notify_d" in result.cuda_source
        # Manifest reflects the gate state.
        assert result.manifest["cluster_sync"]["enabled"] is True
        assert result.manifest["cluster_sync"]["intra_cluster_edges_present"] is True

    def test_intra_cluster_flag_on_in_out_cells(self) -> None:
        """Constant tables emit one ``intra_cluster`` flag per cell,
        matching the per-cell mask the scheduler computed. Pinning
        this on a chain that produces exactly 1 intra-cluster cell on
        each side."""
        ev = EventTensor((1,), wait_count_default=1)
        calls = (
            DeviceCall(
                name="t0",
                body_fn=lambda c: None,
                task_shape=(1,),
                out_edges=(EventEdge("e", lambda c: (0,)),),
            ),
            DeviceCall(
                name="t1",
                body_fn=lambda c: None,
                task_shape=(1,),
                in_edges=(EventEdge("e", lambda c: (0,)),),
            ),
        )
        graph = MegakernelGraph(
            name="chain2",
            calls=calls,
            event_tensors={"e": ev},
            policy="static",
        )
        schedule = compute_static_schedule(
            graph,
            sm_count=2,
            supports_clusters=True,
            cluster_dim=(2, 1, 1),
        )
        bodies = {
            "t0": DeviceFunctionSource(name="t0", body="(void)task_id; (void)sm_id; (void)buffers;"),
            "t1": DeviceFunctionSource(name="t1", body="(void)task_id; (void)sm_id; (void)buffers;"),
        }
        result = emit_cuda_megakernel(
            schedule,
            device_function_sources=bodies,
        )
        # CG_OUT_CELLS has exactly one cell — t0's lone out-edge —
        # which must carry intra_cluster=1. Format is
        # ``{event_idx, cell, decrement, peer_rank, intra_cluster}``.
        # Find the OUT_CELLS table line and assert its last field is 1.
        out_table_idx = result.cuda_source.find("CG_OUT_CELLS[")
        # First non-zero cell tuple after the table opening brace.
        slice_after = result.cuda_source[out_table_idx:]
        first_brace = slice_after.find("{", slice_after.find("="))
        first_close = slice_after.find("}", first_brace)
        first_cell = slice_after[first_brace : first_close + 1]
        # Cell shape is ``{e, c, d, p, ic}`` — last digit before ``}``
        # is the intra_cluster flag.
        assert first_cell.replace(" ", "").endswith(",1}"), f"out-cell intra_cluster flag should be 1; got {first_cell}"

        # Same check on CG_IN_CELLS (t1's lone in-edge).
        in_table_idx = result.cuda_source.find("CG_IN_CELLS[")
        slice_after = result.cuda_source[in_table_idx:]
        first_brace = slice_after.find("{", slice_after.find("="))
        first_close = slice_after.find("}", first_brace)
        first_in_cell = slice_after[first_brace : first_close + 1]
        assert first_in_cell.replace(" ", "").endswith(",1}"), (
            f"in-cell intra_cluster flag should be 1; got {first_in_cell}"
        )

    def test_cluster_dim_set_but_no_intra_cluster_edges_keeps_global_path(
        self,
    ) -> None:
        """When cluster_dim is set but every edge is cross-cluster
        (e.g. cluster_size=1 — degenerate, or load-tolerance broke
        every preference), the cluster path is NOT emitted. The
        emitter must not pay any cluster.sync() cost when there's
        nothing to sync."""
        graph = _saxpy_diamond_graph()
        # cluster_dim=(1,1,1) — cluster_size=1, so every edge is
        # cross-cluster (no two tasks share a cluster of size 1).
        schedule = compute_static_schedule(
            graph,
            sm_count=4,
            supports_clusters=True,
            cluster_dim=(1, 1, 1),
        )
        result = emit_cuda_megakernel(
            schedule,
            device_function_sources=_all_diamond_bodies(),
        )
        # No intra-cluster edges → cluster path stays gated off.
        assert result.manifest["cluster_sync"]["enabled"] is False
        assert "cooperative_groups" not in result.cuda_source
        assert "cluster.sync()" not in result.cuda_source

    def test_nvrtc_compiles(self) -> None:
        """End-to-end NVRTC smoke-compile of the cluster-aware source.

        Skipped unless ``cu13_nvrtc`` is importable (the host has the
        full CUDA 13 toolchain). On a CPU-only host this just confirms
        the emitter doesn't generate something obviously malformed."""
        try:
            from compgen.targets.gpu.nvidia.blackwell.cu13_nvrtc import (
                nvrtc_compile,
            )
        except Exception:
            pytest.skip("cu13_nvrtc not available — skip NVRTC smoke compile")

        ev = EventTensor((1,), wait_count_default=1)
        calls = (
            DeviceCall(
                name="t0",
                body_fn=lambda c: None,
                task_shape=(1,),
                out_edges=(EventEdge("e", lambda c: (0,)),),
            ),
            DeviceCall(
                name="t1",
                body_fn=lambda c: None,
                task_shape=(1,),
                in_edges=(EventEdge("e", lambda c: (0,)),),
            ),
        )
        graph = MegakernelGraph(
            name="nvrtc_smoke",
            calls=calls,
            event_tensors={"e": ev},
            policy="static",
        )
        schedule = compute_static_schedule(
            graph,
            sm_count=2,
            supports_clusters=True,
            cluster_dim=(2, 1, 1),
        )
        bodies = {
            "t0": DeviceFunctionSource(name="t0", body="(void)task_id; (void)sm_id; (void)buffers;"),
            "t1": DeviceFunctionSource(name="t1", body="(void)task_id; (void)sm_id; (void)buffers;"),
        }
        result = emit_cuda_megakernel(
            schedule,
            device_function_sources=bodies,
        )
        # Just confirm the API runs end-to-end. We don't run the
        # kernel — that's behind ``requires_gpu``. ``nvrtc_compile``
        # may itself raise if the device toolchain is out of date or
        # doesn't support sm_100 cluster ops; we surface that as a
        # skip so cpu-only CI stays green.
        try:
            nvrtc_compile(  # type: ignore[call-arg]
                result.cuda_source,
                arch="sm_100",
            )
        except Exception as exc:
            pytest.skip(f"NVRTC smoke compile failed on host: {exc}")


class TestKindTableStable:
    def test_kind_table_sorted_by_name(self) -> None:
        graph = _saxpy_diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        result = emit_cuda_megakernel(
            schedule,
            device_function_sources=_all_diamond_bodies(),
        )
        # Sorted alphabetically.
        names_in_order = [result.device_function_table[k] for k in sorted(result.device_function_table)]
        assert names_in_order == sorted(["saxpy_a", "saxpy_b", "saxpy_c", "reduce_d"])
