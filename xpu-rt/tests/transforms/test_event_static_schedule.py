"""Phase-2 static scheduling compiler pass tests.

Pins the paper Algorithm 1 behaviour:

- Every task lands on exactly one SM.
- Per-SM queue depths are balanced (load-spread metadata faithful).
- Topological order preserved per SM (a consumer never precedes its
  producer on the same queue).
- Event-tensor allocation specs enumerate every tensor in the graph.
- Launch config is cooperative + grid=sm_count for persistent kernels.
- Cluster launch opt-in gated on ``supports_clusters``.
- YAML serialisation round-trips the full schedule for the bundle
  manifest.
- Cost-weighted partition respects ``cost_hints_us``.
"""

from __future__ import annotations

import pytest
from compgen.runtime.event_tensor import EventTensor
from compgen.runtime.megakernel import DeviceCall, EventEdge, MegakernelGraph
from compgen.transforms.event_static_schedule import (
    EventTensorAllocSpec,
    LaunchConfig,
    compute_static_schedule,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain_graph(n: int = 4) -> MegakernelGraph:
    """Linear pipeline: task[0] → task[1] → ... → task[n-1]."""
    # One event per producer-consumer pair.
    events = {f"e{i}": EventTensor((1,), wait_count_default=1) for i in range(n - 1)}

    calls: list[DeviceCall] = []
    for i in range(n):
        in_edges: tuple[EventEdge, ...] = ()
        out_edges: tuple[EventEdge, ...] = ()
        if i > 0:
            in_edges = (EventEdge(f"e{i - 1}", lambda c: (0,)),)
        if i < n - 1:
            out_edges = (EventEdge(f"e{i}", lambda c: (0,)),)
        calls.append(
            DeviceCall(
                name=f"task_{i}",
                body_fn=lambda c, _i=i: None,
                task_shape=(1,),
                in_edges=in_edges,
                out_edges=out_edges,
            )
        )
    return MegakernelGraph(name="chain", calls=tuple(calls), event_tensors=events, policy="static")


def _diamond_graph() -> MegakernelGraph:
    """Diamond DAG: A → (B, C) → D."""
    ab = EventTensor((1,), wait_count_default=1)
    ac = EventTensor((1,), wait_count_default=1)
    bd = EventTensor((1,), wait_count_default=1)
    cd = EventTensor((1,), wait_count_default=1)
    calls = (
        DeviceCall(
            name="A",
            body_fn=lambda c: None,
            task_shape=(1,),
            out_edges=(EventEdge("ab", lambda c: (0,)), EventEdge("ac", lambda c: (0,))),
        ),
        DeviceCall(
            name="B",
            body_fn=lambda c: None,
            task_shape=(1,),
            in_edges=(EventEdge("ab", lambda c: (0,)),),
            out_edges=(EventEdge("bd", lambda c: (0,)),),
        ),
        DeviceCall(
            name="C",
            body_fn=lambda c: None,
            task_shape=(1,),
            in_edges=(EventEdge("ac", lambda c: (0,)),),
            out_edges=(EventEdge("cd", lambda c: (0,)),),
        ),
        DeviceCall(
            name="D",
            body_fn=lambda c: None,
            task_shape=(1,),
            in_edges=(EventEdge("bd", lambda c: (0,)), EventEdge("cd", lambda c: (0,))),
        ),
    )
    return MegakernelGraph(
        name="diamond",
        calls=calls,
        event_tensors={"ab": ab, "ac": ac, "bd": bd, "cd": cd},
        policy="static",
    )


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------


class TestScheduleContract:
    def test_every_task_scheduled_exactly_once(self) -> None:
        graph = _chain_graph(n=8)
        schedule = compute_static_schedule(graph, sm_count=4)
        scheduled_ids = [desc.task_id for q in schedule.sm_queues for desc in q.tasks]
        assert sorted(scheduled_ids) == list(range(8))
        assert schedule.total_tasks == 8

    def test_sm_count_matches_input(self) -> None:
        graph = _chain_graph(n=4)
        schedule = compute_static_schedule(graph, sm_count=3)
        assert schedule.sm_count == 3
        assert len(schedule.sm_queues) == 3

    def test_rejects_zero_sm_count(self) -> None:
        graph = _chain_graph(n=4)
        with pytest.raises(ValueError, match="sm_count"):
            compute_static_schedule(graph, sm_count=0)

    def test_empty_graph_rejected_upstream(self) -> None:
        """``MegakernelGraph`` itself rejects zero-call construction, so
        a zero-task schedule is unreachable via the normal path. This
        test documents the contract: the defensive ``no tasks`` guard
        in ``compute_static_schedule`` is belt-and-suspenders only."""
        et = EventTensor((1,), wait_count_default=0)
        with pytest.raises(ValueError, match="DeviceCall"):
            MegakernelGraph(
                name="empty",
                calls=(),
                event_tensors={"x": et},
                policy="static",
            )


class TestSchedulePreservation:
    def test_topological_order_within_each_sm(self) -> None:
        """If task B depends on task A and both are on the same SM,
        A must appear before B in the queue. This is what makes the
        static scheduler correct without runtime dep-checks."""
        graph = _chain_graph(n=6)
        schedule = compute_static_schedule(graph, sm_count=2)
        for q in schedule.sm_queues:
            ids = [t.task_id for t in q.tasks]
            # chain task_ids are already topo ids — within-SM order
            # must be non-decreasing.
            assert ids == sorted(ids), f"SM {q.sm_id} violates topo: {ids}"

    def test_diamond_preserves_dependencies(self) -> None:
        graph = _diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        # Build task_id → sm_id map.
        sm_of: dict[int, int] = {}
        pos_of: dict[int, int] = {}
        for q in schedule.sm_queues:
            for pos, t in enumerate(q.tasks):
                sm_of[t.task_id] = q.sm_id
                pos_of[t.task_id] = pos
        # A is the only task in its call; its task_id is 0.
        # Within each SM, A < B (if both present), A < C, B < D, C < D.
        a_id = next(t.task_id for q in schedule.sm_queues for t in q.tasks if t.device_func == "A")
        d_id = next(t.task_id for q in schedule.sm_queues for t in q.tasks if t.device_func == "D")
        for q in schedule.sm_queues:
            ids = [t.task_id for t in q.tasks]
            if a_id in ids and d_id in ids:
                assert ids.index(a_id) < ids.index(d_id)


class TestCostBalance:
    def test_cost_weighted_partition_balances_sm_load(self) -> None:
        """With 8 equal-cost tasks across 4 SMs, total cost per SM is 2.0
        (all equal). The cost-balance-ratio should be 1.0."""
        graph = _chain_graph(n=8)
        schedule = compute_static_schedule(
            graph,
            sm_count=4,
            cost_hints_us={f"task_{i}": 1.0 for i in range(8)},
        )
        balance = schedule.scheduling_metadata["cost_balance_ratio"]
        assert balance == pytest.approx(1.0, abs=0.01)

    def test_one_heavy_task_goes_alone(self) -> None:
        """A task 10× heavier than its peers monopolises one SM; the
        other SMs absorb the remaining work. Balance should degrade
        but be deterministic."""
        graph = _chain_graph(n=4)
        hints = {"task_0": 10.0, "task_1": 1.0, "task_2": 1.0, "task_3": 1.0}
        schedule = compute_static_schedule(graph, sm_count=3, cost_hints_us=hints)
        # SM with task_0 has cost 10; peers have up to 2.0 → balance ~0.2.
        assert schedule.scheduling_metadata["cost_balance_ratio"] < 0.3


class TestAllocSpecs:
    def test_event_tensor_allocs_enumerate_all_events(self) -> None:
        graph = _diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        names = [a.name for a in schedule.event_tensor_allocs]
        assert sorted(names) == ["ab", "ac", "bd", "cd"]
        for spec in schedule.event_tensor_allocs:
            assert isinstance(spec, EventTensorAllocSpec)
            assert spec.shape == (1,)
            assert spec.wait_count_default == 1
            assert spec.dtype in {"i32", "i64"}
            assert spec.scope == "device"


class TestLaunchConfig:
    def test_grid_matches_sm_count_and_cooperative(self) -> None:
        graph = _chain_graph(n=4)
        schedule = compute_static_schedule(graph, sm_count=16)
        lc = schedule.launch_config
        assert isinstance(lc, LaunchConfig)
        assert lc.grid_dim == (16, 1, 1)
        assert lc.cooperative is True
        assert lc.cluster_dim is None  # default: no clusters

    def test_cluster_dim_gated_on_supports_clusters(self) -> None:
        graph = _chain_graph(n=4)
        s_off = compute_static_schedule(graph, sm_count=8, supports_clusters=False, cluster_dim=(2, 1, 1))
        assert s_off.launch_config.cluster_dim is None
        s_on = compute_static_schedule(graph, sm_count=8, supports_clusters=True, cluster_dim=(2, 1, 1))
        assert s_on.launch_config.cluster_dim == (2, 1, 1)


class TestSerialisation:
    def test_as_yaml_round_trips_key_fields(self) -> None:
        import yaml

        graph = _diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        doc = yaml.safe_load(schedule.as_yaml())
        assert doc["graph_name"] == "diamond"
        assert doc["sm_count"] == 2
        assert doc["total_tasks"] == 4
        assert len(doc["sm_queues"]) == 2
        assert len(doc["event_tensor_allocs"]) == 4
        assert doc["launch_config"]["cooperative"] is True


class TestClusterAwarePartitioning:
    """Wave 1.6b — when ``supports_clusters=True`` and ``cluster_dim``
    is set, the scheduler should prefer SMs in the same cluster as a
    task's predecessor so the predecessor → successor event-tensor
    edge becomes intra-cluster (eligible for cluster-DSM signaling
    in the Phase-5 emitter, which is the perf lever per bridge #127).
    """

    def test_chain_keeps_dependent_tasks_in_one_cluster(self) -> None:
        """Linear chain of 4 tasks on 8 SMs / cluster_size=4 should
        land all 4 tasks within the same cluster (cluster_id=0)
        because each successor follows its predecessor's cluster."""
        graph = _chain_graph(n=4)
        schedule = compute_static_schedule(
            graph,
            sm_count=8,
            supports_clusters=True,
            cluster_dim=(4, 1, 1),
        )
        # Every task should land in cluster 0 (predecessors land first
        # at sm 0, then each successor prefers cluster 0 within the
        # cost-tolerance band).
        clusters_used: set[int] = set()
        for q in schedule.sm_queues:
            if q.tasks:
                assert q.cluster_id is not None
                clusters_used.add(q.cluster_id)
        assert clusters_used == {0}, f"chain should stay in one cluster; got {clusters_used}"

        meta = schedule.scheduling_metadata
        assert meta["cluster_size"] == 4
        # 3 chain edges, all intra-cluster.
        assert meta["intra_cluster_edges"] == 3
        assert meta["cross_cluster_edges"] == 0
        assert meta["intra_cluster_edge_fraction"] == 1.0

    def test_diamond_keeps_dependents_local(self) -> None:
        """Diamond A → (B, C) → D on 8 SMs / cluster_size=4. All four
        tasks should land in the same cluster because B/C share A's
        cluster and D shares B/C's cluster (transitive)."""
        graph = _diamond_graph()
        schedule = compute_static_schedule(
            graph,
            sm_count=8,
            supports_clusters=True,
            cluster_dim=(4, 1, 1),
        )
        meta = schedule.scheduling_metadata
        # Diamond has 4 dependency edges (A→B, A→C, B→D, C→D).
        assert meta["intra_cluster_edges"] + meta["cross_cluster_edges"] == 4
        # All 4 should be intra-cluster.
        assert meta["intra_cluster_edge_fraction"] == 1.0

    def test_cluster_aware_off_when_supports_clusters_false(self) -> None:
        """``supports_clusters=False`` disables the cluster-aware
        partitioning even if ``cluster_dim`` is passed. Metadata
        should NOT carry cluster fields and SMQueue.cluster_id is None.
        """
        graph = _diamond_graph()
        schedule = compute_static_schedule(
            graph,
            sm_count=4,
            supports_clusters=False,
            cluster_dim=(2, 1, 1),
        )
        for q in schedule.sm_queues:
            assert q.cluster_id is None
        assert "cluster_size" not in schedule.scheduling_metadata
        assert "intra_cluster_edges" not in schedule.scheduling_metadata

    def test_load_balance_falls_back_when_cluster_overloaded(self) -> None:
        """When the preferred cluster is way over the global min, the
        scheduler falls back to global cost-min so load balance isn't
        broken. Test: 1 heavy seed-task on cluster 0 should NOT force
        many cheap successor tasks onto the same cluster — they should
        spread across the cost-balanced SMs."""
        graph = _chain_graph(n=10)
        # Make the first task very expensive; following tasks are cheap.
        cost_hints = {"task_0": 1000.0, **{f"task_{i}": 1.0 for i in range(1, 10)}}
        schedule = compute_static_schedule(
            graph,
            sm_count=8,
            cost_hints_us=cost_hints,
            supports_clusters=True,
            cluster_dim=(4, 1, 1),
        )
        # task_0 lands on sm 0 (cluster 0). Without the load-tolerance
        # fallback, every successor would also pile onto cluster 0
        # (sms 0-3) and clusters 1-3 (sms 4-7) would be empty. Verify
        # at least one task lands outside cluster 0.
        sms_used = {q.sm_id for q in schedule.sm_queues if q.tasks}
        clusters_used = {q.cluster_id for q in schedule.sm_queues if q.tasks}
        assert len(clusters_used) >= 2, (
            f"load-balance fallback should put some tasks outside cluster 0; "
            f"used={clusters_used} sms={sorted(sms_used)}"
        )


class TestMultiProducerClusterCellRace:
    """Bridge #146: when two producers in the same cluster decrement the
    same event-tensor cell with the relaxed (cluster-DSM) path, they
    race and the consumer's wait spins forever.

    Repro shape: two ``A`` tasks fold their out-edges onto a single
    ``ev`` cell that ``B`` consumes. With ``cluster_size=2`` both ``A``
    tasks land on SMs 0-1 (same cluster) before adding ``B``. The
    scheduler must mark BOTH ``A`` tasks' out-cells as
    ``intra_cluster=False`` so the relaxed-decrement race is avoided
    via the global atomic path.
    """

    def test_multi_producer_cluster_cell_forces_global_atomic(self) -> None:
        ev = EventTensor((1,), wait_count_default=2)
        ev_done = EventTensor((1,), wait_count_default=1)
        calls = (
            DeviceCall(
                name="A",
                body_fn=lambda c: None,
                task_shape=(2,),
                out_edges=(EventEdge("ev", lambda c: (0,)),),
            ),
            DeviceCall(
                name="B",
                body_fn=lambda c: None,
                task_shape=(1,),
                in_edges=(EventEdge("ev", lambda c: (0,)),),
                out_edges=(EventEdge("ev_done", lambda c: (0,)),),
            ),
        )
        graph = MegakernelGraph(
            name="multi_producer",
            calls=calls,
            event_tensors={"ev": ev, "ev_done": ev_done},
            policy="static",
        )
        schedule = compute_static_schedule(
            graph,
            sm_count=4,
            supports_clusters=True,
            cluster_dim=(2, 1, 1),
        )
        # Producer A has out_cluster_mask of length 1. With both A
        # tasks landing in cluster 0 (the cluster-aware scheduler
        # prefers locality), each A's only out-cell must be
        # intra_cluster=False to avoid the relaxed-decrement race.
        a_out_masks: list[bool] = []
        for q in schedule.sm_queues:
            for desc in q.tasks:
                if desc.device_func == "A":
                    assert len(desc.out_cluster_mask) == 1
                    a_out_masks.append(desc.out_cluster_mask[0])
        assert len(a_out_masks) == 2, "expected exactly two A producers"
        assert all(not m for m in a_out_masks), (
            "Both A producers share cluster 0 + cell (ev,(0,)); "
            "out_cluster_mask must be False on each (force global "
            f"atomic). Got: {a_out_masks}"
        )

    def test_single_producer_cluster_cell_keeps_intra(self) -> None:
        """Control: a chain with one producer per cell keeps
        ``intra_cluster=True`` — the multi-producer guard must not
        regress the single-producer optimization.
        """
        graph = _chain_graph(n=4)
        schedule = compute_static_schedule(
            graph,
            sm_count=8,
            supports_clusters=True,
            cluster_dim=(4, 1, 1),
        )
        # Every task in the chain has at most one out-edge with one
        # consumer, and one in-edge with one producer; intra-cluster
        # masks should all be True.
        for q in schedule.sm_queues:
            for desc in q.tasks:
                for is_intra in desc.out_cluster_mask:
                    assert is_intra is True, (
                        f"single-producer chain regressed: {desc.device_func} out_cluster_mask={desc.out_cluster_mask}"
                    )


class TestPoolInvariant:
    """Bridge #124: ``num_linear_tasks + num_pointwise_tasks ==
    total_tile_tasks`` is the load-bearing invariant for the cost
    predictor's pool classification. When schedule_hints' tile-grid
    schema drifts (e.g. FFN added a third linear without a third
    tile_grid entry), the predictor silently undercounts one pool
    and the reason string regresses to the ``2/57342`` MLP-1 bug.

    Pin the invariant on the existing matcher patterns (diamond +
    FFN) so future schedule schema changes fail loudly here, not
    six rounds later in the bridge.
    """

    def test_pool_count_matches_total_for_diamond_and_ffn(self) -> None:
        """Run the actual matcher, predict, and assert the pool
        invariant. Catches schedule_hints schema drift.
        """
        import torch
        import torch.nn as nn
        from compgen.kernels.cost.etc_predict import predict_etc_dispatch
        from compgen.runtime.lowering import lower_torch_to_megakernel

        class _Diamond(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Linear(64, 64, bias=False)
                self.b = nn.Linear(64, 64, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (self.a(x) + self.b(x)).relu()

        class _Ffn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.up = nn.Linear(64, 128, bias=False)
                self.down = nn.Linear(128, 64, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.down(torch.relu(self.up(x)))

        backend = {
            "target_arch": "sm_100",
            "tile_shape": [64, 64, 16],
            "use_cublasdx_for_linears": True,
            "cublasdx_precision": "bf16_fp32",
        }
        for model, sample, label in [
            (_Diamond(), (torch.randn(64, 64),), "diamond"),
            (_Ffn(), (torch.randn(64, 64),), "ffn"),
        ]:
            result = lower_torch_to_megakernel(
                model,
                sample,
                allow_generic_fallback=False,
            )
            decision = result.decision.to_dict()
            total = decision.get("total_tile_tasks", 0)
            pred = predict_etc_dispatch(
                sample_input_shape=tuple(sample[0].shape),
                decision=decision,
                backend_choice=backend,
                model_dtype="fp32",
            )
            n_lin = pred.components["num_linear_tasks"]
            n_pw = pred.components["num_pointwise_tasks"]
            assert n_lin + n_pw == total, (
                f"{label}: pool invariant violated — "
                f"num_linear={n_lin} + num_pointwise={n_pw} != "
                f"total_tile_tasks={total} "
                f"(schedule_hints={list(decision['schedule_hints'].keys())})"
            )
            # Both pools should be non-zero for these patterns
            # (diamond has add+relu; FFN has relu).
            assert n_lin > 0, f"{label}: num_linear should be >0"
            assert n_pw > 0, f"{label}: num_pointwise should be >0"


class TestTaskDescriptor:
    def test_in_out_cells_resolved_to_concrete_coords(self) -> None:
        """The Phase-5 emitter consumes concrete cell tuples. For a
        static schedule every edge's index_fn is data-independent, so
        cells must be fully resolved at schedule time."""
        graph = _diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=2)
        for q in schedule.sm_queues:
            for desc in q.tasks:
                for entry in desc.in_cells + desc.out_cells:
                    # Each entry is (event_name, cell, decrement,
                    # peer_rank). peer_rank is None for intra-rank
                    # edges (the common case for diamond_dag).
                    _name, cell, _dec, _peer = entry
                    assert isinstance(cell, tuple)
                    for c in cell:
                        assert isinstance(c, int)


class TestIntraClusterMask:
    """Wave 1.6b emitter half — per-cell intra-cluster classification.

    The scheduler annotates each TaskDescriptor's in_cells / out_cells
    with a parallel ``in_cluster_mask`` / ``out_cluster_mask`` of
    bools. ``True`` means every peer task connected via that cell is
    on an SM in the same Blackwell cluster — eligible for the
    cluster-DSM signaling path the emitter half generates. ``False``
    keeps the cell on the safe global-atomic path.
    """

    def test_chain_intra_cluster_mask_all_true_when_one_cluster(self) -> None:
        """Linear chain of 4 tasks on 8 SMs / cluster_size=4 — every
        task lands in cluster 0, so every in-edge and every out-edge
        is intra-cluster. Mask should be all-True wherever cells exist
        (head/tail tasks have one mask side empty)."""
        graph = _chain_graph(n=4)
        schedule = compute_static_schedule(
            graph,
            sm_count=8,
            supports_clusters=True,
            cluster_dim=(4, 1, 1),
        )
        for q in schedule.sm_queues:
            for desc in q.tasks:
                # Every cell that exists must be intra-cluster.
                assert all(desc.in_cluster_mask), f"task {desc.device_func} has cross-cluster in-cell"
                assert all(desc.out_cluster_mask), f"task {desc.device_func} has cross-cluster out-cell"
                # Mask length must match the parallel cell list.
                assert len(desc.in_cluster_mask) == len(desc.in_cells)
                assert len(desc.out_cluster_mask) == len(desc.out_cells)

    def test_chain_crossing_cluster_boundary_has_mixed_mask(self) -> None:
        """A long chain with one very heavy seed forces the
        load-tolerance fallback to spill some tasks onto a different
        cluster. At least one cell mask entry should be False
        (cross-cluster) somewhere in the schedule.
        """
        graph = _chain_graph(n=10)
        # task_0 dominates cluster 0's cost; later tasks fall back to
        # the global cost-min (different cluster) per the
        # load-balance band.
        cost_hints = {
            "task_0": 1000.0,
            **{f"task_{i}": 1.0 for i in range(1, 10)},
        }
        schedule = compute_static_schedule(
            graph,
            sm_count=8,
            supports_clusters=True,
            cluster_dim=(4, 1, 1),
            cost_hints_us=cost_hints,
        )
        meta = schedule.scheduling_metadata
        assert meta["cross_cluster_edges"] >= 1, f"this fixture should produce cross-cluster edges; got {meta}"

        # At least one False entry must appear somewhere in the masks.
        has_false = False
        for q in schedule.sm_queues:
            for desc in q.tasks:
                if any(not b for b in desc.in_cluster_mask):
                    has_false = True
                if any(not b for b in desc.out_cluster_mask):
                    has_false = True
        assert has_false, "expected at least one cross-cluster cell mask"

    def test_cluster_off_yields_all_false_masks(self) -> None:
        """Cluster-agnostic schedules (no ``supports_clusters``) keep
        the mask all-False so the emitter falls back to the pure
        global-atomic path."""
        graph = _diamond_graph()
        schedule = compute_static_schedule(graph, sm_count=4)
        for q in schedule.sm_queues:
            for desc in q.tasks:
                assert not any(desc.in_cluster_mask)
                assert not any(desc.out_cluster_mask)
                # Mask length still matches cells parallel.
                assert len(desc.in_cluster_mask) == len(desc.in_cells)
                assert len(desc.out_cluster_mask) == len(desc.out_cells)
