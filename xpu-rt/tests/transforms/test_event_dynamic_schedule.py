"""Phase-3 dynamic scheduling compiler pass tests.

Pins paper §3.2 behaviour:

- Initial ready set = tasks with zero predecessors.
- Queue capacity sized with slack to bound push-side contention.
- Successor adjacency recorded per task (so the Phase-5 emitter can
  inline push logic without re-walking the event graph).
- Predecessor counts match the dependency graph.
- Gate: graph with trigger generators on a target without on-device
  scheduler → ``DynamicSchedulingUnavailable``.
- YAML round-trips the full schedule.
- Cluster-launch gating mirrors the static pass.
"""

from __future__ import annotations

import pytest
from compgen.runtime.event_tensor import EventTensor
from compgen.runtime.megakernel import DeviceCall, EventEdge, MegakernelGraph
from compgen.transforms.event_dynamic_schedule import (
    DynamicSchedulingUnavailable,
    TriggerGenerator,
    compute_dynamic_schedule,
)
from compgen.transforms.event_static_schedule import EventTensorAllocSpec, LaunchConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diamond_graph() -> MegakernelGraph:
    """A → (B, C) → D."""
    events = {
        "ab": EventTensor((1,), wait_count_default=1),
        "ac": EventTensor((1,), wait_count_default=1),
        "bd": EventTensor((1,), wait_count_default=1),
        "cd": EventTensor((1,), wait_count_default=1),
    }
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
    return MegakernelGraph(name="diamond", calls=calls, event_tensors=events, policy="dynamic")


def _fanout_graph(n_fanout: int = 5) -> MegakernelGraph:
    """Single producer → n_fanout consumers (stress for queue pushes)."""
    et = EventTensor((n_fanout,), wait_count_default=1)
    calls = [
        DeviceCall(
            name="prod",
            body_fn=lambda c: None,
            task_shape=(1,),
            out_edges=tuple(EventEdge("signal", lambda c, _i=i: (_i,)) for i in range(n_fanout)),
        )
    ]
    for i in range(n_fanout):
        calls.append(
            DeviceCall(
                name=f"cons_{i}",
                body_fn=lambda c: None,
                task_shape=(1,),
                in_edges=(EventEdge("signal", lambda c, _i=i: (_i,)),),
            )
        )
    return MegakernelGraph(
        name="fanout",
        calls=tuple(calls),
        event_tensors={"signal": et},
        policy="dynamic",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScheduleContract:
    def test_initial_ready_set_matches_zero_pred_tasks(self) -> None:
        g = _diamond_graph()
        s = compute_dynamic_schedule(g, sm_count=4)
        # Only task A has 0 predecessors.
        assert len(s.ready_queue.initial_task_ids) == 1

    def test_task_records_carry_successor_adjacency(self) -> None:
        g = _diamond_graph()
        s = compute_dynamic_schedule(g, sm_count=4)
        # Locate A's record and confirm it lists B + C as successors.
        a_rec = next(r for r in s.tasks if r.descriptor.device_func == "A")
        succs = {
            next(r.descriptor.device_func for r in s.tasks if r.descriptor.task_id == sid)
            for sid in a_rec.successor_task_ids
        }
        assert succs == {"B", "C"}

    def test_predecessor_count_matches_dependency_graph(self) -> None:
        g = _diamond_graph()
        s = compute_dynamic_schedule(g, sm_count=4)
        by_name = {r.descriptor.device_func: r.predecessor_count for r in s.tasks}
        assert by_name == {"A": 0, "B": 1, "C": 1, "D": 2}

    def test_rejects_zero_sm_count(self) -> None:
        g = _diamond_graph()
        with pytest.raises(ValueError, match="sm_count"):
            compute_dynamic_schedule(g, sm_count=0)


class TestReadyQueue:
    def test_capacity_has_paper_slack(self) -> None:
        """Capacity = ceil(total_tasks * factor). Default factor 1.5×."""
        g = _fanout_graph(n_fanout=5)  # 6 tasks total
        s = compute_dynamic_schedule(g, sm_count=4)
        assert s.ready_queue.capacity >= 6
        # Default factor 1.5 → ceil(6*1.5)=9.
        assert s.ready_queue.capacity == 9

    def test_custom_capacity_factor_respected(self) -> None:
        g = _fanout_graph(n_fanout=10)  # 11 tasks
        s = compute_dynamic_schedule(g, sm_count=4, queue_capacity_factor=2.0)
        assert s.ready_queue.capacity == 22

    def test_pop_batch_hint_carries_through(self) -> None:
        g = _diamond_graph()
        s = compute_dynamic_schedule(g, sm_count=4, pop_batch_hint=16)
        assert s.ready_queue.pop_batch_hint == 16


class TestTriggerGate:
    def test_trigger_generator_on_supported_target_records_trigger(self) -> None:
        g = _diamond_graph()
        trig = TriggerGenerator(
            target_event="ab",
            source_tensor="exp_indptr",
            target_device_func="B",
            task_shape=(1,),
        )
        s = compute_dynamic_schedule(
            g,
            sm_count=4,
            supports_ondevice_scheduler=True,
            trigger_generators=(trig,),
        )
        assert len(s.trigger_generators) == 1
        assert s.scheduling_metadata["num_triggers"] == 1

    def test_trigger_generator_on_unsupported_target_fails_loudly(self) -> None:
        g = _diamond_graph()
        trig = TriggerGenerator(
            target_event="ab",
            source_tensor="exp_indptr",
            target_device_func="B",
            task_shape=(1,),
        )
        with pytest.raises(DynamicSchedulingUnavailable, match="on-device scheduler"):
            compute_dynamic_schedule(
                g,
                sm_count=4,
                supports_ondevice_scheduler=False,
                trigger_generators=(trig,),
            )

    def test_no_triggers_on_unsupported_target_is_fine(self) -> None:
        """If the graph is data-independent, dynamic-schedule on a
        target without an on-device scheduler is degenerate but
        legal — callers can still prefer static in that case."""
        g = _diamond_graph()
        s = compute_dynamic_schedule(g, sm_count=4, supports_ondevice_scheduler=False)
        assert len(s.trigger_generators) == 0


class TestAllocsAndLaunch:
    def test_event_allocs_enumerate_all_tensors(self) -> None:
        g = _diamond_graph()
        s = compute_dynamic_schedule(g, sm_count=4)
        names = [a.name for a in s.event_tensor_allocs]
        assert sorted(names) == ["ab", "ac", "bd", "cd"]
        for spec in s.event_tensor_allocs:
            assert isinstance(spec, EventTensorAllocSpec)

    def test_launch_config_defaults_to_cooperative(self) -> None:
        g = _diamond_graph()
        s = compute_dynamic_schedule(g, sm_count=8)
        lc = s.launch_config
        assert isinstance(lc, LaunchConfig)
        assert lc.grid_dim == (8, 1, 1)
        assert lc.cooperative is True
        assert lc.cluster_dim is None

    def test_cluster_dim_gated(self) -> None:
        g = _diamond_graph()
        s_off = compute_dynamic_schedule(g, sm_count=8, supports_clusters=False, cluster_dim=(2, 1, 1))
        assert s_off.launch_config.cluster_dim is None
        s_on = compute_dynamic_schedule(g, sm_count=8, supports_clusters=True, cluster_dim=(2, 1, 1))
        assert s_on.launch_config.cluster_dim == (2, 1, 1)


class TestSerialisation:
    def test_yaml_round_trips(self) -> None:
        import yaml

        g = _diamond_graph()
        s = compute_dynamic_schedule(g, sm_count=4)
        doc = yaml.safe_load(s.as_yaml())
        assert doc["graph_name"] == "diamond"
        assert doc["sm_count"] == 4
        assert doc["ready_queue"]["capacity"] >= 4
        assert doc["scheduling_metadata"]["policy"] == "dynamic"
        assert len(doc["tasks"]) == 4


class TestMetadata:
    def test_max_fanout_is_tracked(self) -> None:
        g = _fanout_graph(n_fanout=7)
        s = compute_dynamic_schedule(g, sm_count=4)
        # Producer has 7 successors.
        assert s.scheduling_metadata["max_fanout"] == 7
