"""Tests for runtime/megakernel.py — MegakernelGraph composition.

Verifies the paper's (Jin et al., MLSys '26) megakernel semantics:
one persistent launch executes tasks from **multiple device functions**
under a shared event-tensor-driven scheduler, with DAG-aware dispatch
that minimises wait overhead.

Patterns exercised end-to-end as real MegakernelGraph compositions:

- **GEMM + ReduceScatter** — two DeviceCalls (GEMM tiles + RS rows)
  in one launch. The paper's canonical 1.4× speedup case.
- **MoE dispatch** — expert-routed token producers + per-expert
  consumers, each a separate DeviceCall.
- **Diamond DAG** — one-to-many fan-out, many-to-one fan-in across
  three DeviceCalls.

Plus:

- Static-policy topological ordering actually works (consumers on
  same worker never block).
- Dynamic-policy ready queue never pops a task with unsatisfied
  predecessors.
- Edge validation (unknown event, shape mismatch, non-positive
  decrement) surfaces at graph construction, not launch.
- Cycle detection.
- Error propagation + timeout.
- Static and dynamic produce identical observable output.
- Dispatch-overhead sanity: both policies run a 100-task DAG in well
  under a second on a warm CPU; waits never exceed a tight bound.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch
from compgen.runtime.event_tensor import EventTensor
from compgen.runtime.megakernel import DeviceCall, EventEdge, MegakernelGraph

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_edge_rejects_non_positive_decrement() -> None:
    with pytest.raises(ValueError, match="decrement"):
        EventEdge("e", lambda c: (0,), decrement=0)
    with pytest.raises(ValueError, match="decrement"):
        EventEdge("e", lambda c: (0,), decrement=-1)


def test_device_call_rejects_empty_task_shape() -> None:
    with pytest.raises(ValueError, match="task_shape"):
        DeviceCall(name="x", body_fn=lambda c: None, task_shape=())


def test_graph_rejects_duplicate_call_names() -> None:
    call = DeviceCall(name="a", body_fn=lambda c: None, task_shape=(1,))
    with pytest.raises(ValueError, match="duplicate"):
        MegakernelGraph(name="g", calls=(call, call), event_tensors={})


def test_graph_rejects_edge_to_unknown_event() -> None:
    call = DeviceCall(
        name="a",
        body_fn=lambda c: None,
        task_shape=(2,),
        out_edges=(EventEdge("missing", lambda c: (0,)),),
    )
    with pytest.raises(ValueError, match="unknown event"):
        MegakernelGraph(name="g", calls=(call,), event_tensors={})


def test_graph_rejects_edge_rank_mismatch() -> None:
    """2-d event with a 1-d index_fn should fail at construction."""
    call = DeviceCall(
        name="a",
        body_fn=lambda c: None,
        task_shape=(2,),
        out_edges=(EventEdge("E", lambda c: (0,)),),
    )
    with pytest.raises(ValueError, match="rank"):
        MegakernelGraph(
            name="g",
            calls=(call,),
            event_tensors={"E": EventTensor((3, 4))},
        )


def test_graph_detects_cycle() -> None:
    """Two calls each waiting on the other's event — a cycle."""
    E1 = EventTensor((1,), wait_count_default=1)
    E2 = EventTensor((1,), wait_count_default=1)
    a = DeviceCall(
        name="a",
        body_fn=lambda c: None,
        task_shape=(1,),
        in_edges=(EventEdge("E2", lambda c: (0,)),),
        out_edges=(EventEdge("E1", lambda c: (0,)),),
    )
    b = DeviceCall(
        name="b",
        body_fn=lambda c: None,
        task_shape=(1,),
        in_edges=(EventEdge("E1", lambda c: (0,)),),
        out_edges=(EventEdge("E2", lambda c: (0,)),),
    )
    g = MegakernelGraph(name="cyc", calls=(a, b), event_tensors={"E1": E1, "E2": E2})
    with pytest.raises(ValueError, match="cycle"):
        g.launch(num_workers=2, timeout_s=2.0)


# ---------------------------------------------------------------------------
# Paper pattern: GEMM + ReduceScatter as a single megakernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["static", "dynamic"])
def test_gemm_plus_reduce_scatter_composed_in_one_launch(policy) -> None:
    """Two device functions — GEMM tile (M, N) grid + RS row (M,)
    grid — composed into one :class:`MegakernelGraph` launch.

    Event ``row_events`` has shape ``(M,)`` and ``wait_count=N``:
    each GEMM tile at ``(m, n)`` notifies cell ``m``; the RS task
    at ``m`` waits on cell ``m`` then reads the full row.

    This is the paper's canonical megakernel shape and the
    1.4×-speedup setting.
    """
    M, N = 4, 5
    produced = torch.zeros((M, N), dtype=torch.int32)
    consumed = torch.zeros(M, dtype=torch.int32)

    E_row = EventTensor((M,), wait_count_default=N, sym_name="row_events")

    def gemm_body(coord: tuple[int, ...]) -> None:
        m, n = coord
        produced[m, n] = 1

    def rs_body(coord: tuple[int, ...]) -> None:
        (m,) = coord
        # All N column tiles in row m must have completed.
        assert int(produced[m].sum().item()) == N, f"RS task m={m} saw only {int(produced[m].sum().item())}/{N} tiles"
        consumed[m] = 1

    gemm_call = DeviceCall(
        name="gemm",
        body_fn=gemm_body,
        task_shape=(M, N),
        out_edges=(EventEdge("row_events", lambda c: (c[0],)),),
    )
    rs_call = DeviceCall(
        name="reduce_scatter",
        body_fn=rs_body,
        task_shape=(M,),
        in_edges=(EventEdge("row_events", lambda c: (c[0],)),),
    )

    graph = MegakernelGraph(
        name="gemm_rs",
        calls=(gemm_call, rs_call),
        event_tensors={"row_events": E_row},
        policy=policy,
        sm_count=4,
    )
    graph.launch(timeout_s=5.0)

    assert torch.equal(produced, torch.ones((M, N), dtype=torch.int32))
    assert torch.equal(consumed, torch.ones(M, dtype=torch.int32))
    for m in range(M):
        assert E_row.load(m) == 0


# ---------------------------------------------------------------------------
# Paper pattern: MoE with data-dependent routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["static", "dynamic"])
def test_moe_routing_composed_in_one_launch(policy) -> None:
    """MoE dispatch: token tasks notify ``E_expert[topk[t]]`` with
    decrement=1; expert tasks wait on their own cell with
    ``wait_count`` = number of tokens routed to that expert.

    Paper's Fig. 5 ``"i->topk[i,:]"`` pattern — the index_fn closes
    over the routing tensor.
    """
    NUM_TOKENS = 12
    NUM_EXPERTS = 4

    torch.manual_seed(0)
    topk = torch.randint(low=0, high=NUM_EXPERTS, size=(NUM_TOKENS,)).tolist()

    histogram = [0] * NUM_EXPERTS
    for t in topk:
        histogram[t] += 1

    # One event tensor per expert so we can set per-expert wait counts.
    # (In the IR, UpdateOp writes these at runtime.)
    expert_events = {
        f"E{i}": EventTensor((1,), wait_count_default=max(1, histogram[i]), sym_name=f"E{i}")
        for i in range(NUM_EXPERTS)
    }
    # Experts with zero routed tokens: fire once to stand in for UpdateOp.
    for i, h in enumerate(histogram):
        if h == 0:
            expert_events[f"E{i}"].notify(0)

    tokens_seen: list[set[int]] = [set() for _ in range(NUM_EXPERTS)]
    tokens_lock = threading.Lock()

    def token_body(coord: tuple[int, ...]) -> None:
        (t,) = coord
        with tokens_lock:
            tokens_seen[topk[t]].add(t)

    def expert_body(coord: tuple[int, ...]) -> None:
        (i,) = coord
        expected = {t for t in range(NUM_TOKENS) if topk[t] == i}
        # The expert must observe exactly the tokens routed to it —
        # the megakernel scheduler must have ordered all relevant
        # token tasks before the expert task runs.
        with tokens_lock:
            observed = set(tokens_seen[i])
        assert observed == expected, f"expert {i}: expected {expected}, got {observed}"

    # One DeviceCall per token→expert relation group. Simpler: one
    # DeviceCall per expert (token tasks notify E<topk[t]>).
    def make_token_edge(t_idx: int) -> EventEdge:
        # Closure captures the expert for token t_idx.
        expert = topk[t_idx]
        return EventEdge(f"E{expert}", lambda _c: (0,))

    # One DeviceCall for all tokens; each coord t has a distinct
    # out-edge — model as a single call with N per-task out_edges?
    # MegakernelGraph doesn't support per-coord edges; split into N
    # tiny token DeviceCalls (one per token) so each carries its own
    # expert target. This is faithful to the paper: each task has
    # its own out_edge bound.
    token_calls = tuple(
        DeviceCall(
            name=f"token_{t}",
            body_fn=(lambda _c, t=t: token_body((t,))),
            task_shape=(1,),
            out_edges=(make_token_edge(t),),
        )
        for t in range(NUM_TOKENS)
    )
    expert_calls = tuple(
        DeviceCall(
            name=f"expert_{i}",
            body_fn=(lambda _c, i=i: expert_body((i,))),
            task_shape=(1,),
            in_edges=(EventEdge(f"E{i}", lambda _c: (0,)),),
        )
        for i in range(NUM_EXPERTS)
    )

    graph = MegakernelGraph(
        name="moe",
        calls=token_calls + expert_calls,
        event_tensors=expert_events,
        policy=policy,
        sm_count=4,
    )
    graph.launch(timeout_s=5.0)

    for i in range(NUM_EXPERTS):
        expected = {t for t in range(NUM_TOKENS) if topk[t] == i}
        assert tokens_seen[i] == expected


# ---------------------------------------------------------------------------
# Diamond DAG — three DeviceCalls, cross fan-out + fan-in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["static", "dynamic"])
def test_diamond_dag_three_calls(policy) -> None:
    """Producer (1 task) → 2 workers (2 tasks each) → Sink (1 task).

    Producer notifies E_broadcast[0..1]; each worker waits its own
    cell then notifies E_reduce[0]; sink waits E_reduce[0] with
    wait_count = 2 (for two worker groups).
    """
    E_broadcast = EventTensor((2,), wait_count_default=1)
    E_reduce = EventTensor((1,), wait_count_default=2)

    produced = [False]
    workers_done = [0, 0]
    workers_lock = threading.Lock()
    sink_done = [False]

    def producer_body(_c):
        produced[0] = True

    def worker_body(coord):
        (i,) = coord
        assert produced[0], "worker ran before producer"
        with workers_lock:
            workers_done[i] += 1

    def sink_body(_c):
        assert all(workers_done[i] > 0 for i in range(2)), "sink ran before workers"
        sink_done[0] = True

    calls = (
        DeviceCall(
            name="producer",
            body_fn=producer_body,
            task_shape=(1,),
            out_edges=(
                EventEdge("E_b", lambda _c: (0,)),
                EventEdge("E_b", lambda _c: (1,)),
            ),
        ),
        DeviceCall(
            name="workers",
            body_fn=worker_body,
            task_shape=(2,),
            in_edges=(EventEdge("E_b", lambda c: (c[0],)),),
            out_edges=(EventEdge("E_r", lambda _c: (0,)),),
        ),
        DeviceCall(
            name="sink",
            body_fn=sink_body,
            task_shape=(1,),
            in_edges=(EventEdge("E_r", lambda _c: (0,)),),
        ),
    )

    graph = MegakernelGraph(
        name="diamond",
        calls=calls,
        event_tensors={"E_b": E_broadcast, "E_r": E_reduce},
        policy=policy,
        sm_count=3,
    )
    graph.launch(timeout_s=5.0)

    assert produced[0]
    assert workers_done == [1, 1]
    assert sink_done[0]


# ---------------------------------------------------------------------------
# Static vs dynamic produce identical outputs
# ---------------------------------------------------------------------------


def test_static_and_dynamic_produce_same_output() -> None:
    """Deterministic workload: static and dynamic policies must
    produce the same final tensor state on the diamond DAG.
    """
    outputs: dict[str, tuple[bool, list[int], bool]] = {}

    for policy in ("static", "dynamic"):
        E_broadcast = EventTensor((2,), wait_count_default=1)
        E_reduce = EventTensor((1,), wait_count_default=2)
        produced = [False]
        workers_done = [0, 0]
        workers_lock = threading.Lock()
        sink_done = [False]

        def _p(_c, produced=produced):
            produced[0] = True

        def _w(coord, workers_done=workers_done, workers_lock=workers_lock):
            (i,) = coord
            with workers_lock:
                workers_done[i] += 1

        def _s(_c, sink_done=sink_done):
            sink_done[0] = True

        calls = (
            DeviceCall(
                name="producer",
                body_fn=_p,
                task_shape=(1,),
                out_edges=(
                    EventEdge("E_b", lambda _c: (0,)),
                    EventEdge("E_b", lambda _c: (1,)),
                ),
            ),
            DeviceCall(
                name="workers",
                body_fn=_w,
                task_shape=(2,),
                in_edges=(EventEdge("E_b", lambda c: (c[0],)),),
                out_edges=(EventEdge("E_r", lambda _c: (0,)),),
            ),
            DeviceCall(
                name="sink",
                body_fn=_s,
                task_shape=(1,),
                in_edges=(EventEdge("E_r", lambda _c: (0,)),),
            ),
        )
        graph = MegakernelGraph(
            name=f"diamond_{policy}",
            calls=calls,
            event_tensors={"E_b": E_broadcast, "E_r": E_reduce},
            policy=policy,
            sm_count=3,
        )
        graph.launch(timeout_s=5.0)
        outputs[policy] = (produced[0], list(workers_done), sink_done[0])

    assert outputs["static"] == outputs["dynamic"]


# ---------------------------------------------------------------------------
# Error propagation + timeout
# ---------------------------------------------------------------------------


def test_launch_propagates_body_exception() -> None:
    class _KaBoom(RuntimeError):
        pass

    def good(_c):
        pass

    def bad(_c):
        raise _KaBoom("explode")

    g = MegakernelGraph(
        name="err",
        calls=(
            DeviceCall(name="good", body_fn=good, task_shape=(4,)),
            DeviceCall(name="bad", body_fn=bad, task_shape=(2,)),
        ),
        event_tensors={},
        policy="dynamic",
    )
    with pytest.raises(_KaBoom, match="explode"):
        g.launch(num_workers=2, timeout_s=2.0)


def test_launch_timeout_on_indefinite_body() -> None:
    def slow(_c):
        time.sleep(10.0)

    g = MegakernelGraph(
        name="slow",
        calls=(DeviceCall(name="slow", body_fn=slow, task_shape=(2,)),),
        event_tensors={},
        policy="static",
    )
    with pytest.raises(TimeoutError):
        g.launch(num_workers=2, timeout_s=0.2)


# ---------------------------------------------------------------------------
# Dispatch efficiency — DAG-awareness means waits never block
# ---------------------------------------------------------------------------


def test_dispatch_never_parks_on_wait_under_dag_aware_policies() -> None:
    """Because static topo-sorts and dynamic only pops ready tasks,
    the ``load()`` check inside ``_run_task`` should never need the
    fallback ``wait()``. We verify this by timing: if the fallback
    were hit, the 5s timeout would dominate.

    The workload is a 50-task DAG; both policies should finish in
    well under a second on a warm CPU.
    """
    NUM = 50
    E = EventTensor((NUM,), wait_count_default=1)

    finished = [False] * NUM
    finished_lock = threading.Lock()

    def producer_body(coord):
        (i,) = coord
        time.sleep(0.001)
        with finished_lock:
            finished[i] = True

    def consumer_body(coord):
        (i,) = coord
        with finished_lock:
            assert finished[i], f"consumer {i} ran before producer"

    producer = DeviceCall(
        name="p",
        body_fn=producer_body,
        task_shape=(NUM,),
        out_edges=(EventEdge("E", lambda c: (c[0],)),),
    )
    consumer = DeviceCall(
        name="c",
        body_fn=consumer_body,
        task_shape=(NUM,),
        in_edges=(EventEdge("E", lambda c: (c[0],)),),
    )

    for policy in ("static", "dynamic"):
        E.reset()
        g = MegakernelGraph(
            name=f"eff_{policy}",
            calls=(producer, consumer),
            event_tensors={"E": E},
            policy=policy,
        )
        t0 = time.monotonic()
        g.launch(num_workers=4, timeout_s=5.0, batch_size=8)
        elapsed = time.monotonic() - t0
        # 50 producers * 1ms * 1/4 workers = 12.5ms lower bound; 100x
        # slack for GIL + OS scheduling. The point is to catch a
        # regression where fallback waits are hit, which would be
        # >> 1s.
        assert elapsed < 1.0, f"{policy} launch took {elapsed:.3f}s"
