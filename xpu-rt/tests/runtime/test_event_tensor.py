"""Tests for runtime/event_tensor.py — EventTensor + persistent_launch.

Exercises the paper's four canonical patterns (Jin et al., MLSys '26):

1. Pipeline — producer → event → consumer (single dependency).
2. Fan-in   — M producers → 1 consumer (AllReduce-like).
3. Fan-out  — 1 producer → N consumers (Broadcast-like).
4. GEMM+Reduce-Scatter — matrix of producers notifying row-aggregated events.
5. Data-dependent / MoE — producers notifying ``E[topk[i]]`` (the paper's
   non-trivial routing case).

Plus:
- Static vs dynamic scheduler parity.
- Error propagation — a raising kernel cancels sibling waiters.
- Timeout — deadlocked waiters surface ``TimeoutError`` cleanly.
- Counter reset between launches.
- Index validation.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch
from compgen.runtime.event_tensor import EventTensor, persistent_launch

# ---------------------------------------------------------------------------
# EventTensor — primitive correctness
# ---------------------------------------------------------------------------


def test_event_tensor_init_and_shape() -> None:
    e = EventTensor((3, 4), wait_count_default=2, sym_name="E")
    assert e.shape == (3, 4)
    assert e.wait_count_default == 2
    assert e.load((0, 0)) == 2
    assert e.load((2, 3)) == 2


def test_event_tensor_notify_decrements() -> None:
    e = EventTensor((2,), wait_count_default=3)
    assert e.load(0) == 3
    e.notify(0)
    assert e.load(0) == 2
    e.notify(0, decrement=2)
    assert e.load(0) == 0
    # Over-notify is legal per paper semantics (counter goes negative).
    e.notify(0)
    assert e.load(0) == -1


def test_event_tensor_wait_unblocks_on_notify() -> None:
    """Canonical producer-consumer: consumer blocks, producer notifies,
    consumer unblocks and reads the correct order."""
    e = EventTensor((1,), wait_count_default=1)
    events: list[str] = []
    events_lock = threading.Lock()

    def consumer() -> None:
        e.wait(0)
        with events_lock:
            events.append("consumer_unblocked")

    def producer() -> None:
        # Small delay to ensure the consumer is parked first.
        time.sleep(0.05)
        with events_lock:
            events.append("producer_notifying")
        e.notify(0)

    t_c = threading.Thread(target=consumer)
    t_p = threading.Thread(target=producer)
    t_c.start()
    t_p.start()
    t_c.join(timeout=2.0)
    t_p.join(timeout=2.0)

    assert events == ["producer_notifying", "consumer_unblocked"]


def test_event_tensor_wait_negative_counter_is_unblocked() -> None:
    """If counter starts negative (over-notified before wait), wait
    returns immediately."""
    e = EventTensor((1,), wait_count_default=1)
    e.notify(0)
    e.notify(0)  # counter now -1
    assert e.load(0) == -1
    e.wait(0, timeout_s=0.1)  # must not block


def test_event_tensor_wait_timeout() -> None:
    e = EventTensor((1,), wait_count_default=1)
    with pytest.raises(TimeoutError):
        e.wait(0, timeout_s=0.05)


def test_event_tensor_reset_restores_counters() -> None:
    e = EventTensor((3,), wait_count_default=2)
    e.notify(0)
    e.notify(0)
    e.notify(1)
    assert e.load(0) == 0
    assert e.load(1) == 1
    assert e.load(2) == 2
    e.reset()
    for i in range(3):
        assert e.load(i) == 2


def test_event_tensor_reset_single_cell() -> None:
    e = EventTensor((2,), wait_count_default=3)
    e.notify(0)
    e.notify(1)
    e.reset(0)
    assert e.load(0) == 3
    assert e.load(1) == 2


def test_event_tensor_rejects_bad_dtype() -> None:
    with pytest.raises(ValueError, match="dtype"):
        EventTensor((1,), dtype="f32")


def test_event_tensor_rejects_bad_scope() -> None:
    with pytest.raises(ValueError, match="scope"):
        EventTensor((1,), scope="global")


def test_event_tensor_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        EventTensor((0, 4))
    with pytest.raises(ValueError, match="shape"):
        EventTensor(())


def test_event_tensor_rejects_oob_index() -> None:
    e = EventTensor((2, 3))
    with pytest.raises(IndexError):
        e.notify((2, 0))
    with pytest.raises(IndexError):
        e.load((0,))  # rank mismatch


# ---------------------------------------------------------------------------
# persistent_launch — paper patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheduler", ["static", "dynamic"])
def test_launch_covers_every_task(scheduler) -> None:
    """Every task coordinate must be visited exactly once."""
    visited: set[tuple[int, ...]] = set()
    visited_lock = threading.Lock()

    def kernel(coord, _events):
        with visited_lock:
            assert coord not in visited, f"{coord} visited twice"
            visited.add(coord)

    persistent_launch(kernel, (4, 5), num_workers=3, scheduler=scheduler)

    expected = {(i, j) for i in range(4) for j in range(5)}
    assert visited == expected


# --- Pattern 1: Pipeline ----------------------------------------------------


def test_pattern_pipeline(tmp_path) -> None:
    """Producer task writes a value; consumer task reads it after notify.

    Task grid is 2 tasks: coord (0,) = producer, coord (1,) = consumer.
    """
    e = EventTensor((1,), wait_count_default=1, sym_name="produced")
    shared = torch.zeros(4, dtype=torch.float32)

    def kernel(coord, events):
        E = events["produced"]
        if coord == (0,):
            # Producer: write, then notify.
            shared.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
            E.notify(0)
        elif coord == (1,):
            # Consumer: wait, then read.
            E.wait(0, timeout_s=2.0)
            # If we got here, producer's writes are observable.
            assert torch.equal(shared, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    persistent_launch(
        kernel,
        (2,),
        event_tensors={"produced": e},
        num_workers=2,
        scheduler="static",
        timeout_s=5.0,
    )

    assert e.load(0) == 0


# --- Pattern 2: Fan-in (AllReduce-like) ------------------------------------


@pytest.mark.parametrize("scheduler", ["static", "dynamic"])
def test_pattern_fan_in(scheduler) -> None:
    """M producers each notify E[0]; consumer waits with
    wait_count_default=M and must unblock only after all M notifies.

    Grid layout: tasks 0..M-1 = producers, task M = consumer. The
    consumer is placed **last** deliberately — the static scheduler
    is DAG-unaware (round-robin partition), so a consumer scheduled
    before producers on the same worker would deadlock that worker.
    Real use of the static scheduler requires the compiler to have
    solved a topologically-safe per-worker queue (the paper's
    Algorithm 1); here we hand-place the ordering.
    """
    M = 8
    e = EventTensor((1,), wait_count_default=M, sym_name="all_reduce")
    consumer_observed_at_unblock: list[int] = []

    def kernel(coord, events):
        E = events["all_reduce"]
        if coord == (M,):
            # Consumer — must see counter at 0 or below on unblock.
            E.wait(0, timeout_s=5.0)
            consumer_observed_at_unblock.append(E.load(0))
        else:
            # Small stagger so we exercise the wait-on-progress path.
            time.sleep(0.005 * coord[0])
            E.notify(0)

    persistent_launch(
        kernel,
        (M + 1,),
        event_tensors={"all_reduce": e},
        num_workers=4,
        scheduler=scheduler,
        timeout_s=5.0,
    )

    assert e.load(0) == 0  # exactly M notifies landed
    # Consumer ran exactly once and unblocked only after reaching zero.
    assert len(consumer_observed_at_unblock) == 1
    assert consumer_observed_at_unblock[0] <= 0


# --- Pattern 3: Fan-out (Broadcast-like) -----------------------------------


@pytest.mark.parametrize("scheduler", ["static", "dynamic"])
def test_pattern_fan_out(scheduler) -> None:
    """1 producer notifies E[0..N]; N consumers each wait on their own coord."""
    N = 6
    e = EventTensor((N,), wait_count_default=1, sym_name="broadcast")
    # Grid: task 0 = producer, tasks 1..N = consumers i = coord-1.
    received = torch.zeros(N, dtype=torch.int32)

    def kernel(coord, events):
        E = events["broadcast"]
        if coord == (0,):
            # Producer: notify each index individually.
            for i in range(N):
                E.notify(i)
        else:
            i = coord[0] - 1
            E.wait(i, timeout_s=5.0)
            received[i] = 1

    persistent_launch(
        kernel,
        (N + 1,),
        event_tensors={"broadcast": e},
        num_workers=min(4, N + 1),
        scheduler=scheduler,
        timeout_s=5.0,
    )

    assert torch.equal(received, torch.ones(N, dtype=torch.int32))
    for i in range(N):
        assert e.load(i) == 0


# --- Pattern 4: GEMM + Reduce-Scatter --------------------------------------


def test_pattern_gemm_reduce_scatter() -> None:
    """M*N matrix of producer tiles; each tile (m, n) notifies E[m] by 1.

    Each row event starts at N (one per column in that row). A
    downstream consumer per row waits on E[m] and reads a per-row
    aggregation.

    Task grid layout: tasks 0..M*N-1 = producers, tasks M*N..M*N+M-1 =
    consumers. This matches the paper's GEMM + ReduceScatter shape.
    """
    M, N = 4, 3
    e = EventTensor((M,), wait_count_default=N, sym_name="row_events")
    produced = torch.zeros((M, N), dtype=torch.int32)
    consumed = torch.zeros(M, dtype=torch.int32)

    P = M * N

    def coord_to_tile(c: int) -> tuple[int, int]:
        return c // N, c % N

    def kernel(coord, events):
        c = coord[0]
        E = events["row_events"]
        if c < P:
            m, n = coord_to_tile(c)
            produced[m, n] = 1
            E.notify(m)
        else:
            m = c - P
            E.wait(m, timeout_s=5.0)
            # All N producers for row m have notified — they wrote their cells.
            row_sum = int(produced[m].sum().item())
            assert row_sum == N, f"row {m} sum={row_sum}, expected {N}"
            consumed[m] = 1

    persistent_launch(
        kernel,
        (P + M,),
        event_tensors={"row_events": e},
        num_workers=6,
        scheduler="dynamic",
        timeout_s=5.0,
    )

    assert torch.equal(consumed, torch.ones(M, dtype=torch.int32))
    assert torch.equal(produced, torch.ones((M, N), dtype=torch.int32))


# --- Pattern 5: Data-dependent / MoE-style routing -------------------------


def test_pattern_data_dependent_topk_routing() -> None:
    """MoE-style: each token picks an expert (topk index); producer for
    token i notifies ``E[topk[i]]``. Experts that receive ≥1 token
    wait on their event; their wait_count is set by the topk histogram.

    This mirrors paper Fig. 5: ``"i->topk[i,:]"`` einsum notation.
    """
    NUM_TOKENS = 12
    NUM_EXPERTS = 4
    torch.manual_seed(0)
    topk = torch.randint(low=0, high=NUM_EXPERTS, size=(NUM_TOKENS,))

    # Expert i's wait count = #tokens that routed to i.
    histogram = torch.zeros(NUM_EXPERTS, dtype=torch.int64)
    for t in range(NUM_TOKENS):
        histogram[int(topk[t].item())] += 1

    # Experts with zero tokens: skip — no waits needed (cancel by
    # lowering their wait_count to 0 at init; the paper's UpdateOp
    # does this at runtime).
    per_cell_waits = [max(1, int(histogram[i].item())) for i in range(NUM_EXPERTS)]
    # Model this by building one event tensor per expert so we can
    # set individual wait counts. (In the IR, this is the event.update
    # op writing a data-dependent counter.)
    expert_events = {
        f"expert_{i}": EventTensor((1,), wait_count_default=per_cell_waits[i], sym_name=f"expert_{i}")
        for i in range(NUM_EXPERTS)
    }
    # Pre-seed any expert with 0 tokens: its counter starts at the
    # fallback 1, subtract once to unblock (stands in for UpdateOp).
    for i in range(NUM_EXPERTS):
        if int(histogram[i].item()) == 0:
            expert_events[f"expert_{i}"].notify(0)

    # Grid: NUM_TOKENS + NUM_EXPERTS tasks (producers + consumers).
    tokens_observed_by_expert: list[set[int]] = [set() for _ in range(NUM_EXPERTS)]
    obs_lock = threading.Lock()

    def kernel(coord, events):
        c = coord[0]
        if c < NUM_TOKENS:
            # Producer: token c fires at its expert topk[c].
            e_idx = int(topk[c].item())
            events[f"expert_{e_idx}"].notify(0)
        else:
            # Consumer: expert c - NUM_TOKENS.
            e_idx = c - NUM_TOKENS
            events[f"expert_{e_idx}"].wait(0, timeout_s=5.0)
            with obs_lock:
                for t in range(NUM_TOKENS):
                    if int(topk[t].item()) == e_idx:
                        tokens_observed_by_expert[e_idx].add(t)

    persistent_launch(
        kernel,
        (NUM_TOKENS + NUM_EXPERTS,),
        event_tensors=expert_events,
        num_workers=4,
        scheduler="dynamic",
        timeout_s=5.0,
    )

    for i in range(NUM_EXPERTS):
        expected = {t for t in range(NUM_TOKENS) if int(topk[t].item()) == i}
        assert tokens_observed_by_expert[i] == expected


# --- Error propagation ------------------------------------------------------


def test_launch_propagates_kernel_exception() -> None:
    """A raising kernel's exception must surface; sibling waiters must
    not deadlock."""
    e = EventTensor((1,), wait_count_default=1)

    class _KaBoom(RuntimeError):
        pass

    def kernel(coord, events):
        if coord == (0,):
            # Waiter — would deadlock if nobody notified.
            events["e"].wait(0, timeout_s=5.0)
        else:
            raise _KaBoom(f"task {coord} failed")

    with pytest.raises(_KaBoom, match="task"):
        persistent_launch(
            kernel,
            (3,),
            event_tensors={"e": e},
            num_workers=3,
            scheduler="static",
            timeout_s=3.0,
        )


def test_launch_timeout_on_deadlock() -> None:
    """If all waiters are stuck with no notifier, the overall timeout
    surfaces as TimeoutError."""
    e = EventTensor((1,), wait_count_default=5)

    def kernel(coord, events):
        events["e"].wait(0, timeout_s=10.0)

    with pytest.raises(TimeoutError):
        persistent_launch(
            kernel,
            (2,),
            event_tensors={"e": e},
            num_workers=2,
            scheduler="static",
            timeout_s=0.3,
        )


# --- Static vs dynamic parity -----------------------------------------------


def test_static_and_dynamic_produce_same_observable_outputs() -> None:
    """For a deterministic pattern, both schedulers produce the same
    final tensor state."""
    M, N = 5, 4

    def run(scheduler):
        e = EventTensor((M,), wait_count_default=N)
        accum = torch.zeros(M, dtype=torch.int32)

        def kernel(coord, events):
            c = coord[0]
            P = M * N
            E = events["e"]
            if c < P:
                m, _n = c // N, c % N
                E.notify(m)
            else:
                m = c - P
                E.wait(m, timeout_s=3.0)
                accum[m] = 1

        persistent_launch(
            kernel,
            (M * N + M,),
            event_tensors={"e": e},
            num_workers=4,
            scheduler=scheduler,
            timeout_s=5.0,
        )
        return accum

    assert torch.equal(run("static"), run("dynamic"))


# --- Launch validation ------------------------------------------------------


def test_launch_rejects_empty_grid() -> None:
    with pytest.raises(RuntimeError, match="task_grid_shape"):
        persistent_launch(lambda c, e: None, ())


def test_launch_rejects_bad_scheduler() -> None:
    with pytest.raises(ValueError, match="scheduler"):
        persistent_launch(lambda c, e: None, (2,), scheduler="greedy")  # type: ignore[arg-type]


def test_launch_clamps_workers_to_task_count() -> None:
    """If num_workers > num_tasks, extra workers are silently dropped."""
    seen: set[tuple[int, ...]] = set()
    lk = threading.Lock()

    def kernel(coord, events):
        with lk:
            seen.add(coord)

    persistent_launch(kernel, (3,), num_workers=100, scheduler="static")
    assert seen == {(0,), (1,), (2,)}
