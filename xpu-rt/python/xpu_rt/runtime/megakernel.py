"""Megakernel graph composition — Phase-D3 Python-level groundwork.

Matches the paper's ``GraphOp`` + ``CallDeviceOp`` + ``EventCoordAttr``
semantics (Jin et al., MLSys '26): **one persistent launch composes
multiple device functions into a single scheduling domain**, with
fine-grained task dependencies expressed as event-tensor edges.

This module sits **above** :mod:`compgen.runtime.event_tensor`.

- :class:`EventEdge` — one directed edge: ``(event_name, task_coord ↦
  event_coord, decrement)``. Matches ``event.coord`` (``EventCoordAttr``).
- :class:`DeviceCall` — one device function + task grid + in/out
  edges. Matches ``event.call_device`` (``CallDeviceOp``).
- :class:`MegakernelGraph` — collection of device calls + scheduling
  policy + event-tensor bindings. Matches ``event.graph`` (``GraphOp``).

Why this is not just a loop over ``persistent_launch``:

- **Multi-function composition** — one persistent launch executes
  tasks from *all* device calls, matching the paper's claim that
  GEMM and ReduceScatter tasks interleave on the same SMs under one
  scheduler.
- **Edge-driven wait/notify** — the runtime auto-inserts
  ``event.wait(in_edges)`` before each task body and
  ``event.notify(out_edges)`` after; the user ``body_fn`` stays pure
  compute. Matches what the paper's compiler emits when lowering
  ``CallDeviceOp`` to kernel code.
- **DAG-aware scheduling** — efficient dispatch that hides latency:
  - *static* policy: topologically sort tasks across all calls,
    round-robin partition across workers. Consumers never precede
    their producers on the same worker, so waits never block.
  - *dynamic* policy: maintain a ready queue populated only when a
    task's predecessors have all completed. Workers pull batched;
    never park on a wait.

  These match the efficiency envelope of the paper's Algorithm 1 /
  Algorithm 2 respectively.

- **Overhead-minimised dispatch** — batched queue pulls, a single
  cancellation ``threading.Event`` shared across workers (no 50 ms
  poll), and precomputed predecessor counts (no runtime DAG walks).

Honest scope:
- CPU reference implementation. Python GIL limits true parallelism
  for pure-Python bodies; PyTorch ops release the GIL so real tensor
  workloads do parallelise. The dispatch semantics are the
  contract; the CUDA/RTOS backends land in Phase C + D.
- :class:`UpdateOp` / :class:`TriggerOp` / :class:`MaterializeViewOp`
  (data-dependent extensions, Fig. 4/5 of the paper) are **not** in
  this file. Those need a symbolic-shape runtime + dynamic task-grid
  materialisation; separate, deliberate work.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import structlog

from compgen.runtime.event_tensor import EventTensor

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Edge + DeviceCall + MegakernelGraph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventEdge:
    """One directed edge between a device-call task and an event cell.

    Matches :class:`compgen.ir.event.attrs.EventCoordAttr`: binds an
    event tensor name to a function that maps a task's coordinate to
    a cell in that event tensor.

    Attributes:
        event_name: Must match a key of
            ``MegakernelGraph.event_tensors``.
        index_fn: Maps a task coordinate (one tuple per launched
            task) to an event-cell coordinate. For the paper's
            canonical ``"ij->i"`` pattern: ``lambda c: (c[0],)``.
            For a constant projection (``"*->0"``): ``lambda _: (0,)``.
            For data-dependent routing (``"i->topk[i,:]"``): a
            closure that captures the ``topk`` tensor.
        decrement: Amount ``notify`` subtracts when this edge fires.
            Paper default is 1; values >1 model grouped completion.
        peer_rank: When set, this edge addresses a cell on a *peer*
            rank's event tensor instead of the local rank's. The
            Phase-5 emitter lowers the cell to
            ``cg_rt_cuda_etensor_peer_notify_d`` /
            ``cg_rt_cuda_etensor_peer_wait_d`` and the launcher
            passes the peer's mapped event-tensor pointer alongside
            the local one. ``None`` (default) keeps the local
            (intra-rank) semantics every existing workload uses.
    """

    event_name: str
    index_fn: Callable[[tuple[int, ...]], tuple[int, ...] | int]
    decrement: int = 1
    peer_rank: int | None = None

    def __post_init__(self) -> None:
        if self.decrement <= 0:
            raise ValueError(f"EventEdge.decrement must be positive; got {self.decrement}")
        if self.peer_rank is not None and self.peer_rank < 0:
            raise ValueError(f"EventEdge.peer_rank must be a non-negative rank index or None; got {self.peer_rank}")

    def resolve(self, task_coord: tuple[int, ...]) -> tuple[int, ...]:
        """Return the event-tensor coordinate for this edge at ``task_coord``."""
        raw = self.index_fn(task_coord)
        if isinstance(raw, int):
            return (raw,)
        return tuple(int(x) for x in raw)


@dataclass(frozen=True)
class DeviceCall:
    """One device function + task grid + event-tensor edges.

    Matches :class:`compgen.ir.event.ops.CallDeviceOp`.

    Attributes:
        name: Unique within the parent graph (matches a
            ``SymbolRefAttr`` in the IR).
        body_fn: Pure computation, signature
            ``body_fn(task_coord: tuple[int, ...]) -> None``. Must not
            call ``notify`` / ``wait`` itself — those are inserted by
            the runtime from the declared edges.
        task_shape: Multi-dim task grid. ``(M, N)`` means ``M*N`` tasks.
        in_edges: Edges whose events must reach ``≤ 0`` before the
            task body runs. Inserted as ``wait`` before body.
        out_edges: Edges fired after the task body returns. Inserted
            as ``notify`` after body.
    """

    name: str
    body_fn: Callable[[tuple[int, ...]], None]
    task_shape: tuple[int, ...]
    in_edges: tuple[EventEdge, ...] = ()
    out_edges: tuple[EventEdge, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_shape:
            raise ValueError(f"DeviceCall({self.name!r}): task_shape is empty")
        if any(d <= 0 for d in self.task_shape):
            raise ValueError(f"DeviceCall({self.name!r}): task_shape {self.task_shape} has non-positive dim")


# --- internal task-id type ----------------------------------------------


@dataclass(frozen=True)
class _Task:
    """Internal: one (call, coord) pair with a stable id."""

    task_id: int
    call: DeviceCall
    coord: tuple[int, ...]


# ---------------------------------------------------------------------------
# MegakernelGraph
# ---------------------------------------------------------------------------


@dataclass
class MegakernelGraph:
    """A composition of DeviceCalls sharing a pool of event tensors.

    Matches :class:`compgen.ir.event.ops.GraphOp`. One
    :meth:`launch` call runs every task across every device call in
    one persistent launch — workers see a unified task pool, not
    per-call launches.

    Args:
        name: Symbolic name (for logging / trace).
        calls: The device calls composed by this graph.
        event_tensors: Shared pool of event tensors referenced by
            the calls' edges. Keys must match ``EventEdge.event_name``.
        policy: Scheduling policy. ``"static"`` topologically sorts
            tasks across all calls then round-robin partitions across
            workers. ``"dynamic"`` maintains a ready queue and workers
            only pull tasks whose predecessors have notified.
        sm_count: Hint for number of workers. ``None`` means default
            (see ``launch``). Matches ``GraphOp.sm_count``.

    Raises:
        ValueError: Duplicate call names, or edge references an
            undeclared event, or edge resolves to out-of-bounds cell.
    """

    name: str
    calls: tuple[DeviceCall, ...]
    event_tensors: dict[str, EventTensor]
    policy: Literal["static", "dynamic"] = "static"
    sm_count: int | None = None

    # Populated by _validate + _build_tasks after __post_init__.
    _tasks: tuple[_Task, ...] = field(default_factory=tuple, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.policy not in ("static", "dynamic"):
            raise ValueError(f"policy must be 'static' or 'dynamic'; got {self.policy!r}")
        if self.sm_count is not None and self.sm_count <= 0:
            raise ValueError(f"sm_count must be positive; got {self.sm_count}")
        self._validate_calls()
        self._tasks = tuple(self._enumerate_tasks())

    # --- static validation --------------------------------------------

    def _validate_calls(self) -> None:
        if not self.calls:
            raise ValueError("MegakernelGraph must have at least one DeviceCall")
        seen: set[str] = set()
        for call in self.calls:
            if call.name in seen:
                raise ValueError(f"MegakernelGraph({self.name!r}): duplicate DeviceCall name {call.name!r}")
            seen.add(call.name)

            for edge in itertools.chain(call.in_edges, call.out_edges):
                if edge.event_name not in self.event_tensors:
                    raise ValueError(
                        f"MegakernelGraph({self.name!r}): call {call.name!r} edge "
                        f"references unknown event {edge.event_name!r}"
                    )
                # Probe the index_fn on the task origin to surface
                # obvious shape errors before launch.
                origin = tuple(0 for _ in call.task_shape)
                try:
                    resolved = edge.resolve(origin)
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(
                        f"MegakernelGraph({self.name!r}): call {call.name!r} edge "
                        f"index_fn raised on task_coord {origin}: {exc!r}"
                    ) from exc
                et = self.event_tensors[edge.event_name]
                if len(resolved) != len(et.shape):
                    raise ValueError(
                        f"MegakernelGraph({self.name!r}): call {call.name!r} edge "
                        f"to {edge.event_name!r} resolved to rank-{len(resolved)} coord, "
                        f"event shape is rank-{len(et.shape)}"
                    )

    # --- task enumeration ----------------------------------------------

    def _enumerate_tasks(self) -> list[_Task]:
        tasks: list[_Task] = []
        task_id = 0
        for call in self.calls:
            for coord in itertools.product(*[range(d) for d in call.task_shape]):
                tasks.append(_Task(task_id=task_id, call=call, coord=coord))
                task_id += 1
        return tasks

    # --- dependency analysis -------------------------------------------

    def _build_dependency_graph(
        self,
    ) -> tuple[dict[int, list[int]], dict[int, int]]:
        """Compute ``(successors, predecessor_count)`` for every task.

        A task ``T`` is a predecessor of ``S`` if some out-edge of
        ``T`` and some in-edge of ``S`` target the same event cell.
        Cell-granularity deps, not event-granularity — this captures
        the paper's fine-grained parallelism.

        Returns:
            ``successors[task_id]`` — task ids that depend on ``task_id``.
                Uses a deduplicated list (each pair appears once).
            ``predecessor_count[task_id]`` — how many distinct
                predecessor tasks ``task_id`` waits on.
        """
        # Build a producer map: (event_name, cell_coord) → [task_ids].
        producers: dict[tuple[str, tuple[int, ...]], list[int]] = defaultdict(list)
        for t in self._tasks:
            for e in t.call.out_edges:
                cell = e.resolve(t.coord)
                producers[(e.event_name, cell)].append(t.task_id)

        successors: dict[int, set[int]] = defaultdict(set)
        pred_count: dict[int, int] = {t.task_id: 0 for t in self._tasks}

        for t in self._tasks:
            pred_ids: set[int] = set()
            for e in t.call.in_edges:
                cell = e.resolve(t.coord)
                for p_id in producers.get((e.event_name, cell), ()):
                    if p_id == t.task_id:
                        continue  # self-edges don't count
                    pred_ids.add(p_id)
                    successors[p_id].add(t.task_id)
            pred_count[t.task_id] = len(pred_ids)

        # Freeze to lists so launcher code is straight-line.
        succ_lists = {tid: sorted(ss) for tid, ss in successors.items()}
        # Ensure every task_id has an entry, even if no successors.
        for t in self._tasks:
            succ_lists.setdefault(t.task_id, [])
        return succ_lists, pred_count

    def _topo_sort(self, successors: dict[int, list[int]], pred_count: dict[int, int]) -> list[int]:
        """Return task_ids in a topologically valid order.

        Ties are broken by task_id for determinism. Raises
        ``ValueError`` on a cycle — the paper's megakernel pattern
        is acyclic by construction, but edges with index-fn aliases
        can accidentally create a cycle.
        """
        remaining = dict(pred_count)
        ready: list[int] = sorted([tid for tid, c in remaining.items() if c == 0])
        order: list[int] = []
        while ready:
            # Take deterministically from the head.
            current = ready.pop(0)
            order.append(current)
            for s in successors.get(current, ()):
                remaining[s] -= 1
                if remaining[s] == 0:
                    # Insert in sorted order to keep output deterministic.
                    # Binary search is overkill at this scale.
                    ready.append(s)
                    ready.sort()
        if len(order) != len(self._tasks):
            stuck = [tid for tid, c in remaining.items() if c > 0]
            raise ValueError(
                f"MegakernelGraph({self.name!r}): event-tensor cycle detected — "
                f"{len(stuck)} tasks cannot be scheduled (blocked tasks: {stuck[:10]}...)"
            )
        return order

    # --- launch -------------------------------------------------------

    def launch(
        self,
        *,
        num_workers: int = 0,
        timeout_s: float | None = None,
        batch_size: int = 8,
    ) -> None:
        """Execute the megakernel graph in one persistent launch.

        Args:
            num_workers: Number of worker threads. Defaults to
                ``sm_count`` if set, otherwise ``os.cpu_count()``.
                Clamped to task count.
            timeout_s: Overall timeout. ``None`` means no timeout.
            batch_size: Dynamic-policy only — how many ready tasks a
                worker pulls per lock acquisition. Larger values
                reduce contention but increase tail latency when the
                queue drains unevenly. ``8`` is a reasonable default
                for the paper's task counts (hundreds-to-thousands).

        Raises:
            TimeoutError: If ``timeout_s`` elapses before all tasks
                complete.
            Whatever a ``DeviceCall.body_fn`` raised: propagated from
                the first failing task.
        """
        if num_workers <= 0:
            if self.sm_count is not None and self.sm_count > 0:
                num_workers = self.sm_count
            else:
                import os

                num_workers = max(1, os.cpu_count() or 1)
        num_workers = min(num_workers, len(self._tasks))
        if num_workers <= 0:
            raise RuntimeError("no tasks to run")

        successors, pred_count = self._build_dependency_graph()

        # Shared worker-error + cancellation state.
        cancel_event = threading.Event()
        error_lock = threading.Lock()
        error_holder: dict[str, object] = {"exc": None, "task_id": None}

        def _record_failure(exc: BaseException, task_id: int) -> None:
            with error_lock:
                if error_holder["exc"] is None:
                    error_holder["exc"] = exc
                    error_holder["task_id"] = task_id
            cancel_event.set()
            # Wake every waiter on every tensor — a task body that
            # happens to call wait manually (shouldn't; edges should
            # cover this) won't deadlock.
            for et in self.event_tensors.values():
                et._cancel(exc)

        # Run one task: wait in_edges, run body, notify out_edges.
        def _run_task(t: _Task) -> None:
            if cancel_event.is_set():
                return
            # in_edges — at this point the scheduler should have
            # ensured all predecessors completed (static: topo order,
            # dynamic: ready queue), so wait should be instant.
            # Assertion-level check via load(), then a non-blocking
            # wait on any stragglers (shouldn't happen but guards
            # correctness against bugs).
            for e in t.call.in_edges:
                cell = e.resolve(t.coord)
                et = self.event_tensors[e.event_name]
                if et.load(cell) > 0:
                    # Fallback: something stepped outside the DAG.
                    # Bounded wait so we surface bugs as timeouts
                    # rather than hangs.
                    et.wait(cell, timeout_s=5.0)
            t.call.body_fn(t.coord)
            for e in t.call.out_edges:
                cell = e.resolve(t.coord)
                self.event_tensors[e.event_name].notify(cell, decrement=e.decrement)

        tasks_by_id = {t.task_id: t for t in self._tasks}

        # --- static policy: topo sort + round-robin partition ----------
        if self.policy == "static":
            order = self._topo_sort(successors, pred_count)
            per_worker: list[list[int]] = [[] for _ in range(num_workers)]
            for i, tid in enumerate(order):
                per_worker[i % num_workers].append(tid)

            def _worker_static(wid: int, tids: list[int]) -> None:
                for tid in tids:
                    if cancel_event.is_set():
                        return
                    t = tasks_by_id[tid]
                    try:
                        _run_task(t)
                    except BaseException as exc:  # noqa: BLE001
                        _record_failure(exc, tid)
                        return

            threads = [
                threading.Thread(
                    target=_worker_static,
                    args=(wid, per_worker[wid]),
                    name=f"megakernel-{self.name}-static-{wid}",
                    daemon=True,
                )
                for wid in range(num_workers)
            ]

        # --- dynamic policy: ready queue driven by predecessor count ---
        else:
            remaining_pred: dict[int, int] = dict(pred_count)
            remaining_tasks = len(self._tasks)
            ready: list[int] = [tid for tid, c in remaining_pred.items() if c == 0]
            ready.sort()  # deterministic initial order

            ready_lock = threading.Lock()
            ready_cond = threading.Condition(ready_lock)
            # Shared counters guarded by ready_lock.
            completed = [0]  # boxed int so we mutate inside closures.

            def _worker_dynamic(wid: int) -> None:
                while True:
                    if cancel_event.is_set():
                        return
                    # Pull up to batch_size ready tasks in one lock hold.
                    local_batch: list[int] = []
                    with ready_cond:
                        # Wait until either there's work ready, we're
                        # done, or we're cancelled. wait_for avoids
                        # the 50 ms poll overhead.
                        ready_cond.wait_for(
                            lambda: bool(ready) or completed[0] >= remaining_tasks or cancel_event.is_set()
                        )
                        if cancel_event.is_set():
                            return
                        if completed[0] >= remaining_tasks:
                            # All done — also wake any sibling
                            # waiters and return.
                            ready_cond.notify_all()
                            return
                        while ready and len(local_batch) < batch_size:
                            local_batch.append(ready.pop(0))

                    for tid in local_batch:
                        if cancel_event.is_set():
                            return
                        t = tasks_by_id[tid]
                        try:
                            _run_task(t)
                        except BaseException as exc:  # noqa: BLE001
                            _record_failure(exc, tid)
                            return
                        # Task done — notify successors, push newly
                        # ready ones onto the queue.
                        new_ready: list[int] = []
                        with ready_cond:
                            completed[0] += 1
                            for s in successors[tid]:
                                remaining_pred[s] -= 1
                                if remaining_pred[s] == 0:
                                    new_ready.append(s)
                            if new_ready:
                                ready.extend(new_ready)
                                ready.sort()
                            # Always signal so other workers check
                            # the completion condition.
                            ready_cond.notify_all()

            threads = [
                threading.Thread(
                    target=_worker_dynamic,
                    args=(wid,),
                    name=f"megakernel-{self.name}-dynamic-{wid}",
                    daemon=True,
                )
                for wid in range(num_workers)
            ]

        # --- launch + join -----------------------------------------------
        log.info(
            "megakernel.launch.start",
            graph=self.name,
            policy=self.policy,
            num_workers=num_workers,
            num_tasks=len(self._tasks),
            num_calls=len(self.calls),
            num_events=len(self.event_tensors),
            batch_size=batch_size if self.policy == "dynamic" else None,
        )
        start = time.monotonic()
        for t in threads:
            t.start()

        for t in threads:
            remaining = None if timeout_s is None else max(0.0, timeout_s - (time.monotonic() - start))
            t.join(timeout=remaining)
            if t.is_alive():
                te = TimeoutError(
                    f"MegakernelGraph({self.name!r}) timed out after {timeout_s}s; worker {t.name} still running"
                )
                _record_failure(te, -1)
                for tt in threads:
                    tt.join(timeout=1.0)
                self._clear_tensor_cancels()
                raise te

        self._clear_tensor_cancels()
        elapsed_ms = (time.monotonic() - start) * 1000.0

        with error_lock:
            exc = error_holder["exc"]
            failed_tid = error_holder["task_id"]

        if exc is not None:
            failed_task = tasks_by_id.get(failed_tid) if isinstance(failed_tid, int) else None
            log.warning(
                "megakernel.launch.failed",
                graph=self.name,
                task_id=failed_tid,
                call=failed_task.call.name if failed_task else None,
                coord=failed_task.coord if failed_task else None,
                error=repr(exc),
                elapsed_ms=round(elapsed_ms, 3),
            )
            assert isinstance(exc, BaseException)
            raise exc

        log.info(
            "megakernel.launch.done",
            graph=self.name,
            num_tasks=len(self._tasks),
            elapsed_ms=round(elapsed_ms, 3),
        )

    def _clear_tensor_cancels(self) -> None:
        for et in self.event_tensors.values():
            et._uncancel()


__all__ = [
    "DeviceCall",
    "EventEdge",
    "MegakernelGraph",
]
