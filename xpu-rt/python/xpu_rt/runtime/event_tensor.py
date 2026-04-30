"""Event Tensor runtime primitives — Phase-D3 Python-level groundwork.

Implements the paper's Event Tensor abstraction (Jin et al., MLSys '26)
at the CPU level, matching the semantics of the ``compgen.ir.event``
dialect without the C / CUDA runtime that Phase D3 will eventually
build.

Primitives:

- :class:`EventTensor` — multi-dim int64 counter array with atomic
  ``notify(idx, decrement)`` and blocking ``wait(idx)``.  Each cell
  starts at ``wait_count_default``; notify decrements; wait blocks
  until the cell reaches ``≤ 0``.  Matches ``event.event_tensor`` +
  ``event.notify`` + ``event.wait`` from the dialect.

- :func:`persistent_launch` — runs a user ``kernel_fn(coord,
  event_tensors)`` across every coordinate of a task-grid shape,
  using real worker threads.  Supports both the paper's static
  (pre-computed per-worker queues) and dynamic (shared atomic queue)
  scheduling policies.  Matches ``event.call_device`` +
  ``event.graph.policy`` from the dialect.

Correctness + safety:

- Counter storage is a ``torch.int64`` tensor (matches
  ``EventTensorTypeAttr.counter_dtype``).
- Atomicity + blocking uses :class:`threading.Condition`.  This is
  the Python-level equivalent of the GPU's ``atomicSub`` +
  spin-wait: on the device side the paper spins (cheap when all
  SMs are active); on CPU we sleep on a condition variable so we
  don't burn a whole core per blocked worker.
- Worker exceptions propagate out of ``persistent_launch`` via a
  shared error-holder; the first worker to raise wakes sleepers
  and the exception bubbles to the caller.
- Reentrant lock so a kernel calling both ``notify`` and ``wait``
  on the same tensor in the same thread doesn't deadlock.

This module is the CPU reference implementation.  The GPU / RTOS /
bare-metal equivalents land in Phase B+C+D; the Python semantics
here define what those implementations must preserve.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable
from queue import Empty, Queue
from typing import Any, Literal

import structlog
import torch

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# EventTensor
# ---------------------------------------------------------------------------


_DTYPE_MAP: dict[str, torch.dtype] = {
    "i32": torch.int32,
    "u32": torch.int32,  # torch has no uint32; fall back to int32
    "i64": torch.int64,
    "u64": torch.int64,  # same
}


class EventTensor:
    """A multi-dim array of counter-semaphores with atomic notify/wait.

    Matches the dialect op ``event.event_tensor``. Each cell starts
    at ``wait_count_default``. :meth:`notify` atomically decrements
    one cell (matching ``event.notify``). :meth:`wait` blocks until
    the cell reaches zero or below (matching ``event.wait``).

    Args:
        shape: Tensor shape (multi-dim).
        wait_count_default: Initial value each cell holds. When this
            many notifies have landed the wait unblocks. A counter can
            go negative if more notifies arrive than expected — this
            is legal and matches the paper's over-notify semantics.
        dtype: One of ``"i32" | "u32" | "i64" | "u64"`` (from the
            dialect's ``_VALID_COUNTER_DTYPES``). Torch-backing uses
            ``int32`` or ``int64``; the ``u*`` dtypes are accepted for
            compatibility but stored as signed (the paper's semantics
            never go above ``wait_count_default`` so the MSB is unused).
        scope: ``"workgroup" | "device" | "system"`` — informational on
            CPU (all memory is coherent). Forwarded to the runtime
            adapter if Phase D3 wires this through to real hardware.
        sym_name: Optional name for trace-log readability.
    """

    __slots__ = (
        "shape",
        "wait_count_default",
        "dtype",
        "scope",
        "sym_name",
        "_counters",
        "_cond",
        "_failed",
        "_failure",
    )

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        wait_count_default: int = 1,
        *,
        dtype: str = "i64",
        scope: str = "device",
        sym_name: str = "",
    ) -> None:
        if wait_count_default < 0:
            raise ValueError(f"wait_count_default must be >= 0, got {wait_count_default}")
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"dtype {dtype!r} invalid; expected one of {sorted(_DTYPE_MAP)}")
        if scope not in ("workgroup", "device", "system"):
            raise ValueError(f"scope {scope!r} invalid; expected workgroup|device|system")
        if not shape or any(d <= 0 for d in shape):
            raise ValueError(f"shape must be non-empty with positive dims; got {shape}")

        self.shape = tuple(shape)
        self.wait_count_default = int(wait_count_default)
        self.dtype = dtype
        self.scope = scope
        self.sym_name = sym_name
        self._counters = torch.full(self.shape, wait_count_default, dtype=_DTYPE_MAP[dtype])
        # Single condition per tensor protects all cells. A finer-
        # grained per-cell lock would reduce contention but
        # multiplies memory + lock state; for CPU-emulation of GPU
        # semantics one lock is adequate. Use RLock so a kernel can
        # hold the tensor's lock across notify+wait if needed.
        self._cond = threading.Condition(threading.RLock())
        # Error flag — set when a co-executing worker fails so
        # waiters bail out instead of deadlocking.
        self._failed = False
        self._failure: BaseException | None = None

    # --- atomic ops ----------------------------------------------------

    def notify(self, idx: tuple[int, ...] | int, decrement: int = 1) -> None:
        """Atomically decrement ``self._counters[idx]`` by ``decrement``.

        Wakes any waiter whose cell has reached ≤ 0. ``decrement`` must
        be positive (matches ``event.notify`` verifier).
        """
        if decrement <= 0:
            raise ValueError(f"decrement must be positive, got {decrement}")
        coord = self._canon_idx(idx)
        with self._cond:
            self._counters[coord] = self._counters[coord] - decrement
            # Always notify_all — cheap on CPU, and we don't track
            # which waiters are parked on which cell.
            self._cond.notify_all()

    def wait(
        self,
        idx: tuple[int, ...] | int,
        *,
        timeout_s: float | None = None,
    ) -> None:
        """Block until ``self._counters[idx] <= 0`` or ``timeout_s``.

        Raises ``TimeoutError`` on timeout. Raises the worker's
        exception if the launch is cancelled (see :func:`persistent_launch`).
        """
        coord = self._canon_idx(idx)
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._cond:
            while int(self._counters[coord].item()) > 0:
                if self._failed:
                    # A sibling worker raised; don't deadlock.
                    raise _PersistentLaunchCancelled(self._failure)
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"EventTensor({self.sym_name or 'unnamed'})[{coord}].wait "
                            f"timed out after {timeout_s}s "
                            f"(current={int(self._counters[coord].item())})"
                        )
                    self._cond.wait(remaining)
                else:
                    # No timeout — still wake every 50 ms so we can
                    # spot cancellation flags without hanging forever.
                    self._cond.wait(0.05)

    def load(self, idx: tuple[int, ...] | int) -> int:
        """Non-blocking read of a counter value."""
        coord = self._canon_idx(idx)
        with self._cond:
            return int(self._counters[coord].item())

    def update(self, idx: tuple[int, ...] | int, new_count: int) -> None:
        """Atomically store ``new_count`` into ``self._counters[idx]``.

        Implements ``event.update`` (paper Fig. 5b). The paper's MoE
        pattern uses this to rewrite each expert's wait count from a
        runtime ``topk`` tensor before dependent tiles launch. Any
        parked waiters are woken — if ``new_count <= 0`` they'll
        proceed immediately; otherwise they'll re-check and block
        again.

        Args:
            idx: Target coordinate in the counter array.
            new_count: New counter value. May be zero (all waiters
                proceed) or negative (over-signalled — legal, matches
                the over-notify semantics of ``notify``).
        """
        coord = self._canon_idx(idx)
        with self._cond:
            self._counters[coord] = int(new_count)
            self._cond.notify_all()

    def trigger(self, idx: tuple[int, ...] | int, consumer_count: int) -> None:
        """Reinitialise a counter to ``consumer_count`` and wake waiters.

        Implements ``event.trigger`` (paper Fig. 5b). Symmetric to
        :meth:`update` but carries the ``consumer_count`` semantics
        from the paper: the caller is announcing how many consumers
        will ``wait`` on this cell. Equivalent to ``update(idx,
        consumer_count)`` but named for the downstream-tile-count
        use-case so traces read more clearly.

        Args:
            idx: Target coordinate.
            consumer_count: Number of consumers that will wait for this
                cell before it's considered "triggered complete".
                Negative values are rejected — triggering with zero
                consumers is the degenerate case that immediately
                unblocks any pre-existing waiter.
        """
        if consumer_count < 0:
            raise ValueError(f"trigger consumer_count must be non-negative, got {consumer_count}")
        coord = self._canon_idx(idx)
        with self._cond:
            self._counters[coord] = int(consumer_count)
            self._cond.notify_all()

    def reset(self, idx: tuple[int, ...] | int | None = None) -> None:
        """Reset a cell (or all cells) back to ``wait_count_default``.

        Useful between iterations of a persistent kernel; the paper's
        scheduler resets counters at graph boundaries.
        """
        with self._cond:
            if idx is None:
                self._counters.fill_(self.wait_count_default)
            else:
                coord = self._canon_idx(idx)
                self._counters[coord] = self.wait_count_default
            self._cond.notify_all()

    # --- cancellation hook (used by persistent_launch on worker error) -

    def _cancel(self, exc: BaseException) -> None:
        """Mark the tensor's waiters as cancelled. Internal."""
        with self._cond:
            self._failed = True
            self._failure = exc
            self._cond.notify_all()

    def _uncancel(self) -> None:
        """Clear the cancellation flag. Internal."""
        with self._cond:
            self._failed = False
            self._failure = None

    # --- helpers -------------------------------------------------------

    def _canon_idx(self, idx: tuple[int, ...] | int) -> tuple[int, ...]:
        if isinstance(idx, int):
            idx = (idx,)
        if len(idx) != len(self.shape):
            raise IndexError(
                f"EventTensor({self.sym_name or 'unnamed'}): expected {len(self.shape)}-d index, got {idx!r}"
            )
        out: list[int] = []
        for i, dim in zip(idx, self.shape, strict=True):
            if i < 0 or i >= dim:
                raise IndexError(
                    f"EventTensor({self.sym_name or 'unnamed'}): index {i} out of range for dim size {dim}"
                )
            out.append(int(i))
        return tuple(out)

    def __repr__(self) -> str:
        return (
            f"EventTensor(name={self.sym_name!r}, shape={self.shape}, "
            f"dtype={self.dtype!r}, scope={self.scope!r}, "
            f"wait_count_default={self.wait_count_default})"
        )


def materialize_view(
    template_shape: tuple[int, ...] | list[int],
    concrete_shape: tuple[int, ...] | list[int],
    *,
    wait_count_default: int = 1,
    dtype: str = "i64",
    scope: str = "device",
    sym_name: str | None = None,
) -> EventTensor:
    """Materialize a concrete-shape EventTensor from a symbolic-shape template.

    Implements ``event.materialize_view`` (paper Fig. 4). The template
    shape may have ``0`` or ``-1`` entries indicating symbolic dims;
    the helper produces a fresh :class:`EventTensor` with those dims
    replaced by the supplied concrete extents and every cell
    initialised to ``wait_count_default``.

    Args:
        template_shape: Symbolic template shape (``-1`` / ``0`` mark
            dims that need materialization; positive entries are
            concrete assertions that ``concrete_shape`` must match).
        concrete_shape: Full shape to materialize at. Must have the
            same rank as ``template_shape`` and every dim must be a
            positive integer.
        wait_count_default: Initial counter value for every cell of
            the materialized tensor.
        dtype: Counter dtype.
        scope: EventTensor scope (``"workgroup"``, ``"device"``,
            ``"system"``).
        sym_name: Optional name for the materialized tensor.

    Returns:
        A new concrete-shape :class:`EventTensor`.

    Raises:
        ValueError: rank mismatch, non-positive concrete dims, or a
            template dim with a positive value that doesn't match the
            concrete_shape entry.
    """
    tshape = tuple(int(d) for d in template_shape)
    cshape = tuple(int(d) for d in concrete_shape)
    if len(tshape) != len(cshape):
        raise ValueError(
            f"materialize_view: template rank {len(tshape)} != concrete rank "
            f"{len(cshape)} (template={tshape}, concrete={cshape})"
        )
    resolved: list[int] = []
    for axis, (t_dim, c_dim) in enumerate(zip(tshape, cshape)):
        if c_dim <= 0:
            raise ValueError(f"materialize_view: concrete_shape[{axis}]={c_dim} must be positive")
        if t_dim > 0 and t_dim != c_dim:
            raise ValueError(
                f"materialize_view: template dim {axis} is concrete ({t_dim}) "
                f"but concrete_shape[{axis}]={c_dim} disagrees"
            )
        resolved.append(c_dim)
    return EventTensor(
        shape=tuple(resolved),
        wait_count_default=wait_count_default,
        dtype=dtype,
        scope=scope,
        sym_name=sym_name,
    )


class _PersistentLaunchCancelled(RuntimeError):
    """Raised inside a wait when a sibling worker has failed.

    Used internally by :func:`persistent_launch` to wake stuck waiters.
    The outer launch re-raises the original worker exception — callers
    see the real failure, not this sentinel.
    """

    def __init__(self, original: BaseException | None) -> None:
        super().__init__("persistent_launch cancelled by sibling worker failure")
        self.original = original


# ---------------------------------------------------------------------------
# persistent_launch
# ---------------------------------------------------------------------------


KernelFn = Callable[[tuple[int, ...], dict[str, EventTensor]], None]


def persistent_launch(
    kernel_fn: KernelFn,
    task_grid_shape: tuple[int, ...] | list[int],
    *,
    event_tensors: dict[str, EventTensor] | None = None,
    num_workers: int = 0,
    scheduler: Literal["static", "dynamic"] = "static",
    timeout_s: float | None = None,
) -> None:
    """Run ``kernel_fn`` across a task grid using ``num_workers`` threads.

    Matches the shape of ``event.call_device`` + ``event.graph.policy``
    from the dialect. Each thread takes tasks from a precomputed list
    (``scheduler="static"``) or a shared queue (``scheduler="dynamic"``)
    and invokes ``kernel_fn(coord, event_tensors)`` once per task. The
    kernel is free to call ``notify`` / ``wait`` on the event tensors
    to express inter-task dependencies.

    Args:
        kernel_fn: User function. Takes the task coordinate tuple and
            the event-tensor dict. No return value. May raise — the
            first exception propagates out of this call.
        task_grid_shape: Multi-dim task-grid shape. E.g. ``(M, N)`` →
            ``M*N`` tasks.
        event_tensors: Dict of event tensors the kernel reads. Keys
            match the ``sym_name`` of each ``event.event_tensor`` op in
            the IR.
        num_workers: Number of threads. ``0`` (default) uses
            ``os.cpu_count()`` — roughly matches the paper's per-SM
            worker model.
        scheduler: ``"static"`` pre-partitions tasks across workers
            (round-robin — the simplest static policy from the
            paper). ``"dynamic"`` uses a shared FIFO queue.
        timeout_s: Overall launch timeout in seconds. ``None`` (default)
            means no timeout.

    Raises:
        RuntimeError: If no tasks would be run (e.g. empty grid).
        TimeoutError: If ``timeout_s`` elapses before all workers join.
        Whatever ``kernel_fn`` raised: propagated from the first failing
            worker.
    """
    if scheduler not in ("static", "dynamic"):
        raise ValueError(f"scheduler must be 'static' or 'dynamic', got {scheduler!r}")
    if not task_grid_shape:
        raise RuntimeError("task_grid_shape is empty — nothing to run")

    event_tensors = dict(event_tensors or {})
    tensors_list = list(event_tensors.values())

    if num_workers <= 0:
        import os

        num_workers = max(1, os.cpu_count() or 1)

    # Enumerate all task coords in lexicographic order.
    all_coords: list[tuple[int, ...]] = list(itertools.product(*[range(d) for d in task_grid_shape]))
    if not all_coords:
        raise RuntimeError("task_grid_shape produced zero coordinates")

    # Clamp workers so we don't spin up more threads than tasks.
    num_workers = min(num_workers, len(all_coords))

    # Error propagation — first worker to raise wins. Shared Lock
    # guards the holder so we don't race on exception assignment.
    error_holder: dict[str, Any] = {"exc": None, "coord": None, "worker_id": None}
    error_lock = threading.Lock()

    def _record_failure(exc: BaseException, coord: tuple[int, ...] | None, worker_id: int) -> None:
        with error_lock:
            if error_holder["exc"] is None:
                error_holder["exc"] = exc
                error_holder["coord"] = coord
                error_holder["worker_id"] = worker_id
        # Wake every waiter on every tensor so sibling workers bail out.
        for t in tensors_list:
            t._cancel(exc)

    # --- static scheduler: pre-partition tasks -------------------------
    if scheduler == "static":
        per_worker: list[list[tuple[int, ...]]] = [[] for _ in range(num_workers)]
        for i, coord in enumerate(all_coords):
            per_worker[i % num_workers].append(coord)

        def _worker_static(worker_id: int, tasks: list[tuple[int, ...]]) -> None:
            for coord in tasks:
                with error_lock:
                    if error_holder["exc"] is not None:
                        return
                try:
                    kernel_fn(coord, event_tensors)
                except _PersistentLaunchCancelled:
                    # Another worker already failed; exit quietly.
                    return
                except BaseException as exc:  # noqa: BLE001
                    _record_failure(exc, coord, worker_id)
                    return

        threads = [
            threading.Thread(
                target=_worker_static,
                args=(wid, per_worker[wid]),
                name=f"compgen-rt-worker-{wid}",
                daemon=True,
            )
            for wid in range(num_workers)
        ]

    # --- dynamic scheduler: shared queue ------------------------------
    else:
        q: Queue[tuple[int, ...] | None] = Queue()
        for coord in all_coords:
            q.put(coord)
        for _ in range(num_workers):
            q.put(None)  # sentinel per worker

        def _worker_dynamic(worker_id: int) -> None:
            while True:
                with error_lock:
                    if error_holder["exc"] is not None:
                        return
                try:
                    coord = q.get(timeout=0.5)
                except Empty:
                    # Check error flag + loop.
                    continue
                if coord is None:
                    return
                try:
                    kernel_fn(coord, event_tensors)
                except _PersistentLaunchCancelled:
                    return
                except BaseException as exc:  # noqa: BLE001
                    _record_failure(exc, coord, worker_id)
                    return

        threads = [
            threading.Thread(
                target=_worker_dynamic,
                args=(wid,),
                name=f"compgen-rt-worker-{wid}",
                daemon=True,
            )
            for wid in range(num_workers)
        ]

    # --- launch + join -------------------------------------------------
    log.info(
        "persistent_launch.start",
        scheduler=scheduler,
        num_workers=num_workers,
        num_tasks=len(all_coords),
        task_grid=tuple(task_grid_shape),
        num_event_tensors=len(tensors_list),
    )
    start = time.monotonic()
    for t in threads:
        t.start()

    # Join with optional overall timeout.
    for t in threads:
        remaining = None if timeout_s is None else max(0.0, timeout_s - (time.monotonic() - start))
        t.join(timeout=remaining)
        if t.is_alive():
            # Deadlock or timeout — record the failure to unblock anyone
            # still holding an event wait.
            te = TimeoutError(f"persistent_launch timed out after {timeout_s}s; worker {t.name} still running")
            _record_failure(te, None, -1)
            # Best-effort join to clean up.
            for tt in threads:
                tt.join(timeout=1.0)
            # Clear cancellation flags so the tensors are reusable
            # by the caller after the exception.
            for tensor in tensors_list:
                tensor._uncancel()
            raise te

    # Clear cancellation flags unconditionally — tensors survive the
    # launch and the caller may want to reuse them.
    for tensor in tensors_list:
        tensor._uncancel()

    elapsed_ms = (time.monotonic() - start) * 1000.0
    with error_lock:
        exc = error_holder["exc"]
        coord = error_holder["coord"]
        worker_id = error_holder["worker_id"]

    if exc is not None:
        log.warning(
            "persistent_launch.failed",
            worker_id=worker_id,
            coord=coord,
            error=repr(exc),
            elapsed_ms=round(elapsed_ms, 3),
        )
        raise exc

    log.info(
        "persistent_launch.done",
        num_tasks=len(all_coords),
        elapsed_ms=round(elapsed_ms, 3),
    )


__all__ = ["EventTensor", "KernelFn", "materialize_view", "persistent_launch"]
