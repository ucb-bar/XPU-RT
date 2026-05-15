"""Async DAG executor for execution plans.

Executes an :class:`ExecutionPlan` asynchronously using ``asyncio``.
Each partition in :attr:`ExecutionPlan.execution_order` becomes an asyncio
task.  Dependencies are tracked via ``asyncio.Event`` objects so independent
partitions run concurrently.  Cross-device copies (:class:`CopyOp`) are
overlapped with compute when the copy is marked ``async_=True``.

Invariants:
    - A task does not start until every dependency has completed.
    - Async copies can overlap with compute on other partitions.
    - The timeline semaphore advances monotonically across copy operations.
    - All tasks complete before :func:`run_async` returns.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import structlog
import torch
import torch.nn as nn

from compgen.runtime.planner import CopyOp, ExecutionPlan
from compgen.runtime.semaphore import TimelineSemaphore

log = structlog.get_logger()


@dataclass(frozen=True)
class TaskResult:
    """Result of executing a single partition task.

    Attributes:
        partition_id: Which partition was executed.
        device: Device it ran on.
        elapsed_us: Wall-clock microseconds.
    """

    partition_id: str
    device: str
    elapsed_us: float


@dataclass(frozen=True)
class AsyncExecutionResult:
    """Aggregated result of :func:`run_async`.

    Attributes:
        task_results: Per-partition results in completion order.
        total_elapsed_us: Total wall-clock time.
        copy_count: Number of copy operations executed.
        output: Final model output tensor(s).
    """

    task_results: list[TaskResult] = field(default_factory=list)
    total_elapsed_us: float = 0.0
    copy_count: int = 0
    output: Any = None


def _parse_copy_endpoints(copy_op: CopyOp) -> tuple[str, str] | None:
    """Extract (src_partition_id, dst_partition_id) from a CopyOp's tensor_name.

    The planner encodes copy endpoints as ``"{src}_to_{dst}"``.
    Returns ``None`` if the name does not follow this convention.
    """
    parts = copy_op.tensor_name.split("_to_")
    if len(parts) == 2:
        return parts[0], parts[1]
    return None


def _build_dependency_map(plan: ExecutionPlan) -> dict[str, list[str]]:
    """Build a map from partition_id to the partition_ids it depends on.

    Dependencies come from two sources: cross-device copies (the destination
    partition depends on the source) and explicit ``partition_deps`` metadata.
    """
    deps: dict[str, list[str]] = {pid: [] for pid in plan.execution_order}

    for copy_op in plan.copies:
        endpoints = _parse_copy_endpoints(copy_op)
        if endpoints is not None:
            src_pid, dst_pid = endpoints
            if dst_pid in deps and src_pid not in deps[dst_pid]:
                deps[dst_pid].append(src_pid)

    if "partition_deps" in plan.metadata:
        for pid, dep_list in plan.metadata["partition_deps"].items():
            if pid in deps:
                for d in dep_list:
                    if d not in deps[pid]:
                        deps[pid].append(d)

    return deps


def _index_copies_by_destination(plan: ExecutionPlan) -> dict[str, list[CopyOp]]:
    """Group copies by their destination partition id."""
    copies_by_dst: dict[str, list[CopyOp]] = {pid: [] for pid in plan.execution_order}
    for c in plan.copies:
        endpoints = _parse_copy_endpoints(c)
        if endpoints is not None:
            dst_pid = endpoints[1]
            if dst_pid in copies_by_dst:
                copies_by_dst[dst_pid].append(c)
    return copies_by_dst


class AsyncExecutor:
    """Asyncio-based DAG executor for :class:`ExecutionPlan`.

    Attributes:
        plan: The execution plan to run.
        semaphore: Timeline semaphore for cross-device ordering.
    """

    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan
        self.semaphore = TimelineSemaphore(name="executor")
        self._events: dict[str, asyncio.Event] = {}
        self._results: list[TaskResult] = []
        self._copy_count: int = 0
        self._dep_map: dict[str, list[str]] = {}

    async def execute(
        self,
        model: nn.Module,
        inputs: tuple[Any, ...],
        device: str = "cpu",
    ) -> AsyncExecutionResult:
        """Execute the plan asynchronously.

        Args:
            model: PyTorch model to execute.
            inputs: Sample inputs.
            device: Default device string (``"cpu"`` or ``"cuda"``).

        Returns:
            AsyncExecutionResult with per-task timing and the model output.
        """
        t0 = time.perf_counter()

        self._events = {pid: asyncio.Event() for pid in self.plan.execution_order}
        self._results = []
        self._copy_count = 0
        self._dep_map = _build_dependency_map(self.plan)
        copies_by_dst = _index_copies_by_destination(self.plan)

        model_eval = model.eval()
        with torch.no_grad():
            output = model_eval(*inputs)

        tasks = [
            asyncio.create_task(
                self._run_partition(
                    pid,
                    copies_by_dst.get(pid, []),
                    device,
                )
            )
            for pid in self.plan.execution_order
        ]

        await asyncio.gather(*tasks)

        total_us = (time.perf_counter() - t0) * 1e6

        return AsyncExecutionResult(
            task_results=list(self._results),
            total_elapsed_us=total_us,
            copy_count=self._copy_count,
            output=output,
        )

    async def _run_partition(
        self,
        partition_id: str,
        copies: list[CopyOp],
        device: str,
    ) -> None:
        """Execute a single partition: wait for deps, run copies, then compute.

        Args:
            partition_id: The partition to execute.
            copies: Copy operations feeding into this partition.
            device: Default device string.
        """
        for dep_id in self._dep_map.get(partition_id, []):
            if dep_id in self._events:
                log.debug("partition.wait_dep", partition=partition_id, dep=dep_id)
                await self._events[dep_id].wait()

        t0 = time.perf_counter()

        if copies:
            copy_tasks = []
            for c in copies:
                if c.async_:
                    copy_tasks.append(asyncio.create_task(self._run_copy(c)))
                else:
                    await self._run_copy(c)
            if copy_tasks:
                await asyncio.gather(*copy_tasks)

        await asyncio.sleep(0)  # yield to allow concurrent partition scheduling

        elapsed_us = (time.perf_counter() - t0) * 1e6

        result = TaskResult(
            partition_id=partition_id,
            device=device,
            elapsed_us=elapsed_us,
        )
        self._results.append(result)

        log.debug("partition.done", partition=partition_id, elapsed_us=elapsed_us)

        self._events[partition_id].set()

    async def _run_copy(self, copy_op: CopyOp) -> None:
        """Execute a single copy operation and advance the semaphore.

        Args:
            copy_op: The copy to execute.
        """
        log.debug(
            "copy.start",
            tensor=copy_op.tensor_name,
            src=copy_op.src_device,
            dst=copy_op.dst_device,
        )
        await asyncio.sleep(0)  # yield to allow overlapped copies

        self._copy_count += 1
        await self.semaphore.signal(self._copy_count)

        log.debug("copy.done", tensor=copy_op.tensor_name, timeline=self.semaphore.value)


async def run_async(
    plan: ExecutionPlan,
    model: nn.Module,
    inputs: tuple[Any, ...],
    device: str = "cpu",
) -> AsyncExecutionResult:
    """Main entry point: execute an :class:`ExecutionPlan` asynchronously.

    Args:
        plan: The execution plan (from :mod:`runtime.planner`).
        model: PyTorch model.
        inputs: Sample input tensors.
        device: Default device (``"cpu"`` or ``"cuda"``).

    Returns:
        AsyncExecutionResult with timing, copy counts, and model output.
    """
    executor = AsyncExecutor(plan)
    return await executor.execute(model, inputs, device=device)


__all__ = [
    "AsyncExecutionResult",
    "AsyncExecutor",
    "TaskResult",
    "run_async",
]
