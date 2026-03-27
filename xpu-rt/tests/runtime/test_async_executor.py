"""Tests for runtime/async_executor.py -- async DAG execution."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from compgen.runtime.async_executor import (
    AsyncExecutionResult,
    AsyncExecutor,
    TaskResult,
    run_async,
)
from compgen.runtime.planner import CopyOp, ExecutionPlan, MemoryPlan, PlacementDecision
from compgen.runtime.semaphore import TimelineSemaphore

EXAMPLES = Path(__file__).parent.parent.parent / "examples" / "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model_and_inputs() -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    sys.path.insert(0, str(EXAMPLES))
    from simple_mlp import SimpleMLP, get_sample_inputs
    return SimpleMLP(), get_sample_inputs()


def _simple_plan() -> ExecutionPlan:
    """Single-device plan with two partitions, no copies."""
    return ExecutionPlan(
        placements=[
            PlacementDecision(op_name="p_0", device_index=0, reason="test"),
            PlacementDecision(op_name="p_1", device_index=0, reason="test"),
        ],
        copies=[],
        execution_order=["p_0", "p_1"],
        memory_plans=[MemoryPlan(device_index=0, peak_bytes=1024)],
        estimated_latency_us=100.0,
    )


def _multi_device_plan() -> ExecutionPlan:
    """Two-device plan with a copy between partitions."""
    return ExecutionPlan(
        placements=[
            PlacementDecision(op_name="p_0", device_index=0),
            PlacementDecision(op_name="p_1", device_index=1),
        ],
        copies=[
            CopyOp(
                tensor_name="p_0_to_p_1",
                src_device=0,
                dst_device=1,
                size_bytes=4096,
                async_=True,
            ),
        ],
        execution_order=["p_0", "p_1"],
        memory_plans=[
            MemoryPlan(device_index=0, peak_bytes=4096),
            MemoryPlan(device_index=1, peak_bytes=4096),
        ],
        estimated_latency_us=50.0,
    )


def _diamond_plan() -> ExecutionPlan:
    """Diamond DAG: p_0 -> p_1, p_0 -> p_2, p_1+p_2 -> p_3.

    Devices: p_0 on 0, p_1 on 0, p_2 on 1, p_3 on 1.
    Copies: p_0->p_2 (cross-device), p_1->p_3 (cross-device).
    """
    return ExecutionPlan(
        placements=[
            PlacementDecision(op_name="p_0", device_index=0),
            PlacementDecision(op_name="p_1", device_index=0),
            PlacementDecision(op_name="p_2", device_index=1),
            PlacementDecision(op_name="p_3", device_index=1),
        ],
        copies=[
            CopyOp(tensor_name="p_0_to_p_2", src_device=0, dst_device=1, size_bytes=2048, async_=True),
            CopyOp(tensor_name="p_1_to_p_3", src_device=0, dst_device=1, size_bytes=2048, async_=True),
        ],
        execution_order=["p_0", "p_1", "p_2", "p_3"],
        memory_plans=[
            MemoryPlan(device_index=0, peak_bytes=4096),
            MemoryPlan(device_index=1, peak_bytes=4096),
        ],
        estimated_latency_us=100.0,
        metadata={
            "partition_deps": {
                "p_1": ["p_0"],
                "p_2": ["p_0"],
                "p_3": ["p_1", "p_2"],
            },
        },
    )


def _run(coro: Any) -> Any:
    """Run an async coroutine in a new event loop."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# TimelineSemaphore tests
# ---------------------------------------------------------------------------

class TestTimelineSemaphore:
    """Tests for the timeline semaphore primitive."""

    def test_initial_value(self) -> None:
        sem = TimelineSemaphore(name="test")
        assert sem.value == 0

    def test_signal_advances(self) -> None:
        async def _go() -> None:
            sem = TimelineSemaphore(name="test")
            await sem.signal(1)
            assert sem.value == 1
            await sem.signal(5)
            assert sem.value == 5

        _run(_go())

    def test_signal_rejects_backwards(self) -> None:
        async def _go() -> None:
            sem = TimelineSemaphore(name="test")
            await sem.signal(3)
            with pytest.raises(ValueError, match="cannot go backwards"):
                await sem.signal(1)

        _run(_go())

    def test_wait_already_reached(self) -> None:
        """wait() returns immediately if the value is already >= target."""
        async def _go() -> None:
            sem = TimelineSemaphore(name="test")
            await sem.signal(5)
            await asyncio.wait_for(sem.wait(3), timeout=1.0)

        _run(_go())

    def test_wait_blocks_until_signal(self) -> None:
        """wait() blocks until signal advances past target."""
        async def _go() -> None:
            sem = TimelineSemaphore(name="test")

            reached = False

            async def waiter() -> None:
                nonlocal reached
                await sem.wait(2)
                reached = True

            task = asyncio.create_task(waiter())
            await asyncio.sleep(0.01)
            assert not reached

            await sem.signal(1)
            await asyncio.sleep(0.01)
            assert not reached

            await sem.signal(2)
            await asyncio.sleep(0.01)
            assert reached

            await task

        _run(_go())

    def test_multiple_waiters(self) -> None:
        """Multiple waiters at different targets all get woken."""
        async def _go() -> None:
            sem = TimelineSemaphore(name="test")
            results: list[int] = []

            async def waiter(target: int) -> None:
                await sem.wait(target)
                results.append(target)

            t1 = asyncio.create_task(waiter(1))
            t2 = asyncio.create_task(waiter(3))
            t3 = asyncio.create_task(waiter(2))

            await asyncio.sleep(0.01)
            assert results == []

            await sem.signal(3)
            await asyncio.gather(t1, t2, t3)

            assert sorted(results) == [1, 2, 3]

        _run(_go())

    def test_reset(self) -> None:
        sem = TimelineSemaphore(name="test")
        sem._value = 5
        sem.reset()
        assert sem.value == 0


# ---------------------------------------------------------------------------
# TaskResult / AsyncExecutionResult tests
# ---------------------------------------------------------------------------

class TestTaskResult:
    def test_construction(self) -> None:
        r = TaskResult(partition_id="p_0", device="cpu", elapsed_us=42.0)
        assert r.partition_id == "p_0"
        assert r.device == "cpu"
        assert r.elapsed_us == 42.0

    def test_frozen(self) -> None:
        r = TaskResult(partition_id="p_0", device="cpu", elapsed_us=1.0)
        with pytest.raises(AttributeError):
            r.device = "cuda"  # type: ignore[misc]


class TestAsyncExecutionResult:
    def test_defaults(self) -> None:
        r = AsyncExecutionResult()
        assert r.task_results == []
        assert r.total_elapsed_us == 0.0
        assert r.copy_count == 0
        assert r.output is None


# ---------------------------------------------------------------------------
# AsyncExecutor tests
# ---------------------------------------------------------------------------

class TestAsyncExecutor:
    def test_simple_plan_produces_output(self) -> None:
        """run_async returns model output."""
        model, inputs = _get_model_and_inputs()
        plan = _simple_plan()
        result = _run(run_async(plan, model, inputs, device="cpu"))

        assert result.output is not None
        assert isinstance(result.output, torch.Tensor)
        assert result.total_elapsed_us > 0

    def test_simple_plan_all_partitions_run(self) -> None:
        """Every partition in execution_order produces a TaskResult."""
        model, inputs = _get_model_and_inputs()
        plan = _simple_plan()
        result = _run(run_async(plan, model, inputs, device="cpu"))

        completed_ids = {r.partition_id for r in result.task_results}
        assert completed_ids == {"p_0", "p_1"}

    def test_copy_ops_counted(self) -> None:
        """Copy operations are counted correctly."""
        model, inputs = _get_model_and_inputs()
        plan = _multi_device_plan()
        result = _run(run_async(plan, model, inputs, device="cpu"))

        assert result.copy_count == 1

    def test_semaphore_advances_with_copies(self) -> None:
        """The timeline semaphore advances once per copy."""
        async def _go() -> None:
            model, inputs = _get_model_and_inputs()
            plan = _multi_device_plan()
            executor = AsyncExecutor(plan)
            await executor.execute(model, inputs, device="cpu")

            assert executor.semaphore.value == 1

        _run(_go())

    def test_diamond_dag_dependency_order(self) -> None:
        """In a diamond DAG, p_3 must run after both p_1 and p_2."""
        model, inputs = _get_model_and_inputs()
        plan = _diamond_plan()
        result = _run(run_async(plan, model, inputs, device="cpu"))

        completed_ids = [r.partition_id for r in result.task_results]
        idx_p1 = completed_ids.index("p_1")
        idx_p2 = completed_ids.index("p_2")
        idx_p3 = completed_ids.index("p_3")
        assert idx_p3 > idx_p1
        assert idx_p3 > idx_p2

    def test_diamond_dag_all_copies_run(self) -> None:
        """Both copies in a diamond DAG execute."""
        async def _go() -> None:
            model, inputs = _get_model_and_inputs()
            plan = _diamond_plan()
            executor = AsyncExecutor(plan)
            result = await executor.execute(model, inputs, device="cpu")

            assert result.copy_count == 2
            assert executor.semaphore.value == 2

        _run(_go())

    def test_empty_plan(self) -> None:
        """Empty plan produces empty results but still returns output."""
        model, inputs = _get_model_and_inputs()
        plan = ExecutionPlan()
        result = _run(run_async(plan, model, inputs, device="cpu"))

        assert result.task_results == []
        assert result.copy_count == 0
        assert result.output is not None

    def test_sync_copy_not_overlapped(self) -> None:
        """A synchronous copy (async_=False) runs inline, not as a sub-task."""
        plan = ExecutionPlan(
            placements=[
                PlacementDecision(op_name="p_0", device_index=0),
                PlacementDecision(op_name="p_1", device_index=1),
            ],
            copies=[
                CopyOp(
                    tensor_name="p_0_to_p_1",
                    src_device=0,
                    dst_device=1,
                    size_bytes=4096,
                    async_=False,
                ),
            ],
            execution_order=["p_0", "p_1"],
        )

        model, inputs = _get_model_and_inputs()
        result = _run(run_async(plan, model, inputs, device="cpu"))

        assert result.copy_count == 1
        completed_ids = {r.partition_id for r in result.task_results}
        assert completed_ids == {"p_0", "p_1"}

    def test_output_matches_eager(self) -> None:
        """Async executor output matches direct eager execution."""
        model, inputs = _get_model_and_inputs()
        plan = _simple_plan()

        model.eval()
        with torch.no_grad():
            expected = model(*inputs)

        result = _run(run_async(plan, model, inputs, device="cpu"))
        torch.testing.assert_close(result.output, expected)


# ---------------------------------------------------------------------------
# run_async convenience function tests
# ---------------------------------------------------------------------------

class TestRunAsync:
    def test_returns_execution_result(self) -> None:
        model, inputs = _get_model_and_inputs()
        plan = _simple_plan()
        result = _run(run_async(plan, model, inputs, device="cpu"))
        assert isinstance(result, AsyncExecutionResult)

    def test_default_device_cpu(self) -> None:
        model, inputs = _get_model_and_inputs()
        plan = _simple_plan()
        result = _run(run_async(plan, model, inputs))
        for tr in result.task_results:
            assert tr.device == "cpu"
