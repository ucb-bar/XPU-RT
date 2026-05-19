"""Tests for the greedy first-fit memory planner baseline."""

from __future__ import annotations

from xpu_rt.solve.memory_planner import (
    BufferSpec,
    MemoryPlanInput,
    TierCapacity,
)
from xpu_rt.solve.memory_planner_greedy import plan_memory_greedy


def test_single_tier_no_overlap() -> None:
    sizes = (1024, 2048, 512, 4096)
    bufs = tuple(
        BufferSpec(
            buffer_id=f"b{i}",
            size_bytes=sz,
            lifetime_start=i * 10,
            lifetime_end=i * 10 + 5,
            allowed_tiers=("scratch",),
        )
        for i, sz in enumerate(sizes)
    )
    plan = plan_memory_greedy(
        MemoryPlanInput(
            buffers=bufs,
            tier_capacities=(TierCapacity("scratch", capacity_bytes=1 << 20),),
        )
    )
    assert plan.status == "optimal"
    assert {a.tier for a in plan.buffers} == {"scratch"}
    assert all(a.offset_bytes == 0 for a in plan.buffers)
    assert plan.tier_peak_usage["scratch"] == max(sizes)


def test_single_tier_all_overlap() -> None:
    sizes = (1024, 2048, 512, 4096)
    bufs = tuple(
        BufferSpec(
            buffer_id=f"b{i}",
            size_bytes=sz,
            lifetime_start=0,
            lifetime_end=100,
            allowed_tiers=("scratch",),
        )
        for i, sz in enumerate(sizes)
    )
    plan = plan_memory_greedy(
        MemoryPlanInput(
            buffers=bufs,
            tier_capacities=(TierCapacity("scratch", capacity_bytes=1 << 20),),
        )
    )
    assert plan.status == "optimal"
    assert plan.tier_peak_usage["scratch"] == sum(sizes)
    offsets = sorted(a.offset_bytes for a in plan.buffers)
    assert offsets[0] == 0
    for prev, cur in zip(offsets, offsets[1:]):
        assert cur > prev


def test_two_tier_spill_priority() -> None:
    fast = TierCapacity("fast", capacity_bytes=4096, weight=1.0)
    slow = TierCapacity("slow", capacity_bytes=1 << 30, weight=10.0)
    small_hot = BufferSpec(
        "hot",
        size_bytes=512,
        lifetime_start=0,
        lifetime_end=100,
        allowed_tiers=("fast", "slow"),
        spill_cost=100.0,
    )
    big_cold = BufferSpec(
        "cold",
        size_bytes=8192,
        lifetime_start=0,
        lifetime_end=100,
        allowed_tiers=("fast", "slow"),
        spill_cost=1.0,
    )
    plan = plan_memory_greedy(
        MemoryPlanInput(buffers=(small_hot, big_cold), tier_capacities=(fast, slow))
    )
    assert plan.status == "optimal"
    by_id = {a.buffer_id: a for a in plan.buffers}
    assert by_id["hot"].tier == "fast"
    assert by_id["cold"].tier == "slow"


def test_infeasible_returns_blocked_status() -> None:
    huge = BufferSpec(
        "huge",
        size_bytes=1 << 20,
        lifetime_start=0,
        lifetime_end=10,
        allowed_tiers=("tiny",),
    )
    plan = plan_memory_greedy(
        MemoryPlanInput(
            buffers=(huge,),
            tier_capacities=(TierCapacity("tiny", capacity_bytes=4096),),
        )
    )
    assert plan.status == "infeasible"
    assert plan.buffers == ()
    assert plan.objective_value is None
