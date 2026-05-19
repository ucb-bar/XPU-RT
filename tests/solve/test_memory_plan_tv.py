"""Tests for the memory-planner translation-validation pass."""

from __future__ import annotations

from dataclasses import replace

import pytest
from xpu_rt.solve.memory_plan_tv import translation_validate_memory_plan
from xpu_rt.solve.memory_planner import (
    BufferAllocation,
    BufferSpec,
    MemoryPlanInput,
    MemoryPlanSolved,
    TierCapacity,
)


def _make_clean() -> tuple[MemoryPlanInput, MemoryPlanSolved]:
    buffers = (
        BufferSpec("a", size_bytes=128, lifetime_start=0, lifetime_end=4, allowed_tiers=("sram",), alignment=16),
        BufferSpec("b", size_bytes=64, lifetime_start=2, lifetime_end=6, allowed_tiers=("sram",), alignment=16),
        BufferSpec("c", size_bytes=32, lifetime_start=5, lifetime_end=8, allowed_tiers=("sram",), alignment=16),
    )
    tiers = (TierCapacity("sram", capacity_bytes=4096),)
    problem = MemoryPlanInput(buffers=buffers, tier_capacities=tiers)
    allocations = (
        BufferAllocation("a", tier="sram", offset_bytes=0),
        BufferAllocation("b", tier="sram", offset_bytes=128),
        BufferAllocation("c", tier="sram", offset_bytes=0),
    )
    solution = MemoryPlanSolved(
        schema_version="memory_plan_solver_v1",
        solver_backend="test",
        status="optimal",
        buffers=allocations,
        tier_peak_usage={"sram": 192},
        objective_value=0.0,
        formulation_hash="deadbeefcafefeed",
    )
    return problem, solution


def test_clean_solution_proves() -> None:
    problem, solution = _make_clean()
    result = translation_validate_memory_plan(problem, solution)
    assert result.proved, result.violations
    assert result.violations == []
    assert result.n_pairs_checked >= 1


def test_buffer_overlap_caught() -> None:
    problem, solution = _make_clean()
    # Force b's offset to overlap a within their shared lifetime.
    corrupt = list(solution.buffers)
    corrupt[1] = replace(corrupt[1], offset_bytes=64)
    bad = replace(solution, buffers=tuple(corrupt))
    result = translation_validate_memory_plan(problem, bad)
    assert not result.proved
    kinds = {v.kind for v in result.violations}
    assert "buffer_overlap" in kinds


def test_tier_capacity_caught() -> None:
    problem, solution = _make_clean()
    tight = replace(problem, tier_capacities=(TierCapacity("sram", capacity_bytes=64),))
    result = translation_validate_memory_plan(tight, solution)
    assert not result.proved
    kinds = {v.kind for v in result.violations}
    assert "tier_capacity_exceeded" in kinds


def test_alignment_caught() -> None:
    problem, solution = _make_clean()
    corrupt = list(solution.buffers)
    # Buffer "b" has alignment=16; force an offset of 129 (not divisible).
    corrupt[1] = replace(corrupt[1], offset_bytes=129)
    bad = replace(solution, buffers=tuple(corrupt))
    result = translation_validate_memory_plan(problem, bad)
    assert not result.proved
    kinds = {v.kind for v in result.violations}
    assert "alignment_violated" in kinds


def test_fixed_assignment_caught() -> None:
    problem, solution = _make_clean()
    # Declare a fixed-assignment to a tier the solution does not honor;
    # extend allowed_tiers so the fixed-assignment passes input validation.
    buffers = tuple(
        replace(b, allowed_tiers=("sram", "dram")) for b in problem.buffers
    )
    tiers = problem.tier_capacities + (TierCapacity("dram", capacity_bytes=8192),)
    forced = replace(
        problem,
        buffers=buffers,
        tier_capacities=tiers,
        fixed_assignments={"a": "dram"},
    )
    result = translation_validate_memory_plan(forced, solution)
    assert not result.proved
    kinds = {v.kind for v in result.violations}
    assert "fixed_assignment_violated" in kinds


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
