"""Tests for solve/memory.py -- memory allocation."""

from __future__ import annotations

from compgen.solve.memory import BufferLifetime, MemoryAllocation, solve_memory


def test_buffer_lifetime_construction() -> None:
    bl = BufferLifetime(buffer_name="buf0", size_bytes=1024, device_index=0, start_us=0.0, end_us=100.0)
    assert bl.buffer_name == "buf0"
    assert bl.size_bytes == 1024


def test_memory_allocation_defaults() -> None:
    ma = MemoryAllocation()
    assert ma.offsets == {}
    assert ma.feasible is False


def test_solve_memory_feasible() -> None:
    lifetimes = [
        BufferLifetime("a", 1024, 0, 0.0, 50.0),
        BufferLifetime("b", 2048, 0, 0.0, 100.0),
    ]
    result = solve_memory(lifetimes, {0: 1_000_000})
    assert result.feasible
    assert "a" in result.offsets
    assert "b" in result.offsets
    assert result.peak_per_device[0] <= 1_000_000


def test_solve_memory_infeasible() -> None:
    lifetimes = [
        BufferLifetime("a", 500, 0, 0.0, 100.0),
        BufferLifetime("b", 600, 0, 0.0, 100.0),
    ]
    # Capacity only 800 but both overlap → need 1100
    result = solve_memory(lifetimes, {0: 800})
    assert not result.feasible


def test_solve_memory_reuse() -> None:
    # Non-overlapping lifetimes can share space
    lifetimes = [
        BufferLifetime("a", 1024, 0, 0.0, 50.0),
        BufferLifetime("b", 1024, 0, 60.0, 100.0),
    ]
    result = solve_memory(lifetimes, {0: 2048})
    assert result.feasible
    # Peak should be 1024 (reuse) not 2048
    assert result.peak_per_device[0] <= 2048


def test_solve_memory_multi_device() -> None:
    lifetimes = [
        BufferLifetime("a", 1024, 0, 0.0, 100.0),
        BufferLifetime("b", 2048, 1, 0.0, 100.0),
    ]
    result = solve_memory(lifetimes, {0: 4096, 1: 4096})
    assert result.feasible
    assert 0 in result.peak_per_device
    assert 1 in result.peak_per_device
