"""Tests for solver objective definitions."""

from __future__ import annotations

from compgen.solve.objectives import (
    CompositeCost,
    EnergyCost,
    LatencyCost,
    MemoryCost,
    ThroughputCost,
)
from compgen.solve.partition import Partition


def test_latency_cost() -> None:
    c = LatencyCost(weight=2.0)
    assert c.weight == 2.0


def test_composite_cost() -> None:
    composite = CompositeCost(terms=[LatencyCost(weight=1.0), MemoryCost(weight=0.5)])
    assert len(composite.terms) == 2


def _make_partitions() -> list[Partition]:
    return [
        Partition(partition_id="p0", op_names=["matmul"], estimated_cost_us=100.0, memory_bytes=1024),
        Partition(partition_id="p1", op_names=["relu"], estimated_cost_us=50.0, memory_bytes=512),
    ]


def test_latency_evaluate() -> None:
    parts = _make_partitions()
    cost = LatencyCost(weight=2.0).evaluate(parts)
    assert cost == (100.0 + 50.0) * 2.0


def test_throughput_evaluate() -> None:
    parts = _make_partitions()
    cost = ThroughputCost(weight=1.0, batch_size=4).evaluate(parts)
    assert cost == (100.0 + 50.0) / 4


def test_memory_evaluate() -> None:
    parts = _make_partitions()
    cost = MemoryCost(weight=1.0).evaluate(parts)
    assert cost == 1024.0  # peak memory


def test_energy_evaluate() -> None:
    parts = _make_partitions()
    cost = EnergyCost(weight=1.0, energy_per_us=0.001).evaluate(parts)
    assert cost == 150.0 * 0.001


def test_composite_evaluate() -> None:
    parts = _make_partitions()
    composite = CompositeCost(terms=[LatencyCost(weight=1.0), MemoryCost(weight=0.5)])
    cost = composite.evaluate(parts)
    assert cost == 150.0 + 1024.0 * 0.5


def test_empty_partitions() -> None:
    empty: list[Partition] = []
    assert LatencyCost().evaluate(empty) == 0.0
    assert MemoryCost().evaluate(empty) == 0.0
    assert CompositeCost(terms=[LatencyCost()]).evaluate(empty) == 0.0
