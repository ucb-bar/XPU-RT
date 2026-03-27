"""Tests for solver objective definitions."""

from __future__ import annotations

from compgen.solve.objectives import CompositeCost, LatencyCost, MemoryCost


def test_latency_cost() -> None:
    c = LatencyCost(weight=2.0)
    assert c.weight == 2.0


def test_composite_cost() -> None:
    composite = CompositeCost(terms=[LatencyCost(weight=1.0), MemoryCost(weight=0.5)])
    assert len(composite.terms) == 2
