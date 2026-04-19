"""Tests for liveness_gate."""

from __future__ import annotations

from compgen.agent.gates import liveness_gate
from compgen.solve.memory import BufferLifetime


def test_deferred_when_missing_ctx() -> None:
    assert liveness_gate({})["status"] == "deferred"


def test_accepts_fitting_allocation() -> None:
    lifetimes = [
        BufferLifetime("b0", 1024, 0, 0.0, 1.0),
        BufferLifetime("b1", 2048, 0, 2.0, 3.0),    # non-overlap with b0
    ]
    r = liveness_gate(
        {}, lifetimes=lifetimes, device_capacities={0: 4096}
    )
    assert r["status"] == "accepted"
    assert r["details"]["peak_per_device"][0] <= 4096


def test_rejects_overflow() -> None:
    lifetimes = [BufferLifetime("b0", 10000, 0, 0.0, 1.0)]
    r = liveness_gate({}, lifetimes=lifetimes, device_capacities={0: 4096})
    assert r["status"] == "rejected"
    assert 0 in r["details"]["overflow_by_device"]
    assert r["details"]["overflow_by_device"][0]["peak"] == 10000


def test_reports_reuse_count() -> None:
    # Two non-overlapping small buffers → should be reusable
    lifetimes = [
        BufferLifetime("b0", 100, 0, 0.0, 1.0),
        BufferLifetime("b1", 100, 0, 2.0, 3.0),
    ]
    r = liveness_gate({}, lifetimes=lifetimes, device_capacities={0: 1024})
    assert r["status"] == "accepted"
    assert r["details"]["reuse_count"] >= 0
