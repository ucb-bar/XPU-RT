#!/usr/bin/env python3
"""Tests for heft_edf's laxity gate (metaheuristics.py).

Plain asserts, no pytest dependency, matching test_granularity_advisor.py. Run
either way:

    python3 xpu-rt/tests/test_laxity_gate.py
    pytest xpu-rt/tests/

The fixture is one machine on purpose. `heft_edf_schedule` is a *priority rule*;
with a single lane the priority order is the only thing that can move the
schedule, so every number below is a hand-checkable consequence of the ordering
and not of the placement heuristic underneath it.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metaheuristics import (heft_edf_schedule, heft_schedule, laxity,
                            laxity_gates, laxity_levels, total_float)
from schedule_decoder import DecoderContext, evaluate
from workload import Operation, Workload


def _chain(durations, min_start_t=None, max_end_t=None):
    """A linear chain of ops, each with one processing time (one combination)."""
    ops, prev = [], None
    for d in durations:
        op = Operation([float(d)], predecessors=[prev] if prev else None,
                       min_start_t=min_start_t, max_end_t=max_end_t)
        ops.append(op)
        prev = op
    return ops


def _fixture():
    """One lane, three jobs:

      - a 60 ms non-periodic chain (3 x 20), the makespan-critical work;
      - a *tight* periodic op, 5 ms with a 6 ms window: 1 ms of float, and it
        misses unless something puts it first;
      - a *slack* periodic chain, 5 x 2 ms with a 200 ms window: 190 ms of
        float, so banding it above the critical chain is pure loss.

    Which is the wl_sweep "quad" situation in miniature.
    """
    nonper = _chain([20.0, 20.0, 20.0])
    tight = _chain([5.0], max_end_t=6.0)
    slack = _chain([2.0] * 5, max_end_t=200.0)
    ops = nonper + tight + slack
    w = Workload(ops, ["A"], np.zeros((1, 1)), machine_combinations=[["A"]])
    return w, nonper, tight, slack


def _score(w, **kw):
    ctx = DecoderContext(w)
    t, alpha = heft_edf_schedule(w, **kw)
    obj, misses, _ = evaluate(ctx, t, alpha, True)
    return round(obj, 6), misses


def test_total_float_is_chain_aware():
    """Every op of a periodic chain gets the same float, and it is the window
    minus the *whole* chain's work — not minus its own duration. Without the
    backward pass the head of the slack chain would read 198 ms of float and
    the tail 190, which is the bug that makes raw `max_end - eft` useless as a
    laxity signal."""
    w, nonper, tight, slack = _fixture()
    ctx = DecoderContext(w)
    fl = total_float(ctx)
    idx = {id(op): i for i, op in enumerate(ctx.ops)}

    assert fl[idx[id(tight[0])]] == 6.0 - 5.0
    slack_floats = [fl[idx[id(op)]] for op in slack]
    assert slack_floats == [200.0 - 10.0] * 5, slack_floats
    assert all(np.isinf(fl[idx[id(op)]]) for op in nonper)


def test_laxity_is_scale_free():
    """Scaling every duration and every window by the same factor leaves laxity
    unchanged — that is what lets one gate mean the same thing on a 35 ms
    control workload and a 3.7 s vision one."""
    w, _, _, _ = _fixture()
    big_ops = [Operation([d * 1000.0 for d in op.processing_times],
                         predecessors=None,
                         max_end_t=None if op.max_end_t is None else op.max_end_t * 1000.0)
               for op in w.operations]
    for op, src in zip(big_ops, w.operations):
        for p in src.predecessors:
            op.add_predecessor(big_ops[w.operations.index(p)])
    big = Workload(big_ops, ["A"], np.zeros((1, 1)), machine_combinations=[["A"]])

    a = laxity(DecoderContext(w))
    b = laxity(DecoderContext(big))
    assert np.allclose(a[np.isfinite(a)], b[np.isfinite(b)])


def test_laxity_levels_merge_floating_point_noise():
    """Ops of one instance are equally tight but their floats are sums in
    different association orders, so they differ in the last ulp. Gating inside
    that spread splits a chain at a boundary decided by rounding."""
    noisy = np.array([1.0, 1.0 + 1e-15, 1.0 + 2e-15, 2.0, 2.0 + 3e-16])
    assert laxity_levels(noisy).size == 2
    assert laxity_levels(np.array([1.0, 1.0 + 1e-3])).size == 2
    assert laxity_levels(np.array([])).size == 0


def test_gate_enumeration_covers_both_endpoints():
    """The fixture has two laxity levels, so three gates: lift nothing, lift the
    tight op, lift everything. Both endpoints must always be present — they are
    what bound the result below by plain HEFT and by the unconditional band."""
    w, _, tight, _ = _fixture()
    ctx = DecoderContext(w)
    gates = laxity_gates(ctx)

    assert len(gates) == 3, [int(g.sum()) for g in gates]
    assert not gates[0].any()                        # lift nothing
    assert np.array_equal(gates[-1], ctx.periodic)   # lift everything
    assert [int(g.sum()) for g in gates] == [0, 1, 6]


def test_gate_recovers_makespan_without_dropping_a_deadline():
    """The point of the whole change. Banding every periodic op costs 10 ms
    here (75 vs 65) and buys nothing, because the slack chain had 190 ms of
    float; banding none of them drops the tight window."""
    w, _, _, _ = _fixture()

    unconditional = _score(w, gate=np.inf)     # the old behaviour
    none_lifted = _score(w, gate=-np.inf)      # plain HEFT ordering
    gated = _score(w)

    assert unconditional == (75.0, 0)
    assert none_lifted == (60.0, 1)            # tight window missed
    assert gated == (65.0, 0)                  # valid *and* 10 ms cheaper


def test_gate_never_loses_to_any_single_gate():
    """The guard: every gate is decoded and scored, and the best (misses,
    makespan) wins, so the answer is at least as good as each gate on its own —
    including the two endpoints. This is what makes a gate this aggressive safe
    to ship: a gate that would drop a deadline is discarded before it is
    returned, not shipped and hoped for."""
    w, _, _, _ = _fixture()
    ctx = DecoderContext(w)
    got = _score(w)
    for g in laxity_gates(ctx):
        thresh = np.inf if g.all() else (laxity(ctx)[g].max() + 1e-12 if g.any()
                                         else -np.inf)
        one = _score(w, gate=thresh)
        assert (got[1], got[0]) <= (one[1], one[0]), (got, one, int(g.sum()))


def test_workload_without_windows_is_plain_heft():
    """No periodic ops means no band to gate, and no extra decodes either."""
    ops = _chain([20.0, 20.0, 20.0]) + _chain([3.0, 3.0])
    w = Workload(ops, ["A"], np.zeros((1, 1)), machine_combinations=[["A"]])
    ctx = DecoderContext(w)

    t_edf, a_edf = heft_edf_schedule(w)
    t_heft, a_heft = heft_schedule(w)
    assert np.allclose(t_edf, t_heft)
    assert np.array_equal(a_edf, a_heft)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failures.append(t.__name__)
            print(f"FAIL: {t.__name__}: {e}")
    if failures:
        print(f"\n{len(failures)}/{len(tests)} failed: {failures}")
        return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
