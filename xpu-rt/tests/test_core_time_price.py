#!/usr/bin/env python3
"""Tests for the core-time price in the SGS decoder (`decode(price=...)`), the
`core_time_price` policy of heft / heft_edf (a number, or "auto"), and the
decoder family's own width pinning under the codegen contract.

Every fixture is small enough to check by hand; the arithmetic is in the
docstrings. See docs/handoff_and_width_pricing.md for the design.

    pytest xpu-rt/tests/test_core_time_price.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metaheuristics import (PRICE_LEVELS, heft_edf_schedule, heft_schedule,
                            solve_with_core_time_price)
from schedule_decoder import (DecoderContext, _earliest_slot, _priority_order,
                              decode, evaluate)
from workload import Operation, Workload


# --------------------------------------------------------------------------
# 1. Width price
# --------------------------------------------------------------------------

_W_COMBOS = [["A"], ["B"], ["A", "B"]]


def _one_op(w1, w2, max_end_t=None, price=None):
    op = Operation([w1, w1, w2], job_id=0, max_end_t=max_end_t,
                   min_start_t=0.0 if max_end_t is not None else None)
    w = Workload([op], ["A", "B"], np.zeros((2, 2)), job_names=["net0"],
                 machine_combinations=_W_COMBOS)
    if price is not None:
        w.core_time_price = price
    return w


def _width(w, alpha, i=0):
    return len(w.get_machine_combinations()[int(np.argmax(alpha[i]))])


def test_price_zero_splits_and_a_high_enough_price_does_not():
    """Durations {w1: 1.0, w2: 0.6}: core-ms 1.0 against 1.2, excess 0.2.

        score(w1) = 1.0
        score(w2) = 0.6 + price * 0.2

    so the split is kept while price < 2 and dropped above it. Note what that
    says about price 1.0: a split that speeds an op up by more than 1.5x at
    width 2 (d2 < 2/3 d1) still wins at price 1, and 0.6 is on that side.
    """
    for price, width in ((0.0, 2), (1.0, 2), (1.99, 2), (2.01, 1), (3.0, 1)):
        w = _one_op(1.0, 0.6)
        ctx = DecoderContext(w)
        _t, alpha = decode(ctx, np.zeros(1), price=price)
        assert _width(w, alpha) == width, (price, width)


def test_price_one_picks_width_one_when_the_split_gains_less_than_1p5x():
    """Durations {w1: 1.0, w2: 0.7}: excess 0.4, score(w2) = 0.7 + price * 0.4.
    Price 0 -> width 2 (0.7 < 1.0); price 1 -> width 1 (1.1 > 1.0)."""
    for price, width in ((0.0, 2), (None, 2), (0.5, 2), (1.0, 1)):
        w = _one_op(1.0, 0.7)
        _t, alpha = decode(DecoderContext(w), np.zeros(1), price=price)
        assert _width(w, alpha) == width, (price, width)
    # The same through the workload attribute and both HEFT entry points.
    for fn in (heft_schedule, heft_edf_schedule):
        assert _width(_one_op(1.0, 0.7), fn(_one_op(1.0, 0.7))[1]) == 2
        w = _one_op(1.0, 0.7, price=1.0)
        assert _width(w, fn(w)[1]) == 1
        w = _one_op(1.0, 0.7, price=1.0)
        assert _width(w, fn(w, core_time_price=0.0)[1]) == 2    # argument wins
        assert w.core_time_price_result["level"] == "fixed"


def test_price_also_prices_a_slow_kind_placement():
    """Width is not the only way to burn core time. Core A runs the op in
    1.0 ms, core B in 1.5; A is busy until 0.8. Earliest finish takes B (1.5
    against 1.8); at price 1 the extra 0.5 core-ms on B outweighs finishing
    0.3 ms sooner."""
    def build(price):
        blocker = Operation([0.8, 9.0], job_id=0)
        op = Operation([1.0, 1.5], job_id=1)
        w = Workload([blocker, op], ["A", "B"], np.zeros((2, 2)),
                     job_names=["x", "y"], machine_combinations=[["A"], ["B"]])
        _t, alpha = decode(DecoderContext(w), np.array([2.0, 1.0]), price=price)
        return int(np.argmax(alpha[1]))
    assert build(None) == 1
    assert build(1.0) == 0


def test_auto_keeps_width_one_when_the_window_is_met_at_width_one():
    for fn in (heft_schedule, heft_edf_schedule):
        w = _one_op(1.0, 0.7, max_end_t=1.0, price="auto")
        t, alpha = fn(w)
        assert _width(w, alpha) == 1
        res = w.core_time_price_result
        assert res["policy"] == "auto" and res["level"] == "1"
        assert res["misses"] == 0
        assert res["tried"] == [{"level": "1", "misses": 0}]   # stops at once


def test_auto_splits_when_the_window_can_only_be_met_at_width_two():
    """Window 0.8: width 1 finishes at 1.0 and misses, width 2 at 0.7 meets.
    Price 1 misses; the op's network is the only one, so there is no staged
    step, and the first global level that splits (0.5: 0.7 + 0.5 * 0.4 = 0.9
    < 1.0) is the highest price with no miss."""
    for fn in (heft_schedule, heft_edf_schedule):
        w = _one_op(1.0, 0.7, max_end_t=0.8, price="auto")
        t, alpha = fn(w)
        assert _width(w, alpha) == 2
        res = w.core_time_price_result
        assert res["level"] == "0.5" and res["price"] == 0.5
        assert res["misses"] == 0
        assert [r["level"] for r in res["tried"]] == ["1", "0.5", "0.25", "0"]
        assert res["tried"][0]["misses"] == 1


def test_auto_lowers_the_price_only_for_the_network_that_missed():
    """Two networks, four cores, pairs [A,B] and [C,D]. `tight` needs the split
    (window 0.8); `roomy` does not (window 5). At price 1 both run at width 1
    and `tight` misses. The staged step lowers the price for `tight` alone and
    reaches zero misses, so it is preferred over the global 0.5 that would have
    split `roomy` as well."""
    def build(price):
        combos = [["A"], ["B"], ["C"], ["D"], ["A", "B"], ["C", "D"]]
        pt = [1.0, 1.0, 1.0, 1.0, 0.7, 0.7]
        tight = Operation(list(pt), job_id=0, min_start_t=0.0, max_end_t=0.8)
        roomy = Operation(list(pt), job_id=1, min_start_t=0.0, max_end_t=5.0)
        w = Workload([tight, roomy], ["A", "B", "C", "D"], np.zeros((4, 4)),
                     job_names=["tight0", "roomy0"], machine_combinations=combos)
        w.core_time_price = price
        return w

    w = build("auto")
    t, alpha = heft_edf_schedule(w)
    assert _width(w, alpha, 0) == 2 and _width(w, alpha, 1) == 1
    res = w.core_time_price_result
    assert res["level"] == "staged 0.5"
    assert res["lowered_networks"] == ["tight"] and res["lowered_price"] == 0.5
    assert res["price"] == 1.0 and res["misses"] == 0
    assert w._core_time_price_vector == [0.5, 1.0]
    # The global lowering it was preferred over splits both.
    w = build(0.5)
    t, alpha = heft_edf_schedule(w)
    assert _width(w, alpha, 0) == 2 and _width(w, alpha, 1) == 2


def test_auto_prefers_the_higher_price_at_equal_misses():
    """A window nothing can meet: every level misses once, so the top level
    (the least core time) is kept."""
    w = _one_op(1.0, 0.7, max_end_t=0.1, price="auto")
    t, alpha = heft_edf_schedule(w)
    assert w.core_time_price_result["level"] == "1"
    assert w.core_time_price_result["misses"] == 1
    assert _width(w, alpha) == 1
    assert PRICE_LEVELS == (1.0, 0.5, 0.25, 0.0)


def _reference_decode(ctx, priority):
    """The decoder as it was before the price existed (earliest finish, first
    combination wins ties), kept here as the yardstick for 'unchanged'."""
    n, n_combos = ctx.n, ctx.n_combos
    t = np.zeros(n)
    alpha = np.zeros((n, n_combos))
    chosen = np.full(n, -1, dtype=int)
    busy = [[] for _ in range(n_combos)]
    for i in _priority_order(ctx, priority):
        pred_ready = 0.0
        for p in ctx.pred[i]:
            pred_ready = max(pred_ready, t[p] + ctx.dur[p, chosen[p]])
        floor_base = max(pred_ready, ctx.min_start[i])
        best = None
        for c in range(n_combos):
            d = ctx.dur[i, c]
            if not np.isfinite(d):
                continue
            floor = floor_base
            for p in ctx.pred[i]:
                pc = chosen[p]
                floor = max(floor, t[p] + ctx.dur[p, pc]
                            + ctx.transfer[ctx.first_machine[pc], ctx.first_machine[c]])
            start = _earliest_slot(ctx, busy, c, floor, d, True)
            if best is None or start + d < best[0]:
                best = (start + d, c, start)
        _, c, start = best
        t[i], chosen[i] = start, c
        alpha[i, c] = 1.0
        d = ctx.dur[i, c]
        if d > 0:
            lst = busy[c]
            k = 0
            while k < len(lst) and lst[k][0] < start:
                k += 1
            lst.insert(k, (start, start + d))
    return t, alpha


def _random_workload(seed, transfer_ms=0.0):
    rng = np.random.default_rng(seed)
    machines = ["P#0", "P#1", "E#0", "E#1"]
    combos = [["P#0"], ["P#1"], ["P#0", "P#1"], ["E#0"], ["E#1"], ["E#0", "E#1"]]
    ops, names = [], []
    for job in range(5):
        names.append(f"net{job}_")
        periodic = job % 2 == 0
        prev = []
        for _k in range(int(rng.integers(4, 9))):
            p1, e1 = rng.uniform(0.2, 2.0), rng.uniform(0.2, 2.0)
            pt = [p1, p1, p1 * rng.uniform(0.5, 1.1), e1, e1, e1 * rng.uniform(0.5, 1.1)]
            preds = ([prev[int(rng.integers(len(prev)))]] if prev else [])
            if len(prev) > 2 and rng.random() < 0.3:
                extra = prev[int(rng.integers(len(prev)))]
                if extra not in preds:
                    preds.append(extra)
            op = Operation([float(x) for x in pt], predecessors=preds or None,
                           job_id=job,
                           min_start_t=2.0 * job if periodic else None,
                           max_end_t=2.0 * job + 12.0 if periodic else None)
            ops.append(op)
            prev.append(op)
    tr = np.full((4, 4), transfer_ms)
    np.fill_diagonal(tr, 0.0)
    return Workload(ops, machines, tr, job_names=names, machine_combinations=combos)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("transfer_ms", [0.0, 0.05])
def test_no_price_reproduces_the_old_schedule_exactly(seed, transfer_ms):
    w = _random_workload(seed, transfer_ms)
    ctx = DecoderContext(w)
    assert ctx.price is None and ctx.core_time_price is None
    for prio in (ctx.upward_rank(), np.random.default_rng(seed).random(ctx.n)):
        t_ref, a_ref = _reference_decode(ctx, prio)
        for price in (None, 0, 0.0, np.zeros(ctx.n)):
            t, alpha = decode(ctx, prio, price=price)
            assert np.array_equal(t, t_ref) and np.array_equal(alpha, a_ref)
    # HEFT is the reference decode of the upward rank, with and without an
    # explicit "no price".
    t_ref, a_ref = _reference_decode(ctx, ctx.upward_rank())
    for kw in ({}, {"core_time_price": None}, {"core_time_price": 0.0}):
        t, alpha = heft_schedule(_random_workload(seed, transfer_ms), **kw)
        assert np.array_equal(t, t_ref) and np.array_equal(alpha, a_ref)
    t0, a0 = heft_edf_schedule(_random_workload(seed, transfer_ms))
    for kw in ({"core_time_price": None}, {"core_time_price": 0.0}):
        t, alpha = heft_edf_schedule(_random_workload(seed, transfer_ms), **kw)
        assert np.array_equal(t, t0) and np.array_equal(alpha, a0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_price_never_raises_core_time_per_op_choice(seed):
    """Priced schedules are valid schedules (no overlap, precedence kept), and
    on these workloads the price does what it is for: less total core time."""
    def run(price):
        w = _random_workload(seed, 0.05)
        ctx = DecoderContext(w)
        t, alpha = decode(ctx, ctx.upward_rank(), price=price)
        combo = np.argmax(alpha, axis=1)
        core_ms = float(sum(ctx.core_ms[i, combo[i]] for i in range(ctx.n)))
        end = t + ctx.dur[np.arange(ctx.n), combo]
        for i in range(ctx.n):
            for p in ctx.pred[i]:
                tr = ctx.transfer[ctx.first_machine[combo[p]],
                                  ctx.first_machine[combo[i]]]
                assert t[i] >= end[p] + tr - 1e-9
            for j in range(i):
                if ctx.conflict[combo[i]][combo[j]]:
                    assert t[i] >= end[j] - 1e-9 or t[j] >= end[i] - 1e-9
        return core_ms
    assert run(1.0) < run(None)


def test_a_policy_string_is_refused_by_decode_itself():
    w = _one_op(1.0, 0.7)
    with pytest.raises(ValueError):
        decode(DecoderContext(w), np.zeros(1), price="auto")
    with pytest.raises(ValueError):
        heft_schedule(_one_op(1.0, 0.7), core_time_price=-1.0)


def test_solve_with_core_time_price_ranks_levels_by_misses_then_order():
    """The level ranking on its own, with a scripted solver: misses per level
    decide, and among equal misses the earlier entry of
    1, staged 0.5, staged 0.25, staged 0, 0.5, 0.25, 0 wins."""
    combos = [["A"], ["B"], ["A", "B"]]
    a = Operation([1.0, 1.0, 0.7], job_id=0, min_start_t=0.0, max_end_t=0.8)
    b = Operation([1.0, 1.0, 0.7], job_id=1, min_start_t=0.0, max_end_t=9.0)
    w = Workload([a, b], ["A", "B"], np.zeros((2, 2)), job_names=["a0", "b0"],
                 machine_combinations=combos)
    ctx = DecoderContext(w)
    seen = []

    def solve(price):
        seen.append(tuple(np.asarray(price, dtype=float)))
        return decode(ctx, np.array([2.0, 1.0]), price=price)

    t, alpha, info = solve_with_core_time_price(ctx, "auto", solve)
    assert seen == [(1.0, 1.0), (0.5, 1.0), (0.25, 1.0), (0.0, 1.0),
                    (0.5, 0.5), (0.25, 0.25), (0.0, 0.0)]
    assert [r["level"] for r in info["tried"]] == [
        "1", "staged 0.5", "staged 0.25", "staged 0", "0.5", "0.25", "0"]
    assert info["level"] == "staged 0.5" and info["misses"] == 0
    assert evaluate(ctx, t, alpha)[1] == 0


# --------------------------------------------------------------------------
# 2. Codegen contract: a price does not undo the width pin
# --------------------------------------------------------------------------

def test_price_does_not_undo_the_contract_width_pin():
    """Two instances of one convolution dispatch. Unpinned, earliest finish
    gives the first instance the pair and the second (whose pair is taken by a
    blocker) a single core: two widths for one dispatch, which the compiler
    cannot build. On this tree the contract is applied to the workload itself
    (codegen_contract.apply_to_workload, called from run_xpurt_schedule's
    _build_workload for every solver), so the decoder sees the pinned widths;
    a core-time price must not undo that pin."""
    import codegen_contract
    contract = {"rules": {"uniform_width_across_instances": {
        "severity": "refuse", "applies_to_ops": ["conv2d_s8"]}}}

    def build():
        combos = [["A"], ["B"], ["A", "B"]]
        pt = [1.0, 1.0, 0.9]
        conv0 = Operation(list(pt), job_id=0, operation_id=7,
                          operation_name="net0_dispatch_7")
        # Released at 2.0; B is taken by a long blocker from 1.9, so the
        # pair is not free and a single core finishes sooner.
        conv1 = Operation(list(pt), job_id=1, operation_id=7,
                          operation_name="net1_dispatch_7", min_start_t=2.0)
        blocker = Operation([9.0, 3.0, 9.0], job_id=2, operation_id=0,
                            operation_name="other_dispatch_0", min_start_t=1.9)
        return Workload([conv0, conv1, blocker], ["A", "B"], np.zeros((2, 2)),
                        job_names=["net0", "net1", "other0"],
                        machine_combinations=combos)

    def module_name_of(op):
        if str(op.operation_name).startswith("net"):
            return "net$dispatch_7_ime_k3_a100_conv2d_s8_OC16", "net"
        return None, None

    def widths(w, price=None):
        w.core_time_price = price
        t, alpha = heft_edf_schedule(w)
        return [_width(w, alpha, 0), _width(w, alpha, 1)]

    w = build()
    t, alpha = decode(DecoderContext(w), np.array([3.0, 1.0, 2.0]))
    assert [_width(w, alpha, 0), _width(w, alpha, 1)] == [2, 1]
    for price in (None, 1.0, "auto"):
        w = build()
        st = codegen_contract.apply_to_workload(w, module_name_of, contract)
        assert st["pinned_dispatches"] == 1
        got = widths(w, price)
        assert got[0] == got[1], (price, got)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
