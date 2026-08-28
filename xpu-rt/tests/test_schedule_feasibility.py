"""Unit tests for scripts/check_schedule_feasibility.py.

The check exists for schedules assembled OUTSIDE a solver's constraint set --
`mosek_decompose_by_network.py`'s sequential stitch,
`packing.combine_solved_windows`, anything hand-merged or hot-swapped. A
solver's own output satisfies no-overlap by construction and this is a
tautology on it.

What makes it worth having: the walker runs one worker per (core_kind, hart)
and does not preempt, so a double-booked core does not fail on the board. It
SERIALISES, the run comes out slower than predicted, and in a results table
that is indistinguishable from contention or an optimistic profile.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


chk = _load(os.path.join(_REPO, "scripts", "check_schedule_feasibility.py"),
            "_feasibility")

TOL = 1e-6


def _d(start, dur, target="CPU_P#0", deps=()):
    return {"start_time": start, "duration": dur, "hardware_target": target,
            "dependencies": list(deps)}


class DoubleBookingIsFoundInFull(unittest.TestCase):

    def test_a_clean_schedule_reports_nothing(self):
        ivs = chk.intervals_by_machine({
            "a": _d(0.0, 1.0), "b": _d(1.0, 1.0), "c": _d(2.0, 1.0)})
        self.assertEqual(chk.find_overlaps(ivs, TOL), [])

    def test_back_to_back_is_not_an_overlap(self):
        """An op ending exactly when the next starts is the normal case."""
        ivs = chk.intervals_by_machine({"a": _d(0.0, 1.0), "b": _d(1.0, 1.0)})
        self.assertEqual(chk.find_overlaps(ivs, TOL), [])

    def test_a_long_dispatch_spanning_several_short_ones_reports_all(self):
        """The bug an adjacent-pair scan has, and the reason this sweeps.

        With A=[0,100], B=[10,20], C=[30,40] the neighbour scan finds A-B, then
        compares B against C -- which do not overlap -- and reports ONE
        conflict where there are two. Under-reporting a double-booking is the
        wrong direction: the serialisation cost scales with what was hidden.
        """
        ivs = chk.intervals_by_machine({
            "A": _d(0.0, 100.0), "B": _d(10.0, 10.0), "C": _d(30.0, 10.0)})
        pairs = {(r["a"], r["b"]) for r in chk.find_overlaps(ivs, TOL)}
        self.assertEqual(pairs, {("A", "B"), ("A", "C")})

    def test_different_machines_never_conflict(self):
        ivs = chk.intervals_by_machine({
            "a": _d(0.0, 10.0, "CPU_P#0"), "b": _d(0.0, 10.0, "CPU_E#0")})
        self.assertEqual(chk.find_overlaps(ivs, TOL), [])

    def test_a_sharded_dispatch_occupies_every_machine_it_holds(self):
        """It is one dispatch precisely because it holds them all at once, so
        anything else on either core in that window is a conflict."""
        ivs = chk.intervals_by_machine({
            "shard": _d(0.0, 10.0, "CPU_P#0+CPU_P#1"),
            "other": _d(5.0, 1.0, "CPU_P#1")})
        self.assertEqual(sorted(ivs), ["CPU_P#0", "CPU_P#1"])
        bad = chk.find_overlaps(ivs, TOL)
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["machine"], "CPU_P#1")


class TiesAreNotForwardEdges(unittest.TestCase):
    """The false positive this check shipped with for one iteration.

    A zero-duration op (`view`, `chunk2_c1`) finishes at the instant it starts,
    so its successor starts at exactly the same timestamp. Ordering by
    `(start_time, key)` then puts `..._dispatch_10` before `..._dispatch_9`,
    because "1" sorts before "9", and a perfectly good schedule is called
    infeasible. Measured on
    `scheduled__iter_baseline_decomposed_profiled.json`, where both start at
    14.66098 ms and dispatch 9 has duration 0.0.
    """

    def test_a_zero_duration_predecessor_at_the_same_instant_is_fine(self):
        d = {"n_dispatch_9": _d(14.66098, 0.0, deps=["n_dispatch_8"]),
             "n_dispatch_10": _d(14.66098, 0.4, deps=["n_dispatch_9"]),
             "n_dispatch_8": _d(14.0, 0.66098)}
        self.assertEqual(chk.find_forward_edges(d, TOL), [])
        self.assertEqual(chk.find_dependency_violations(d, TOL), [])

    def test_a_strictly_later_dependency_is_still_caught(self):
        d = {"a": _d(0.0, 1.0, deps=["b"]), "b": _d(5.0, 1.0)}
        bad = chk.find_forward_edges(d, TOL)
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["delta_ms"], 5.0)

    def test_a_same_instant_dependency_that_has_not_finished_is_caught(self):
        """Not by the forward-edge rule -- by the dependency rule, which is
        where it belongs."""
        d = {"a": _d(1.0, 1.0, deps=["b"]), "b": _d(1.0, 3.0)}
        self.assertEqual(chk.find_forward_edges(d, TOL), [])
        self.assertEqual(len(chk.find_dependency_violations(d, TOL)), 1)


class TargetsMustExistOnTheBoard(unittest.TestCase):

    def test_a_core_index_past_the_cluster_is_refused(self):
        d = {"a": _d(0.0, 1.0, "CPU_P#7")}
        self.assertEqual(len(chk.find_out_of_range_targets(d, 4)), 1)

    def test_every_hart_of_the_cluster_is_allowed(self):
        d = {f"d{i}": _d(float(i), 1.0, f"CPU_P#{i}") for i in range(4)}
        self.assertEqual(chk.find_out_of_range_targets(d, 4), [])


if __name__ == "__main__":
    unittest.main()
