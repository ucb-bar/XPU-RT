"""Tests for the deadline-aware scheduler advisor (xpu-rt/advisor.py)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workload import Operation, Workload  # noqa: E402
from profiling import SchedulerReport  # noqa: E402
from advisor import advise_schedule  # noqa: E402

MACHINES = ["FAST", "SLOW"]   # combos default to singletons: idx 0=FAST, 1=SLOW
INFEAS_SLOW = {1}


def _report(ops, t, alpha, **kw):
    wl = Workload(ops, MACHINES, np.zeros((2, 2), dtype=float))
    return SchedulerReport.from_solver_state(
        wl, np.array(t, dtype=float), np.array(alpha, dtype=float),
        solver_name="test", solve_wall_s=0.0, **kw)


def _independent_ops(deadline):
    # 4 independent ops, all cheap on FAST; op2/op3 are infeasible on SLOW.
    return [
        Operation(processing_times=[10.0, 30.0], operation_name="a0", deadline_us=deadline),
        Operation(processing_times=[10.0, 30.0], operation_name="a1", deadline_us=deadline),
        Operation(processing_times=[10.0, 1e9], operation_name="b2",
                  infeasible_combinations=INFEAS_SLOW, deadline_us=deadline),
        Operation(processing_times=[10.0, 1e9], operation_name="b3",
                  infeasible_combinations=INFEAS_SLOW, deadline_us=deadline),
    ]


# all 4 placed serially on FAST (the bottleneck); SLOW idle
SERIAL_T = [0.0, 10.0, 20.0, 30.0]
ALL_ON_FAST = [[1.0, 0.0]] * 4


class AdvisorTests(unittest.TestCase):

    def test_rebalance_fires_and_respects_feasibility(self):
        rep = _report(_independent_ops(deadline=30.0), SERIAL_T, ALL_ON_FAST)
        diag = advise_schedule(rep)
        self.assertFalse(diag.meets_deadline)            # makespan 40 > deadline 30
        self.assertEqual(diag.bottleneck_backend, "FAST")
        self.assertIn("SLOW", diag.idle_backends)
        rebs = [r for r in diag.recommendations if r.kind == "rebalance"]
        self.assertEqual(len(rebs), 1)
        moved = set(rebs[0].detail["op_ids"])
        # op0/op1 are feasible on SLOW; op2/op3 are NOT and must be excluded.
        self.assertTrue(moved.issubset({0, 1}))
        self.assertNotIn(2, moved)
        self.assertNotIn(3, moved)
        # saving = sum(durs) - max(durs) for the moved antichain (10+10 - 10).
        self.assertAlmostEqual(rebs[0].expected_savings_us, 10.0, places=3)
        # projection reduces makespan but never beats the critical path.
        self.assertLess(diag.projected_makespan_us, diag.makespan_us)
        self.assertGreaterEqual(diag.projected_makespan_us, rep.critical_path - 1e-6)

    def test_no_rebalance_when_idle_backend_is_infeasible(self):
        # every op infeasible on SLOW => nothing can move => no rebalance rec.
        ops = [
            Operation(processing_times=[10.0, 1e9], operation_name=f"x{i}",
                      infeasible_combinations=INFEAS_SLOW, deadline_us=30.0)
            for i in range(4)
        ]
        diag = advise_schedule(_report(ops, SERIAL_T, ALL_ON_FAST))
        self.assertEqual([r for r in diag.recommendations if r.kind == "rebalance"], [])

    def test_chain_has_no_parallel_antichain(self):
        # op0->op1->op2->op3 chain: each at a distinct depth => max antichain 1.
        a = Operation(processing_times=[10.0, 30.0], operation_name="c0", deadline_us=100.0)
        b = Operation(processing_times=[10.0, 30.0], operation_name="c1", predecessors=[a], deadline_us=100.0)
        c = Operation(processing_times=[10.0, 30.0], operation_name="c2", predecessors=[b], deadline_us=100.0)
        dd = Operation(processing_times=[10.0, 30.0], operation_name="c3", predecessors=[c], deadline_us=100.0)
        diag = advise_schedule(_report([a, b, c, dd], SERIAL_T, ALL_ON_FAST))
        self.assertEqual([r for r in diag.recommendations if r.kind == "rebalance"], [])

    def test_no_deadline_mode(self):
        rep = _report(_independent_ops(deadline=None), SERIAL_T, ALL_ON_FAST)
        diag = advise_schedule(rep)              # no deadline on ops, none passed
        self.assertIsNone(diag.meets_deadline)
        self.assertIsNone(diag.deadline_us)
        # rebalance is deadline-independent, so it should still surface.
        self.assertTrue(any(r.kind == "rebalance" for r in diag.recommendations))

    def test_explicit_deadline_overrides(self):
        rep = _report(_independent_ops(deadline=None), SERIAL_T, ALL_ON_FAST)
        diag = advise_schedule(rep, deadline_us=100.0)
        self.assertTrue(diag.meets_deadline)     # makespan 40 <= 100
        self.assertEqual(diag.deadline_us, 100.0)

    def test_terminal_gantt_renders(self):
        import plot_gantt
        rep = _report(_independent_ops(deadline=25.0), SERIAL_T, ALL_ON_FAST)
        txt = plot_gantt.render_terminal_gantt(rep.to_dict(), deadline_us=25.0, width=40)
        self.assertIn("FAST", txt)
        self.assertIn("SLOW", txt)
        self.assertIn("deadline", txt.lower())
        self.assertIn("x", txt)  # some dispatch finishes after the 25us deadline

    def test_terminal_gantt_degrades_without_dispatches(self):
        import plot_gantt
        txt = plot_gantt.render_terminal_gantt({"solver_name": "x"}, deadline_us=10.0)
        self.assertIn("schema>=2", txt)

    def test_never_recommends_infeasible_target(self):
        rep = _report(_independent_ops(deadline=30.0), SERIAL_T, ALL_ON_FAST)
        diag = advise_schedule(rep)
        d = rep.to_dict()
        feas = {disp["id"]: set(disp["feasible_targets"]) for disp in d["dispatches"]}
        for r in diag.recommendations:
            if r.kind == "rebalance":
                for oid in r.detail["op_ids"]:
                    self.assertTrue(set(r.detail["to_candidates"]).issubset(feas[oid]))


if __name__ == "__main__":
    unittest.main()
