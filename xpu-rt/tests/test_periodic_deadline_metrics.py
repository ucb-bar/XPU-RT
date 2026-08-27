"""Periodic deadline misses must be counted.

THE BUG: compute_metrics counted a miss only when `op.deadline_us` was set. But
workload_factory expands `period` / `window_duration` into `min_start_t` /
`max_end_t` and never touches `deadline_us` -- that field is only populated by
create_workload_from_dependencies, a different entry point. So for every
periodic workload, which is the entire point of this scheduler, the deadline
metric was structurally incapable of being non-zero.

It surfaced on the first real K1 schedule. All ten dronet instances overran
their 33.3 ms window by ~80 ms each -- DroNet measures 113 ms of work per
instance on one core -- and the metrics file reported deadline_miss_count = 0.
The schedule was not wrong; the number describing it was, which is the more
dangerous of the two. Anyone reading the metrics would have concluded the
workload was feasible.

The fix takes whichever of deadline_us / max_end_t is tighter, so both entry
points are covered and neither overrides the other.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _REPO)
sys.path.insert(0, _XPURT)

from metrics import compute_metrics  # noqa: E402
from workload import Operation, Workload  # noqa: E402

M = ["CPU_P#0"]
COMBOS = [["CPU_P#0"]]


def _wl(ops):
    return Workload(operations=ops, machines=M,
                    transfer_times=np.zeros((1, 1)),
                    machine_combinations=COMBOS)


def _metrics(ops, starts):
    wl = _wl(ops)
    t = np.array(starts, dtype=float)
    alpha = np.ones((len(ops), 1))
    return compute_metrics(wl, t, alpha, scheduler_name="test")


class PeriodicDeadlineTests(unittest.TestCase):
    def test_periodic_overrun_is_counted(self):
        """The regression: max_end_t alone must be enough to register a miss."""
        op = Operation([10.0], min_start_t=0.0, max_end_t=5.0)
        m = _metrics([op], [0.0])
        self.assertEqual(m["deadline_miss_count"], 1)
        self.assertAlmostEqual(m["total_lateness_us"], 5.0)
        self.assertAlmostEqual(m["max_lateness_us"], 5.0)

    def test_periodic_inside_its_window_is_not_counted(self):
        op = Operation([2.0], min_start_t=0.0, max_end_t=5.0)
        m = _metrics([op], [0.0])
        self.assertEqual(m["deadline_miss_count"], 0)

    def test_exactly_meeting_the_deadline_is_not_a_miss(self):
        """Float reconstruction of t+duration must not turn an exact hit into a miss."""
        op = Operation([5.0], min_start_t=0.0, max_end_t=5.0)
        m = _metrics([op], [0.0])
        self.assertEqual(m["deadline_miss_count"], 0)

    def test_deadline_us_still_works(self):
        """The pre-existing path must not regress."""
        op = Operation([10.0], deadline_us=4.0)
        m = _metrics([op], [0.0])
        self.assertEqual(m["deadline_miss_count"], 1)
        self.assertAlmostEqual(m["total_lateness_us"], 6.0)

    def test_the_tighter_of_the_two_bounds_wins(self):
        op = Operation([10.0], max_end_t=9.0, deadline_us=4.0)
        m = _metrics([op], [0.0])
        self.assertEqual(m["deadline_miss_count"], 1)
        self.assertAlmostEqual(m["total_lateness_us"], 6.0, places=6)

        op2 = Operation([10.0], max_end_t=4.0, deadline_us=9.0)
        m2 = _metrics([op2], [0.0])
        self.assertAlmostEqual(m2["total_lateness_us"], 6.0, places=6)

    def test_an_op_with_no_bound_is_never_a_miss(self):
        op = Operation([1000.0])
        m = _metrics([op], [0.0])
        self.assertEqual(m["deadline_miss_count"], 0)

    def test_counts_scale_with_instances(self):
        """A ten-instance periodic job that always overruns must report ten."""
        ops = [Operation([10.0], min_start_t=float(i) * 5.0,
                         max_end_t=float(i) * 5.0 + 5.0) for i in range(10)]
        m = _metrics(ops, [float(i) * 5.0 for i in range(10)])
        self.assertEqual(m["deadline_miss_count"], 10)
        self.assertAlmostEqual(m["deadline_miss_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
