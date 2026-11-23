"""Hand-off (thread wake-up) cost table for the planner.

On the board a layer whose input was produced on another core's worker thread
starts later than "input done": that worker has to be woken. The planner
already has a per-core-pair `transfer_times` table (all zeros so far). The
`scheduler.handoff_ms` config key fills it by core kind:

    "handoff_ms": {"cpu_p": {"cpu_p": 0.0115, "cpu_e": 0.0153},
                   "cpu_e": {"cpu_p": 0.0137, "cpu_e": 0.0153}}

(from kind -> to kind -> ms, charged between two DIFFERENT cores; the same
core always costs 0). Absent key = zeros = today's behaviour.

Run: python3 -m unittest xpu-rt/tests/test_handoff_table.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np

_XPU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_XPU)
sys.path.insert(0, _XPU)

from greedy_scheduler import decomposed_schedule  # noqa: E402
from workload import Operation, Workload  # noqa: E402
from workload_factory import build_handoff_transfer_times  # noqa: E402

MACHINES = ["CPU_P#0", "CPU_P#1", "CPU_E#0", "CPU_E#1"]
TABLE = {"cpu_p": {"cpu_p": 0.010, "cpu_e": 0.015},
         "cpu_e": {"cpu_p": 0.014, "cpu_e": 0.016}}


class BuildTable(unittest.TestCase):
    def test_absent_table_is_all_zeros(self):
        for empty in (None, {}):
            tt = build_handoff_transfer_times(MACHINES, empty)
            self.assertEqual(tt.shape, (4, 4))
            self.assertFalse(tt.any())

    def test_values_follow_from_kind_to_kind(self):
        tt = build_handoff_transfer_times(MACHINES, TABLE)
        self.assertEqual(tt[0, 1], 0.010)   # X100 -> another X100
        self.assertEqual(tt[0, 2], 0.015)   # X100 -> A100
        self.assertEqual(tt[3, 1], 0.014)   # A100 -> X100
        self.assertEqual(tt[2, 3], 0.016)   # A100 -> another A100

    def test_same_core_is_free(self):
        tt = build_handoff_transfer_times(MACHINES, TABLE)
        self.assertFalse(np.diag(tt).any())

    def test_kind_names_are_case_insensitive(self):
        upper = {k.upper(): {k2.upper(): v for k2, v in row.items()} for k, row in TABLE.items()}
        np.testing.assert_array_equal(build_handoff_transfer_times(MACHINES, upper),
                                      build_handoff_transfer_times(MACHINES, TABLE))

    def test_missing_pair_is_an_error(self):
        broken = {"cpu_p": {"cpu_p": 0.010, "cpu_e": 0.015}, "cpu_e": {"cpu_p": 0.014}}
        with self.assertRaisesRegex(ValueError, "cpu_e.*cpu_e"):
            build_handoff_transfer_times(MACHINES, broken)

    def test_negative_or_non_number_is_an_error(self):
        for bad in (-0.001, "fast", None, float("nan")):
            table = {"cpu_p": {"cpu_p": bad, "cpu_e": 0.015}, "cpu_e": {"cpu_p": 0.014, "cpu_e": 0.016}}
            with self.assertRaises(ValueError):
                build_handoff_transfer_times(MACHINES, table)

    def test_wrong_shape_is_an_error(self):
        for bad in ([0.01, 0.02], 0.01, {"cpu_p": 0.01, "cpu_e": 0.02}):
            with self.assertRaises(ValueError):
                build_handoff_transfer_times(MACHINES, bad)

    def test_a_kind_with_one_core_needs_no_same_kind_value(self):
        tt = build_handoff_transfer_times(["CPU_P#0", "CPU_E#0"],
                                          {"cpu_p": {"cpu_e": 0.015}, "cpu_e": {"cpu_p": 0.014}})
        self.assertEqual(tt.tolist(), [[0.0, 0.015], [0.014, 0.0]])


def _chain(durations_a, durations_b, transfer):
    """Two layers of one periodic job, B reads A's output; one option per core."""
    machines = ["CPU_P#0", "CPU_E#0"]
    a = Operation(list(durations_a), operation_id=0, operation_name="a", job_id=0,
                  min_start_t=0.0, max_end_t=100.0)
    b = Operation(list(durations_b), predecessors=[a], operation_id=1, operation_name="b", job_id=0,
                  min_start_t=0.0, max_end_t=100.0)
    wl = Workload([a, b], machines, np.array(transfer, dtype=float), job_names=["net"],
                  machine_combinations=[[m] for m in machines])
    t, alpha = decomposed_schedule(wl)
    return t, alpha.argmax(axis=1)


class KnownAnswer(unittest.TestCase):
    HAND_OFF = [[0.0, 0.5], [0.5, 0.0]]
    NONE = [[0.0, 0.0], [0.0, 0.0]]

    def test_layer_on_another_core_starts_one_table_value_later(self):
        t0, c0 = _chain([1.0, 100.0], [100.0, 1.0], self.NONE)
        t1, c1 = _chain([1.0, 100.0], [100.0, 1.0], self.HAND_OFF)
        self.assertEqual(c0.tolist(), [0, 1])
        self.assertEqual(c1.tolist(), [0, 1])
        self.assertAlmostEqual(t0[1], 1.0)
        self.assertAlmostEqual(t1[1], 1.5)

    def test_layer_on_the_same_core_does_not_shift(self):
        t0, c0 = _chain([1.0, 100.0], [1.0, 100.0], self.NONE)
        t1, c1 = _chain([1.0, 100.0], [1.0, 100.0], self.HAND_OFF)
        self.assertEqual(c1.tolist(), [0, 0])
        self.assertEqual(t1.tolist(), t0.tolist())
        self.assertAlmostEqual(t1[1], 1.0)

    def test_planner_keeps_a_layer_on_its_core_when_the_hand_off_costs_more_than_it_saves(self):
        # B is 0.2 ms faster on the other core, but the hand-off costs 0.5 ms.
        _, without = _chain([1.0, 100.0], [1.2, 1.0], self.NONE)
        _, with_cost = _chain([1.0, 100.0], [1.2, 1.0], self.HAND_OFF)
        self.assertEqual(without.tolist(), [0, 1])
        self.assertEqual(with_cost.tolist(), [0, 0])


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "run_xpurt_schedule_handoff_test", os.path.join(_REPO, "scripts", "run_xpurt_schedule.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DriverConfig(unittest.TestCase):
    def _cfg(self, scheduler: dict) -> dict:
        spec = {"hardware": {"machines": {"cpu_p": 2, "cpu_e": 2}}, "scheduler": scheduler, "networks": {}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(spec, f)
        self.addCleanup(os.unlink, f.name)
        return _load_driver().load_networks_config(f.name)[1]

    def test_key_absent_means_no_table(self):
        self.assertIsNone(self._cfg({})["handoff_ms"])

    def test_key_is_passed_through(self):
        self.assertEqual(self._cfg({"handoff_ms": TABLE})["handoff_ms"], TABLE)


if __name__ == "__main__":
    unittest.main()
