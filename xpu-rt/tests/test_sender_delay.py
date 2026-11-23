"""Sender's delay: the core that hands work off starts its own next layer late.

On the board a worker thread that finishes a layer some OTHER thread is asleep waiting for has to
wake that thread before it can start its own next layer (about 8 us on the K3). The hand-off table
(`scheduler.handoff_ms`) charges the receiver's wait; `scheduler.sender_ms` ({core kind: ms})
charges this cost on the sender: `sender_delay.apply_sender_delay` re-times a finished schedule so
that the sender's next layer on the same core, and whatever depends on it, start that much later.
It only ever moves starts later, keeps every placement, and is not called when the key is absent.

Run: python3 -m unittest xpu-rt/tests/test_sender_delay.py
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sender_delay import apply_sender_delay  # noqa: E402
from workload import Operation, Workload  # noqa: E402

MACHINES = ["CPU_P#0", "CPU_P#1", "CPU_E#0"]
COMBOS = [["CPU_P#0"], ["CPU_P#1"], ["CPU_E#0"], ["CPU_P#1", "CPU_P#0"]]
HAND_OFF = 0.5
SEND = {"cpu_p": 0.2, "cpu_e": 0.3}


def transfer():
    tt = np.full((3, 3), HAND_OFF)
    np.fill_diagonal(tt, 0.0)
    return tt


def schedule(spec):
    """spec: list of (name, combo index, start, duration, [predecessor names]) -> (workload, t, alpha)."""
    ops, by_name = [], {}
    for k, (name, combo, _start, dur, preds) in enumerate(spec):
        times = [1.0e6] * len(COMBOS)
        times[combo] = dur
        op = Operation(times, predecessors=[by_name[p] for p in preds], operation_id=k, operation_name=name, job_id=0)
        ops.append(op)
        by_name[name] = op
    wl = Workload(ops, MACHINES, transfer(), job_names=["net"], machine_combinations=COMBOS)
    t = np.array([s[2] for s in spec], dtype=float)
    alpha = np.zeros((len(spec), len(COMBOS)))
    for k, s in enumerate(spec):
        alpha[k, s[1]] = 1.0
    return wl, t, alpha


def starts(spec, sender_ms):
    wl, t, alpha = schedule(spec)
    return apply_sender_delay(wl, t, alpha, sender_ms).tolist()


class SenderDelay(unittest.TestCase):
    #: a finishes on core 0; b (other core) waits for it; c is core 0's own next layer
    FORK = [("a", 0, 0.0, 1.0, []),
            ("b", 1, 1.5, 1.0, ["a"]),
            ("c", 0, 1.0, 1.0, ["a"])]

    def test_next_layer_on_the_senders_core_starts_late_and_the_receiver_does_not(self):
        self.assertEqual(starts(self.FORK, SEND), [0.0, 1.5, 1.2])

    def test_no_table_means_no_change(self):
        for empty in (None, {}):
            self.assertEqual(starts(self.FORK, empty), [0.0, 1.5, 1.0])

    def test_input_is_not_modified(self):
        wl, t, alpha = schedule(self.FORK)
        before = t.copy()
        apply_sender_delay(wl, t, alpha, SEND)
        np.testing.assert_array_equal(t, before)

    def test_no_hand_off_no_delay(self):
        chain = [("a", 0, 0.0, 1.0, []), ("c", 0, 1.0, 1.0, ["a"])]
        self.assertEqual(starts(chain, SEND), [0.0, 1.0])

    def test_receiver_that_is_not_waiting_for_the_sender_costs_nothing(self):
        # b starts long after a's result arrived (it was busy or waiting for something else),
        # so nobody is asleep on a when a finishes: a wakes no one.
        spec = [("a", 0, 0.0, 1.0, []), ("b", 1, 3.0, 1.0, ["a"]), ("c", 0, 1.0, 1.0, ["a"])]
        self.assertEqual(starts(spec, SEND), [0.0, 3.0, 1.0])

    def test_planned_idle_time_absorbs_the_delay(self):
        spec = [("a", 0, 0.0, 1.0, []), ("b", 1, 1.5, 1.0, ["a"]), ("c", 0, 1.5, 1.0, ["a"])]
        self.assertEqual(starts(spec, SEND), [0.0, 1.5, 1.5])

    def test_delay_ripples_to_what_follows(self):
        spec = self.FORK + [("d", 0, 2.0, 1.0, ["c"]),      # same core, right after c
                            ("e", 2, 2.5, 1.0, ["c"])]      # another core, waits for c
        got = starts(spec, SEND)
        self.assertAlmostEqual(got[2], 1.2)                 # c
        # c now wakes e, so d (core 0's next layer) pays c's sender delay too: 2.2 + 0.2
        self.assertAlmostEqual(got[3], 2.4)
        self.assertAlmostEqual(got[4], 2.2 + HAND_OFF)      # e: c's new end + hand-off

    def test_waking_two_threads_costs_one_delay(self):
        spec = self.FORK + [("b2", 2, 1.5, 1.0, ["a"])]
        self.assertAlmostEqual(starts(spec, SEND)[2], 1.2)

    def test_value_follows_the_senders_core_kind(self):
        spec = [("a", 2, 0.0, 1.0, []), ("b", 0, 1.5, 1.0, ["a"]), ("c", 2, 1.0, 1.0, ["a"])]
        self.assertAlmostEqual(starts(spec, SEND)[2], 1.3)

    def test_only_the_senders_own_thread_is_delayed_not_a_layer_led_by_another_core(self):
        # x is led by core 1 and uses core 0 as a helper right when a ends: core 0 is unlocked
        # before the wake-up is sent, so x is not held up.
        spec = [("a", 0, 0.0, 1.0, []), ("b", 2, 1.5, 1.0, ["a"]), ("x", 3, 1.0, 1.0, [])]
        self.assertEqual(starts(spec, SEND), [0.0, 1.5, 1.0])

    def test_zero_cost_layer_is_never_moved_by_a_busy_core(self):
        # the planner may place a zero-cost layer while its core is busy; that must stay as planned
        spec = [("a", 0, 0.0, 2.0, []), ("z", 0, 1.0, 0.0, [])]
        self.assertEqual(starts(spec, SEND), [0.0, 1.0])

    def test_schedule_without_senders_is_returned_unchanged_even_with_a_table(self):
        spec = [("a", 0, 0.0, 1.0, []), ("b", 0, 1.0, 1.0, ["a"]), ("x", 1, 0.0, 3.0, []), ("y", 2, 0.25, 0.5, [])]
        self.assertEqual(starts(spec, SEND), [0.0, 1.0, 0.0, 0.25])

    def test_bad_table_is_an_error(self):
        for bad in ({"cpu_p": -0.1, "cpu_e": 0.3}, {"cpu_p": "slow", "cpu_e": 0.3}, {"cpu_p": 0.2}, [0.2, 0.3], 0.2):
            with self.assertRaises(ValueError):
                starts(self.FORK, bad)


class DriverConfig(unittest.TestCase):
    def _cfg(self, scheduler: dict) -> dict:
        import importlib.util
        import json
        import tempfile
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        spec = importlib.util.spec_from_file_location(
            "run_xpurt_schedule_sender_test", os.path.join(repo, "scripts", "run_xpurt_schedule.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"hardware": {"machines": {"cpu_p": 2, "cpu_e": 1}}, "scheduler": scheduler, "networks": {}}, f)
        self.addCleanup(os.unlink, f.name)
        return mod.load_networks_config(f.name)[1]

    def test_key_absent_means_not_applied(self):
        self.assertIsNone(self._cfg({})["sender_ms"])

    def test_key_is_passed_through(self):
        self.assertEqual(self._cfg({"sender_ms": SEND})["sender_ms"], SEND)


if __name__ == "__main__":
    unittest.main()
