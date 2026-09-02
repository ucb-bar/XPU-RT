"""Regression tests for repeatable steady-state frame qualification."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import repeat_window  # noqa: E402


def _dispatch(job, start, duration=1.0, deps=()):
    return {"job_name": job, "start_time": start, "duration": duration,
            "dependencies": list(deps), "hardware_target": "CPU_P#0"}


class RepeatWindowTests(unittest.TestCase):

    def test_trailing_periodic_instances_are_excluded_after_anchor_frame(self):
        schedule = {"dispatches": {
            "a": _dispatch("large0", 0, 25),
            "d0": _dispatch("dronet0", 0),
            "d1": _dispatch("dronet1", 10),
            "d2": _dispatch("dronet2", 20),
            "d3": _dispatch("dronet3", 30),
            "d4": _dispatch("dronet4", 40),
        }}
        report = repeat_window.find(
            schedule, {"large": 100, "dronet": 10}, "large",
            {"large", "dronet"}, quantum_ms=10, max_window_ms=50)
        self.assertEqual(report["window_ms"], 30)
        self.assertEqual(report["dispatches_excluded"], 2)
        self.assertEqual(report["models"]["dronet"]["complete_instances"], 3)

    def test_crossing_job_pushes_frame_to_next_quantum(self):
        schedule = {"dispatches": {
            "a": _dispatch("large0", 0, 25),
            "d0": _dispatch("dronet0", 0),
            "d1": _dispatch("dronet1", 10),
            "d2": _dispatch("dronet2", 20, 11),
            "d3": _dispatch("dronet3", 30),
        }}
        report = repeat_window.find(
            schedule, {"large": 100, "dronet": 10}, "large",
            {"large", "dronet"}, quantum_ms=10, max_window_ms=40)
        self.assertEqual(report["window_ms"], 40)
        self.assertTrue(report["boundary_clear"])

    def test_insufficient_instances_refuses_repeat_claim(self):
        schedule = {"dispatches": {
            "a": _dispatch("large0", 0, 25),
            "d0": _dispatch("dronet0", 0),
            "d1": _dispatch("dronet1", 10),
        }}
        with self.assertRaisesRegex(ValueError, "no repeatable"):
            repeat_window.find(
                schedule, {"large": 100, "dronet": 10}, "large",
                {"large", "dronet"}, quantum_ms=10, max_window_ms=40)

    def test_extract_frame_materializes_only_the_qualified_prefix(self):
        schedule = {"dispatches": {
            "a": _dispatch("large0", 0, 25),
            "d0": _dispatch("dronet0", 0),
            "d1": _dispatch("dronet1", 10),
            "d2": _dispatch("dronet2", 20),
            "d3": _dispatch("dronet3", 30),
        }, "metadata": {"makespan": 31, "num_operations": 5}}
        report = repeat_window.find(
            schedule, {"large": 100, "dronet": 10}, "large",
            {"large", "dronet"}, quantum_ms=10, max_window_ms=40)
        frame = repeat_window.extract_frame(schedule, report)
        self.assertEqual(set(frame["dispatches"]), {"a", "d0", "d1", "d2"})
        self.assertEqual(frame["metadata"]["makespan"], 30)
        self.assertEqual(frame["metadata"]["num_operations"], 4)
        self.assertEqual(frame["repeat_frame"]["mode"], "repeat_indefinitely")
        self.assertEqual(len(frame["repeat_frame"]["source_schedule_sha256"]), 64)

    def test_common_frame_uses_one_boundary_for_every_schedule(self):
        fast = {"dispatches": {
            "a": _dispatch("large0", 0, 25),
            "d0": _dispatch("dronet0", 0),
            "d1": _dispatch("dronet1", 10),
            "d2": _dispatch("dronet2", 20),
            "d3": _dispatch("dronet3", 30),
        }}
        slow = {"dispatches": {
            "a": _dispatch("large0", 0, 35),
            "d0": _dispatch("dronet0", 0),
            "d1": _dispatch("dronet1", 10),
            "d2": _dispatch("dronet2", 20),
            "d3": _dispatch("dronet3", 30),
        }}
        reports = repeat_window.find_common(
            [fast, slow], {"large": 100, "dronet": 10}, "large",
            {"large", "dronet"}, quantum_ms=10, max_window_ms=40)
        self.assertEqual([r["window_ms"] for r in reports], [40, 40])
        self.assertTrue(all(r["status"] == "pass" for r in reports))


if __name__ == "__main__":
    unittest.main()
