"""A network name may end in a digit, and the scorer must survive it.

`<network><instance>` has no separator. While no network name ends in a digit
the split is trivial, and it was written independently five times on that
assumption. `yolov8_nano_64x96` — the deployed detector, at 64x96 — breaks it.

Four of those five copies were cosmetic: a legend read `yolov8_nano_64x`, a
colour fell back to grey, an `#include` named a header that did not exist. The
fifth was `trace_metrics`, which is the one place in this repo allowed to say
what a periodic run achieved, and there the failure was silent and favourable.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import job_names  # noqa: E402
import schedule_trace  # noqa: E402
import trace_metrics  # noqa: E402

KNOWN = {"yolov8_nano_64x96", "yolov8_nano", "dronet", "mlp_control",
         "fused_full"}


class TheSplitNeedsTheRealNames(unittest.TestCase):

    def test_a_digit_ending_network_splits_at_the_right_place(self):
        self.assertEqual(job_names.split_job_name("yolov8_nano_64x960", KNOWN),
                         ("yolov8_nano_64x96", 0))
        self.assertEqual(job_names.split_job_name("yolov8_nano_64x963", KNOWN),
                         ("yolov8_nano_64x96", 3))

    def test_longest_match_wins(self):
        """`yolov8_nano` also prefixes the job name; the longer one is right."""
        self.assertEqual(job_names.model_of("yolov8_nano_64x960", KNOWN),
                         "yolov8_nano_64x96")

    def test_ordinary_names_are_unaffected(self):
        for job, want in (("dronet3", ("dronet", 3)),
                          ("mlp_control12", ("mlp_control", 12)),
                          ("fused_full0", ("fused_full", 0))):
            self.assertEqual(job_names.split_job_name(job, KNOWN), want)
            self.assertEqual(job_names.split_job_name(job), want,
                             "a name not ending in a digit must not need the "
                             "known set")

    def test_without_the_known_set_it_falls_back_and_is_wrong(self):
        """Pinned deliberately. The fallback is what every caller did before,
        and it is correct for every network whose name does not end in a
        digit — which is why this went unnoticed. Callers that can supply the
        names must."""
        self.assertEqual(job_names.split_job_name("yolov8_nano_64x960"),
                         ("yolov8_nano_64x", 960))

    def test_an_unknown_job_is_not_guessed_into_a_known_name(self):
        self.assertEqual(job_names.model_of("something_else7", KNOWN),
                         "something_else")


class TheScorerReportsAMeasurementNotAStructuralZero(unittest.TestCase):
    """The bug, end to end. `instance_index` returned 960, the deadline is
    `k*T + D`, so the detector's deadline became 960 * 50 ms = 48 seconds and
    it could not miss. Measured on the featured schedule: 0 misses with
    `response_p50 = -47954.45 ms`."""

    PERIOD = 50.0

    def _rows(self):
        """Four instances of a digit-ending network, each ~46 ms of work."""
        rows = []
        for inst in range(4):
            st = inst * self.PERIOD * 1000.0
            rows.append({
                "dispatch_key": f"yolov8_nano_64x96{inst}_dispatch_0",
                "job_name": f"yolov8_nano_64x96{inst}",
                "start_us": st, "end_us": st + 46_000.0,
                "run_us": 46_000.0, "queue_delay_us": 0.0,
                "target": "CPU_P", "cores": "0",
            })
        return rows

    def test_response_time_is_relative_to_this_instances_release(self):
        pm = trace_metrics.summarise_trace(
            self._rows(), {"yolov8_nano_64x96": self.PERIOD})["per_model"]
        v = pm["yolov8_nano_64x96"]
        self.assertEqual(v["instances"], 4)
        self.assertAlmostEqual(v["response_p50_ms"], 46.0, places=1)
        self.assertGreater(v["response_p50_ms"], 0.0,
                           "a negative response time means the release index "
                           "was read from the network's own name")

    def test_a_real_miss_is_reported_as_a_miss(self):
        """The property the bug destroyed: with k mis-read as ~960 the
        deadline is 48 s and nothing can ever be late."""
        rows = self._rows()
        for r in rows:                      # 60 ms of work in a 50 ms window
            r["end_us"] = r["start_us"] + 60_000.0
            r["run_us"] = 60_000.0
        pm = trace_metrics.summarise_trace(
            rows, {"yolov8_nano_64x96": self.PERIOD})["per_model"]
        v = pm["yolov8_nano_64x96"]
        self.assertEqual(v["instance_deadline_misses"], 4)
        self.assertAlmostEqual(v["worst_lateness_ms"], 10.0, places=1)

    def test_the_window_is_the_deadline_not_the_period(self):
        """`D = windows_ms.get(m, T)`. A tighter window than the period is how
        the workload spec expresses a deadline, and scoring without it scores
        against the wrong one."""
        pm = trace_metrics.summarise_trace(
            self._rows(), {"yolov8_nano_64x96": self.PERIOD},
            {"yolov8_nano_64x96": 40.0})["per_model"]
        v = pm["yolov8_nano_64x96"]
        self.assertEqual(v["deadline_ms"], 40.0)
        self.assertEqual(v["instance_deadline_misses"], 4,
                         "46 ms of work does not fit a 40 ms window")


class OldSchedulesStillScore(unittest.TestCase):
    """Schedules already on disk recorded the stripped name. They must keep
    scoring, and must be repairable when the real names are available."""

    def _sched(self, key):
        return {"metadata": {"periodic_networks": {key: 50.0}},
                "dispatches": {}}

    def test_a_stripped_key_passes_through_untouched_without_the_names(self):
        self.assertEqual(schedule_trace.periods_ms(self._sched("yolov8_nano_64x")),
                         {"yolov8_nano_64x": 50.0})

    def test_the_real_names_repair_a_stripped_key(self):
        self.assertEqual(
            schedule_trace.periods_ms(self._sched("yolov8_nano_64x"), KNOWN),
            {"yolov8_nano_64x96": 50.0})

    def test_a_correct_key_is_left_alone(self):
        self.assertEqual(
            schedule_trace.periods_ms(self._sched("dronet"), KNOWN),
            {"dronet": 50.0})

    def test_a_key_matching_nothing_known_is_kept_not_guessed(self):
        self.assertEqual(
            schedule_trace.periods_ms(self._sched("mystery"), KNOWN),
            {"mystery": 50.0})


if __name__ == "__main__":
    unittest.main()
