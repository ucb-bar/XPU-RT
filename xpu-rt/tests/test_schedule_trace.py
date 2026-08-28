"""Regression tests for rendering a predicted schedule into trace rows.

THE BUG THIS FILE EXISTS FOR: a predicted schedule and a measured run were
scored by two different definitions of "deadline miss".

`output_scheduled_json` writes `start_time` and `duration` in **ms**;
`trace_metrics` reads a trace in **us**. Because nothing converted between them,
the only host-side scorer in the tree (`scripts/k1_baselines.predicted()`) grew
its own fourth copy of the instance collapse -- per instance, no rate, no
response time, no utilization -- while `trace_metrics` (the module written
specifically to stop that happening) counted instances properly. Two numbers,
same name, different meaning, and no error anywhere.

`schedule_trace` removes the second definition rather than adding a third. These
tests pin the two things that make that safe: the unit conversion, and the fact
that `trace_metrics.summarise_trace` on the rendered rows agrees with the
schedule it came from.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _REPO)
sys.path.insert(0, _XPURT)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import schedule_trace  # noqa: E402
import trace_metrics  # noqa: E402


def _schedule():
    """Two 10 ms-period MLP instances and one DroNet instance that overruns.

    dronet0 spans 0 -> 40 ms against a 33.3 ms deadline: exactly one instance
    miss, and *two* late dispatches. The two counts differing is the point.
    """
    return {
        "dispatches": {
            "mlp0_dispatch_0": {
                "id": 0, "dependencies": [], "hardware_target": "CPU_P#0",
                "start_time": 0.0, "duration": 0.5, "job_name": "mlp0",
                "release_policy": "phase_locked", "release_us": 0.0,
            },
            "mlp1_dispatch_0": {
                "id": 0, "dependencies": [], "hardware_target": "CPU_P#0",
                "start_time": 12.0, "duration": 0.5, "job_name": "mlp1",
                "release_policy": "phase_locked", "release_us": 10000.0,
            },
            "dronet0_dispatch_0": {
                "id": 0, "dependencies": [], "hardware_target": "CPU_E#0+CPU_E#1",
                "start_time": 0.0, "duration": 35.0, "job_name": "dronet0",
                "release_policy": "phase_locked", "release_us": 0.0,
            },
            "dronet0_dispatch_1": {
                "id": 1, "dependencies": ["dronet0_dispatch_0"],
                "hardware_target": "CPU_E#0+CPU_E#1",
                "start_time": 35.0, "duration": 5.0, "job_name": "dronet0",
            },
        },
        "metadata": {
            "makespan": 40.0,
            "machines": ["CPU_P#0", "CPU_E#0", "CPU_E#1"],
            "periodic_networks": {"mlp": 10.0, "dronet": 33.3},
        },
    }


class TestUnits(unittest.TestCase):
    def test_ms_fields_become_us(self):
        rows = {r["dispatch_key"]: r
                for r in schedule_trace.trace_rows_from_schedule(_schedule())}
        r = rows["dronet0_dispatch_1"]
        self.assertEqual(r["start_us"], 35_000.0)
        self.assertEqual(r["end_us"], 40_000.0)
        self.assertEqual(r["run_us"], 5_000.0)

    def test_release_us_is_not_scaled_twice(self):
        """`release_us` is ALREADY in us in the schedule JSON.

        Scaling it again would put mlp1's release at 10 000 ms and make its
        queue delay hugely negative, which clamps to zero and silently hides
        the wait. Pinned because the surrounding fields all do need scaling.
        """
        rows = {r["dispatch_key"]: r
                for r in schedule_trace.trace_rows_from_schedule(_schedule())}
        self.assertEqual(rows["mlp1_dispatch_0"]["ready_us"], 10_000.0)
        self.assertEqual(rows["mlp1_dispatch_0"]["queue_delay_us"], 2_000.0)

    def test_ready_follows_dependency_completion(self):
        rows = {r["dispatch_key"]: r
                for r in schedule_trace.trace_rows_from_schedule(_schedule())}
        self.assertEqual(rows["dronet0_dispatch_1"]["ready_us"], 35_000.0)
        self.assertEqual(rows["dronet0_dispatch_1"]["queue_delay_us"], 0.0)


class TestCoreAttribution(unittest.TestCase):
    def test_sharded_target_splits_into_cluster_and_cores(self):
        cluster, cores = schedule_trace.split_hardware_target("CPU_E#0+CPU_E#1")
        self.assertEqual(cluster, "CPU_E")
        self.assertEqual(cores, ["0", "1"])

    def test_utilization_is_reported_per_core(self):
        """Without a `cores` column `trace_metrics` refuses to report
        utilization at all -- correctly, since summing run_us by cluster label
        takes a 4-core cluster past 100%. The rendered rows must therefore
        carry it, or every predicted cell loses the utilization term."""
        s = _schedule()
        rows = schedule_trace.trace_rows_from_schedule(s)
        summary = trace_metrics.summarise_trace(rows, schedule_trace.periods_ms(s))
        util = summary["per_core_utilization_pct"]
        self.assertIsNotNone(util)
        # dronet holds both E cores for the whole 40 ms span.
        self.assertEqual(util["CPU_E#0"], 100.0)
        self.assertEqual(util["CPU_E#1"], 100.0)

    def test_no_observed_cpu_columns(self):
        """A predicted run observed nothing; emitting the columns would make
        `summarise_trace` report a migration count for a run that never ran."""
        rows = schedule_trace.trace_rows_from_schedule(_schedule())
        s = trace_metrics.summarise_trace(rows, {"mlp": 10.0, "dronet": 33.3})
        self.assertIsNone(s.get("dispatches_that_migrated_mid_run"))


class TestScoring(unittest.TestCase):
    def test_misses_are_counted_per_instance_not_per_dispatch(self):
        s = _schedule()
        rows = schedule_trace.trace_rows_from_schedule(s)
        summary = trace_metrics.summarise_trace(rows, schedule_trace.periods_ms(s))
        dronet = summary["per_model"]["dronet"]
        self.assertEqual(dronet["instances"], 1)
        self.assertEqual(dronet["instance_deadline_misses"], 1)
        self.assertEqual(dronet["instance_deadline_miss_rate_pct"], 100.0)
        # Two dronet dispatches end after the deadline; the instance count is 1.
        late_dispatches = sum(1 for r in rows
                              if r["job_name"] == "dronet0"
                              and r["end_us"] / 1000.0 > 33.3)
        self.assertEqual(late_dispatches, 2)

    def test_response_is_measured_from_the_nominal_release(self):
        """mlp1 completes at 12.5 ms. Measured from its own start that is
        0.5 ms and it looks perfect; measured from its release k*T = 10 ms it
        is 2.5 ms. The second is the one that can see a model failing to keep
        its rate by running invocations back to back."""
        s = _schedule()
        summary = trace_metrics.summarise_trace(
            schedule_trace.trace_rows_from_schedule(s),
            schedule_trace.periods_ms(s))
        self.assertAlmostEqual(summary["per_model"]["mlp"]["response_p99_ms"],
                               2.5, places=3)

    def test_windows_override_the_period_as_deadline(self):
        s = _schedule()
        rows = schedule_trace.trace_rows_from_schedule(s)
        tight = trace_metrics.summarise_trace(rows, {"mlp": 10.0}, {"mlp": 1.0})
        self.assertEqual(tight["per_model"]["mlp"]["instance_deadline_misses"], 1)

    def test_standalone_service_is_the_serial_sum_in_us(self):
        self.assertAlmostEqual(
            schedule_trace.standalone_service_us(_schedule()), 41_000.0, places=3)


class TestSweepPlumbing(unittest.TestCase):
    """The sweep locates each cell's schedule by reconstructing the filename
    `run_xpurt_schedule.py` chose. That rule lives in two files, so it is
    pinned here: if the naming changes on one side, this fails instead of the
    sweep silently reporting `no schedule at ...` for every cell."""

    def test_output_tag_matches_run_xpurt_schedule(self):
        import profile_schedulers as ps
        self.assertEqual(ps.output_tag("mosek"), "")       # no infix, by design
        self.assertEqual(ps.output_tag("greedy"), "_greedy")
        self.assertEqual(ps.output_tag("greedy_periodic"), "_greedy_periodic")
        self.assertEqual(ps.output_tag("heft"), "_heft")
        src = open(os.path.join(_REPO, "scripts",
                                "run_xpurt_schedule.py")).read()
        self.assertIn('solver_tag = "" if scheduler == "mosek" '
                      'else f"_{scheduler}"', src)
        self.assertIn('solver_tag = "_greedy"', src)

    def test_unavailable_is_distinguished_from_error(self):
        import profile_schedulers as ps
        self.assertEqual(ps.classify_failure(
            "ModuleNotFoundError: No module named 'ortools'"), "unavailable")
        self.assertEqual(ps.classify_failure(
            "MSK_RES_ERR_LICENSE_EXPIRED"), "unavailable")
        self.assertEqual(ps.classify_failure(
            "ValueError: infeasible"), "error")


class TestHeavyModelWithoutAPeriod(unittest.TestCase):
    """A one-shot background net (a YOLO pass) has no period, so
    `trace_metrics` -- correctly -- does not report it: there is no k*T to
    measure a response from. But `candidate_objective` terms 5 and 6 are about
    exactly that model, and `from_trace_summary` can only fill them from
    `per_model`. Left alone, both terms silently read 0.0 for every candidate
    and drop out of the order."""

    def test_latency_and_throughput_are_derived_from_the_rows(self):
        import profile_schedulers as ps
        s = _schedule()
        s["dispatches"]["yolo0_dispatch_0"] = {
            "id": 0, "dependencies": [], "hardware_target": "CPU_P#0",
            "start_time": 50.0, "duration": 30.0, "job_name": "yolo0"}
        rows = schedule_trace.trace_rows_from_schedule(s)
        latency, hz = ps.heavy_stats(rows, "yolo")
        self.assertAlmostEqual(latency, 30.0, places=3)
        self.assertAlmostEqual(hz, 1 / 0.080, places=3)

    def test_absent_model_is_zero_not_an_exception(self):
        import profile_schedulers as ps
        rows = schedule_trace.trace_rows_from_schedule(_schedule())
        self.assertEqual(ps.heavy_stats(rows, "not_a_model"), (0.0, 0.0))


class TestGanttRenderer(unittest.TestCase):
    """The sweep reuses `scripts/plot_k1_evolution.py`'s renderer rather than
    adding a fourth Gantt. These pin the parameters it now has to honour."""

    def test_lanes_come_from_the_schedule(self):
        import plot_k1_evolution as pk
        cores = pk.cores_from_schedule(_schedule()["dispatches"])
        self.assertEqual(cores, ["CPU_E#0", "CPU_E#1", "CPU_P#0"])

    def test_lane_order_is_numeric_not_lexicographic(self):
        import plot_k1_evolution as pk
        sched = {"a": {"hardware_target": "CPU_P#10"},
                 "b": {"hardware_target": "CPU_P#2"}}
        self.assertEqual(pk.cores_from_schedule(sched),
                         ["CPU_P#2", "CPU_P#10"])

    def test_published_figure_colours_are_unchanged(self):
        import plot_k1_evolution as pk
        cols = pk.model_colours({"dronet", "mlp", "yolov8_nano"})
        self.assertEqual(cols["dronet"], pk.C_DRONET)
        self.assertEqual(cols["mlp"], pk.C_MLP)
        self.assertNotIn(cols["yolov8_nano"], (pk.C_DRONET, pk.C_MLP))

    def test_renders_a_multi_panel_figure(self):
        import tempfile
        import plot_k1_evolution as pk
        s = _schedule()
        rows = schedule_trace.trace_rows_from_schedule(s)
        panels = [{"title": "a", "rows": rows, "sched": s["dispatches"]},
                  {"title": "b", "rows": rows, "sched": s["dispatches"]}]
        with tempfile.TemporaryDirectory() as d:
            png, pdf = pk.render_gantt_panels(
                panels, os.path.join(d, "g"),
                periods=schedule_trace.periods_ms(s), window_ms=50.0,
                deadline_model="dronet")
            self.assertTrue(os.path.getsize(png) > 0)
            self.assertTrue(os.path.getsize(pdf) > 0)


if __name__ == "__main__":
    unittest.main()
