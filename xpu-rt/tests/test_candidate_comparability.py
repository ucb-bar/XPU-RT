"""Two amounts of work are not two graphs.

A verdict between two solved schedules only means something if both sides
scheduled the same workload. Nothing in a schedule records how many refinement
iterations produced it, so this is not checkable from the file's own metadata
-- it has to be recomputed from the dispatches.

THE FAILURE THIS PINS is not hypothetical. The 4 Hz baseline was re-solved
without `--max-periodic-iters 1`; the refinement loop grew `mlp_control` from
32 instances to 91 and `dronet` from 10 to 28, and the result was written under
the baseline's own filename. Three verdicts had already been recorded against
the 826-dispatch version. Nothing downstream complained:

  * `pdb_hash` still differed from every candidate's, so the guard that proves
    the two solves read different measured COSTS was satisfied;
  * every one of the nine terms still computed, on more instances;
  * and the figure rendered from it reported ACCEPT for the DroNet x2 rung,
    which had been adjudicated REJECT on heavy-model max latency.

A rewrite changes how many dispatches an instance is made of. It must not
change how many instances there are -- so unequal instance counts mean the
solver flags differed, and the difference being measured is not the graph.
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import schedule_scoring  # noqa: E402

KNOWN = {"yolov8_nano_64x96", "yolov8_nano", "dronet", "mlp_control"}


def _sched(jobs):
    """A schedule holding one dispatch per (job, index) pair."""
    return {"dispatches": {
        f"d{i}": {"job_name": j, "hardware_target": "CPU_P#0",
                  "start_time": float(i), "duration": 1.0}
        for i, j in enumerate(jobs)}}


class InstanceCountTests(unittest.TestCase):

    def test_counts_instances_not_dispatches(self):
        """A split multiplies dispatches per instance; the count must not move."""
        one = _sched(["dronet0", "dronet0", "dronet1"])
        # the same two instances, each now made of three dispatches
        split = _sched(["dronet0"] * 3 + ["dronet1"] * 3)
        self.assertEqual(schedule_scoring.instances_per_model(one, KNOWN),
                         {"dronet": 2})
        self.assertEqual(schedule_scoring.instances_per_model(split, KNOWN),
                         {"dronet": 2})

    def test_refinement_growth_is_visible(self):
        """The actual regression: 32 mlp instances against 91."""
        short = _sched([f"mlp_control{i}" for i in range(32)])
        grown = _sched([f"mlp_control{i}" for i in range(91)])
        self.assertNotEqual(schedule_scoring.instances_per_model(short, KNOWN),
                            schedule_scoring.instances_per_model(grown, KNOWN))

    def test_digit_suffixed_network_name(self):
        """The detector's own name ends in digits; it is one model, not many.

        Without the known set, `yolov8_nano_64x960` splits at the wrong place
        and every instance counts as its own model -- so this guard would fire
        on two schedules that are in fact perfectly comparable.
        """
        s = _sched([f"yolov8_nano_64x96{i}" for i in range(4)])
        self.assertEqual(schedule_scoring.instances_per_model(s, KNOWN),
                         {"yolov8_nano_64x96": 4})

    def test_ignores_dispatches_without_a_job(self):
        s = _sched(["dronet0"])
        s["dispatches"]["nojob"] = {"job_name": "", "hardware_target": "CPU_P#0"}
        self.assertEqual(schedule_scoring.instances_per_model(s, KNOWN),
                         {"dronet": 1})


if __name__ == "__main__":
    unittest.main()


class ImplTagTests(unittest.TestCase):
    """`impl` must name the implementation that ran EACH dispatch.

    The field was first read from a loop variable left over from an earlier
    pass over the operations, so every dispatch inherited whatever the LAST
    operation happened to be assigned -- which tagged an entire heterogeneous
    schedule `rvv` while the durations were plainly the IME costs. It was
    caught by cross-checking against the report's own `combo_idx`, not by
    reading the field, which is the point of this test.
    """

    def test_impl_matches_the_combination_actually_chosen(self):
        import glob, json, os
        sched = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                             "schedules",
                             "scheduled_networks_k1_ffn_ime_greedy_profiled.json")
        report = sched.replace(".json", "_report.json")
        if not (os.path.exists(sched) and os.path.exists(report)):
            self.skipTest("impl-aware schedule not solved in this tree")
        d = json.load(open(sched))
        impls = (d.get("metadata") or {}).get("combo_impls")
        self.assertIsNotNone(impls, "an impl-aware solve must record combo_impls")
        chosen = {e["name"]: e["combo_idx"] for e in json.load(open(report))["dispatches"]}
        mismatched = [n for n, v in d["dispatches"].items()
                      if n in chosen and v.get("impl") != impls[chosen[n]]]
        self.assertEqual(mismatched, [],
                         "impl disagrees with the combination the solver chose")

    def test_more_than_one_implementation_is_actually_used(self):
        """A heterogeneous schedule that uses one implementation is not one."""
        import json, os
        sched = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                             "schedules",
                             "scheduled_networks_k1_ffn_ime_greedy_profiled.json")
        if not os.path.exists(sched):
            self.skipTest("impl-aware schedule not solved in this tree")
        d = json.load(open(sched))
        used = {v.get("impl") for v in d["dispatches"].values() if v.get("impl")}
        self.assertIn("ime", used)
        self.assertIn("rvv", used)
