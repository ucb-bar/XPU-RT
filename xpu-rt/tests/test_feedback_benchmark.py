"""Contract tests for the original-vs-feedback experiment evaluator."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
sys.path.insert(0, _XPURT)

import feedback_benchmark as fb  # noqa: E402


def _schedule(duration, pdb_hash, jobs=("heavy0",)):
    return {
        "dispatches": {
            f"d{i}": {"id": i, "dependencies": [],
                       "hardware_target": "CPU_P#0",
                       "start_time": i * duration,
                       "duration": duration, "job_name": job}
            for i, job in enumerate(jobs)
        },
        "metadata": {"machines": ["CPU_P#0"], "pdb_hash": pdb_hash,
                     "periodic_networks": {}},
    }


class FeedbackBenchmarkTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        self.workload = "workload.json"
        with open(os.path.join(self.repo, self.workload), "w") as f:
            json.dump({"networks": {"heavy": {"identifier": "heavy"}}}, f)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_schedule(self, name, duration, jobs=("heavy0",)):
        path = f"{name}.json"
        with open(os.path.join(self.repo, path), "w") as f:
            json.dump(_schedule(duration, name, jobs), f)
        return path

    def _manifest(self):
        base_g = self._write_schedule("base_g", 20.0)
        base_c = self._write_schedule("base_c", 18.0)
        cand_g = self._write_schedule("cand_g", 12.0)
        cand_c = self._write_schedule("cand_c", 10.0)
        feedback = "feedback.json"
        with open(os.path.join(self.repo, feedback), "w") as f:
            json.dump({
                "schema_version": 1,
                "dispatches": {
                    "heavy0_dispatch_0": {"hints": ["prefer_finer"]},
                },
            }, f)
        return {
            "schema_version": 1,
            "experiment_id": "fixture",
            "common": {"solvers": ["greedy", "cpsat", "mosek"],
                       "heavy_model": "heavy"},
            "gates": {name: {"status": "pass", "evidence": "fixture"}
                      for name in fb.REQUIRED_GATES},
            "phases": {
                "original": {"networks_json": self.workload,
                             "cells": [
                                 {"solver": "greedy", "status": "validated",
                                  "schedule": base_g},
                                 {"solver": "cpsat", "status": "validated",
                                  "schedule": base_c},
                                 {"solver": "mosek", "status": "timeout"}]},
                "feedback": {"networks_json": self.workload,
                             "transformation": {
                                 "kind": "unfuse",
                                 "targets": ["heavy"],
                                 "feedback_artifact": feedback,
                                 "source_schedule_sha256": fb.sha256_file(
                                     os.path.join(self.repo, base_g)),
                             },
                             "cells": [
                                 {"solver": "greedy", "status": "validated",
                                  "schedule": cand_g},
                                 {"solver": "cpsat", "status": "validated",
                                  "schedule": cand_c},
                                 {"solver": "mosek", "status": "timeout"}]},
            },
        }

    def test_feedback_must_beat_every_validated_original(self):
        result = fb.evaluate(self._manifest(), self.repo)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["feedback_best"], "cpsat")
        self.assertEqual({c["baseline_solver"] for c in result["comparisons"]},
                         {"greedy", "cpsat"})

    def test_timeout_is_reported_but_not_ranked(self):
        result = fb.evaluate(self._manifest(), self.repo)
        mosek = [c for c in result["cells"] if c["solver"] == "mosek"]
        self.assertEqual([c["status"] for c in mosek], ["timeout", "timeout"])
        self.assertNotIn("mosek", result["original_co_best"])

    def test_failed_gate_rejects_an_objective_win(self):
        manifest = self._manifest()
        manifest["gates"]["correctness"]["status"] = "fail"
        self.assertFalse(fb.evaluate(manifest, self.repo)["accepted"])

    def test_different_instance_counts_are_refused(self):
        manifest = self._manifest()
        extra = self._write_schedule("cand_extra", 10.0,
                                     ("heavy0", "heavy1"))
        manifest["phases"]["feedback"]["cells"][1]["schedule"] = extra
        with self.assertRaisesRegex(fb.ManifestError, "instance counts differ"):
            fb.evaluate(manifest, self.repo)

    def test_schedule_marked_validated_must_pass_feasibility(self):
        manifest = self._manifest()
        bad = _schedule(10.0, "bad", ("heavy0", "heavy0"))
        bad["dispatches"]["d1"]["start_time"] = 5.0
        with open(os.path.join(self.repo, "bad.json"), "w") as f:
            json.dump(bad, f)
        manifest["phases"]["feedback"]["cells"][1]["schedule"] = "bad.json"
        with self.assertRaisesRegex(fb.ManifestError, "infeasible"):
            fb.evaluate(manifest, self.repo)

    def test_feedback_target_needs_prefer_finer_provenance(self):
        manifest = self._manifest()
        manifest["phases"]["feedback"]["transformation"]["targets"] = [
            "missing"]
        with self.assertRaisesRegex(fb.ManifestError, "without prefer_finer"):
            fb.evaluate(manifest, self.repo)

    def test_feedback_must_point_to_a_validated_original_schedule(self):
        manifest = self._manifest()
        manifest["phases"]["feedback"]["transformation"][
            "source_schedule_sha256"] = "not-a-schedule-hash"
        with self.assertRaisesRegex(fb.ManifestError, "does not identify"):
            fb.evaluate(manifest, self.repo)

    def test_undeclared_workload_change_is_refused(self):
        manifest = self._manifest()
        feedback_workload = "feedback_workload.json"
        with open(os.path.join(self.repo, feedback_workload), "w") as f:
            json.dump({"networks": {"heavy": {"identifier": "heavy",
                                                "period": 10}}}, f)
        manifest["phases"]["feedback"]["networks_json"] = feedback_workload
        with self.assertRaisesRegex(fb.ManifestError,
                                    "do not match transformation contract"):
            fb.evaluate(manifest, self.repo)

    def test_declared_workload_change_is_recorded(self):
        manifest = self._manifest()
        feedback_workload = "feedback_workload.json"
        with open(os.path.join(self.repo, feedback_workload), "w") as f:
            json.dump({"networks": {"heavy": {"identifier": "heavy",
                                                "period": 10}}}, f)
        manifest["phases"]["feedback"]["networks_json"] = feedback_workload
        change = {"path": "networks.heavy.period", "from": None, "to": 10}
        manifest["phases"]["feedback"]["transformation"][
            "allowed_workload_changes"] = [change]
        self.assertEqual(fb.evaluate(manifest, self.repo)["workload_changes"],
                         [change])


if __name__ == "__main__":
    unittest.main()
