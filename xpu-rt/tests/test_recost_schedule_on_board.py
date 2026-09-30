"""Regression tests for board recosting of digit-suffixed network names."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "recost_schedule_on_board.py"


class BoardRecostUsesAuthoritativeNetworkNames(unittest.TestCase):

    def test_yolo_resolution_digits_are_not_parsed_as_the_instance(self):
        schedule = {
            "dispatches": {
                "d0": {
                    "id": "d0",
                    "job_name": "yolov8_nano_64x960",
                    "module_name": "dispatch_0",
                    "hardware_target": "CPU_P#0",
                    "start_time": 0.0,
                    "duration": 24.0,
                    "dependencies": [],
                },
                "d1": {
                    "id": "d1",
                    "job_name": "yolov8_nano_64x961",
                    "module_name": "dispatch_0",
                    "hardware_target": "CPU_P#1",
                    "start_time": 23.0,
                    "duration": 2.0,
                    "dependencies": [],
                },
            }
        }
        spec = {"networks": {"yolov8_nano_64x96": {
            "period": 23.0, "window_duration": 23.0,
        }}}
        calibration = {"aggregate_multiplier": 1.0}

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = {name: root / f"{name}.json"
                     for name in ("schedule", "spec", "calibration", "out")}
            for name, value in (("schedule", schedule), ("spec", spec),
                                ("calibration", calibration)):
                paths[name].write_text(json.dumps(value))
            subprocess.run([
                sys.executable, str(SCRIPT),
                "--schedule", str(paths["schedule"]),
                "--spec", str(paths["spec"]),
                "--calibration", str(paths["calibration"]),
                "--out", str(paths["out"]),
            ], check=True, capture_output=True, text=True)

            recosted = json.loads(paths["out"].read_text())
            metrics = json.loads((root / "out_metrics.json").read_text())

        first = recosted["dispatches"]["d0"]
        second = recosted["dispatches"]["d1"]
        self.assertTrue(first["deadline_miss"])
        self.assertAlmostEqual(first["deadline_overrun_us"], 1000.0)
        self.assertAlmostEqual(second["start_time"], 23.0)
        self.assertFalse(second["deadline_miss"])
        self.assertEqual(metrics["deadline_miss_count"], 1)
        self.assertAlmostEqual(metrics["total_lateness_ms"], 1.0)
        self.assertAlmostEqual(recosted["metadata"]["makespan"], 25.0)
        self.assertAlmostEqual(recosted["metadata"]["yolo_frame_latency_ms"], 24.0)
        self.assertEqual(recosted["metadata"]["deadline_miss_count"], 1)


if __name__ == "__main__":
    unittest.main()
