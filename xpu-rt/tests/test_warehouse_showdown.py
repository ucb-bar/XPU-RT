"""Truthfulness checks for the warehouse figure's schedule comparison."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "sims" / "scripts" / "compose_warehouse_showdown.py"
SPEC = importlib.util.spec_from_file_location("compose_warehouse_showdown", SCRIPT)
warehouse = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(warehouse)


class DigitSuffixedDetectorNames(unittest.TestCase):

    def test_detector_resolution_is_not_mistaken_for_instance_number(self):
        self.assertEqual(warehouse.netinfo("yolov8_nano_64x961")[0], "yolo")
        self.assertEqual(warehouse.inst_of("yolov8_nano_64x961"), 1)

    def test_frame_response_is_measured_from_release_not_first_dispatch(self):
        dispatches = {
            "d": {
                "job_name": "yolov8_nano_64x961",
                "start_time": 30.0,
                "duration": 12.0,
                "hardware_target": "CPU_P#0",
            }
        }
        instance, release, finish = warehouse._frame_window(dispatches, 1, 23.0)
        _, response = warehouse._hart_blocks(dispatches, (release, finish))
        self.assertEqual(instance, 1)
        self.assertEqual(release, 23.0)
        self.assertEqual(finish, 42.0)
        self.assertEqual(response, 19.0)


if __name__ == "__main__":
    unittest.main()
