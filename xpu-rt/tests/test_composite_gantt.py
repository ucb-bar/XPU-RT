"""Smoke test for the before/after composite Gantt (xpu-rt/plot_gantt.py)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _fixture(makespan_ms):
    # minimal ModelBlaster-style schedule fixture
    return {
        "dispatches": {
            "yolov8_nano_dispatch_0": {
                "id": 0, "hardware_target": "CPU_P#0", "start_time": 0.0,
                "duration": makespan_ms, "job_name": "yolov8_nano"},
            "mlp_control0_dispatch_0": {
                "id": 1, "hardware_target": "CPU_E#0", "start_time": 0.0,
                "duration": makespan_ms / 2, "job_name": "mlp_control0"},
        }
    }


class CompositeGanttTests(unittest.TestCase):
    def test_composite_renders_png(self):
        try:
            import matplotlib  # noqa: F401
        except Exception:
            self.skipTest("matplotlib not installed")
        import plot_gantt
        with tempfile.TemporaryDirectory() as d:
            before = os.path.join(d, "before.json")
            after = os.path.join(d, "after.json")
            with open(before, "w") as fb:
                json.dump(_fixture(20.0), fb)
            with open(after, "w") as fa:
                json.dump(_fixture(12.0), fa)
            out = os.path.join(d, "composite.png")
            info = plot_gantt.render_composite_gantt(before, after, out,
                                                     titles=("BEFORE", "AFTER"))
            self.assertTrue(os.path.exists(out) and os.path.getsize(out) > 0)
            self.assertIn("before", info)
            self.assertIn("after", info)


if __name__ == "__main__":
    unittest.main()
