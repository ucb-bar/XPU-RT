"""Tests for postmortem.compare_trace."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from postmortem import compare_trace  # noqa: E402


def _write_trace(path: str, rows: list[dict]) -> None:
    """Emit an xpurt_trace.csv with the same header ModelBlaster writes."""
    header = (
        "entry_id,network,instance,dispatch_id,op,name,core_kind,hart,"
        "predicted_start_ms,predicted_duration_ms,worker_kind_idx,"
        "actual_start_cycles,actual_end_cycles"
    )
    with open(path, "w") as f:
        f.write(header + "\n\n")
        for r in rows:
            line = (
                f"{r['entry_id']},{r['network']},{r['instance']},{r['dispatch_id']},"
                f"{r['op']},{r['name']},{r['core_kind']},{r['hart']},"
                f"{r['predicted_start']:.6f},{r['predicted_duration']:.6f},"
                f"{r['worker_kind_idx']},{r['actual_start']},{r['actual_end']}"
            )
            f.write(line + "\n\n")


class CompareTraceTests(unittest.TestCase):

    def _base_row(self, **kw):
        row = dict(
            entry_id=0, network="dronet", instance=0, dispatch_id=0,
            op="conv2d_s8", name="conv0", core_kind="gemmini", hart=0,
            predicted_start=0.0, predicted_duration=1000.0,
            worker_kind_idx=0, actual_start=0, actual_end=1000,
        )
        row.update(kw)
        return row

    def test_perfect_predictions(self):
        # actual / predicted = 1.0 on every dispatch → median_ratio=1, error=0
        rows = [
            self._base_row(entry_id=i, dispatch_id=i,
                           predicted_duration=1000.0,
                           actual_start=i * 1000, actual_end=(i + 1) * 1000)
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as td:
            tp = os.path.join(td, "trace.csv")
            _write_trace(tp, rows)
            out = compare_trace(tp)
        self.assertEqual(out["n_rows"], 5)
        self.assertAlmostEqual(out["median_ratio"], 1.0)
        self.assertAlmostEqual(out["rms_error_pct"], 0.0)

    def test_constant_clock_domain_ratio(self):
        # Same predicted_duration on every row, actuals are 0.5× — uniform.
        # Postmortem should report median_ratio=0.5 and zero error.
        rows = [
            self._base_row(entry_id=i, dispatch_id=i,
                           predicted_duration=1000.0,
                           actual_start=0, actual_end=500)
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as td:
            tp = os.path.join(td, "trace.csv")
            _write_trace(tp, rows)
            out = compare_trace(tp)
        self.assertAlmostEqual(out["median_ratio"], 0.5)
        self.assertAlmostEqual(out["rms_error_pct"], 0.0)

    def test_outlier_surfaces_in_top_outliers(self):
        # 4 perfectly-predicted rows + 1 outlier (2× expected duration).
        rows = []
        for i in range(4):
            rows.append(self._base_row(
                entry_id=i, dispatch_id=i,
                predicted_duration=1000.0,
                actual_start=0, actual_end=1000,
            ))
        rows.append(self._base_row(
            entry_id=4, dispatch_id=4, name="outlier",
            predicted_duration=1000.0,
            actual_start=0, actual_end=2000,
        ))
        with tempfile.TemporaryDirectory() as td:
            tp = os.path.join(td, "trace.csv")
            _write_trace(tp, rows)
            out = compare_trace(tp, n_outliers=3)
        self.assertEqual(out["top_outliers"][0]["name"], "outlier")
        self.assertGreater(out["top_outliers"][0]["deviation_pct"], 50.0)
        self.assertAlmostEqual(out["top_outliers"][0]["ratio"], 2.0)

    def test_write_to_emits_file(self):
        rows = [self._base_row()]
        with tempfile.TemporaryDirectory() as td:
            tp = os.path.join(td, "trace.csv")
            outp = os.path.join(td, "sub", "postmortem.json")
            _write_trace(tp, rows)
            compare_trace(tp, write_to=outp)
            with open(outp) as f:
                payload = json.load(f)
            self.assertEqual(payload["n_rows"], 1)


if __name__ == "__main__":
    unittest.main()
