"""Tests for the ModelBlaster profile_db -> XPU-RT results.csv bridge.

The behaviour that matters is the strictness boundary: a dispatch with no
measurement must either be a known zero-cost alias (emit 0, say so) or a hard
error. Silently substituting a cost is how a schedule gets built against
fictional timings, which is the failure this whole bridge exists to avoid.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_SCRIPT = os.path.join(_REPO, "scripts", "export_profile_db_to_results_csv.py")

_spec = importlib.util.spec_from_file_location("_profile_bridge", _SCRIPT)
bridge = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(bridge)


class Harness:
    """Builds a minimal fake ModelBlaster tree + emitted graph in a tmpdir."""

    def __init__(self, root: str, *, ops, profiled_ids):
        self.root = root
        self.mb = os.path.join(root, "mb")
        self.graph_root = os.path.join(root, "gen", "vmfb")
        self.out_root = os.path.join(root, "gen", "profile")
        self.model = "toynet"
        self.quant = "int8"
        self.target = "firesim_gemmini_opu"

        # ModelBlaster IR: ops is a list of (dispatch_id, op_name)
        ir_dir = os.path.join(self.mb, "examples", self.model, self.quant, "generated")
        os.makedirs(ir_dir, exist_ok=True)
        with open(os.path.join(ir_dir, "graph.json"), "w") as f:
            json.dump(
                {"ops": [{"dispatch_id": d, "op": o, "depends_on": []} for d, o in ops]},
                f,
            )

        # profile_db: only `profiled_ids` get measurements
        db_dir = os.path.join(self.mb, "benchmarks", "profile_db")
        os.makedirs(db_dir, exist_ok=True)
        with open(
            os.path.join(db_dir, f"{self.model}__gemmini__{self.quant}.jsonl"), "w"
        ) as f:
            for d in profiled_ids:
                # two samples so the median is exercised
                for c in (1000 * (d + 1), 1000 * (d + 1) + 100):
                    f.write(
                        json.dumps(
                            {
                                "dispatch_id": d,
                                "cycles": c,
                                "op_type": dict(ops)[d],
                            }
                        )
                        + "\n"
                    )

        # emitted XPU-RT dispatch graph: every op in the IR
        basename = f"{self.model}.{self.quant}"
        gdir = os.path.join(
            self.graph_root, self.model, self.target, "gemmini", basename
        )
        os.makedirs(gdir, exist_ok=True)
        with open(os.path.join(gdir, f"{basename}_dispatch_graph.json"), "w") as f:
            json.dump(
                {
                    "dot_file": "",
                    "dispatches": {
                        f"dispatch_{d}": {
                            "id": d,
                            "ordinal": 1,
                            "total": 1,
                            "dependencies": [],
                        }
                        for d, _ in ops
                    },
                },
                f,
            )

    def export(self, **kw):
        params = dict(
            model=self.model,
            backend="gemmini",
            quant=self.quant,
            target=self.target,
            topo_tag="topo_0",
            mb_root=self.mb,
            out_root=self.out_root,
            graph_root=self.graph_root,
            clock_mhz=1000.0,
            xpurt_root=_REPO,
        )
        params.update(kw)
        return bridge.export_one(**params)

    def rows(self):
        p = os.path.join(
            self.out_root,
            "gemmini",
            self.target,
            self.model,
            f"{self.model}.{self.quant}",
            "topo_0",
            "results.csv",
        )
        with open(p) as f:
            return list(csv.DictReader(f))

    def provenance(self):
        p = os.path.join(
            self.out_root,
            "gemmini",
            self.target,
            self.model,
            f"{self.model}.{self.quant}",
            "topo_0",
            "_provenance.json",
        )
        with open(p) as f:
            return json.load(f)


class Strictness(unittest.TestCase):
    def test_fully_profiled_model_exports_every_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            h = Harness(
                d,
                ops=[(0, "conv2d_s8"), (1, "linear_s8")],
                profiled_ids=[0, 1],
            )
            res = h.export()
            self.assertEqual(res["n_graph"], 2)
            self.assertEqual(res["n_profiled"], 2)
            self.assertEqual(res["n_zero"], 0)
            rows = h.rows()
            self.assertEqual([r["source"] for r in rows], ["firesim_measured"] * 2)
            # median of (1000, 1100) = 1050 cycles -> 0.00105 ms at 1 GHz
            self.assertEqual(rows[0]["cycles"], "1050")
            self.assertAlmostEqual(float(rows[0]["mean_time"]), 0.00105, places=9)

    def test_zero_cost_alias_is_emitted_as_zero_and_labelled(self):
        """chunk2_c1 and view appear in the graph but never in a profile run.
        They must survive as zero-duration rows so precedence edges hold."""
        with tempfile.TemporaryDirectory() as d:
            h = Harness(
                d,
                ops=[(0, "conv2d_s8"), (1, "chunk2_c1"), (2, "linear_s8")],
                profiled_ids=[0, 2],
            )
            res = h.export()
            self.assertEqual(res["n_zero"], 1)
            rows = {int(r["dispatch_id"]): r for r in h.rows()}
            self.assertEqual(len(rows), 3, "the alias must still be present")
            self.assertEqual(rows[1]["source"], "zero_cost_by_construction")
            self.assertEqual(rows[1]["cycles"], "0")
            self.assertEqual(float(rows[1]["mean_time"]), 0.0)
            self.assertEqual(rows[0]["source"], "firesim_measured")

    def test_unprofiled_real_op_is_a_hard_error(self):
        """The whole point: an unmeasured conv must stop the export, not get a
        substituted cost."""
        with tempfile.TemporaryDirectory() as d:
            h = Harness(
                d,
                ops=[(0, "conv2d_s8"), (1, "conv2d_s8")],
                profiled_ids=[0],
            )
            with self.assertRaises(RuntimeError) as cm:
                h.export()
            msg = str(cm.exception)
            self.assertIn("id=1", msg)
            self.assertIn("conv2d_s8", msg)
            self.assertIn("fictional", msg)

    def test_missing_profile_db_names_what_is_available(self):
        with tempfile.TemporaryDirectory() as d:
            h = Harness(d, ops=[(0, "conv2d_s8")], profiled_ids=[0])
            with self.assertRaises(FileNotFoundError) as cm:
                h.export(backend="does_not_exist")
            self.assertIn("Available", str(cm.exception))


class Provenance(unittest.TestCase):
    def test_sidecar_records_the_clock_assumption(self):
        with tempfile.TemporaryDirectory() as d:
            h = Harness(d, ops=[(0, "conv2d_s8")], profiled_ids=[0])
            h.export(clock_mhz=1000.0)
            p = h.provenance()
            self.assertEqual(p["timing_source"], "firesim_measured")
            self.assertEqual(p["clock_mhz_assumed"], 1000.0)
            self.assertAlmostEqual(p["scaling_factor_cycles_to_ms"], 1e-6)
            # The note must say this is not the FPGA frequency, because the
            # bitstreams close timing at 25-30 MHz.
            self.assertIn("NOT the Alveo", p["clock_note"])
            self.assertIn("cycles", p["clock_note"])

    def test_clock_changes_derived_ms_but_not_raw_cycles(self):
        """Raw cycles are the measurement; ms is derived. Changing the assumed
        clock must move only the derived column."""
        with tempfile.TemporaryDirectory() as d:
            h = Harness(d, ops=[(0, "conv2d_s8")], profiled_ids=[0])
            h.export(clock_mhz=1000.0)
            fast = h.rows()[0]
            h.export(clock_mhz=25.0)
            slow = h.rows()[0]
            self.assertEqual(fast["cycles"], slow["cycles"])
            self.assertAlmostEqual(
                float(slow["mean_time"]) / float(fast["mean_time"]), 40.0, places=6
            )

    def test_zero_cost_ids_are_listed_in_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            h = Harness(
                d,
                ops=[(0, "conv2d_s8"), (1, "view"), (2, "chunk2_c1")],
                profiled_ids=[0],
            )
            h.export()
            p = h.provenance()
            self.assertEqual(p["zero_cost_dispatch_ids"], [1, 2])
            self.assertEqual(p["zero_cost_ops"], ["chunk2_c1", "view"])
            self.assertEqual(p["n_profiled"], 1)


class LoaderCompatibility(unittest.TestCase):
    def test_emitted_csv_is_readable_by_profile_loader(self):
        """The bridge exists to feed profile_loader; assert that end to end
        rather than trusting the column list."""
        sys.path.insert(0, os.path.dirname(_HERE))
        from profile_loader import find_profile_csv, load_profiled_times

        with tempfile.TemporaryDirectory() as d:
            h = Harness(
                d,
                ops=[(0, "conv2d_s8"), (1, "chunk2_c1")],
                profiled_ids=[0],
            )
            h.export()
            # find_profile_csv looks under <root>/gen/profile/...
            found = find_profile_csv(
                d,
                model=h.model,
                target=h.target,
                hw="gemmini",
                basename=f"{h.model}.{h.quant}",
                topo_tag="topo_0",
            )
            self.assertIsNotNone(found, "profile_loader must locate the emitted CSV")
            times = load_profiled_times(found)
            self.assertEqual(sorted(times), [0, 1])
            self.assertAlmostEqual(times[0]["time_ms"], 0.00105, places=9)
            self.assertEqual(times[1]["time_ms"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
