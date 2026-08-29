"""The `results.csv` profile producer, and the join key that survives a rewrite.

Two regressions are guarded here.

**1. `compile_advice` could only read one of the two profile producers.**
`load_profiles` reads `profile.jsonl`, which the retired IREE path wrote.
ModelBlaster's `pipeline/profile_writer.py` writes an IREE-shape `results.csv`
at a path with an extra spec directory, and the corrected `rvv_x60` builds --
the ones where curated kernels are no longer silently falling back to the scalar
reference -- exist ONLY in that format. Regenerating advice against the
corrected costs was therefore impossible, and the advisor kept citing costs from
a build labelled `rvv` that had run `scalar`.

**2. `cv_pct` must stay absent for a single-sample profile.** `results.csv`
carries one `mean_time` per dispatch. Inventing a `cv_pct` of 0 would promote
every recommendation to "high" confidence on n=1. The advisors default a missing
`cv_pct` to 100, which yields "medium" -- so the correct behaviour is to omit
the key, and a well-meaning later edit that fills it in is a silent
overclaim.

**3. The lineage join key.** `apply_fusion_hint` / `apply_split_hint` reassign
dispatch ids contiguously, so an id join across a rewrite compares different
ops. The op signature carried inside `module_name` is what survives.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "xpu-rt"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import compile_advice as ca  # noqa: E402

_HEADER = ("dispatch_id,module_name,vmfb_path,mlir_path,mean_time,mean_unit,"
           "mean_time_ns,returncode,log_path,source,op,shape,cycles,"
           "implementation")


def _write_csv(root, impl, target, model, basename, topo, rows, spec=True):
    """Lay a results.csv down in the real directory shape and return its path."""
    parts = [root, "profile", impl, target, model, basename]
    if spec:
        parts.append(f"{model}_{target}_{impl}_{basename}")
    parts.append(topo)
    d = os.path.join(*parts)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "results.csv")
    with open(p, "w") as f:
        f.write(_HEADER + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    return p


def _row(did, op, shape, ms, impl, model="m", backend="rvv_x60"):
    tag = "x".join(kv.replace("=", "") for kv in shape.split(";") if kv) or "noshape"
    return (did, f"{model}$dispatch_{did}_{backend}_{op}_{tag}", "", "",
            f"{ms:.6f}", "ms", f"{ms * 1e6:.6f}", 0, "", "k1", op, shape,
            int(ms * 24000), impl)


class ResultsCsvLoaderTest(unittest.TestCase):

    def test_reads_the_spec_directory_layout(self):
        with tempfile.TemporaryDirectory() as t:
            _write_csv(t, "rvv_x60", "spacemit_x60", "m", "m.int8", "topo_0",
                       [_row(0, "linear_s8", "M=1;K=16;N=256", 0.028667,
                             "curated[rvv]/direct")])
            got = ca.load_profiles_csv(t, "spacemit_x60", "m", "m.int8",
                                       ["rvv_x60", "scalar"])
            self.assertEqual(sorted(got), ["rvv_x60"])
            rec = got["rvv_x60"][0]
            self.assertAlmostEqual(rec["median_ms"], 0.028667)
            self.assertEqual(rec["implementation"], "curated[rvv]/direct")
            self.assertEqual(rec["n_cores"], 1)

    def test_reads_the_layout_without_a_spec_directory(self):
        with tempfile.TemporaryDirectory() as t:
            _write_csv(t, "rvv_x60", "spacemit_x60", "m", "m.int8", "topo_0",
                       [_row(0, "elu_s8", "n=256", 0.015125, "curated[rvv]/lut")],
                       spec=False)
            got = ca.load_profiles_csv(t, "spacemit_x60", "m", "m.int8",
                                       ["rvv_x60"])
            self.assertEqual(len(got["rvv_x60"]), 1)

    def test_cv_pct_is_absent_so_confidence_stays_medium(self):
        with tempfile.TemporaryDirectory() as t:
            _write_csv(t, "rvv_x60", "spacemit_x60", "m", "m.int8", "topo_0",
                       [_row(0, "linear_s8", "M=1;K=16;N=256", 1.0,
                             "curated[rvv]/direct")])
            _write_csv(t, "scalar", "spacemit_x60", "m", "m.int8", "topo_0",
                       [_row(0, "linear_s8", "M=1;K=16;N=256", 0.5, "",
                             backend="scalar")])
            profs = ca.load_profiles_csv(t, "spacemit_x60", "m", "m.int8",
                                         ["rvv_x60", "scalar"])
            self.assertNotIn("cv_pct", profs["rvv_x60"][0])
            self.assertEqual(profs["rvv_x60"][0]["stat_basis"],
                             "single_sample_mean")
            adv = ca.implementation_advice("m", profs, "rvv_x60")
            self.assertEqual(len(adv), 1)
            self.assertEqual(adv[0].recommendation, "choose_implementation")
            self.assertEqual(adv[0].confidence, "medium")
            ev = adv[0].as_dict()["evidence"]
            self.assertEqual(ev["baseline_kernel"], "curated[rvv]/direct")
            self.assertEqual(ev["stat_basis"], "single_sample_mean")

    def test_negative_result_also_reports_medium_on_one_sample(self):
        with tempfile.TemporaryDirectory() as t:
            _write_csv(t, "rvv_x60", "spacemit_x60", "m", "m.int8", "topo_0",
                       [_row(0, "linear_s8", "M=1;K=16;N=256", 1.0,
                             "curated[rvv]/direct")])
            profs = ca.load_profiles_csv(t, "spacemit_x60", "m", "m.int8",
                                         ["rvv_x60"])
            adv = ca.implementation_advice("m", profs, "rvv_x60")
            self.assertEqual(adv[0].recommendation, "unchanged")
            self.assertEqual(adv[0].confidence, "medium")

    def test_one_core_count_cannot_justify_a_shard(self):
        """A tree profiled at one core count cannot support `shard`.

        `shard` is a claim about how cost changes with core count. With a single
        topo there is nothing to measure, so the only legitimate output is
        `unchanged` carrying "no multi-core profile" -- never a shard justified
        by op size alone. `emit_compile_advice` does not even call this when
        fewer than two core counts exist, so in a real run the model is silent
        rather than recording a refusal it was never asked for.
        """
        with tempfile.TemporaryDirectory() as t:
            _write_csv(t, "rvv_x60", "spacemit_x60", "m", "m.int8", "topo_0",
                       [_row(0, "conv2d_s8", "OC=32", 50.0, "curated[rvv]/x")])
            by_cores = ca.load_profiles_by_cores_csv(t, "spacemit_x60", "m",
                                                     "m.int8", "rvv_x60")
            self.assertEqual(sorted(by_cores), [1])
            adv = ca.shard_advice("m", by_cores, free_slot_ms=1.0)
            self.assertEqual([a.recommendation for a in adv], ["unchanged"])
            self.assertIn("no multi-core profile",
                          adv[0].evidence.extra["detail"])

    def test_core_count_comes_from_the_topo_tag(self):
        with tempfile.TemporaryDirectory() as t:
            for topo, ms in (("topo_0", 8.0), ("topo_0_1_2_3", 2.5)):
                _write_csv(t, "rvv_x60", "spacemit_x60", "m", "m.int8", topo,
                           [_row(0, "conv2d_s8", "OC=32", ms, "curated[rvv]/x")])
            by_cores = ca.load_profiles_by_cores_csv(t, "spacemit_x60", "m",
                                                     "m.int8", "rvv_x60")
            self.assertEqual(sorted(by_cores), [1, 4])
            adv = ca.shard_advice("m", by_cores, free_slot_ms=4.0)
            self.assertEqual([a.recommendation for a in adv], ["shard"])
            self.assertEqual(adv[0].constraints["n_cores"], 4)


class GranularityGateTest(unittest.TestCase):
    """`scripts/diff_dispatch_graph.py` must refuse a no-op rewrite.

    The precedent it exists for: `gen/vmfb/mlp/spacemit_x60/RVV_fused/` holds
    the same five dispatch names as its baseline. Nothing fused, and the run was
    recorded as a fusion result.
    """

    _SCRIPT = os.path.join(_ROOT, "scripts", "diff_dispatch_graph.py")

    @staticmethod
    def _graph(ops):
        return {"name": "g", "ops": ops}

    def _run(self, before, after):
        with tempfile.TemporaryDirectory() as t:
            pb = os.path.join(t, "before.json")
            pa = os.path.join(t, "after.json")
            json.dump(before, open(pb, "w"))
            json.dump(after, open(pa, "w"))
            r = subprocess.run(
                [sys.executable, self._SCRIPT, "--before", pb, "--after", pa],
                capture_output=True, text=True)
            return r.returncode, r.stdout

    def test_identical_graph_fails_the_gate(self):
        g = self._graph([
            {"dispatch_id": 0, "op": "linear_s8", "shape": {"M": 1, "K": 16, "N": 256}},
            {"dispatch_id": 1, "op": "elu_s8", "shape": {"n": 256}},
        ])
        code, out = self._run(g, json.loads(json.dumps(g)))
        self.assertEqual(code, 3, out)
        self.assertIn("GRANULARITY UNCHANGED", out)

    def test_renumbering_alone_still_fails_the_gate(self):
        """Ids shift on every rewrite. A shift is not a granularity change.

        This is the case a `dispatch_id` join would report as two removals and
        two additions.
        """
        before = self._graph([
            {"dispatch_id": 0, "op": "linear_s8", "shape": {"M": 1, "K": 16, "N": 256}},
            {"dispatch_id": 1, "op": "elu_s8", "shape": {"n": 256}},
        ])
        after = self._graph([
            {"dispatch_id": 7, "op": "linear_s8", "shape": {"M": 1, "K": 16, "N": 256}},
            {"dispatch_id": 9, "op": "elu_s8", "shape": {"n": 256}},
        ])
        code, out = self._run(before, after)
        self.assertEqual(code, 3, out)

    def test_real_fusion_passes_the_gate(self):
        before = self._graph([
            {"dispatch_id": 0, "op": "linear_s8", "shape": {"M": 1, "K": 16, "N": 256}},
            {"dispatch_id": 1, "op": "elu_s8", "shape": {"n": 256}},
        ])
        after = self._graph([
            {"dispatch_id": 0, "op": "linear_s8_elu_s8", "shape": None,
             "sub_ops": [
                 {"op": "linear_s8", "shape": {"M": 1, "K": 16, "N": 256}},
                 {"op": "elu_s8", "shape": {"n": 256}}]},
        ])
        code, out = self._run(before, after)
        self.assertEqual(code, 0, out)
        self.assertIn("GRANULARITY CHANGED", out)
        self.assertIn("op count delta: -1", out)

    def test_repeated_signatures_are_counted_not_deduplicated(self):
        """Three identical maxpools are three dispatches, not one."""
        three = [{"dispatch_id": i, "op": "maxpool2d_s8",
                  "shape": {"C": 128, "K": 5}} for i in range(3)]
        two = three[:2]
        code, out = self._run(self._graph(three), self._graph(two))
        self.assertEqual(code, 0, out)
        self.assertIn("op count delta: -1", out)


if __name__ == "__main__":
    unittest.main()
