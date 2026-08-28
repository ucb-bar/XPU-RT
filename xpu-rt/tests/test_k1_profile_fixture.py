"""The K1 (`spacemit_x60`) profile parsers, against a committed real profile.

WHY THIS FILE EXISTS. Every K1 measurement in this tree lives under
`gen/profile_mb/`, which is untracked *and* unreachable: the loaders glob
`<gen_root>/profile/...`, so a directory named `profile_mb` cannot be addressed
by any `gen_root` (documented in `docs/k1_modelblaster_xpurt_closed_loop.md`,
§"`PROFILE_OUT_ROOT` must end in `profile`"). The consequence was that the
ModelBlaster/K1 side of the profile parsers -- a different directory depth, a
different module-name convention, and a 14th column that the older schema does
not have -- had no test data at all, so nothing exercised it.

`fixtures/k1_profile/` is that data: a verbatim copy of two real 21-dispatch
DroNet profiles (rvv_x60 and scalar), placed at a `gen_root`-addressable path.
See its README for exactly what is measured and what is derived.

The defects these tests guard, in the order they would bite:

* **The `<input_tag>` directory level.** ModelBlaster's writer interposes
  `<model>_<target>_<impl>_<basename>/` between the basename and the topo tag;
  the IREE-era writer does not. A loader that handles only one layout silently
  finds nothing, and "no profile" is indistinguishable from "no advice".

* **The `implementation` column.** It records which kernel actually ran. It
  exists because curated kernels are looked up by exact op name, so an op with
  no curated entry fell back to the scalar reference *inside a build labelled
  `rvv_x60`*. A profile that cannot say which kernel it timed cannot support
  advice about that kernel -- and a parser that chokes on the extra column, or
  drops it, removes the only way to notice.

* **Unit handling.** `mean_time` is accompanied by `mean_unit`. Reading the
  number without the unit is a 1000x error that looks like a plausible schedule.

* **Two schema generations coexisting.** The scalar profile in the fixture has
  13 columns and no `implementation`; the rvv_x60 one has 14. Both are on disk
  in real trees and both have to parse.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _REPO)
sys.path.insert(0, _XPURT)

import compile_advice  # noqa: E402
from profile_loader import find_profile_csv, load_profiled_times  # noqa: E402

FIXTURE = os.path.join(_HERE, "fixtures", "k1_profile")
LOOKUP = dict(model="dronet", target="spacemit_x60", basename="dronet.int8",
              topo_tag="topo_0")

#: Facts about the measurement itself, so a fixture edit that changed the data
#: cannot pass unnoticed as a parser test.
N_DISPATCHES = 21
RVV_TOTAL_MS = 9.789
SCALAR_TOTAL_MS = 157.377
#: dispatch 1, `maxpool2d_s8` 56x56 -> 27x27. A mid-sized row, so a us/ms
#: confusion or an off-by-one row offset both show up.
RVV_DISPATCH_1_MS = 0.80175


class TheFixtureIsFindable(unittest.TestCase):
    """No `skipUnless`: this data is committed, which is the point of it."""

    def test_the_modelblaster_input_tag_layout_resolves(self):
        p = find_profile_csv(FIXTURE, hw="rvv_x60", **LOOKUP)
        self.assertIsNotNone(
            p, "the <input_tag> level between basename and topo_tag must be "
               "globbed; ModelBlaster's writer always emits it")
        self.assertTrue(p.endswith(os.path.join("topo_0", "results.csv")))
        self.assertIn("dronet_spacemit_x60_rvv_x60_dronet.int8", p)

    def test_both_implementations_resolve_independently(self):
        rvv = find_profile_csv(FIXTURE, hw="rvv_x60", **LOOKUP)
        scalar = find_profile_csv(FIXTURE, hw="scalar", **LOOKUP)
        self.assertNotEqual(rvv, scalar)
        self.assertIn(os.sep + "rvv_x60" + os.sep, rvv)
        self.assertIn(os.sep + "scalar" + os.sep, scalar)

    def test_a_wrong_hw_returns_none_rather_than_the_other_build(self):
        """Falling back to a neighbouring implementation is the worst outcome.

        It produces a full, plausible cost table measured on a binary nobody
        asked about. `None` lets the caller's strict mode raise instead.
        """
        self.assertIsNone(find_profile_csv(FIXTURE, hw="ime_x60", **LOOKUP))
        self.assertIsNone(find_profile_csv(FIXTURE, gen_root="gen_absent",
                                           hw="rvv_x60", **LOOKUP))

    def test_a_wrong_topo_tag_returns_none_rather_than_the_one_core_run(self):
        """`topo_0_1_2_3` is a four-core measurement; this tree has none.

        Returning the one-core profile for a four-core request would report a
        4x speedup that was never measured -- exactly the claim `shard_advice`
        is built to refuse.
        """
        self.assertIsNone(find_profile_csv(
            FIXTURE, hw="rvv_x60", model="dronet", target="spacemit_x60",
            basename="dronet.int8", topo_tag="topo_0_1_2_3"))


class TheSchedulerSideParser(unittest.TestCase):
    """`profile_loader.load_profiled_times` -- what the solver's costs come from."""

    def _load(self, hw):
        path = find_profile_csv(FIXTURE, hw=hw, **LOOKUP)
        self.assertIsNotNone(path, hw)
        return load_profiled_times(path)

    def test_the_14_column_schema_parses(self):
        prof = self._load("rvv_x60")
        self.assertEqual(len(prof), N_DISPATCHES)
        self.assertEqual(sorted(prof), list(range(N_DISPATCHES)))
        self.assertAlmostEqual(prof[1]["time_ms"], RVV_DISPATCH_1_MS, places=6)
        self.assertAlmostEqual(sum(r["time_ms"] for r in prof.values()),
                               RVV_TOTAL_MS, places=3)

    def test_the_13_column_schema_parses_too(self):
        """The scalar profile predates the `implementation` column.

        A parser that started requiring it would drop every older profile in the
        tree, and `strict=True` would then declare the network unprofiled.
        """
        prof = self._load("scalar")
        self.assertEqual(len(prof), N_DISPATCHES)
        self.assertAlmostEqual(sum(r["time_ms"] for r in prof.values()),
                               SCALAR_TOTAL_MS, places=3)

    def test_module_names_carry_the_backend_tag(self):
        """It is what tells two builds of the same graph apart downstream.

        Both profiles have dispatch_ids 0..20 and identical shapes, so the only
        thing distinguishing a scalar cost from a vector one in the schedule JSON
        is the tag inside `module_name`.
        """
        rvv = self._load("rvv_x60")
        scalar = self._load("scalar")
        self.assertIn("_rvv_x60_", rvv[0]["module_name"])
        self.assertIn("_scalar_", scalar[0]["module_name"])
        self.assertNotEqual(rvv[0]["module_name"], scalar[0]["module_name"])

    def test_the_unit_column_is_honoured(self):
        """A `us` row must come back 1000x smaller, not 1000x larger.

        Derived from the real fixture row by rewriting the unit and the number
        consistently, so the expected value is the fixture's own measurement.
        """
        src = find_profile_csv(FIXTURE, hw="rvv_x60", **LOOKUP)
        with open(src, newline="") as f:
            rows = list(csv.DictReader(f))
            fields = list(rows[0])
        for row in rows:
            row["mean_time"] = f"{float(row['mean_time']) * 1000.0:.6f}"
            row["mean_unit"] = "us"
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "results.csv")
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            prof = load_profiled_times(p)
        self.assertAlmostEqual(prof[1]["time_ms"], RVV_DISPATCH_1_MS, places=6)

    def test_a_missing_file_is_an_empty_profile_not_an_exception(self):
        """`_load_all_topo_profiles` relies on this to aggregate every gap
        before raising, so one missing tree does not hide the other four."""
        self.assertEqual(load_profiled_times(os.path.join(FIXTURE, "nope.csv")),
                         {})


class TheAdvisorSideParser(unittest.TestCase):
    """`compile_advice.load_profiles_csv` -- what the advice's evidence is."""

    def setUp(self):
        if not hasattr(compile_advice, "load_profiles_csv"):
            self.skipTest("compile_advice has no results.csv ingester")
        self.gen = os.path.join(FIXTURE, "gen")

    def test_both_implementations_load_under_one_call(self):
        profs = compile_advice.load_profiles_csv(
            self.gen, "spacemit_x60", "dronet", "dronet.int8",
            ["rvv_x60", "scalar", "ime_x60"])
        self.assertEqual(sorted(profs), ["rvv_x60", "scalar"],
                         "an implementation with no profile must be absent, "
                         "not present and empty")
        self.assertEqual(len(profs["rvv_x60"]), N_DISPATCHES)

    def test_the_kernel_that_actually_ran_is_carried_into_the_records(self):
        """The regression the `implementation` column was added for.

        Curated kernels are matched by exact op name, so an op without one fell
        back to the scalar reference inside a build labelled `rvv_x60`. If the
        parser drops this column, that fallback is invisible and the advice
        claims to be about a vector kernel that never ran.
        """
        profs = compile_advice.load_profiles_csv(
            self.gen, "spacemit_x60", "dronet", "dronet.int8", ["rvv_x60"])
        impls = {r["implementation"] for r in profs["rvv_x60"].values()}
        self.assertTrue(impls, "the implementation column must reach the record")
        self.assertTrue(all(i.startswith("curated[rvv]/") for i in impls), impls)
        # Five distinct curated kernels across 21 dispatches -- so this is not a
        # constant the parser could be inventing.
        self.assertGreaterEqual(len(impls), 4, impls)

    def test_the_older_schema_yields_a_blank_implementation_not_a_crash(self):
        profs = compile_advice.load_profiles_csv(
            self.gen, "spacemit_x60", "dronet", "dronet.int8", ["scalar"])
        self.assertEqual(len(profs["scalar"]), N_DISPATCHES)
        self.assertEqual({r["implementation"] for r in profs["scalar"].values()},
                         {""})

    def test_a_single_sample_profile_says_so(self):
        """`results.csv` has one `mean_time` per dispatch, no distribution.

        Carrying it under the key `median_ms` (which every advisor reads) is
        deliberate, but then the record has to state what the number IS, or
        advice drawn from n=1 is indistinguishable from advice drawn from a
        warm median.
        """
        profs = compile_advice.load_profiles_csv(
            self.gen, "spacemit_x60", "dronet", "dronet.int8", ["rvv_x60"])
        for did, r in profs["rvv_x60"].items():
            self.assertEqual(r["stat_basis"], "single_sample_mean", did)
            self.assertNotIn("cv_pct", r,
                             "a dispersion that was never measured must be "
                             "absent, not zero")

    def test_one_core_tree_can_never_justify_a_shard(self):
        """A tree profiled at one core count cannot support a shard.

        Sharding is a claim about how cost changes with core count, so with one
        core count measured the only honest answers are "no shard" and "here is
        why". This tree has only `topo_0`, so the loader must report exactly one
        core count, and every dispatch that overruns the slot must come back
        refused *with the reason* -- not recommended from op size, and not
        silently omitted either, since a later round would then re-propose it.
        """
        by_cores = compile_advice.load_profiles_by_cores_csv(
            self.gen, "spacemit_x60", "dronet", "dronet.int8", "rvv_x60")
        self.assertEqual(sorted(by_cores), [1])

        adv = compile_advice.shard_advice("dronet", by_cores, free_slot_ms=1.0)
        self.assertEqual([a for a in adv if a.recommendation == "shard"], [])
        # DroNet has three dispatches over 1 ms; each must say what was missing.
        self.assertEqual(len(adv), 3, [a.dispatch_id for a in adv])
        for a in adv:
            self.assertEqual(a.recommendation, "unchanged")
            self.assertEqual(a.evidence.extra["detail"], "no multi-core profile")


class TheDerivedJsonl(unittest.TestCase):
    """`profile.jsonl` in the fixture is generated from the CSV beside it.

    These tests pin the two honesty properties its README claims, because a
    later regeneration is where they would quietly be lost.
    """

    def _jsonl(self, impl, tag):
        p = os.path.join(FIXTURE, "gen", "profile", impl, "spacemit_x60",
                         "dronet", "dronet.int8", tag, "topo_0", "profile.jsonl")
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_medians_are_exactly_the_csv_numbers(self):
        """Derived means derived: no rounding, no rescaling, no clock guess."""
        recs = self._jsonl("rvv_x60", "dronet_spacemit_x60_rvv_x60_dronet.int8")
        csv_path = find_profile_csv(FIXTURE, hw="rvv_x60", **LOOKUP)
        with open(csv_path, newline="") as f:
            expect = {int(r["dispatch_id"]): float(r["mean_time"])
                      for r in csv.DictReader(f)}
        self.assertEqual(len(recs), N_DISPATCHES)
        for r in recs:
            self.assertEqual(r["median_ms"], expect[r["dispatch_id"]])

    def test_no_dispersion_is_invented(self):
        """`cv_pct` present would make `implementation_advice` claim
        `confidence="high"` off statistics nobody measured."""
        for impl, tag in (("rvv_x60", "dronet_spacemit_x60_rvv_x60_dronet.int8"),
                          ("scalar", "dronet_spacemit_x60_scalar_dronet.int8")):
            for r in self._jsonl(impl, tag):
                for absent in ("cv_pct", "samples_ms", "stdev_ms", "p99_ms"):
                    self.assertNotIn(absent, r, f"{impl}/{r['dispatch_id']}")

    def test_n_cores_is_recorded_so_a_multi_core_tree_can_be_keyed_on_it(self):
        """`load_profiles_by_cores` keys on the recorded `n_cores`, not the
        directory name, so the field has to be there even when it is 1."""
        recs = self._jsonl("rvv_x60", "dronet_spacemit_x60_rvv_x60_dronet.int8")
        self.assertEqual({r["n_cores"] for r in recs}, {1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
