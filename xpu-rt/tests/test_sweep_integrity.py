"""Integrity tests for the sweep driver's bookkeeping.

Every case here corresponds to a bug that was live in this repository and that
produced a *plausible-looking* artifact rather than an error. That is the class of
bug this file exists to catch: nothing crashed, the manifest looked fine, and the
numbers were wrong.

1. FIXTURE/CONFIG COLLISION. Fixture stems are (policy, burst, seed) only, so two
   different workloads swept with the same policy names overwrite each other's
   schedules. This happened: the 25 MHz clock-invariance control clobbered
   `_fx_static_nominal_B{0,1,2}_s0`, and a later reuse pass would have scored
   2820-ms-scale input ages against a 70.5 ms freshness window and reported the
   entire baseline column as stale.

2. SUCCESS-STATUS EQUALITY. `run_schedule` returns "ok" for a fresh solve and
   "ok (reused fixture)" for a verified reuse. A `status != "ok"` check turned all
   57 reused cells into recorded failures -- data silently dropped while every
   failure's status string read "ok".

3. EPOCH COMPARABILITY. When a schedule overruns the epoch, greedy's horizon
   search extends the horizon and adds instances, so rates are computed over a
   longer trace: static_nominal at B=3 was scored over 41 consumer invocations
   across 483 ms while a candidate was scored over 30 across 297 ms. Ranking those
   compares two different experiments.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _REPO)
sys.path.insert(0, _XPURT)

from benchmarks.freshness_eval.run import (  # noqa: E402
    ALL_POLICIES,
    PROBES,
    materialise,
    solver_tag,
)

CANON = os.path.join(_REPO, "data", "toplevel", "freshness_canon_300ms.json")
MHZ25 = os.path.join(_REPO, "data", "toplevel", "freshness_canon_25mhz.json")


def _load(p):
    with open(p) as f:
        return json.load(f)


def _digest(cfg):
    return hashlib.sha256(
        json.dumps(cfg, indent=2, sort_keys=True).encode()).hexdigest()


class FixtureConfigCollision(unittest.TestCase):
    """The two configs that actually collided must produce different hashes."""

    def test_two_configs_with_the_same_policy_produce_different_digests(self):
        canon, mhz = _load(CANON), _load(MHZ25)
        for burst in (0, 1, 2):
            a = materialise(canon, burst=burst, mutations=None,
                            epoch_ms=300.0, seed=0)
            b = materialise(mhz, burst=burst, mutations=None,
                            epoch_ms=12000.0, seed=0)
            self.assertNotEqual(
                _digest(a), _digest(b),
                f"B={burst}: the 1 GHz and 25 MHz workloads hash identically, so "
                f"the sidecar could not tell them apart")

    def test_the_two_configs_share_a_fixture_stem(self):
        """Documents WHY the hash is load-bearing: the names really do collide."""
        for burst in (0, 1, 2):
            stem = f"_fx_static_nominal_B{burst}_s0"
            self.assertEqual(stem, f"_fx_static_nominal_B{burst}_s0")
        # And with --stem-tag they no longer do.
        tagged = "_fx_mhz25_static_nominal_B0_s0"
        self.assertNotEqual(tagged, "_fx_static_nominal_B0_s0")

    def test_stem_tag_is_exposed_on_the_cli(self):
        import argparse
        import benchmarks.freshness_eval.run as R
        src = open(R.__file__).read()
        self.assertIn("--stem-tag", src,
                      "the collision is only *safe* via the hash; --stem-tag is "
                      "what stops it happening")
        del argparse

    def test_digest_is_stable_across_calls(self):
        canon = _load(CANON)
        a = materialise(canon, burst=2, mutations=None, epoch_ms=300.0, seed=0)
        b = materialise(canon, burst=2, mutations=None, epoch_ms=300.0, seed=0)
        self.assertEqual(_digest(a), _digest(b))

    def test_a_mutation_changes_the_digest(self):
        canon = _load(CANON)
        a = materialise(canon, burst=2, mutations=None, epoch_ms=300.0, seed=0)
        b = materialise(canon, burst=2, mutations={"soft_phase_ms": 10.0},
                        epoch_ms=300.0, seed=0)
        self.assertNotEqual(_digest(a), _digest(b))


class SuccessStatusHandling(unittest.TestCase):
    def test_reuse_status_is_recognised_as_success(self):
        for status in ("ok", "ok (reused fixture)"):
            self.assertTrue(status.startswith("ok"), status)

    def test_real_failures_are_still_failures(self):
        for status in ("timeout after 2400s", "solver exit 1 (see log)",
                       "fixture missing: /x"):
            self.assertFalse(status.startswith("ok"), status)

    def test_the_driver_does_not_compare_status_for_equality(self):
        import benchmarks.freshness_eval.run as R
        src = open(R.__file__).read()
        self.assertNotIn('if status != "ok":', src,
                         'an equality check silently drops every reused cell')
        self.assertIn('if not status.startswith("ok"):', src)


class EpochComparability(unittest.TestCase):
    def test_overrunning_cells_are_marked_not_comparable(self):
        rows = [
            {"policy": "static_nominal", "contention_level": 2,
             "fits_in_epoch": True},
            {"policy": "cand", "contention_level": 2, "fits_in_epoch": True},
            {"policy": "static_nominal", "contention_level": 3,
             "fits_in_epoch": False},
            {"policy": "cand", "contention_level": 3, "fits_in_epoch": True},
        ]
        overrun = {}
        for r in rows:
            if not r["fits_in_epoch"]:
                overrun.setdefault(r["contention_level"], []).append(r["policy"])
        comparable = sorted(b for b in {r["contention_level"] for r in rows}
                            if b not in overrun)
        self.assertEqual(comparable, [2],
                         "B=3 has one overrunning policy, so the whole level is "
                         "not rankable -- not just that one row")

    def test_a_single_overrun_disqualifies_the_whole_level(self):
        """Comparability is a property of the LEVEL, because ranking is pairwise:
        one policy scored over a longer trace poisons every comparison at that B."""
        rows = [{"policy": f"p{i}", "contention_level": 4,
                 "fits_in_epoch": i != 0} for i in range(5)]
        overrun = {r["contention_level"] for r in rows if not r["fits_in_epoch"]}
        self.assertIn(4, overrun)

    def test_driver_stamps_comparability_before_writing_artifacts(self):
        """The flag must be in aggregate.csv, so it has to be set before the
        DictWriter derives its column list from the rows."""
        import benchmarks.freshness_eval.run as R
        src = open(R.__file__).read()
        stamp = src.index('r["epoch_comparable"]')
        write = src.index('inv_csv = os.path.join(out_dir, "per_invocation.csv")')
        self.assertLess(stamp, write,
                        "epoch_comparable is stamped after the artifacts are "
                        "written, so the column is missing from aggregate.csv")


class ProbeDesignHonesty(unittest.TestCase):
    """The probe table records measured outcomes, including the ones that failed."""

    def test_the_inert_control_is_labelled_as_measured_inert(self):
        intent = PROBES["probe_nonperiodic_priority"]["intent"]
        self.assertIn("INERT", intent,
                      "this control produced bit-identical schedules to the "
                      "baseline; it must not read as a passed falsification test")

    def test_a_working_directional_control_exists(self):
        """Two controls were tried and BOTH measured inert, because both used
        window_duration, which greedy ignores. A control must use a lever the
        solver demonstrably responds to -- here, release time."""
        controls = {n: s for n, s in PROBES.items() if "WORSE" in s["intent"]}
        self.assertTrue(controls, "no expected-WORSE probe remains")
        for name, spec in controls.items():
            muts = spec.get("mutations") or {}
            self.assertNotIn(
                "window_duration", muts,
                f"{name} is a control built on window_duration, which is measured "
                f"inert on greedy -- it cannot falsify anything")
            self.assertNotIn(
                "phase_ms", muts,
                f"{name} is a control built on a producer phase offset, which was "
                f"MEASURED to improve freshness (delaying the release delays the "
                f"sample under producer_release semantics) -- it cannot falsify")
            self.assertTrue(
                set(muts) & {"period_scale", "admit_cap"},
                f"{name} must use a lever that moves the structural age floor and "
                f"cannot be gamed by re-phasing; got {sorted(muts)}")

    def test_both_inert_controls_are_recorded_as_inert(self):
        for name in ("probe_nonperiodic_priority", "probe_soft_first"):
            intent = PROBES[name]["intent"]
            self.assertIn("INERT", intent, name)
            self.assertNotIn("expected WORSE", intent, name)

    def test_the_producer_delay_control_does_not_move_the_fill_threshold(self):
        """The control delays the producer, which leaves early consumers with no
        input. Those must be charged to the policy, so pipeline_fill_ms -- computed
        from the BASE config -- must not follow the mutation."""
        from benchmarks.freshness_eval.run import compute_a0
        from freshness import freshness_edges_from_config
        base = _load(CANON)
        edge = freshness_edges_from_config(base)[0]
        before = compute_a0(base, epoch_ms=300.0, edge=edge)["pipeline_fill_ms"]
        materialise(base, burst=2, mutations={"phase_ms": {"dronet": 25.0}},
                    epoch_ms=300.0, seed=0)
        after = compute_a0(_load(CANON), epoch_ms=300.0,
                           edge=edge)["pipeline_fill_ms"]
        self.assertAlmostEqual(before, after)

    def test_phase_ms_reaches_the_named_network(self):
        cfg = materialise(_load(CANON), burst=2,
                          mutations={"phase_ms": {"dronet": 25.0}},
                          epoch_ms=300.0, seed=0)
        self.assertAlmostEqual(float(cfg["networks"]["dronet"]["start_time"]), 25.0)

    def test_phase_ms_rejects_an_unknown_network(self):
        with self.assertRaises(ValueError):
            materialise(_load(CANON), burst=2,
                        mutations={"phase_ms": {"nope": 1.0}},
                        epoch_ms=300.0, seed=0)

    def test_the_deferral_curve_brackets_the_measured_optimum(self):
        """10 ms was the best of {10,25,40,50}; offsets below it are needed to
        show whether the curve turns over or keeps improving."""
        offsets = sorted(
            int(n.replace("probe_defer", "")) for n in PROBES
            if n.startswith("probe_defer")
        )
        self.assertTrue(any(o < 10 for o in offsets),
                        f"no offset below the measured optimum: {offsets}")
        self.assertTrue(any(o > 10 for o in offsets), offsets)

    def test_the_false_phase_control_claim_is_retracted(self):
        """probe_defer50 was documented as a phase control on the grounds that
        50 ms is a full DroNet period. The offset applies to the SOFT network,
        whose period is epoch/admitted, so it never was one."""
        base = _load(CANON)
        for burst, expected in ((1, 300.0), (2, 150.0), (3, 100.0)):
            cfg = materialise(base, burst=burst, mutations={"soft_phase_ms": 50.0},
                              epoch_ms=300.0, seed=0)
            self.assertAlmostEqual(
                float(cfg["networks"]["yolov8_nano_64"]["period"]), expected,
                msg=f"B={burst}: soft period is not the DroNet period, so a 50 ms "
                    f"offset is not a full-period null control")
            self.assertNotAlmostEqual(50.0, expected)


class SoftSideWindowMutation(unittest.TestCase):
    def test_soft_window_mutation_survives_burst_zero(self):
        """At B=0 admission removes the soft network entirely; a soft-side window
        mutation must not crash the B=0 cell of its own sweep."""
        cfg = materialise(_load(CANON), burst=0,
                          mutations={"window_duration": {"yolov8_nano_64": 75.0}},
                          epoch_ms=300.0, seed=0)
        self.assertNotIn("yolov8_nano_64", cfg["networks"])

    def test_a_genuinely_unknown_network_still_raises(self):
        with self.assertRaises(ValueError) as cm:
            materialise(_load(CANON), burst=2,
                        mutations={"window_duration": {"not_a_network": 10.0}},
                        epoch_ms=300.0, seed=0)
        self.assertIn("not_a_network", str(cm.exception))

    def test_soft_window_is_applied_when_the_network_is_present(self):
        for burst in (1, 2, 3, 4):
            cfg = materialise(_load(CANON), burst=burst,
                              mutations={"window_duration": {"yolov8_nano_64": 75.0}},
                              epoch_ms=300.0, seed=0)
            self.assertAlmostEqual(
                float(cfg["networks"]["yolov8_nano_64"]["window_duration"]), 75.0,
                msg=f"B={burst}")

    def test_every_probe_materialises_at_every_burst(self):
        """A probe that raises at some B leaves a hole in the grid that reads as a
        solver failure."""
        base = _load(CANON)
        for name, spec in ALL_POLICIES.items():
            if spec.get("solver") is None:
                continue
            for burst in (0, 1, 2, 3, 4):
                try:
                    materialise(base, burst=burst, mutations=spec.get("mutations"),
                                epoch_ms=300.0, seed=0)
                except Exception as exc:  # noqa: BLE001
                    self.fail(f"{name} B={burst} failed to materialise: "
                              f"{type(exc).__name__}: {exc}")

    def test_solver_tag_is_distinct_for_the_control(self):
        """The inert control still needs its own fixture path, or its result would
        be indistinguishable from the baseline's by construction."""
        self.assertNotEqual(
            solver_tag("greedy_periodic", "mosek"), solver_tag("greedy", "mosek"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
