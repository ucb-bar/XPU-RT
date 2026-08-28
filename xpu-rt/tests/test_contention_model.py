"""Tests for the measured co-runner contention model (xpu-rt/contention_model.py)
and its additive, off-by-default wiring into the greedy scheduler.

The load-bearing test in here is `test_cross_cluster_is_worse_than_same_cluster`,
which pins a hardware measurement that contradicts the obvious intuition. Read
its comment before "fixing" anything about cluster placement.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import contention_model  # noqa: E402
import greedy_scheduler  # noqa: E402
from greedy_scheduler import greedy_schedule  # noqa: E402
from workload import Operation, Workload  # noqa: E402


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
MEASURED_ARTIFACT = os.path.join(REPO_ROOT, "artifacts", "k1_run", "contention.json")


def _v2_artifact(same=1.20, other=1.50, per_module=None):
    """Minimal v2 artifact with two placements."""
    def _meas(placement, cpus, ratio, mods):
        pm = {m: {"solo_ms": 1.0, "co_ms": r, "ratio": r} for m, r in mods.items()}
        return {
            "placement": placement,
            "cpu_under_test": 0,
            "co_cpus": cpus,
            "n_co_runners": len(cpus),
            "co_runner": {"remote_dir": "/root/mb_k1/bench/dronet_RVV",
                          "build": "RVV"},
            "per_module": pm,
            "median_ratio": ratio,
        }

    mods = per_module or {}
    return {
        "schema": contention_model.SCHEMA,
        "host": "k1",
        "cores_per_cluster": 4,
        "cpu_under_test": 0,
        "solo_ms": {},
        "measurements": {
            "same_cluster": _meas("same_cluster", [1], same, mods),
            "other_cluster": _meas("other_cluster", [4], other, mods),
        },
    }


def _write(tmpdir, data, name="contention.json"):
    path = os.path.join(tmpdir, name)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _two_op_workload(dur_p=10.0, dur_e=20.0):
    """A->B chain over singleton CPU_P / CPU_E combinations."""
    op_a = Operation(processing_times=[dur_p, dur_e], operation_name="dronet0_dispatch_12")
    op_b = Operation(
        processing_times=[dur_p, dur_e],
        predecessors=[op_a],
        operation_name="dronet0_dispatch_13",
    )
    return Workload([op_a, op_b], ["CPU_P", "CPU_E"], np.zeros((2, 2)))


class TestCanonicalKey(unittest.TestCase):
    """One dispatch has three spellings across profile, trace and board."""

    def test_all_spellings_agree(self):
        for name in (
            "module_dronet$async_dispatch_12_embedded_elf_riscv_64_benchmark.vmfb",
            "dronet0_dispatch_12",
            "dronet_dispatch_12",
            "dronet3_dispatch_12",
        ):
            self.assertEqual(contention_model.canonical_key(name), "dronet:12", name)

    def test_version_digits_survive(self):
        self.assertEqual(
            contention_model.canonical_key("yolov8_dispatch_3"), "yolov8:3"
        )

    def test_operation_object(self):
        op = Operation(processing_times=[1.0], operation_name="dronet0_dispatch_12")
        self.assertEqual(contention_model.canonical_key(op), "dronet:12")

    def test_unknown_is_not_an_error(self):
        self.assertEqual(contention_model.canonical_key("whatever"), "whatever")
        self.assertEqual(contention_model.canonical_key(None), "")


class TestMissingArtifactIsANoOp(unittest.TestCase):
    """The whole reason this can be wired in additively."""

    def test_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(contention_model.load(os.path.join(d, "nope.json")))

    def test_load_garbage_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "contention.json")
            with open(path, "w") as f:
                f.write("{not json")
            self.assertIsNone(contention_model.load(path))

    def test_factor_without_model_is_one(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(
                contention_model.contention_factor(
                    "dronet0_dispatch_12", "same_cluster",
                    model=contention_model.load(os.path.join(d, "nope.json")),
                ),
                1.0,
            )

    def test_config_flag_off_by_default(self):
        self.assertIsNone(contention_model.load_if_enabled(None))
        self.assertIsNone(contention_model.load_if_enabled({}))
        self.assertIsNone(
            contention_model.load_if_enabled({"contention": {"enabled": False}})
        )

    def test_scheduler_unchanged_when_no_model(self):
        """Missing artifact => byte-identical schedule."""
        with tempfile.TemporaryDirectory() as d:
            missing = contention_model.load(os.path.join(d, "nope.json"))
            t_base, a_base = greedy_schedule(_two_op_workload())
            t_off, a_off = greedy_schedule(_two_op_workload(), contention=missing)
            np.testing.assert_array_equal(t_base, t_off)
            np.testing.assert_array_equal(a_base, a_off)


class TestKnownFactorIsApplied(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = _write(
            self.tmp.name,
            _v2_artifact(
                same=1.20,
                other=1.50,
                per_module={
                    "module_dronet$async_dispatch_12_embedded_elf_riscv_64_benchmark.vmfb": 1.60,
                },
            ),
        )
        self.model = contention_model.load(self.path)

    def tearDown(self):
        self.tmp.cleanup()
        greedy_scheduler.configure_contention(None)

    def test_per_dispatch_factor_beats_median(self):
        # dispatch 12 was measured individually at 1.60x; 13 was not, so it
        # falls back to the placement median.
        self.assertAlmostEqual(
            self.model.contention_factor("dronet0_dispatch_12", "same_cluster"), 1.60
        )
        self.assertAlmostEqual(
            self.model.contention_factor("dronet0_dispatch_13", "same_cluster"), 1.20
        )

    def test_solo_placement_is_neutral(self):
        self.assertEqual(
            self.model.contention_factor("dronet0_dispatch_12", "solo"), 1.0
        )
        self.assertEqual(
            self.model.contention_factor("dronet0_dispatch_12", None), 1.0
        )

    def test_unknown_placement_is_neutral(self):
        self.assertEqual(
            self.model.contention_factor("dronet0_dispatch_12", "on_the_moon"), 1.0
        )

    def test_co_runner_identity_and_count_are_recorded(self):
        self.assertEqual(self.model.n_co_runners("same_cluster"), 1)
        self.assertEqual(self.model.co_runner("same_cluster")["build"], "RVV")

    def test_scheduler_durations_are_scaled(self):
        """Same workload, contention on => strictly later completion."""
        t_off, _ = greedy_schedule(_two_op_workload())
        t_on, _ = greedy_schedule(
            _two_op_workload(),
            contention=self.model,
            contention_placement="same_cluster",
        )
        # Op A (dispatch 12) is scaled 1.60x, so B starts 6ms later.
        self.assertAlmostEqual(float(t_on[1]) - float(t_off[1]), 10.0 * 0.60, places=6)

    def test_contention_never_mutates_the_solo_profile(self):
        """The multiplier is applied at lookup time only. If it were folded
        into the profile, a second schedule would compound it."""
        wl = _two_op_workload()
        before = list(wl.operations[0].processing_times)
        greedy_schedule(wl, contention=self.model,
                        contention_placement="same_cluster")
        self.assertEqual(wl.operations[0].processing_times, before)
        t_after, _ = greedy_schedule(wl)
        self.assertAlmostEqual(float(t_after[1]), 10.0, places=6)

    def test_global_configure_is_reverted_after_a_scoped_call(self):
        greedy_scheduler.configure_contention(None)
        greedy_schedule(_two_op_workload(), contention=self.model,
                        contention_placement="same_cluster")
        self.assertEqual(greedy_scheduler.get_contention(), (None, None))

    def test_config_flag_on(self):
        model = contention_model.load_if_enabled(
            {"contention": {"enabled": True, "path": self.path}}
        )
        self.assertIsNotNone(model)
        self.assertIn("same_cluster", model.placements())


class TestPlacementsDiffer(unittest.TestCase):
    def test_same_and_cross_cluster_give_different_factors(self):
        with tempfile.TemporaryDirectory() as d:
            model = contention_model.load(_write(d, _v2_artifact(same=1.20, other=1.50)))
            same = model.contention_factor("dronet0_dispatch_12", "same_cluster")
            other = model.contention_factor("dronet0_dispatch_12", "other_cluster")
            self.assertNotAlmostEqual(same, other)
            self.assertAlmostEqual(same, 1.20)
            self.assertAlmostEqual(other, 1.50)

    def test_placement_derived_from_machine_combination(self):
        """A combination spanning both clusters is cross-cluster work."""
        with tempfile.TemporaryDirectory() as d:
            model = contention_model.load(_write(d, _v2_artifact()))
            self.assertEqual(
                model.placement_for_combination(["CPU_P"]), "same_cluster"
            )
            self.assertEqual(
                model.placement_for_combination(["CPU_P", "CPU_E"]), "other_cluster"
            )
            # An unmapped machine yields no placement, so the duration is left
            # alone rather than guessed at.
            self.assertIsNone(model.placement_for_combination(["NPU"]))

    def test_derived_placement_changes_the_schedule(self):
        with tempfile.TemporaryDirectory() as d:
            model = contention_model.load(_write(d, _v2_artifact(same=1.20, other=1.50)))
            wl = Workload(
                [Operation(processing_times=[10.0, 10.0, 10.0],
                           operation_name="dronet0_dispatch_12")],
                ["CPU_P", "CPU_E"],
                np.zeros((2, 2)),
                machine_combinations=[["CPU_P"], ["CPU_E"], ["CPU_P", "CPU_E"]],
            )
            self.assertAlmostEqual(
                greedy_scheduler._duration(
                    wl.operations[0], 0, wl.get_machine_combinations(), wl.machines
                ), 10.0)
            with greedy_scheduler._contention_scope(model):
                mc = wl.get_machine_combinations()
                single = greedy_scheduler._duration(
                    wl.operations[0], 0, mc, wl.machines)
                spread = greedy_scheduler._duration(
                    wl.operations[0], 2, mc, wl.machines)
            self.assertAlmostEqual(single, 12.0)
            self.assertAlmostEqual(spread, 15.0)
            # The sharded/cross-cluster placement is the more expensive one --
            # see TestMeasuredInversion for why that is not a typo.
            self.assertGreater(spread, single)


class TestMeasuredInversion(unittest.TestCase):
    """REGRESSION PIN: on the K1 (SpaceMiT X60, 8 harts, 2 clusters of 4, one
    512K L2 per cluster) a co-runner on the OTHER cluster hurts MORE than a
    co-runner sharing your own L2:

        median slowdown, co-runner on the SAME cluster  : 1.043x
        median slowdown, co-runner on the OTHER cluster : 1.185x

    This contradicts the shared-L2 intuition that k1_contention.py's own
    docstring used to assert -- if the L2 were the bottleneck, sharing it would
    be the expensive case and spreading across clusters would be free. It is
    the other way round, so "spread work across clusters" is the WRONG default
    on this board, and any scheduler heuristic that spreads shards across
    clusters to 'avoid cache contention' is optimising backwards.

    If this test fails because the measured artifact changed, do NOT flip the
    assertion to match a theory -- re-run runtime/scripts/k1_contention.py and
    update the numbers from the board.
    """

    def test_measured_artifact_shows_cross_cluster_worse(self):
        if not os.path.exists(MEASURED_ARTIFACT):
            self.skipTest(f"no measured artifact at {MEASURED_ARTIFACT}")
        model = contention_model.load(MEASURED_ARTIFACT)
        self.assertIsNotNone(model)
        same = model.median_factor("same_cluster")
        other = model.median_factor("other_cluster")
        self.assertGreater(other, same,
                           "cross-cluster contention must stay worse than "
                           "same-cluster on this board; see docstring")
        # Both placements cost something, and the gap is large enough to be a
        # real effect rather than run-to-run noise (measured 1.043 vs 1.185).
        self.assertGreater(same, 1.0)
        self.assertGreater(other / same, 1.05)

    def test_v1_artifact_is_readable(self):
        """The run that found the inversion was written in the flat v1 shape.
        The loader must keep reading it, or the evidence becomes unreadable."""
        v1 = {
            "results": [
                {"module": "module_dronet$async_dispatch_12_embedded_elf_riscv_64_benchmark.vmfb",
                 "solo_ms": 9.581, "same_cluster_ms": 10.089,
                 "other_cluster_ms": 11.817,
                 "same_ratio": 1.0530216052604113,
                 "other_ratio": 1.2333785617367707},
            ],
            "median_same_cluster_ratio": 1.0434858585769335,
            "median_other_cluster_ratio": 1.1852990729103046,
            "cpu": 0, "same_cluster_cpu": 1, "other_cluster_cpu": 4,
        }
        with tempfile.TemporaryDirectory() as d:
            model = contention_model.load(_write(d, v1))
            self.assertEqual(
                sorted(model.placements()), ["other_cluster", "same_cluster"])
            self.assertAlmostEqual(model.median_factor("same_cluster"), 1.043486, places=5)
            self.assertAlmostEqual(model.median_factor("other_cluster"), 1.185299, places=5)
            self.assertGreater(model.median_factor("other_cluster"),
                               model.median_factor("same_cluster"))
            self.assertAlmostEqual(
                model.contention_factor("dronet0_dispatch_12", "same_cluster"),
                1.0530216052604113)
            self.assertEqual(model.n_co_runners("same_cluster"), 1)


if __name__ == "__main__":
    unittest.main()
