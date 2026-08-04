"""Tests for the candidate-mutation vocabulary and the fixed-reference-window
deadline metric.

Two failure modes motivate this file, both already hit once in this project:

1. A DOCUMENTED-BUT-INERT knob. `num_instances` was honoured on some code paths
   and silently dropped on others; `preferred_hw` naming a cluster instead of a
   profile hw matched nothing and quietly penalised everything. Both produced
   plausible numbers from a config that was not doing what it said. So
   `test_every_documented_mutation_is_observable` asserts each key in
   MUTATION_KEYS changes the emitted config -- a probe built on an inert
   mutation would otherwise "work" and report the baseline's result under a
   different name.

2. A MOVING MEASUREMENT BAR. One protection mechanism tightens the producer's
   own window, which also moves that producer's `Invocation.deadline`. Comparing
   its `deadline_success_rate` against the baseline's then compares two
   different questions, and the candidate looks like a regression purely for
   having accepted a harder target.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _REPO)
sys.path.insert(0, _XPURT)

from freshness import Invocation  # noqa: E402
from benchmarks.freshness_eval.run import (  # noqa: E402
    ALL_POLICIES,
    MUTATION_KEYS,
    PROBES,
    materialise,
)
from benchmarks.freshness_eval.trace import deadline_compliance  # noqa: E402

CONFIG = os.path.join(_REPO, "data", "toplevel", "freshness_canon_300ms.json")
EPOCH = 300.0
SOFT = "yolov8_nano_64"
PRODUCER = "dronet"


def _base():
    with open(CONFIG) as f:
        return json.load(f)


def _mat(mutations, burst=3):
    return materialise(_base(), burst=burst, mutations=mutations,
                       epoch_ms=EPOCH, seed=0)


class MutationVocabulary(unittest.TestCase):
    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError) as cm:
            _mat({"reserve_gemmini": True})
        self.assertIn("reserve_gemmini", str(cm.exception))

    def test_every_documented_mutation_is_observable(self):
        """A key in MUTATION_KEYS that materialise ignores would make any probe
        built on it silently inert -- it would rerun the baseline under a new
        name and the result would look like a real finding."""
        probes = {
            "preferred_hw": {"preferred_hw": {PRODUCER: "gemmini"}},
            "window_duration": {"window_duration": {PRODUCER: 25.0}},
            "admit_cap": {"admit_cap": 1},
            "soft_phase_ms": {"soft_phase_ms": 25.0},
        }
        self.assertEqual(
            set(probes), set(MUTATION_KEYS),
            "MUTATION_KEYS and this test's coverage have drifted apart",
        )
        baseline = _mat({})
        for key, mut in probes.items():
            mutated = _mat(mut)
            self.assertNotEqual(
                _strip(baseline), _strip(mutated),
                f"mutation {key!r} is documented but produced an identical "
                f"config -- it is inert",
            )

    def test_every_probe_uses_only_known_keys_and_at_most_one(self):
        for name, spec in PROBES.items():
            muts = spec.get("mutations") or {}
            self.assertLessEqual(
                set(muts), set(MUTATION_KEYS),
                f"{name} uses an undocumented mutation",
            )
            self.assertLessEqual(
                len(muts), 1,
                f"{name} bundles {len(muts)} mutations; an outcome would not be "
                f"attributable to a single mechanism",
            )


def _strip(cfg):
    c = copy.deepcopy(cfg)
    c.pop("_materialised", None)
    return json.dumps(c, sort_keys=True)


class AdmissionControl(unittest.TestCase):
    def test_cap_reduces_admitted_and_respreads_releases(self):
        cfg = _mat({"admit_cap": 1}, burst=3)
        self.assertEqual(cfg["networks"][SOFT]["num_instances"], 1)
        # period must follow the ADMITTED count, else one instance would be
        # released at epoch/3 spacing and the rest never exist.
        self.assertAlmostEqual(cfg["networks"][SOFT]["period"], EPOCH / 1)
        self.assertEqual(cfg["_materialised"]["admitted_soft_instances"], 1)

    def test_offered_burst_is_recorded_separately_from_admitted(self):
        """contention_level must stay the OFFERED load, or shedding work would
        look free instead of looking like the tradeoff it is."""
        cfg = _mat({"admit_cap": 1}, burst=4)
        self.assertEqual(cfg["_materialised"]["offered_burst"], 4)
        self.assertEqual(cfg["_materialised"]["admitted_soft_instances"], 1)

    def test_cap_above_offered_is_a_noop(self):
        self.assertEqual(_strip(_mat({"admit_cap": 9}, burst=3)),
                         _strip(_mat({}, burst=3)))

    def test_cap_zero_removes_the_soft_network(self):
        cfg = _mat({"admit_cap": 0}, burst=3)
        self.assertNotIn(SOFT, cfg["networks"])

    def test_burst_zero_removes_the_soft_network(self):
        self.assertNotIn(SOFT, _mat({}, burst=0)["networks"])


class ProducerWindowTightening(unittest.TestCase):
    def test_window_is_tightened(self):
        cfg = _mat({"window_duration": {PRODUCER: 25.0}})
        self.assertEqual(cfg["networks"][PRODUCER]["window_duration"], 25.0)
        # The period must NOT move: the sampling rate is a property of the
        # workload, not of the candidate. Changing both would confound the
        # mechanism with a rate change, and A0 itself depends on the period.
        self.assertEqual(cfg["networks"][PRODUCER]["period"],
                         _base()["networks"][PRODUCER]["period"])

    def test_window_exceeding_the_period_raises(self):
        with self.assertRaises(ValueError) as cm:
            _mat({"window_duration": {PRODUCER: 80.0}})
        self.assertIn("period", str(cm.exception))

    def test_unknown_network_raises(self):
        with self.assertRaises(ValueError) as cm:
            _mat({"window_duration": {"no_such_net": 10.0}})
        self.assertIn("no_such_net", str(cm.exception))


class SoftPhaseOffset(unittest.TestCase):
    def test_phase_sets_start_time(self):
        self.assertEqual(_mat({"soft_phase_ms": 25.0})["networks"][SOFT]["start_time"],
                         25.0)

    def test_phase_is_absent_when_not_requested(self):
        self.assertNotIn("start_time", _mat({})["networks"][SOFT])


GRAPH_PROBE = os.path.join(
    _REPO, "gen/vmfb/dronet/firesim_gemmini_opu/gemmini/dronet.int8/"
           "dronet.int8_dispatch_graph.json")


@unittest.skipUnless(os.path.exists(GRAPH_PROBE),
                     "bridged dispatch graphs absent; run "
                     "scripts/export_profile_db_to_results_csv.py")
class MutationsReachTheSolver(unittest.TestCase):
    """A mutation that changes the config but that the SOLVER ignores would be
    inert in the way that matters, and `test_every_documented_mutation_is_
    observable` above cannot catch it -- it only inspects the config.

    These two knobs are the ones that must survive the config -> Workload
    boundary, because both also feed the evaluator's release/deadline arithmetic
    independently. If workload_factory and trace.py ever disagree about what
    `start_time` or `window_duration` mean, input ages are silently wrong (or,
    if the schedule starts before the release the evaluator computed,
    invocations_from_fixture raises and the cell fails loudly).
    """

    @staticmethod
    def _build(cfg):
        import numpy as np
        from workload_factory import create_workload_from_network_hierarchy
        machines = ["CPU_P#0", "CPU_E#0"]
        return create_workload_from_network_hierarchy(
            {"networks": {PRODUCER: cfg["networks"][PRODUCER]}, "edges": []},
            _REPO, machines, np.zeros((2, 2)), p_core_speedup=1.0,
            random_seed=0, machine_combinations=[["CPU_P#0"], ["CPU_E#0"]],
        )

    def _windows(self, cfg):
        """instance index -> (min_start_t, max_end_t) for the producer."""
        out = {}
        for op in self._build(cfg).get_operations():
            base = op.operation_name.split("_dispatch_")[0]
            i = int(base[len(PRODUCER):])
            lo, hi = out.get(i, (float("inf"), float("-inf")))
            out[i] = (min(lo, float(op.min_start_t)), max(hi, float(op.max_end_t)))
        return out

    def test_window_tightening_reaches_max_end_t(self):
        base = _base()
        base["networks"][PRODUCER]["num_instances"] = 3
        loose = self._windows(materialise(base, burst=0, mutations={},
                                          epoch_ms=EPOCH, seed=0))
        tight = self._windows(materialise(
            base, burst=0, mutations={"window_duration": {PRODUCER: 25.0}},
            epoch_ms=EPOCH, seed=0))
        period = float(base["networks"][PRODUCER]["period"])
        for i in sorted(loose):
            self.assertAlmostEqual(loose[i][1], i * period + period, places=6)
            self.assertAlmostEqual(tight[i][1], i * period + 25.0, places=6,
                                   msg="window_duration did not reach the solver")
            self.assertAlmostEqual(tight[i][0], loose[i][0], places=6,
                                   msg="tightening must not move the release")

    def test_phase_offset_reaches_min_start_t_and_matches_the_evaluator(self):
        """workload_factory uses start_time + i*period for min_start_t; trace.py
        uses the same expression for `release`. Assert they agree, since a
        divergence makes every input age wrong."""
        base = _base()
        # Apply the phase to the PRODUCER so this test can use the one network
        # whose dispatch graph is guaranteed present.
        base["networks"][PRODUCER]["num_instances"] = 3
        base["networks"][PRODUCER]["start_time"] = 25.0
        got = self._windows(base)
        period = float(base["networks"][PRODUCER]["period"])

        from benchmarks.freshness_eval.trace import periodic_spec
        spec = periodic_spec(base)[PRODUCER]
        for i in sorted(got):
            expected_release = float(spec["start_time"]) + i * period
            self.assertAlmostEqual(got[i][0], expected_release, places=6)
            self.assertAlmostEqual(got[i][0], 25.0 + i * period, places=6)


class FixedReferenceWindow(unittest.TestCase):
    """The producer runs 18 ms of work released every 50 ms. Under a tightened
    25 ms window an instance finishing at t=30 misses its OWN deadline but is
    comfortably inside the baseline 50 ms bar. Both numbers are true and they
    answer different questions; the comparison across candidates needs the
    fixed one."""

    @staticmethod
    def _invs(window):
        # instance 0 released at 0, finishing at 30 ms
        return [Invocation(task=PRODUCER, instance=0, release_time=0.0,
                           start_time=5.0, end_time=30.0, deadline=window)]

    def test_self_relative_and_reference_rates_differ(self):
        out = deadline_compliance(self._invs(25.0), PRODUCER, reference_window=50.0)
        self.assertEqual(out[f"{PRODUCER}_deadline_success_rate"], 0.0,
                         "it missed the window it was given")
        self.assertEqual(out[f"{PRODUCER}_deadline_success_rate_vs_ref"], 1.0,
                         "but it met the fixed bar every candidate is scored on")
        self.assertEqual(out[f"{PRODUCER}_reference_window_ms"], 50.0)

    def test_reference_uses_release_not_the_recorded_deadline(self):
        """The reference bar must be rebuilt from release_time + ref_window. If
        it reused Invocation.deadline it would inherit the candidate's own
        tightening and the metric would be circular."""
        out = deadline_compliance(self._invs(25.0), PRODUCER, reference_window=20.0)
        self.assertEqual(out[f"{PRODUCER}_deadline_success_rate_vs_ref"], 0.0)
        self.assertAlmostEqual(out[f"{PRODUCER}_max_lateness_ms_vs_ref"], 10.0)

    def test_no_reference_window_emits_no_ref_keys(self):
        out = deadline_compliance(self._invs(50.0), PRODUCER)
        self.assertFalse([k for k in out if k.endswith("_vs_ref")])

    def test_absent_task_is_reported_as_empty_not_crashing(self):
        out = deadline_compliance([], PRODUCER, reference_window=50.0)
        self.assertEqual(out[f"{PRODUCER}_n_invocations"], 0)
        self.assertIsNone(out[f"{PRODUCER}_deadline_success_rate_vs_ref"])


class PolicyRegistry(unittest.TestCase):
    def test_probe_names_do_not_collide_with_deployable_policies(self):
        from benchmarks.freshness_eval.run import POLICIES
        self.assertFalse(set(POLICIES) & set(PROBES))

    def test_falsification_control_is_present(self):
        """The probe set must contain a mechanism expected to make freshness
        WORSE. Without one, a metric that responds to nothing would be
        indistinguishable from a metric where every mechanism helps."""
        ctrl = ALL_POLICIES["probe_nonperiodic_priority"]
        self.assertEqual(ctrl["solver"], "greedy_periodic")
        self.assertIn("WORSE", ctrl["intent"])

    def test_all_probes_hold_the_solver_fixed_except_the_control(self):
        """A probe that also changed solver would confound mechanism with
        scheduler."""
        for name, spec in PROBES.items():
            if name == "probe_nonperiodic_priority":
                continue
            self.assertEqual(
                spec["solver"], ALL_POLICIES["static_nominal"]["solver"],
                f"{name} changes the solver as well as the mechanism",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
