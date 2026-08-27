"""The K1 resource model: IME is a property of cores 0-3, not a separate engine.

THE BUG THIS PREVENTS: the natural way to let the scheduler choose IME is to
declare it as extra machines -- `{"cpu_p": 4, "cpu_e": 4, "ime": 4}`. That
produces schedules that cannot run. The IME "machine" is marked busy while
CPU_P#2, the core the IME instruction actually executes on, is still marked
idle, so the solver happily places an unrelated RVV dispatch there at the same
instant. Nothing in the model objects, the Gantt looks great, and the schedule
is physically impossible.

The measured fact behind all of this, from a per-core SIGILL probe on the board
(artifacts/k1_bringup/*/ime_capability_probe.txt): `smt.vmadot` executes on
cores 0-3 and traps on cores 4-7. Note "traps" -- an IME kernel sent to cluster
1 does not degrade gracefully, so legality has to be enforced when the workload
is built rather than discovered at runtime.

The representation we use instead: alternative implementations are separate
combinations that share the *same* machine names. `combinations_overlap` is set
intersection, and both schedulers refuse to overlap intersecting combinations,
so mutual exclusion falls out of machinery that already exists and double
booking is unrepresentable rather than merely discouraged.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _REPO)
sys.path.insert(0, _XPURT)

from capabilities import (  # noqa: E402
    K1_CAPABILITIES,
    IllegalPlacement,
    build_machine_combinations_with_impls,
    check_implementation_legality,
    core_ids_for_combination,
    legal_combination_indices,
)
from workload import Workload  # noqa: E402
from workload_factory import build_machine_combinations  # noqa: E402

K1_CORES = {"CPU_P": 4, "CPU_E": 4}
K1_IMPLS = {"CPU_P": ["rvv", "ime"], "CPU_E": ["rvv"]}


def _workload(machines, combinations):
    """A Workload carrying only the resource model -- no operations needed."""
    n = len(machines)
    return Workload(operations=[], machines=machines,
                    transfer_times=np.zeros((n, n)),
                    machine_combinations=combinations)


class LegalityTests(unittest.TestCase):
    def test_ime_is_rejected_on_cluster_1(self):
        with self.assertRaises(IllegalPlacement) as ctx:
            check_implementation_legality({"CPU_E": ["ime"]})
        self.assertIn("ime", str(ctx.exception))
        self.assertIn("CPU_E", str(ctx.exception))

    def test_ime_is_accepted_on_cluster_0(self):
        check_implementation_legality({"CPU_P": ["ime"]})  # must not raise

    def test_rvv_and_scalar_are_legal_on_both_clusters(self):
        check_implementation_legality({"CPU_P": ["scalar", "rvv"],
                                       "CPU_E": ["scalar", "rvv"]})

    def test_every_violation_is_reported_at_once(self):
        """One run should tell you everything that is wrong, not just the first."""
        with self.assertRaises(IllegalPlacement) as ctx:
            check_implementation_legality({"CPU_E": ["ime", "gemmini"]})
        msg = str(ctx.exception)
        self.assertIn("ime", msg)
        self.assertIn("gemmini", msg)

    def test_unknown_machine_kind_is_rejected(self):
        with self.assertRaises(IllegalPlacement):
            check_implementation_legality({"NPU": ["ime"]})

    def test_a_kind_with_no_implementations_is_rejected(self):
        """Silently unschedulable is worse than a loud failure."""
        with self.assertRaises(IllegalPlacement):
            build_machine_combinations_with_impls({"CPU_P": 2}, {"CPU_P": []})

    def test_capability_table_matches_what_was_measured_on_the_board(self):
        self.assertIn("ime", K1_CAPABILITIES["CPU_P"])
        self.assertNotIn("ime", K1_CAPABILITIES["CPU_E"])
        for kind in ("CPU_P", "CPU_E"):
            self.assertIn("rvv", K1_CAPABILITIES[kind])
            self.assertIn("scalar", K1_CAPABILITIES[kind])


class NoDoubleBookingTests(unittest.TestCase):
    """The core invariant, checked through the real Workload, not a stand-in."""

    def setUp(self):
        self.machines, self.combos, self.impls = \
            build_machine_combinations_with_impls(K1_CORES, K1_IMPLS)
        self.wl = _workload(self.machines, self.combos)

    def test_rvv_and_ime_on_the_same_core_are_mutually_exclusive(self):
        rvv = self.combos.index(["CPU_P#0"])
        ime = next(i for i, (c, m) in enumerate(zip(self.combos, self.impls))
                   if c == ["CPU_P#0"] and m == "ime")
        self.assertNotEqual(rvv, ime, "they must be distinct combinations")
        self.assertTrue(
            self.wl.combinations_overlap(rvv, ime),
            "an IME dispatch and an RVV dispatch on CPU_P#0 must never be "
            "allowed to run concurrently -- they are the same physical core",
        )

    def test_every_pair_sharing_a_core_overlaps(self):
        """Exhaustive: no pair of combinations sharing any machine is free to overlap."""
        for i, ci in enumerate(self.combos):
            for j, cj in enumerate(self.combos):
                if i >= j:
                    continue
                shares = bool(set(ci) & set(cj))
                self.assertEqual(
                    shares, self.wl.combinations_overlap(i, j),
                    f"combination {i} {ci} ({self.impls[i]}) vs "
                    f"{j} {cj} ({self.impls[j]})",
                )

    def test_disjoint_clusters_may_run_concurrently(self):
        """The model must not over-serialise: cluster 0 and cluster 1 are independent."""
        p = self.combos.index(["CPU_P#0"])
        e = self.combos.index(["CPU_E#0"])
        self.assertFalse(self.wl.combinations_overlap(p, e))

    def test_a_multicore_implementation_reserves_all_of_its_cores(self):
        """A 4-core combination must block every combination overlapping it."""
        four = self.combos.index(["CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3"])
        blocked = 0
        for i, combo in enumerate(self.combos):
            if i == four:
                continue
            if set(combo) & {"CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3"}:
                self.assertTrue(
                    self.wl.combinations_overlap(four, i),
                    f"a 4-core cluster-0 dispatch must exclude {combo} "
                    f"({self.impls[i]})",
                )
                blocked += 1
        self.assertGreater(blocked, 0)

    def test_combinations_are_cumulative_prefixes_so_a_cluster_is_ONE_resource(self):
        """Document a real limit of the existing model, so nobody assumes 8-way.

        `build_machine_combinations` emits cumulative prefixes: ['CPU_P#0'],
        ['CPU_P#0','CPU_P#1'], ... There is deliberately no ['CPU_P#1'] on its
        own. Every cluster-0 combination therefore contains CPU_P#0, so every
        pair of them intersects, so **at most one dispatch runs on cluster 0 at
        a time** no matter how many cores are declared.

        `machines: {cpu_p: 4}` means "one dispatch may be given up to 4 cores",
        NOT "four dispatches may run side by side". Concurrency in this model
        comes from having several machine KINDS (cluster 0 vs cluster 1), which
        for the K1 caps genuinely-parallel dispatches at 2.

        That is a defensible model -- it matches the profile tree, where
        topo_0_1_2_3 is the 4-hart time for one dispatch -- but it is a ceiling
        the periodic MLP+DroNet+YOLO workload will run into, and it should be a
        deliberate choice rather than a surprise.
        """
        cluster0 = [i for i, c in enumerate(self.combos)
                    if all(m.startswith("CPU_P#") for m in c)]
        self.assertGreater(len(cluster0), 1)
        for i in cluster0:
            self.assertIn("CPU_P#0", self.combos[i],
                          "every cluster-0 combination contains core 0")
        for a in cluster0:
            for b in cluster0:
                if a < b:
                    self.assertTrue(
                        self.wl.combinations_overlap(a, b),
                        "cluster 0 serialises: no two dispatches can share it",
                    )

    def test_the_broken_alternative_would_have_failed_this(self):
        """Guard the actual mistake: IME as its own machine kind.

        Modelled as separate machines, an IME combination and a CPU_P
        combination do NOT intersect, so the scheduler is free to run both at
        once -- on hardware that is one core doing two things.
        """
        machines, combos = build_machine_combinations(
            {"CPU_P": 4, "CPU_E": 4, "IME": 4})
        wl = _workload(machines, combos)
        ime = combos.index(["IME#0"])
        cpu = combos.index(["CPU_P#0"])
        self.assertFalse(
            wl.combinations_overlap(ime, cpu),
            "this is the bug: as separate machines they look independent",
        )


class BackwardCompatibilityTests(unittest.TestCase):
    def test_single_implementation_matches_the_existing_builder(self):
        """One impl per kind must reproduce the current combinations exactly."""
        machines_old, combos_old = build_machine_combinations(K1_CORES)
        machines_new, combos_new, impls = build_machine_combinations_with_impls(
            K1_CORES, {"CPU_P": ["rvv"], "CPU_E": ["rvv"]})
        self.assertEqual(machines_old, machines_new)
        self.assertEqual(combos_old, combos_new)
        self.assertEqual(len(impls), len(combos_new))

    def test_combination_count_is_cores_times_implementations(self):
        _, combos, _ = build_machine_combinations_with_impls(K1_CORES, K1_IMPLS)
        # CPU_P: 4 prefixes x 2 impls, CPU_E: 4 prefixes x 1 impl
        self.assertEqual(len(combos), 4 * 2 + 4 * 1)

    def test_topo_tag_size_convention_is_preserved(self):
        """Profile lookup keys off combination size; impl variants must not shift it.

        ModelBlaster writes 4-core profiles under topo_0_1_2_3
        (pipeline/profile_writer.py), and workload_factory derives the tag from
        combination length, so the two agree only if adding an implementation
        does not change the machine set.
        """
        _, combos, impls = build_machine_combinations_with_impls(K1_CORES, K1_IMPLS)
        for combo, impl in zip(combos, impls):
            if len(combo) == 4 and impl == "ime":
                self.assertEqual(combo,
                                 ["CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3"])
                break
        else:
            self.fail("expected a 4-core IME combination")


class ImplementationSelectionTests(unittest.TestCase):
    def test_indices_for_a_required_implementation(self):
        _, combos, impls = build_machine_combinations_with_impls(K1_CORES, K1_IMPLS)
        ime_idx = legal_combination_indices(combos, impls, "ime")
        self.assertTrue(ime_idx)
        for i in ime_idx:
            self.assertEqual(impls[i], "ime")
            for m in combos[i]:
                self.assertTrue(m.startswith("CPU_P#"),
                                "IME must never be offered on cluster 1")

    def test_a_dispatch_with_no_ime_kernel_gets_no_ime_combination(self):
        _, combos, impls = build_machine_combinations_with_impls(K1_CORES, K1_IMPLS)
        rvv_only = legal_combination_indices(combos, impls, "rvv")
        self.assertTrue(all(impls[i] == "rvv" for i in rvv_only))


class PhysicalCoreIdTests(unittest.TestCase):
    """Affinity masks need physical ids, and cluster 1 does not start at 0."""

    def test_cluster0_indices_are_identity(self):
        self.assertEqual(core_ids_for_combination(["CPU_P#0", "CPU_P#3"]), [0, 3])

    def test_cluster1_is_offset_to_cores_4_through_7(self):
        self.assertEqual(core_ids_for_combination(["CPU_E#0"]), [4])
        self.assertEqual(core_ids_for_combination(["CPU_E#3"]), [7])

    def test_all_eight_cores_are_reachable_and_distinct(self):
        _, combos, _ = build_machine_combinations_with_impls(
            K1_CORES, {"CPU_P": ["rvv"], "CPU_E": ["rvv"]})
        ids = set()
        for combo in combos:
            ids.update(core_ids_for_combination(combo))
        self.assertEqual(ids, set(range(8)))

    def test_out_of_range_core_is_rejected(self):
        with self.assertRaises(IllegalPlacement):
            core_ids_for_combination(["CPU_E#9"])


if __name__ == "__main__":
    unittest.main()
