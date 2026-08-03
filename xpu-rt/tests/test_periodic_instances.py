"""Regression tests for three silent-failure modes found while building the
freshness sweep. Each one produced a plausible-looking but wrong result with no
warning, which is the class of bug that quietly invalidates an experiment.

1. An explicit per-network `num_instances` was ignored when EVERY network was
   periodic: that case took an early return in the instance-count estimator that
   hard-coded 1 instance. A 30-instance control task silently became a
   1-instance one.

2. The adjacent auto-merge post-pass rewrote the emitted fixture by default,
   collapsing dispatches and shifting start times, so any cross-policy
   comparison was really policy+automerge.

3. `preferred_hw` naming the CLUSTER ("cpu_p") instead of the profile hw
   ("gemmini") matched no combination, so every combination was treated as
   non-preferred and received the pin penalty -- inflating that network's cost
   by ~100 ms per dispatch and producing a nonsense schedule.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _XPURT)

from workload_factory import create_workload_from_network_hierarchy  # noqa: E402
from postprocessing import automerge_enabled  # noqa: E402
from profile_loader import load_profiled_processing_times  # noqa: E402

GRAPH = (
    "gen/vmfb/{m}/firesim_gemmini_opu/gemmini/{m}.int8/{m}.int8_dispatch_graph.json"
)
MACHINES = ["CPU_P#0", "CPU_E#0"]
COMBOS = [["CPU_P#0"], ["CPU_E#0"]]


def _net(name: str, nid: int, period=None, window=None, instances=None, **extra):
    info = {
        "id": nid,
        "identifier": name,
        "dispatch_deps_path": GRAPH.format(m=name),
    }
    if period is not None:
        info["period"] = period
        info["window_duration"] = window if window is not None else period
    if instances is not None:
        info["num_instances"] = instances
    info.update(extra)
    return info


def _build(networks):
    return create_workload_from_network_hierarchy(
        {"networks": networks, "edges": []},
        _REPO,
        MACHINES,
        np.zeros((2, 2)),
        p_core_speedup=1.0,
        random_seed=0,
        machine_combinations=COMBOS,
    )


def _instances(workload):
    """job_name -> op count, keyed by the expanded per-instance identifier."""
    counts = {}
    for op in workload.get_operations():
        base = op.operation_name.split("_dispatch_")[0]
        counts[base] = counts.get(base, 0) + 1
    return counts


@unittest.skipUnless(
    os.path.exists(os.path.join(_REPO, GRAPH.format(m="dronet"))),
    "bridged dispatch graphs absent; run scripts/export_profile_db_to_results_csv.py",
)
class NumInstancesOverride(unittest.TestCase):
    def test_override_honoured_when_all_networks_are_periodic(self):
        """The regression: with no aperiodic network the estimator has no horizon
        to derive a count from and used to return 1 per network, discarding the
        explicit num_instances."""
        wl = _build({
            "mlp_control": _net("mlp_control", 0, period=10, instances=30),
            "dronet": _net("dronet", 1, period=50, instances=6),
        })
        counts = _instances(wl)
        mlp = sorted(k for k in counts if k.startswith("mlp_control"))
        dro = sorted(k for k in counts if k.startswith("dronet"))
        self.assertEqual(len(mlp), 30, f"expected 30 mlp_control instances, got {mlp}")
        self.assertEqual(len(dro), 6, f"expected 6 dronet instances, got {dro}")
        # 30 x 7 + 6 x 21 dispatches
        self.assertEqual(len(wl.get_operations()), 30 * 7 + 6 * 21)

    def test_without_override_all_periodic_defaults_to_one_each(self):
        """The pre-existing behaviour is preserved when nothing is stated: with
        no aperiodic work there is no horizon, so one instance each."""
        wl = _build({
            "mlp_control": _net("mlp_control", 0, period=10),
            "dronet": _net("dronet", 1, period=50),
        })
        counts = _instances(wl)
        self.assertEqual(len([k for k in counts if k.startswith("mlp_control")]), 1)
        self.assertEqual(len([k for k in counts if k.startswith("dronet")]), 1)

    def test_releases_and_windows_follow_the_period(self):
        """Instance i must be released at i*period and close at +window, since
        the freshness evaluator derives consumer deadlines from exactly that."""
        wl = _build({"dronet": _net("dronet", 0, period=50, window=50, instances=4)})
        by_inst = {}
        for op in wl.get_operations():
            base = op.operation_name.split("_dispatch_")[0]
            by_inst.setdefault(base, []).append(op)
        for base, ops in sorted(by_inst.items()):
            i = int(base[len("dronet"):])
            self.assertAlmostEqual(min(o.min_start_t for o in ops), i * 50.0, places=6)
            self.assertAlmostEqual(max(o.max_end_t for o in ops), i * 50.0 + 50.0,
                                   places=6)


class AutomergeDefault(unittest.TestCase):
    def test_automerge_is_off_unless_asked_for(self):
        for var in ("XPURT_AUTOMERGE", "XPURT_NO_AUTOMERGE"):
            os.environ.pop(var, None)
        self.assertFalse(automerge_enabled())

    def test_automerge_opt_in_and_force_off(self):
        try:
            os.environ["XPURT_AUTOMERGE"] = "1"
            self.assertTrue(automerge_enabled())
            os.environ["XPURT_NO_AUTOMERGE"] = "1"
            self.assertFalse(automerge_enabled(), "force-off must win")
        finally:
            os.environ.pop("XPURT_AUTOMERGE", None)
            os.environ.pop("XPURT_NO_AUTOMERGE", None)


@unittest.skipUnless(
    os.path.exists(os.path.join(
        _REPO, "gen/profile/gemmini/firesim_gemmini_opu/dronet/dronet.int8/"
               "topo_0/results.csv")),
    "bridged profile CSVs absent; run scripts/export_profile_db_to_results_csv.py",
)
class PreferredHwValidation(unittest.TestCase):
    NETS = {
        "dronet": {
            "id": 0,
            "identifier": "dronet",
            "dispatch_deps_path": GRAPH.format(m="dronet"),
        }
    }

    def _load(self, preferred_hw):
        nets = {k: dict(v) for k, v in self.NETS.items()}
        if preferred_hw is not None:
            nets["dronet"]["preferred_hw"] = preferred_hw
        return load_profiled_processing_times(
            nets, _REPO, COMBOS, ["gemmini", "rvv_opu"],
            "firesim_gemmini_opu", "gemmini", "rvv_opu",
            np.random.default_rng(0), 1.0,
            topo_tag_override="topo_0",
        )

    def test_cluster_name_instead_of_profile_hw_raises(self):
        """'cpu_p' is a cluster, not a profile hw. It used to match nothing and
        silently penalise every combination."""
        with self.assertRaises(ValueError) as cm:
            self._load("cpu_p")
        msg = str(cm.exception)
        self.assertIn("preferred_hw", msg)
        self.assertIn("cpu_p", msg)
        self.assertIn("gemmini", msg, "the error must name the valid options")

    def test_valid_profile_hw_penalises_only_the_others(self):
        pinned, _, _, _ = self._load("gemmini")
        plain, _, _, _ = self._load(None)
        # gemmini is combination 0, rvv_opu is combination 1.
        for name, times in pinned.items():
            self.assertAlmostEqual(
                times[0], plain[name][0], places=9,
                msg=f"{name}: the preferred combination must be unpenalised",
            )
            self.assertGreater(
                times[1], plain[name][1],
                msg=f"{name}: the non-preferred combination must be penalised",
            )


class EdfUsesTheWindowClose(unittest.TestCase):
    """`edf` read only op.deadline_us, which run_xpurt_schedule.py never sets, so
    it silently fell back to upward rank -- it ran, and reported itself as "edf",
    while ordering by something else entirely. Periodic workloads express the
    deadline as the window close (max_end_t), which the MILP itself enforces."""

    @staticmethod
    def _op(oid, name, jid, max_end_t, deadline_us=None):
        from workload import Operation
        return Operation(
            [1.0, 1.0], operation_id=oid, operation_name=name, job_id=jid,
            min_start_t=0, max_end_t=max_end_t, deadline_us=deadline_us,
        )

    def _wl(self, ops):
        from workload import Workload
        return Workload(ops, MACHINES, np.zeros((2, 2)), machine_combinations=COMBOS)

    def test_earlier_window_close_gets_higher_priority(self):
        from scheduler_heft import _deadline_priority
        late = self._op(0, "late_dispatch_0", 0, max_end_t=100.0)
        soon = self._op(1, "soon_dispatch_0", 1, max_end_t=10.0)
        prio = _deadline_priority(self._wl([late, soon]))
        self.assertGreater(prio[1], prio[0])

    def test_explicit_deadline_us_still_wins(self):
        from scheduler_heft import _deadline_priority
        a = self._op(0, "a_dispatch_0", 0, max_end_t=10.0, deadline_us=999.0)
        b = self._op(1, "b_dispatch_0", 1, max_end_t=100.0, deadline_us=5.0)
        prio = _deadline_priority(self._wl([a, b]))
        self.assertGreater(prio[1], prio[0], "deadline_us must take precedence")

    def test_no_deadline_and_no_window_falls_back_without_crashing(self):
        from scheduler_heft import _deadline_priority
        a = self._op(0, "a_dispatch_0", 0, max_end_t=None)
        b = self._op(1, "b_dispatch_0", 1, max_end_t=None)
        prio = _deadline_priority(self._wl([a, b]))
        self.assertEqual(len(prio), 2)
        self.assertTrue(all(isinstance(p, float) for p in prio))

    def test_ordering_differs_from_upward_rank(self):
        """If it matched upward rank the fix would be inert. Give the op with the
        LATER window the higher upward rank, so the two rules disagree."""
        from scheduler_heft import _deadline_priority, _upward_rank
        # a is a predecessor of b, so a has the higher upward rank; but a's
        # window closes later, so EDF must rank b first.
        a = self._op(0, "a_dispatch_0", 0, max_end_t=100.0)
        b = self._op(1, "b_dispatch_0", 1, max_end_t=10.0)
        b.add_predecessor(a)
        wl = self._wl([a, b])
        rank = _upward_rank(wl)
        prio = _deadline_priority(wl)
        self.assertGreater(rank[0], rank[1], "a should outrank b by upward rank")
        self.assertGreater(prio[1], prio[0], "b should outrank a by deadline")


if __name__ == "__main__":
    unittest.main(verbosity=2)
