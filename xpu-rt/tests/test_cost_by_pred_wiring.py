"""cost_by_pred must reach the solver on the path the scheduler actually uses.

THE FAILURES THIS PINS were both silent. Phase E measured a real per-edge
cross-cluster cost and handed it over as a per-dispatch
{"CPU_P#0->CPU_E#0": ms} map that `workload_factory` already parses. Two things
then dropped it on the floor without a word:

  1. The predecessor-aware makespan term (processing_times_by_pred -> a gamma
     linearisation) existed ONLY in scheduler.schedule_window(). But
     `--scheduler mosek` forwards to scheduler.schedule(), a different function
     that had no such term -- so the map was consumed by nothing on the used
     path, and every schedule came out identical to cost_by_pred-absent.

  2. The PROFILED workload builder is create_workload_from_network_hierarchy,
     not create_workload_from_dependencies. Only the latter read cost_by_pred
     (and infeasible_machines); the profiled builder built Operations without
     either field, so the map never even reached an Operation.

Both are the "labelled-but-not-wired" class: the data is present, the code that
should act on it is elsewhere, and nothing errors. These tests assert the map
arrives on the Operation from the profiled builder, and that schedule() (the
used entry point) actually moves a placement because of it.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import scheduler as S  # noqa: E402
from workload import Workload, Operation  # noqa: E402
from workload_factory import (  # noqa: E402
    create_workload_from_network_hierarchy,
    parse_cost_by_pred,
    parse_infeasible_combinations,
)

MACHINES = ["CPU_P", "CPU_E"]


def _write_graph(tmpdir, name, dispatches):
    """Write a minimal dispatch graph; the loader takes basename from the parent
    dir, so nest it under a dir named for the basename."""
    d = os.path.join(tmpdir, name, f"{name}.int8")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{name}.int8_dispatch_graph.json")
    json.dump({"dispatches": dispatches}, open(p, "w"))
    return p


class CostByPredParsing(unittest.TestCase):
    def test_names_map_to_indices(self):
        info = {"cost_by_pred": {"CPU_P->CPU_E": 9.0, "CPU_E->CPU_E": 2.0}}
        m = parse_cost_by_pred(info, MACHINES)
        self.assertEqual(m[(0, 1)], 9.0)
        self.assertEqual(m[(1, 1)], 2.0)

    def test_unknown_machine_side_skipped(self):
        # a hart the spec does not have must be dropped, not crash
        info = {"cost_by_pred": {"CPU_P#7->CPU_E": 9.0, "CPU_P->CPU_E": 3.0}}
        m = parse_cost_by_pred(info, MACHINES)
        self.assertEqual(m, {(0, 1): 3.0})

    def test_infeasible_names_map(self):
        self.assertEqual(
            parse_infeasible_combinations({"infeasible_machines": ["CPU_E"]}, MACHINES),
            {1},
        )


class ProfiledBuilderPlumbing(unittest.TestCase):
    """Bug 2: the profiled builder must carry cost_by_pred AND infeasible_machines
    onto the Operation, and keep them distinct per network."""

    def _build(self, tmp):
        gA = _write_graph(tmp, "neta", {
            "d0": {"id": 0, "dependencies": []},
            "d1": {"id": 1, "dependencies": ["d0"],
                   "cost_by_pred": {"CPU_P->CPU_E": 11.0, "CPU_P->CPU_P": 1.0},
                   "infeasible_machines": ["CPU_E"]},
        })
        gB = _write_graph(tmp, "netb", {
            "d0": {"id": 0, "dependencies": []},
            "d1": {"id": 1, "dependencies": ["d0"],
                   "cost_by_pred": {"CPU_P->CPU_E": 77.0, "CPU_P->CPU_P": 7.0}},
        })
        nd = {"networks": {
            "neta": {"id": 0, "identifier": "neta", "dispatch_deps_path": gA,
                     "period": 100, "window_duration": 100, "num_instances": 1},
            "netb": {"id": 1, "identifier": "netb", "dispatch_deps_path": gB,
                     "period": 100, "window_duration": 100, "num_instances": 1},
        }, "edges": []}
        return create_workload_from_network_hierarchy(
            nd, repo_base_path=tmp, machines=MACHINES,
            transfer_times=np.zeros((2, 2)), processing_times=None, random_seed=0)

    def test_cost_by_pred_reaches_ops_per_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            wl = self._build(tmp)
            maps = {op.operation_name: getattr(op, "processing_times_by_pred", {})
                    for op in wl.operations}
            a1 = [v for k, v in maps.items() if "neta" in k and v]
            b1 = [v for k, v in maps.items() if "netb" in k and v]
            self.assertTrue(a1, "neta's d1 lost its cost_by_pred in the profiled builder")
            self.assertTrue(b1, "netb's d1 lost its cost_by_pred in the profiled builder")
            # distinct per network (the whole point of per-network keying)
            self.assertEqual(a1[0][(0, 1)], 11.0)
            self.assertEqual(b1[0][(0, 1)], 77.0)

    def test_infeasible_machines_reaches_ops(self):
        with tempfile.TemporaryDirectory() as tmp:
            wl = self._build(tmp)
            infe = [getattr(op, "infeasible_combinations", set())
                    for op in wl.operations if "neta" in op.operation_name
                    and getattr(op, "infeasible_combinations", set())]
            self.assertTrue(infe, "infeasible_machines dropped by the profiled builder")
            self.assertIn(1, infe[0])  # CPU_E excluded


class GammaOnUsedPath(unittest.TestCase):
    """Bug 1 + guard: schedule() (the used entry) must move a placement because
    of cost_by_pred, and must NOT misindex when combinations aren't singletons."""

    def _chain(self, with_cbp, combos=None):
        a = Operation([5.0, 50.0], operation_name="A"); a.min_start_t = 0.0
        pmap = {} if not with_cbp else {(0, 0): 10.0, (0, 1): 100.0,
                                        (1, 0): 100.0, (1, 1): 2.0}
        b = Operation([10.0, 2.0], operation_name="B",
                      processing_times_by_pred=pmap); b.min_start_t = 0.0
        b.add_predecessor(a)
        return Workload([a, b], MACHINES, np.zeros((2, 2)), machine_combinations=combos)

    def _place(self, wl):
        t, alpha, _, _ = S.schedule(wl, time_limit=60,
                                    restrict_makespan_to_nonperiodic=False)
        self.assertIsNotNone(t, "schedule() returned no solution")
        return {op.operation_name: MACHINES[int(np.argmax(alpha[i]))]
                for i, op in enumerate(wl.operations)}

    def test_gamma_flips_placement_on_schedule(self):
        # OFF: B prefers its cheaper raw CPU_E cost (2 vs 10)
        self.assertEqual(self._place(self._chain(False))["B"], "CPU_E")
        # ON: crossing P->E costs 100, so B must join A on CPU_P despite raw cost
        self.assertEqual(self._place(self._chain(True))["B"], "CPU_P")

    def test_guard_drops_map_when_not_singletons(self):
        # swapped-order combos: identity combo[k]==[machines[k]] is broken
        combos = [["CPU_E"], ["CPU_P"]]
        # must still solve (map dropped, no misindex / crash)
        pl = self._place(self._chain(True, combos=combos))
        self.assertIn(pl["B"], MACHINES)


if __name__ == "__main__":
    unittest.main()
