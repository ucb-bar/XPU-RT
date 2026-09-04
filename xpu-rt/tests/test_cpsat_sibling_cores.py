#!/usr/bin/env python3
"""Sibling-core semantics for the CP-SAT backend.

Both bugs fixed in 95db5778 were silent, and both were only reachable once a
machine kind had more than one core -- exactly what sharding produces. Neither
was caught, because nothing here exercised sibling-core combinations and the
only visible symptom was an objective value that looked plausible (85.42 ms on
control_mix_gempair, against 60.07 from a heuristic).

So these tests assert the model's *semantics* rather than its objective:

  - two singleton combinations on different harts CAN overlap,
  - a two-core combination CANNOT overlap either singleton,
  - two operations on the same singleton hart cannot overlap.

An objective-only test would pass on a model that serialises sibling harts, as
the old one effectively did. Run with pytest, or directly:

    python3 xpu-rt/tests/test_cpsat_sibling_cores.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from workload import Operation, Workload
from workload_factory import build_machine_combinations
from schedule_decoder import DecoderContext, evaluate
from cpsat_scheduler import build_payload, cpsat_available, cpsat_schedule, _integerize

INF = float("inf")

# machines = ['CPU_P#0', 'CPU_P#1']
# combos   = [['CPU_P#0'], ['CPU_P#0','CPU_P#1'], ['CPU_P#1']]
#              ^ solo #0     ^ both harts          ^ solo #1
SOLO0, PAIR, SOLO1 = 0, 1, 2


def _two_hart_workload(pt_a, pt_b):
    """Two independent operations on one two-core machine kind.

    `pt_a`/`pt_b` give each operation's duration per combination; INF marks a
    combination that operation may not use, which is how a test pins an
    operation to a particular hart.
    """
    machines, combos = build_machine_combinations({"CPU_P": 2})
    assert machines == ["CPU_P#0", "CPU_P#1"], machines
    assert combos == [["CPU_P#0"], ["CPU_P#0", "CPU_P#1"], ["CPU_P#1"]], combos
    ops = [Operation(processing_times=list(pt_a), operation_id="a"),
           Operation(processing_times=list(pt_b), operation_id="b")]
    w = Workload(operations=ops, machines=machines,
                 transfer_times=np.zeros((len(machines), len(machines))),
                 machine_combinations=combos)
    return w


def _solve(w, time_limit=10.0):
    t, alpha = cpsat_schedule(w, time_limit=time_limit, workers=1,
                              restrict_to_nonperiodic=False)
    ctx = DecoderContext(w)
    obj, misses, all_end = evaluate(ctx, t, alpha, False)
    return t, alpha, all_end


def _needs_cpsat():
    if cpsat_available() is None:
        try:
            import pytest
            pytest.skip("no interpreter with ortools (set XPURT_CPSAT_PYTHON)")
        except ImportError:
            print("  SKIP: no ortools interpreter")
        return True
    return False


# --------------------------------------------------------------------------
# The payload: which machines each combination occupies.
# --------------------------------------------------------------------------

def test_payload_lists_every_machine_a_combination_occupies():
    """combo_machines must name *both* harts for the two-core combination.

    The pre-95db5778 model derived a single machine per combination from
    `first_machine`, which cannot express "this dispatch also holds CPU_P#1".
    """
    w = _two_hart_workload([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    ctx = DecoderContext(w)
    payload = build_payload(ctx, time_limit=1.0)

    assert payload["combo_machines"] == [[0], [0, 1], [1]], payload["combo_machines"]
    # first_machine alone is ambiguous: it reports 0 for both the solo-#0 and
    # the two-hart combination, which is precisely the information loss.
    assert payload["first_machine"] == [0, 0, 1], payload["first_machine"]


def test_payload_machine_sets_match_the_conflict_matrix():
    """Sharing a machine and conflicting must be the same relation.

    If they disagree, the no-overlap grouping and the replay in `_integerize`
    are enforcing different things.
    """
    w = _two_hart_workload([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    ctx = DecoderContext(w)
    payload = build_payload(ctx, time_limit=1.0)
    cm = [set(x) for x in payload["combo_machines"]]
    for a in range(ctx.n_combos):
        for b in range(ctx.n_combos):
            shares = bool(cm[a] & cm[b])
            assert shares == bool(payload["conflict"][a][b]), (a, b, shares)
    # and specifically: the two singletons are disjoint, the pair meets both
    assert not (cm[SOLO0] & cm[SOLO1])
    assert cm[PAIR] & cm[SOLO0] and cm[PAIR] & cm[SOLO1]


# --------------------------------------------------------------------------
# The replay used to build the warm-start hint.
# --------------------------------------------------------------------------

def test_integerize_lets_sibling_harts_run_concurrently():
    """Two ops pinned to different harts may share a start instant."""
    w = _two_hart_workload([10.0, INF, INF], [INF, INF, 10.0])
    ctx = DecoderContext(w)
    dur_int = build_payload(ctx, time_limit=1.0)["dur"]
    alpha = np.zeros((2, 3))
    alpha[0, SOLO0] = 1.0
    alpha[1, SOLO1] = 1.0
    starts, ends = _integerize(ctx, np.array([0.0, 0.0]), alpha, dur_int)
    assert starts == [0, 0], starts
    assert ends == [10000, 10000], ends


def test_integerize_serialises_a_two_core_op_against_a_singleton():
    """The pair combination holds CPU_P#1, so a solo-#1 op must wait."""
    w = _two_hart_workload([INF, 10.0, INF], [INF, INF, 10.0])
    ctx = DecoderContext(w)
    dur_int = build_payload(ctx, time_limit=1.0)["dur"]
    alpha = np.zeros((2, 3))
    alpha[0, PAIR] = 1.0
    alpha[1, SOLO1] = 1.0
    starts, ends = _integerize(ctx, np.array([0.0, 0.0]), alpha, dur_int)
    # op b cannot start until the two-hart dispatch releases CPU_P#1
    assert starts == [0, 10000], starts
    assert max(ends) == 20000, ends


# --------------------------------------------------------------------------
# The model itself, solved.
# --------------------------------------------------------------------------

def test_model_runs_sibling_harts_in_parallel():
    """Independent ops, one per hart: makespan is one duration, not two.

    This is the assertion the old model failed. It had all three combinations
    on a single machine's no-overlap list, so it returned 20 here and called
    it optimal.
    """
    if _needs_cpsat():
        return
    w = _two_hart_workload([10.0, INF, INF], [INF, INF, 10.0])
    t, alpha, all_end = _solve(w)
    assert abs(all_end - 10.0) < 1e-6, f"sibling harts serialised: makespan {all_end}"
    assert int(np.argmax(alpha[0])) == SOLO0
    assert int(np.argmax(alpha[1])) == SOLO1
    assert abs(t[0] - t[1]) < 1e-6, f"expected concurrent starts, got {t}"


def test_model_forbids_a_two_core_op_overlapping_a_singleton():
    """A dispatch holding both harts excludes a solo dispatch on either."""
    if _needs_cpsat():
        return
    for solo, name in ((SOLO0, "CPU_P#0"), (SOLO1, "CPU_P#1")):
        pt_b = [INF, INF, INF]
        pt_b[solo] = 10.0
        w = _two_hart_workload([INF, 10.0, INF], pt_b)
        t, alpha, all_end = _solve(w)
        assert abs(all_end - 20.0) < 1e-6, (
            f"two-core op overlapped a {name} op: makespan {all_end}")
        # they must not overlap in time
        lo, hi = sorted([float(t[0]), float(t[1])])
        assert hi - lo >= 10.0 - 1e-6, f"overlapping intervals {t}"


def test_model_serialises_two_ops_on_the_same_hart():
    """The control: same singleton twice really is a conflict."""
    if _needs_cpsat():
        return
    w = _two_hart_workload([10.0, INF, INF], [10.0, INF, INF])
    t, alpha, all_end = _solve(w)
    assert abs(all_end - 20.0) < 1e-6, f"same hart not serialised: {all_end}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")
