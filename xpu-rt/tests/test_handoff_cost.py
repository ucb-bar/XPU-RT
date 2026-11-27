#!/usr/bin/env python3
"""Tests for the hand-off cost: what the SGS decoder does with the transfer
matrix, the K3 calibration written as `scheduler.handoff_ms`, and the
schedule-level hand-off metrics.

Every fixture is small enough to check by hand; the arithmetic is in the
docstrings. See docs/handoff_and_width_pricing.md for the design.

    pytest xpu-rt/tests/test_handoff_cost.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import handoff_metrics
from metaheuristics import heft_edf_schedule, heft_schedule
from schedule_decoder import DecoderContext, decode
from workload import Operation, Workload


# --------------------------------------------------------------------------
# 1. Affinity: the transfer matrix keeps a chain on its core
# --------------------------------------------------------------------------

def _affinity_fixture(handoff_ms: float):
    """Two identical cores A, B and three chains.

      P: p1 (1.0) -> p2 (1.0)        the chain under test
      F: f1 (0.6) -> f2 (0.6)        competitor, ends up on B until 1.2
      G: g1 (0.5), released at 1.0   competitor, takes A at [1.0, 1.5]

    Decode order (explicit priorities): p1, f1, f2, g1, p2.

      p1 -> A [0, 1.0]       (tie between A and B, first combination wins)
      f1 -> B [0, 0.6]       (A is busy until 1.0)
      f2 -> B [0.6, 1.2]     (A would finish at 1.6; with a hand-off cost A is
                              later still, so f2 stays on B in every variant)
      g1 -> A [1.0, 1.5]     (B is busy until 1.2 and would finish at 1.7)
      p2: its own core A is busy until 1.5  -> start 1.5, finish 2.5
          core B is free from 1.2           -> start max(1.2, 1.0 + handoff)

    So the gain from moving p2 to B is 0.3 ms, and A is busy for 0.5 ms after
    p1 ends:  handoff 0    -> B finishes 2.2 < 2.5, the chain moves
              handoff 0.1  -> B finishes 2.2 < 2.5, the chain still moves
              handoff 1.0  -> B finishes 3.0 > 2.5, the chain stays on A
    """
    p1 = Operation([1.0, 1.0], job_id=0)
    p2 = Operation([1.0, 1.0], predecessors=[p1], job_id=0)
    f1 = Operation([0.6, 0.6], job_id=1)
    f2 = Operation([0.6, 0.6], predecessors=[f1], job_id=1)
    g1 = Operation([0.5, 0.5], job_id=2, min_start_t=1.0)
    ops = [p1, p2, f1, f2, g1]
    transfer = np.array([[0.0, handoff_ms], [handoff_ms, 0.0]])
    w = Workload(ops, ["A", "B"], transfer, job_names=["P", "F", "G"],
                 machine_combinations=[["A"], ["B"]])
    priority = np.array([5.0, 1.0, 4.0, 3.0, 2.0])     # p1, f1, f2, g1, p2
    return w, priority


def _decode_affinity(handoff_ms):
    w, prio = _affinity_fixture(handoff_ms)
    ctx = DecoderContext(w)
    t, alpha = decode(ctx, prio)
    return t, np.argmax(alpha, axis=1)


def test_free_handoff_moves_the_chain():
    t, combo = _decode_affinity(0.0)
    assert combo[0] == 0 and combo[1] == 1, "p2 should hop from A to B"
    assert t[1] == pytest.approx(1.2)


def test_handoff_dearer_than_the_gain_keeps_the_chain_on_its_core():
    t, combo = _decode_affinity(1.0)
    assert combo[0] == 0 and combo[1] == 0
    assert t[1] == pytest.approx(1.5)            # waits for g1 on its own core


def test_handoff_cheaper_than_a_real_gain_still_moves():
    """A is busy for 0.5 ms after p1; a 0.1 ms hand-off is worth paying."""
    t, combo = _decode_affinity(0.1)
    assert combo[0] == 0 and combo[1] == 1
    assert t[1] == pytest.approx(1.2)
    # ... and the successor does not start before predecessor end + hand-off.
    assert t[1] >= 1.0 + 0.1 - 1e-12


def test_zero_gain_hop_disappears_with_any_handoff_cost():
    """The hop that costs nothing and gains nothing: k blocks A briefly, c1
    goes to B, and when c2 becomes ready both cores finish it at 2.0, so the
    first combination (A) wins the tie and the chain moves for no reason. Any
    positive hand-off cost breaks the tie in favour of staying."""
    def build(h):
        k = Operation([0.5, 0.5], job_id=0)
        c1 = Operation([1.0, 1.0], job_id=1)
        c2 = Operation([1.0, 1.0], predecessors=[c1], job_id=1)
        w = Workload([k, c1, c2], ["A", "B"], np.array([[0.0, h], [h, 0.0]]),
                     job_names=["K", "C"], machine_combinations=[["A"], ["B"]])
        t, alpha = decode(DecoderContext(w), np.array([3.0, 2.0, 1.0]))
        return np.argmax(alpha, axis=1)
    assert list(build(0.0)) == [0, 1, 0]
    assert list(build(0.01)) == [0, 1, 1]


def test_heft_family_reads_the_matrix_from_the_workload():
    """heft_schedule / heft_edf_schedule go through the same decode."""
    for h, moved in ((0.0, True), (1.0, False)):
        w, _ = _affinity_fixture(h)
        for fn in (heft_schedule, heft_edf_schedule):
            t, alpha = fn(w)
            combo = np.argmax(alpha, axis=1)
            # Whatever order HEFT picks, a successor never starts before its
            # predecessor's end plus the hand-off between their cores.
            for i, op in enumerate(w.operations):
                for p in op.predecessors:
                    j = w.operations.index(p)
                    gap = w.transfer_times[combo[j], combo[i]]
                    assert t[i] >= t[j] + p.processing_times[combo[j]] + gap - 1e-9
            if not moved:
                assert combo[0] == combo[1] and combo[2] == combo[3]


def test_k3_calibration_as_handoff_ms_gives_the_calibrated_matrix():
    """A calibration measured as same_kind_ms per kind and cross_kind_ms per
    direction, written as this tree's `scheduler.handoff_ms` table
    ({from kind: {to kind: ms}}), gives: 0 on the same core, same_kind_ms
    between two cores of one kind, cross_kind_ms from a core of kind A to a
    core of kind B. (The deprecated history built this matrix in its own
    handoff_calibration module; this tree's builder is the one used.)"""
    from workload_factory import build_handoff_transfer_times
    machines = ["CPU_P#0", "CPU_P#1", "CPU_E#0", "CPU_E#1"]
    same = {"CPU_P": 0.0, "CPU_E": 0.0050934343434359125}
    cross = {"CPU_P->CPU_E": 0.0005169270832787731, "CPU_E->CPU_P": 0.0005169270832787731}
    table = {"cpu_e": {"cpu_e": same["CPU_E"], "cpu_p": cross["CPU_E->CPU_P"]},
             "cpu_p": {"cpu_p": same["CPU_P"], "cpu_e": cross["CPU_P->CPU_E"]}}
    m = build_handoff_transfer_times(machines, table)
    assert m.shape == (4, 4)
    assert np.all(np.diag(m) == 0.0)
    assert m[0, 1] == m[1, 0] == same["CPU_P"]
    assert m[2, 3] == m[3, 2] == same["CPU_E"]
    assert m[0, 2] == m[1, 3] == cross["CPU_P->CPU_E"]
    assert m[2, 0] == m[3, 1] == cross["CPU_E->CPU_P"]


# --------------------------------------------------------------------------
# 2. handoff_metrics on a hand-written schedule
# --------------------------------------------------------------------------

def _handwritten_schedule():
    """Two instances.

    net0, in start order:
        d0  CPU_P#0            [0.0, 1.0]
        d1  CPU_P#0            [1.0, 1.5]   deps d0        same core
        d2  CPU_P#1            [1.5, 2.5]   deps d1        core change (same kind)
        d3  CPU_E#0+CPU_E#1    [2.5, 3.0]   deps d1, d2    core + kind change, width 2
        d4  CPU_E#0            [3.0, 3.25]  deps d3        master unchanged
      core changes 2, kind changes 1, cores touched {P0, P1, E0, E1} = 4,
      master cores {P0, P1, E0} = 3, composite 1,
      core-ms 1.0 + 0.5 + 1.0 + 2 * 0.5 + 0.25 = 3.75,
      dependency edges 5 (d0-d1, d1-d2, d1-d3, d2-d3, d3-d4); three of them
      change core: d1-d2 (P0->P1), d1-d3 (P0->E0), d2-d3 (P1->E0); and two of
      those, d1-d3 and d2-d3, change kind.

    net1: a single dispatch on CPU_E#1 [0, 2.0]; its dependency names a
    dispatch of another instance, which is not an edge inside the instance.
    """
    def d(job, did, target, start, dur, deps):
        return {"job_name": job, "id": did, "hardware_target": target,
                "start_time": start, "duration": dur, "dependencies": deps}
    return {"dispatches": {
        # deliberately not in start order
        "net0_d3": d("net0", 3, "CPU_E#0+CPU_E#1", 2.5, 0.5, ["net0_d1", "net0_d2"]),
        "net0_d0": d("net0", 0, "CPU_P#0", 0.0, 1.0, []),
        "net0_d2": d("net0", 2, "CPU_P#1", 1.5, 1.0, ["net0_d1"]),
        "net0_d1": d("net0", 1, "CPU_P#0", 1.0, 0.5, ["net0_d0"]),
        "net0_d4": d("net0", 4, "CPU_E#0", 3.0, 0.25, ["net0_d3"]),
        "net1_d0": d("net1", 0, "CPU_E#1", 0.0, 2.0, ["net0_d0"]),
    }}


def test_handoff_metrics_on_a_handwritten_schedule():
    m = handoff_metrics.schedule_metrics(_handwritten_schedule())
    a = m["instances"]["net0"]
    assert a["dispatches"] == 5
    assert a["core_changes"] == 2
    assert a["kind_changes"] == 1
    assert a["distinct_cores"] == 4
    assert a["distinct_master_cores"] == 3
    assert a["composite"] == 1
    assert a["core_ms"] == pytest.approx(3.75)
    assert a["dep_edges"] == 5
    assert a["dep_core_changes"] == 3
    assert a["dep_kind_changes"] == 2
    b = m["instances"]["net1"]
    assert (b["dispatches"], b["core_changes"], b["kind_changes"],
            b["distinct_cores"], b["composite"], b["dep_edges"]) == (1, 0, 0, 1, 0, 0)
    assert b["core_ms"] == pytest.approx(2.0)
    tot = m["totals"]
    assert tot["instances"] == 2 and tot["dispatches"] == 6
    assert tot["core_changes"] == 2 and tot["kind_changes"] == 1
    assert tot["composite"] == 1 and tot["distinct_cores"] == 4
    assert tot["core_ms"] == pytest.approx(5.75)
    assert handoff_metrics.median_over(m, r"^net\d+$") == 1.0
    assert handoff_metrics.median_over(m, r"^nothing") is None
    # A bare dispatches dict is accepted too.
    assert handoff_metrics.schedule_metrics(
        _handwritten_schedule()["dispatches"])["totals"] == tot


def test_handoff_metrics_can_leave_out_zero_duration_dispatches():
    """A zero-duration dispatch parked on another kind adds a hop there and a
    hop back; leaving it out shows the chain never moved otherwise."""
    def d(did, target, start, dur, deps):
        return {"job_name": "net0", "id": did, "hardware_target": target,
                "start_time": start, "duration": dur, "dependencies": deps}
    sched = {"dispatches": {
        "a": d(0, "CPU_E#0", 0.0, 1.0, []),
        "z": d(1, "CPU_P#0", 1.0, 0.0, ["a"]),
        "b": d(2, "CPU_E#0", 1.0, 1.0, ["z"]),
    }}
    m = handoff_metrics.schedule_metrics(sched)["instances"]["net0"]
    assert (m["core_changes"], m["kind_changes"], m["dep_core_changes"]) == (2, 2, 2)
    m = handoff_metrics.schedule_metrics(
        sched, skip_zero_duration=True)["instances"]["net0"]
    assert (m["dispatches"], m["core_changes"], m["kind_changes"]) == (2, 0, 0)
    assert m["dep_edges"] == 0              # the edges ran through the dropped dispatch


def test_handoff_metrics_cli(tmp_path, capsys):
    p = tmp_path / "sched.json"
    p.write_text(json.dumps(_handwritten_schedule()))
    assert handoff_metrics.main([str(p), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["instances"]["net0"]["core_changes"] == 2
    assert handoff_metrics.main([str(p), "--network", "^net0"]) == 0
    text = capsys.readouterr().out
    assert "net0" in text and "TOTAL" in text and "net1" not in text
    assert "median core_changes over /^net0/: 2" in text


def test_solution_metrics_match_the_schedule_file_metrics():
    """The solver-side helper (used by `--solver auto`'s log) sees what the
    schedule file will say."""
    w, prio = _affinity_fixture(0.0)
    for op, name in zip(w.operations, ("p1", "p2", "f1", "f2", "g1")):
        op.operation_name = name
    t, alpha = decode(DecoderContext(w), prio)
    m = handoff_metrics.solution_metrics(w, t, alpha)
    assert m["instances"]["P"]["core_changes"] == 1          # p2 hopped A -> B
    assert m["instances"]["F"]["core_changes"] == 0
    assert m["totals"]["core_ms"] == pytest.approx(1.0 + 1.0 + 0.6 + 0.6 + 0.5)
    assert m["totals"]["composite"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
