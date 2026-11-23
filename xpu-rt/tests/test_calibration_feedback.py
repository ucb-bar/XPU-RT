"""calibration_feedback + walker_sim on a hand-built two-network table."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calibration_feedback as cfb  # noqa: E402
import walker_sim as W  # noqa: E402


def _sched(tmp_path, entries):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"dispatches": entries}))
    return p


def _e(job, did, target, start, dur, deps=(), tdep=None, zero=False):
    d = {"id": did, "dependencies": list(deps), "hardware_target": target,
         "start_time": start, "duration": dur, "job_name": job}
    if not zero:
        d["module_name"] = f"{job}$dispatch_{did}_x"
    if tdep:
        d["time_dependency"] = tdep
    return f"{job}_dispatch_{did}", d


def test_table_order_gates_and_time_deps(tmp_path):
    s = dict([_e("big0", 0, "CPU_P#0", 0.0, 1.0),
              _e("fast0", 0, "CPU_P#0", 1.0, 0.1, tdep="big0_dispatch_0"),
              _e("fast0", 1, "CPU_P#0", 1.1, 0.1, deps=["fast0_dispatch_0"])])
    tab = W.load_table(_sched(tmp_path, s))
    assert [e.key for e in tab] == ["big0_dispatch_0", "fast0_dispatch_0", "fast0_dispatch_1"]
    assert [e.gate for e in tab] == [True, True, False]
    assert tab[1].tdep == 0 and tab[2].deps == [1]


def test_static_blocking_and_pool_relief(tmp_path):
    """A 2 ms 'big' op overruns to 3 ms on hart 0 and delays 'fast' (planned at
    1 ms). Pooling 'big' lets another hart run it, so 'fast' starts on time."""
    s = dict([_e("big0", 0, "CPU_P#0", 0.0, 1.0),
              _e("fast0", 0, "CPU_P#0", 1.0, 0.1, tdep="big0_dispatch_0")])
    tab = W.load_table(_sched(tmp_path, s))
    dur = lambda e: 3.0 if e.net == "big" else 0.1
    r = W.simulate(tab, dur, policy="static", overhead_ms=0.0)
    assert r.start[1] == pytest.approx(3.0)
    # With the planned 1 ms as its estimate, hart 0 thinks 'big' fits before its
    # own 'fast' (due at 1 ms), takes it, and the 3 ms overrun still blocks 'fast'.
    r = W.simulate(tab, dur, policy="pool", aperiodic={"big"}, overhead_ms=0.0,
                   all_harts={"cpu_p": [0, 1]})
    assert r.ran_on[0] == 0 and r.start[1] == pytest.approx(3.0)
    # With a calibrated estimate (3 ms), hart 0 refuses; idle hart 1 takes it.
    r = W.simulate(tab, dur, policy="pool", aperiodic={"big"}, overhead_ms=0.0,
                   all_harts={"cpu_p": [0, 1]}, estimate=lambda e: 3.0)
    assert r.ran_on[0] == 1 and r.start[1] == pytest.approx(1.0)
    assert r.n_refused >= 1


def test_update_blames_cross_network_blocking_and_reserves(tmp_path):
    # fast (period 1 ms, window 0.5) queued behind 'big' on hart 0 in every instance
    ents = []
    for i in range(4):
        ents.append(_e(f"big{i}", 0, "CPU_P#0", i * 1.0, 0.2))
        ents.append(_e(f"fast{i}", 0, "CPU_P#0", i * 1.0 + 0.2, 0.05, tdep=f"big{i}_dispatch_0"))
    tab = W.load_table(_sched(tmp_path, dict(ents)))
    dur = lambda e: 0.9 if e.net == "big" else 0.05      # big overruns 0.2 -> 0.9
    res = W.simulate(tab, dur, policy="static", overhead_ms=0.0)
    p = tmp_path / "t.csv"
    W.write_trace(tab, res, p)
    tr = W.read_trace(p, tab)
    names = {h: f"CPU_P#{h}" for h in range(4)}
    doc = cfb.update(cfb.empty(), tab, tr, "r0", windows={"fast": 0.5}, periods={"fast": 1.0},
                     hart_names=names, all_harts={"cpu_p": [0, 1, 2, 3]})
    assert doc["last_round"]["missed"]["fast"] == 4
    assert doc["blame"]["fast"]["blocked_by_ms"]["big"] > 0
    assert doc["reserve"] == {"fast": ["CPU_P#0"]}
    # big's factor is a quantile of measured / raw (0.9 / 0.2)
    assert doc["ops"]["big|dispatch_0|cpu_p"]["value"] == pytest.approx(4.0)   # clamped at 4


def test_apply_prices_other_networks_off_reserved_harts():
    doc = cfb.empty()
    doc["reserve"] = {"fast": ["CPU_P#0"]}
    doc["ops"]["big|dispatch_0|cpu_p"] = {"mode": "ratio", "value": 1.5}
    doc["ops"]["fast|dispatch_0|cpu_p"] = {"mode": "add", "value": 0.01}
    doc["gap_ms"]["big|cpu_p"] = 0.002
    pt = {"big_dispatch_0": [1.0, 1.0], "fast_dispatch_0": [0.005, 0.005]}
    st = cfb.apply(pt, ["big", "fast"], [["CPU_P#0"], ["CPU_P#1"]], ["cpu_p", "cpu_p"], doc)
    assert pt["big_dispatch_0"][0] == pytest.approx(1.5 + 0.002 + cfb.PENALTY_MS)
    assert pt["big_dispatch_0"][1] == pytest.approx(1.502)
    assert pt["fast_dispatch_0"] == [pytest.approx(0.015), pytest.approx(0.015)]
    assert st["values_priced_off_reserved"] == 1


def test_tiny_ops_get_an_additive_correction(tmp_path):
    ents = dict(_e(f"t{i}", 0, "CPU_P#0", i * 1.0, 0.001) for i in range(5))
    tab = W.load_table(_sched(tmp_path, ents))
    res = W.simulate(tab, lambda e: 0.011, policy="static", overhead_ms=0.0)
    p = tmp_path / "t.csv"
    W.write_trace(tab, res, p)
    doc = cfb.update(cfb.empty(), tab, W.read_trace(p, tab), "r0", windows={}, periods={})
    rec = doc["ops"]["t|dispatch_0|cpu_p"]
    assert rec["mode"] == "add" and rec["value"] == pytest.approx(0.010)
