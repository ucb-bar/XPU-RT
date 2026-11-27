#!/usr/bin/env python3
"""G2 (hyperperiod wrap-around) and G3 (A100 L2 co-location price) in the SGS decoder (2026-11-26).

    pytest xpu-rt/tests/test_wrap_l2.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from schedule_decoder import DecoderContext, decode
from workload import Operation, Workload


def _core(w, alpha, i):
    return w.get_machine_combinations()[int(np.argmax(alpha[i]))][0]


# ---- G2 -------------------------------------------------------------------
# H = 10. L (6 ms on A, 30 on B), released at 6, ends at 12 on A: 2 ms past H.
# R (1 ms on A, 2 on B), released at 0. One cycle: R on A at 0 (A is free there).
# With the wrap, A is busy [0, 2] at the start of every cycle: R on A would end
# at 3, on B at 2, so R goes to B.
def _wl_wrap(wrap=None):
    L = Operation([6.0, 30.0], job_id=0, min_start_t=6.0, max_end_t=20.0)
    R = Operation([1.0, 2.0], job_id=1, min_start_t=0.0, max_end_t=10.0)
    w = Workload([L, R], ["A", "B"], np.zeros((2, 2)), job_names=["L", "R"],
                 machine_combinations=[["A"], ["B"]])
    if wrap is not None:
        w.wrap_hyper_ms = wrap
    return w


def test_wrap_off_puts_the_root_behind_nothing():
    w = _wl_wrap()
    t, alpha = decode(DecoderContext(w), np.array([1.0, 0.0]))
    assert (_core(w, alpha, 0), _core(w, alpha, 1), t[1]) == ("A", "A", 0.0)


def test_wrap_on_moves_the_root_off_the_core_the_overflow_occupies():
    w = _wl_wrap(10.0)
    t, alpha = decode(DecoderContext(w), np.array([1.0, 0.0]))
    assert (_core(w, alpha, 0), _core(w, alpha, 1), t[1]) == ("A", "B", 0.0)


def test_wrap_env_variable(monkeypatch):
    monkeypatch.setenv("XPURT_WRAP_HYPER_MS", "10")
    w = _wl_wrap()
    _t, alpha = decode(DecoderContext(w), np.array([1.0, 0.0]))
    assert _core(w, alpha, 1) == "B"


# ---- G3 -------------------------------------------------------------------
# A100 cores CPU_E#0, CPU_E#1 (cluster 0) and CPU_E#4 (cluster 1). X (large
# weights) is placed first on E#0 [0, 10]. Y (large): 10 ms on E#0/E#1, 12 ms on
# E#4. Without the price Y goes to E#1 (finish 10); with it, E#1 co-runs with X
# in cluster 0 (k = 2, 1.9x -> 19 ms) and Y goes to E#4 (12 ms).
def _wl_l2(tmp_path):
    X = Operation([10.0, 40.0, 40.0], job_id=0, operation_id=1, min_start_t=0.0, max_end_t=40.0)
    Y = Operation([10.0, 10.0, 12.0], job_id=1, operation_id=1, min_start_t=0.0, max_end_t=40.0)
    w = Workload([X, Y], ["CPU_E#0", "CPU_E#1", "CPU_E#4"], np.zeros((3, 3)),
                 job_names=["yolov8_nano0", "yolov8_nano_b0"],
                 machine_combinations=[["CPU_E#0"], ["CPU_E#1"], ["CPU_E#4"]])
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"yolov8_nano": {"1": 600000}}))
    return w, str(p)


def test_l2_off(tmp_path):
    w, _p = _wl_l2(tmp_path)
    _t, alpha = decode(DecoderContext(w), np.array([1.0, 0.0]))
    assert (_core(w, alpha, 0), _core(w, alpha, 1)) == ("CPU_E#0", "CPU_E#1")


def test_l2_on_spreads_large_weights_over_clusters(tmp_path, monkeypatch):
    w, p = _wl_l2(tmp_path)
    monkeypatch.setenv("XPURT_L2_WEIGHTS", p)
    _t, alpha = decode(DecoderContext(w), np.array([1.0, 0.0]))
    assert (_core(w, alpha, 0), _core(w, alpha, 1)) == ("CPU_E#0", "CPU_E#4")


def test_l2_small_weights_are_not_priced(tmp_path, monkeypatch):
    w, _p = _wl_l2(tmp_path)
    q = tmp_path / "small.json"
    q.write_text(json.dumps({"yolov8_nano": {"1": 36000}}))
    monkeypatch.setenv("XPURT_L2_WEIGHTS", str(q))
    _t, alpha = decode(DecoderContext(w), np.array([1.0, 0.0]))
    assert _core(w, alpha, 1) == "CPU_E#1"
