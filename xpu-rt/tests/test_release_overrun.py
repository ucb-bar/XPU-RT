#!/usr/bin/env python3
"""F5 release-overrun guard in the SGS decoder (2026-11-26).

A long op L (10 ms on core A, 30 ms on B) and a short time-released root R
(min_start 10 ms; 1 ms on A, 2 ms on B). Priority places L first, on A
[0, 10]. Earliest finish puts R on A at 10 (finish 11) rather than B (finish
12): right behind L. With margin 0.25, L's chain (released at 0) ends at
worst at 0 + 1.25 * 10 = 12.5, so R on A scores 11 + 2.5 = 13.5 against 12
on B, and R moves to B.

    pytest xpu-rt/tests/test_release_overrun.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from schedule_decoder import DecoderContext, decode
from workload import Operation, Workload

_COMBOS = [["A"], ["B"]]


def _wl(margin=None):
    long_ = Operation([10.0, 30.0], job_id=0)
    root = Operation([1.0, 2.0], job_id=1, min_start_t=10.0, max_end_t=20.0)
    w = Workload([long_, root], ["A", "B"], np.zeros((2, 2)), job_names=["L", "R"],
                 machine_combinations=_COMBOS)
    if margin is not None:
        w.release_overrun_margin = margin
    return w


def _core(alpha, i):
    return _COMBOS[int(np.argmax(alpha[i]))][0]


def test_off_keeps_the_root_behind_the_long_op():
    for m in (None, 0.0):
        t, alpha = decode(DecoderContext(_wl(m)), np.array([1.0, 0.0]))
        assert (_core(alpha, 0), _core(alpha, 1), t[1]) == ("A", "A", 10.0)


def test_margin_moves_the_root_off_the_core_with_the_long_op():
    t, alpha = decode(DecoderContext(_wl(0.25)), np.array([1.0, 0.0]))
    assert (_core(alpha, 0), _core(alpha, 1), t[1]) == ("A", "B", 10.0)


def test_small_margin_whose_overrun_does_not_beat_the_slower_core_keeps_it():
    """margin 0.05: A scores 11 + 0.5 = 11.5 < 12 on B -> stays on A."""
    _t, alpha = decode(DecoderContext(_wl(0.05)), np.array([1.0, 0.0]))
    assert _core(alpha, 1) == "A"


def test_env_variable_sets_the_margin(monkeypatch):
    monkeypatch.setenv("XPURT_RELEASE_OVERRUN_MARGIN", "0.25")
    _t, alpha = decode(DecoderContext(_wl()), np.array([1.0, 0.0]))
    assert _core(alpha, 1) == "B"
