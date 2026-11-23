"""measured/predicted scheduling feedback: update rule and apply step."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import duration_feedback as dfb  # noqa: E402


def _ob(meas, pred, net="dronet", disp="dispatch_12", kind="cpu_p"):
    return {"network": net, "dispatch": disp, "kind": kind,
            "measured_ms": meas, "predicted_ms": pred}


def test_default_factor_is_one():
    assert dfb.factor(dfb.empty(), "dronet", "dispatch_12", "cpu_p") == 1.0
    assert dfb.load(None)["ops"] == {}


def test_round_averages_over_instances():
    doc = dfb.update(dfb.empty(), [_ob(1.1, 1.0), _ob(1.3, 1.0)], "r0")
    assert dfb.factor(doc, "dronet", "dispatch_12", "cpu_p") == pytest.approx(1.2)
    assert doc["ops"]["dronet|dispatch_12|cpu_p"]["n_instances"] == [2]


def test_second_round_is_anchored_to_raw_profile_not_compounded():
    # raw profile 1.0 ms. Round 0 (factor 1): measured 1.16 -> factor 1.16.
    d0 = dfb.update(dfb.empty(), [_ob(1.16, 1.0)], "r0")
    assert dfb.factor(d0, "dronet", "dispatch_12", "cpu_p") == pytest.approx(1.16)
    # Round 1 was solved with predicted = 1.16; board measures 1.17 (raw x1.17).
    d1 = dfb.update(d0, [_ob(1.17, 1.16)], "r1")
    rec = d1["ops"]["dronet|dispatch_12|cpu_p"]
    assert rec["samples"] == pytest.approx([1.16, 1.17])
    assert rec["factor"] == pytest.approx(1.165)          # mean, not 1.16*1.0086
    assert d1["rounds"] == ["r0", "r1"]


def test_update_does_not_mutate_input_and_keeps_unobserved_ops():
    d0 = dfb.update(dfb.empty(), [_ob(2.0, 1.0, disp="dispatch_1"),
                                  _ob(1.5, 1.0, disp="dispatch_2")], "r0")
    snapshot = repr(d0)
    d1 = dfb.update(d0, [_ob(2.2, 2.0, disp="dispatch_1")], "r1")
    assert repr(d0) == snapshot
    assert dfb.factor(d1, "dronet", "dispatch_2", "cpu_p") == pytest.approx(1.5)
    assert d1["ops"]["dronet|dispatch_2|cpu_p"]["samples"] == pytest.approx([1.5])
    assert dfb.factor(d1, "dronet", "dispatch_1", "cpu_p") == pytest.approx((2.0 + 2.2) / 2)


def test_kinds_are_separate_and_bad_samples_ignored():
    doc = dfb.update(dfb.empty(), [_ob(1.2, 1.0, kind="CPU_P"), _ob(0.8, 1.0, kind="cpu_e"),
                                   _ob(1.0, 0.0), _ob(0.0, 1.0)], "r0")
    assert dfb.factor(doc, "dronet", "dispatch_12", "cpu_p") == pytest.approx(1.2)
    assert dfb.factor(doc, "dronet", "dispatch_12", "cpu_e") == pytest.approx(0.8)


def test_machine_kind():
    assert dfb.machine_kind("CPU_P#3") == "cpu_p"
    assert dfb.machine_kind("CPU_E#0+CPU_E#1") == "cpu_e"


def test_apply_scales_matching_kind_only_and_skips_sentinels():
    doc = dfb.update(dfb.empty(), [_ob(1.5, 1.0, net="yolov8_nano", disp="dispatch_3")], "r0")
    pt = {"yolov8_nano_dispatch_3": [2.0, 4.0, 1e6],
          "yolov8_nano_dispatch_4": [2.0, 4.0, 1.0],
          "dronet_dispatch_3": [2.0, 4.0, 1.0]}
    st = dfb.apply(pt, ["dronet", "yolov8_nano"], ["cpu_p", "cpu_e", "cpu_p"], doc)
    assert pt["yolov8_nano_dispatch_3"] == pytest.approx([3.0, 4.0, 1e6])
    assert pt["yolov8_nano_dispatch_4"] == [2.0, 4.0, 1.0]
    assert pt["dronet_dispatch_3"] == [2.0, 4.0, 1.0]
    assert st == {"ops_scaled": 1, "values_scaled": 1}


def test_apply_with_empty_doc_is_identity():
    pt = {"dronet_dispatch_0": [1.0, 2.0]}
    assert dfb.apply(pt, ["dronet"], ["cpu_p", "cpu_e"], dfb.empty())["values_scaled"] == 0
    assert pt == {"dronet_dispatch_0": [1.0, 2.0]}


def test_roundtrip(tmp_path):
    doc = dfb.update(dfb.empty(), [_ob(1.16, 1.0)], "r0")
    p = tmp_path / "f.json"
    dfb.save(doc, p)
    assert dfb.load(p) == doc
