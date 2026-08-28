#!/usr/bin/env python3
"""Predict a conv-dominated model's K1 service time at a different shape.

Why this exists: choosing a yolov8 resolution/width by training each candidate
costs a training run apiece. The 57 measured `conv2d_batchnorm2d_silu_s8`
dispatches in the current build already pin the cost of a convolution on this
board, so the candidates can be ranked before any of them is trained.

THE MODEL

    t_ms = a*MACs + b*out_elems + c*in_elems + d

Every term is physical rather than fitted decoration:

    a   arithmetic, ~1.5 GMAC/s on rvv_x60
    b   per-output-element: the BN + SiLU epilogue and the store
    c   per-INPUT-element: the input is re-read once per OC block. This term
        is independent confirmation of what the B3 rung measured directly --
        an OC-16 tile cost 80% of the OC-32 dispatch, not 50%, because each
        tile re-reads the whole input.
    d   fixed per-dispatch overhead, ~190 us. It does not scale with shape,
        which is why there is a floor no resolution change can go under.

WHAT IT IS NOT. A holdout (fit on the 47 smallest convs, predict the 10
largest) comes back ~30% LOW, so the model underestimates when extrapolating
toward bigger shapes. Predictions downward are better supported, but treat
absolute numbers as +/-30% and use this to RANK candidates, not to quote a
service time. A predicted number is not a measurement and must never be
written into a profile CSV.

SHAPES DO NOT COME FROM THE PROFILE. A fused conv records `noshape` there --
the shape lives on the fused op's first sub_op -- so this joins the IR graph
to the profile on dispatch_id. That join is safe here because both come from
the SAME build; across a rewrite it would not be (see diff_dispatch_graph.py).

Usage:
    scripts/predict_conv_cost.py --model yolov8_nano \
        --candidate 64:0.25 --candidate 96:0.125
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: yolov8 downsamples by 32 at its deepest level, so a non-multiple of 32 is
#: not a buildable configuration -- ModelBlaster enforces this on
#: MODELBLASTER_YOLOV8N_INPUT. Caught candidate D (imgsz 48) in review.
STRIDE_CONSTRAINT = 32

FUSED_CONV = "conv2d_batchnorm2d_silu_s8"


def _feats(s):
    return [s["OC"] * s["OH"] * s["OW"] * s["IC"] * s["KH"] * s["KW"],
            s["OC"] * s["OH"] * s["OW"],
            s["IC"] * s["IH"] * s["IW"],
            1.0]


def load(model, graph_path, profile_path):
    prof = {}
    with open(profile_path, newline="") as f:
        for r in csv.DictReader(f):
            prof[int(r["dispatch_id"])] = (r["op"], float(r["mean_time"]))
    g = json.load(open(graph_path))
    shapes, times, other = [], [], 0.0
    for o in g["ops"]:
        did = o.get("dispatch_id")
        if did is None or did not in prof:
            continue
        op, ms = prof[did]
        if op != FUSED_CONV:
            other += ms
            continue
        s = ((o.get("sub_ops") or [{}])[0]).get("shape") or {}
        if s:
            shapes.append(s)
            times.append(ms)
    return shapes, np.array(times), other


def fit(shapes, times):
    A = np.array([_feats(s) for s in shapes])
    coef, *_ = np.linalg.lstsq(A, times, rcond=None)
    pred = A @ coef
    r2 = 1 - ((times - pred) ** 2).sum() / ((times - times.mean()) ** 2).sum()
    return coef, r2


def holdout(shapes, times, n_big=10):
    """Fit without the largest convs and predict them. The honesty check."""
    A = np.array([_feats(s) for s in shapes])
    idx = np.argsort(times)
    small, big = idx[:-n_big], idx[-n_big:]
    c, *_ = np.linalg.lstsq(A[small], times[small], rcond=None)
    return float((A[big] @ c).sum()), float(times[big].sum())


def predict(shapes, coef, other_ms, base_res, base_w, res, w):
    rf, wf = res / base_res, w / base_w
    tot = 0.0
    for i, s in enumerate(shapes):
        t = dict(s)
        for k in ("IH", "IW", "OH", "OW"):
            t[k] = max(1, int(round(s[k] * rf)))
        t["OC"] = max(1, int(round(s["OC"] * wf)))
        # The first conv's IC is the image's channel count, not a width knob.
        t["IC"] = s["IC"] if i == 0 else max(1, int(round(s["IC"] * wf)))
        tot += float(np.array(_feats(t)) @ coef)
    return tot + other_ms * rf * rf * wf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8_nano")
    ap.add_argument("--graph", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--base-res", type=float, default=None,
                    help="default: read from the first conv's IH")
    ap.add_argument("--base-width", type=float, default=0.25)
    ap.add_argument("--candidate", action="append", default=[],
                    metavar="IMGSZ:WIDTH")
    a = ap.parse_args()

    m = a.model
    graph = a.graph or os.path.join(
        REPO, "ModelBlaster", "build", "k1_xpurt", m, "int8", "graph.json")
    profile = a.profile or os.path.join(
        REPO, "gen", "profile_mb", "rvv_x60", "spacemit_x60", m, f"{m}.int8",
        f"{m}_spacemit_x60_rvv_x60_{m}.int8", "topo_0", "results.csv")
    for p in (graph, profile):
        if not os.path.exists(p):
            print(f"missing input: {p}", file=sys.stderr)
            return 2

    shapes, times, other = load(m, graph, profile)
    if len(shapes) < 8:
        print(f"only {len(shapes)} shaped convs; too few to fit", file=sys.stderr)
        return 2
    coef, r2 = fit(shapes, times)
    base_res = a.base_res or float(shapes[0]["IH"])
    hp, hm = holdout(shapes, times)

    print(f"{m}: {len(shapes)} fused convs, {times.sum():.3f} ms; "
          f"other ops {other:.3f} ms")
    print(f"base shape: imgsz {base_res:.0f}, width {a.base_width}")
    print(f"model  t_ms = {coef[0]:.3e}*MACs + {coef[1]:.3e}*out "
          f"+ {coef[2]:.3e}*in + {coef[3]:.4f}   R^2={r2:.4f}")
    print(f"holdout (predict the 10 largest from the 47 smallest): "
          f"{hp:.1f} vs {hm:.1f} ms ({100 * (hp / hm - 1):+.0f}%) "
          f"-- treat predictions as +/-30% and use them to RANK")
    floor = coef[3] * len(shapes)
    print(f"per-dispatch overhead floor: {len(shapes)} x {coef[3]*1000:.0f} us "
          f"= {floor:.1f} ms ({1000/floor:.0f} Hz). Below this only reducing "
          f"the DISPATCH COUNT helps, not the dispatch sizes.\n")

    print(f"{'imgsz':>6s} {'width':>6s} {'pred ms':>9s} {'Hz':>7s}   note")
    for c in a.candidate:
        try:
            res_s, w_s = c.split(":")
            res, w = float(res_s), float(w_s)
        except ValueError:
            print(f"bad --candidate {c!r}, want IMGSZ:WIDTH", file=sys.stderr)
            return 2
        note = ("NOT BUILDABLE: imgsz must be a multiple of "
                f"{STRIDE_CONSTRAINT}") if int(res) % STRIDE_CONSTRAINT else ""
        ms = predict(shapes, coef, other, base_res, a.base_width, res, w)
        print(f"{res:6.0f} {w:6.3f} {ms:9.1f} {1000/ms:7.1f}   {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
