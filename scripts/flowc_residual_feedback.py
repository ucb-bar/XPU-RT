#!/usr/bin/env python3
"""Learn QRB5165 cost corrections from schedules that actually ran.

THE PROBLEM THIS EXISTS FOR. K1 and Chipyard schedule well off static solo
profiles because their dynamic terms are small: `contention_model.py` records
that K1 co-runner slowdown is *below the measurement's resolution* at the
co-runner counts we run. QRB5165 is not like that. A dispatch there crosses
FastRPC, may re-quantize at a tile boundary, and shares a backend with
whatever else the runtime has resident, so the solo cell is a systematically
wrong estimate -- and wrong in a DIFFERENT DIRECTION per backend.

Measured over the four committed board traces (400 dispatches):

    HTA   actual/predicted median 1.04 - 1.20   (solo cell UNDER-estimates)
    DSP   actual/predicted median 0.89 - 0.98   (solo cell OVER-estimates)
    CPU   actual/predicted median 0.98 - 1.15   (depends on co-residents)

The DSP direction corroborates `docs/Qualcomm/qualcomm-qrb5165.md` §2, which found the
recorded DSP column ~16% pessimistic and suspected it was captured under a
slower host clock.

WHAT THIS DOES. It reads (predicted, actual) pairs out of a run trace, fits a
correction, and writes it as a SEPARATE artifact -- the same two invariants
`contention_model.py` sets out:

  1. the correction is never folded into the solo profile on disk, so a
     re-profile cannot double-count it;
  2. a missing artifact is a no-op.

`apply` materialises a corrected profile tree under its own `gen_root`, so a
spec opts in by naming that root and nothing else in the toolchain changes.

Fitting on the runs you have and evaluating on the runs you do not is the
whole point: `transfer` holds out each trace in turn, so the number reported
is out-of-sample.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

SCHEMA = "xpurt.flowc_residual/v1"
EXCLUDE_STALLED = False   # set by --exclude-stalled
MIN_PRED_MS = 0.05      # below this the ratio is dominated by timer noise


def _corunners(rows: list[dict]) -> None:
    """Annotate each row with how many other dispatches overlapped it.

    This is the contention signal the solo profile cannot contain. It is
    computed from the trace's OWN actual intervals, so it needs no extra
    instrumentation -- a schedule that ran is already a contention
    measurement, which is the whole idea.
    """
    iv = [(r["_start"], r["_end"], i) for i, r in enumerate(rows)]
    for s0, e0, i in iv:
        n = sum(1 for s1, e1, j in iv if j != i and s1 < e0 and e1 > s0)
        rows[i]["co"] = n
        rows[i]["co_bucket"] = "solo" if n == 0 else ("1" if n == 1 else "2+")


def read_trace(path: str) -> list[dict]:
    out = []
    for r in csv.DictReader(open(path)):
        try:
            pred = float(r["predicted_duration_ms"])
            act = float(r["actual_end_ms"]) - float(r["actual_start_ms"])
            gate = float(r["gate_done_ms"])
            start = float(r["actual_start_ms"])
        except (KeyError, ValueError):
            continue
        if pred <= 0 or act <= 0:
            continue
        out.append({
            "_start": start,
            "_end": float(r["actual_end_ms"]),
            "trace": os.path.basename(os.path.dirname(path)),
            "network": r.get("network", ""),
            "seg": r.get("backend_label", ""),
            "backend": r.get("actual_backend", "?"),
            "pred_ms": pred,
            "act_ms": act,
            "ratio": act / pred,
            "stall_ms": max(0.0, start - gate),
            "usable": pred >= MIN_PRED_MS,
            "stalled": (start - gate) > 1.0,
        })
    _corunners(out)
    if EXCLUDE_STALLED:
        for r in out:
            if r["stalled"]:
                r["usable"] = False
    return out


def fit(rows: list[dict], conditioned: bool = False) -> dict:
    """Per-backend multiplicative correction, plus the stall distribution.

    The median is used, not the mean: the stall column is heavy-tailed (a
    context eviction is ~10^4 x the typical 6 us setup) and one eviction would
    otherwise move the whole correction.
    """
    by = defaultdict(list)
    for r in rows:
        if r["usable"]:
            key = (f"{r['backend']}|{r['co_bucket']}" if conditioned else r["backend"])
            by[key].append(r["ratio"])
    corr = {}
    for b, v in by.items():
        v = sorted(v)
        corr[b] = {
            "n": len(v),
            "factor": round(st.median(v), 4),
            "p10": round(v[len(v) // 10], 4),
            "p90": round(v[9 * len(v) // 10], 4),
            "geo_dispersion": round(
                math.exp(st.pstdev([math.log(x) for x in v])), 4) if len(v) > 1 else 1.0,
        }
    stalls = sorted(r["stall_ms"] for r in rows)
    return {
        "schema": SCHEMA,
        "conditioned": conditioned,
        "n_dispatches": len(rows),
        "backends": corr,
        "stall_ms": {
            "median": round(st.median(stalls), 4) if stalls else 0.0,
            "p99": round(stalls[int(0.99 * (len(stalls) - 1))], 4) if stalls else 0.0,
            "max": round(max(stalls), 4) if stalls else 0.0,
            "n_over_1ms": sum(1 for s in stalls if s > 1.0),
        },
    }


def error(rows: list[dict], model: dict | None) -> dict:
    """Prediction error, with and without the correction applied.

    `logerr` is the median |ln(actual/predicted)|: symmetric in over- and
    under-estimation, which a percentage is not. `mae_ms` is what a scheduler
    actually accumulates.
    """
    le, ae, n = [], [], 0
    for r in rows:
        if not r["usable"]:
            continue
        f = 1.0
        if model:
            b = model.get("backends", {})
            key = (f"{r['backend']}|{r['co_bucket']}"
                   if model.get("conditioned") else r["backend"])
            got = b.get(key)
            if got is None and model.get("conditioned"):
                # unseen bucket: fall back to any bucket for this backend
                cand = [v for k, v in b.items() if k.split("|")[0] == r["backend"]]
                got = cand[0] if cand else None
            f = (got or {}).get("factor", 1.0)
        pred = r["pred_ms"] * f
        le.append(abs(math.log(r["act_ms"] / pred)))
        ae.append(abs(r["act_ms"] - pred))
        n += 1
    return {"n": n,
            "logerr_median": round(st.median(le), 4) if le else None,
            "mae_ms": round(st.mean(ae), 4) if ae else None}


def cmd_learn(a) -> int:
    rows = []
    for p in sorted(glob.glob(a.traces)):
        rows += read_trace(p)
    if not rows:
        print(f"  no usable rows in {a.traces}")
        return 1
    model = fit(rows, a.conditioned)
    model["fit_on"] = sorted({r["trace"] for r in rows})
    with open(a.out, "w") as f:
        json.dump(model, f, indent=1)
    print(f"  fit on {len(rows)} dispatches from {len(model['fit_on'])} trace(s)")
    for b, c in sorted(model["backends"].items()):
        print(f"    {b:4s} n={c['n']:3d}  factor={c['factor']:.3f}  "
              f"[p10 {c['p10']:.3f}, p90 {c['p90']:.3f}]  dispersion={c['geo_dispersion']:.3f}")
    s = model["stall_ms"]
    print(f"    stalls: median={s['median']*1000:.1f} us  p99={s['p99']:.1f} ms  "
          f"max={s['max']:.1f} ms  ({s['n_over_1ms']} over 1 ms)")
    before, after = error(rows, None), error(rows, model)
    print(f"    in-sample  logerr {before['logerr_median']:.4f} -> {after['logerr_median']:.4f}"
          f"   MAE {before['mae_ms']:.3f} -> {after['mae_ms']:.3f} ms")
    print(f"  wrote {a.out}")
    return 0


def cmd_transfer(a) -> int:
    """Leave-one-trace-out: fit on the rest, score the held-out trace."""
    per = {}
    for p in sorted(glob.glob(a.traces)):
        per[os.path.basename(os.path.dirname(p))] = read_trace(p)
    if len(per) < 2:
        print("  need >= 2 traces for a transfer test")
        return 1
    print(f"  leave-one-out over {len(per)} traces "
          f"({sum(len(v) for v in per.values())} dispatches)\n")
    print(f"  {'held-out trace':28s} {'n':>4s} {'logerr':>16s} {'MAE ms':>18s}  verdict")
    wins = 0
    rows_out = []
    for held, rows in per.items():
        train = [r for k, v in per.items() if k != held for r in v]
        model = fit(train, a.conditioned)
        b, af = error(rows, None), error(rows, model)
        better = af["logerr_median"] < b["logerr_median"]
        wins += better
        print(f"  {held:28s} {af['n']:4d} "
              f"{b['logerr_median']:.4f} -> {af['logerr_median']:.4f} "
              f"{b['mae_ms']:8.3f} -> {af['mae_ms']:7.3f}  "
              f"{'improves' if better else 'WORSE'}")
        rows_out.append({"held_out": held, "before": b, "after": af,
                         "improved": better,
                         "factors": {k: v["factor"] for k, v in model["backends"].items()}})
    print(f"\n  out-of-sample: correction helps on {wins}/{len(per)} held-out traces")
    if a.json:
        json.dump(rows_out, open(a.json, "w"), indent=1)
        print(f"  wrote {a.json}")
    return 0


def cmd_apply(a) -> int:
    """Materialise a corrected profile tree under its own gen_root."""
    model = json.load(open(a.model))
    src, dst = a.gen_root, a.out_root
    n_files = n_rows = 0
    for path in glob.glob(os.path.join(src, "profile", "*", a.target, "**", "results.csv"),
                          recursive=True):
        backend = os.path.relpath(path, os.path.join(src, "profile")).split(os.sep)[0]
        factor = (model.get("backends", {}).get(backend, {}) or {}).get("factor")
        if factor is None:
            continue
        rel = os.path.relpath(path, src)
        out = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(path) as f:
            rd = list(csv.DictReader(f))
            cols = rd[0].keys() if rd else []
        for r in rd:
            for c in ("mean_time", "mean_time_ns"):
                if c in r and r[c]:
                    try:
                        r[c] = f"{float(r[c]) * factor:.6f}"
                    except ValueError:
                        pass
            n_rows += 1
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cols))
            w.writeheader()
            w.writerows(rd)
        n_files += 1
    print(f"  wrote {n_files} corrected results.csv ({n_rows} rows) under {dst}")
    print(f"  the solo tree at {src} is untouched; a spec opts in with "
          f'"gen_root": "{os.path.basename(dst)}"')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("learn", cmd_learn), ("transfer", cmd_transfer), ("apply", cmd_apply)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if name in ("learn", "transfer"):
            p.add_argument("--traces", default="runs/*/trace.csv")
            p.add_argument("--exclude-stalled", action="store_true",
                           help="drop dispatches whose start was delayed >1 ms by a "
                                "context stall, isolating the duration model from "
                                "the residency model")
            p.add_argument("--conditioned", action="store_true",
                           help="key the correction on (backend, co-runner bucket) "
                                "instead of backend alone")
        if name == "learn":
            p.add_argument("--out", default="artifacts/flowc_residual.json")
        if name == "transfer":
            p.add_argument("--json", default=None)
        if name == "apply":
            p.add_argument("--model", default="artifacts/flowc_residual.json")
            p.add_argument("--gen-root", default="gen")
            p.add_argument("--out-root", default="gen_corrected")
            p.add_argument("--target", default="qrb5165_v66")
    a = ap.parse_args()
    global EXCLUDE_STALLED
    EXCLUDE_STALLED = bool(getattr(a, "exclude_stalled", False))
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
