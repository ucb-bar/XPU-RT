#!/usr/bin/env python3
"""Feedback-stage ablation for QRB5165: what each knob is actually worth.

The K1 feedback loop turns ONE knob -- per-dispatch core width, read from
`topo_0 / topo_0_1 / ...` profiles. QRB5165 has no such profiles (every cell
is `topo_0`) and its lanes are heterogeneous accelerators, so that knob is
inert there. The knobs that *do* exist on this board are where a model is cut
and what each piece runs on.

This ladder adds one degree of freedom per stage, so each row's delta is
attributable to exactly one knob:

  S0 monolith        whole network, CPU int8 -- the no-accelerator fallback
  S1 +backend        whole network, best backend at int8
  S2 +precision      whole network, best (backend, precision)
  S3 +slice          best CONTIGUOUS cut set, per-tile (backend, precision),
                     tiles serialised
  S4 +branch         non-contiguous tiles allowed; one lane per backend kind,
                     so independent tiles overlap

Costs come from the measured slicing study (`experiments.jsonl`), pooled
across sweeps exactly as `analyze.py` does -- this is not a re-measurement,
it is an attribution of the measurements already taken.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(HERE, "..", "qnn_models", "slicing_study")
sys.path.insert(0, STUDY)

import analyze  # noqa: E402


def _is_branch(e: dict) -> bool:
    """True when two tiles are mutually independent (a real fork)."""
    d = analyze.deps(e["tiles"])
    idx = sorted(d)
    for i in idx:
        for j in idx:
            if i < j and j not in d[i] and i not in d[j]:
                # independent only if neither is a transitive ancestor
                if not _reaches(d, i, j) and not _reaches(d, j, i):
                    return True
    return False


def _reaches(d: dict[int, set[int]], src: int, dst: int) -> bool:
    seen, stack = set(), [dst]
    while stack:
        n = stack.pop()
        for p in d.get(n, ()):
            if p == src:
                return True
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return False


def _cell(e: dict, ti: int, backend: str | None, precision: str | None):
    """Cheapest cell for a tile under an optional backend/precision filter."""
    c = [(v, b, p) for (t, b, p), v in e["cell"].items()
         if t == ti
         and (backend is None or b == backend)
         and (precision is None or p == precision)]
    return min(c) if c else None


def _serial(e: dict, backend=None, precision=None):
    """Sum of per-tile best cells: no concurrency, DAG order irrelevant."""
    total, picks = 0.0, []
    for t in sorted(x["index"] for x in e["tiles"]):
        got = _cell(e, t, backend, precision)
        if got is None:
            return None, []
        v, b, p = got
        total += v
        picks.append(f"t{t}:{b}@{p}")
    return total / 1000.0, picks


def stages(net: str, pooled: dict) -> list[dict]:
    exps = {k: v for k, v in pooled.items() if v["network"] == net}
    if not exps:
        return []
    whole = [e for e in exps.values() if len(e["tiles"]) == 1
             and e["label"].endswith("_k1_whole")]
    cut = [e for e in exps.values() if len(e["tiles"]) > 1]
    contiguous = [e for e in cut if not _is_branch(e)]
    branch = [e for e in cut if _is_branch(e)]

    rows: list[dict] = []

    def add(stage, knob, ms, detail, label, tiles):
        rows.append({"stage": stage, "knob": knob, "ms": ms,
                     "detail": detail, "label": label, "tiles": tiles})

    if whole:
        w = whole[0]
        got = _cell(w, 0, "cpu", "int8")
        add("S0", "monolith (cpu int8)",
            got[0] / 1000.0 if got else None,
            "cpu@int8" if got else "no cpu int8 cell", w["label"], 1)
        got = _cell(w, 0, None, "int8")
        add("S1", "+backend", got[0] / 1000.0 if got else None,
            f"{got[1]}@{got[2]}" if got else "-", w["label"], 1)
        got = _cell(w, 0, None, None)
        add("S2", "+precision", got[0] / 1000.0 if got else None,
            f"{got[1]}@{got[2]}" if got else "-", w["label"], 1)

    best_c = None
    for e in contiguous:
        ms, picks = _serial(e)
        if ms is not None and (best_c is None or ms < best_c[0]):
            best_c = (ms, picks, e)
    if best_c:
        add("S3", "+slice (contiguous)", best_c[0], " ".join(best_c[1]),
            best_c[2]["label"], len(best_c[2]["tiles"]))

    best_b = None
    for e in branch:
        mk, assign, serial = analyze.best_makespan(e)
        if mk is not None and (best_b is None or mk < best_b[0]):
            best_b = (mk, assign, serial, e)
    if best_b:
        mk, assign, serial, e = best_b
        picks = " ".join(f"t{k}:{v}" for k, v in sorted(assign.items()))
        add("S4", "+branch (1 lane/kind)", mk / 1000.0, picks,
            e["label"], len(e["tiles"]))
        rows[-1]["serial_ms"] = serial / 1000.0 if serial else None
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--networks", default="vint,yolov8n,fused_full,dronet,mlp_control")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    pooled = analyze.pool(analyze.load())
    out = {}
    for net in [n.strip() for n in a.networks.split(",") if n.strip()]:
        rows = stages(net, pooled)
        if not rows:
            print(f"\n=== {net}: no slice records ===")
            continue
        out[net] = rows
        base = next((r["ms"] for r in rows if r["stage"] == "S0"), None)
        print(f"\n=== {net} ===")
        print(f"  {'stage':4s} {'knob':22s} {'ms':>9s} {'vs S0':>8s} {'step':>8s}  tiles  detail")
        prev = None
        for r in rows:
            ms = r["ms"]
            cum = f"{base/ms:.2f}x" if (base and ms) else "-"
            step = f"{prev/ms:.2f}x" if (prev and ms) else "-"
            shown = f"{ms:9.3f}" if ms is not None else f"{'n/a':>9s}"
            print(f"  {r['stage']:4s} {r['knob']:22s} {shown} {cum:>8s} {step:>8s}"
                  f"  {r['tiles']:5d}  {r['detail'][:52]}")
            if ms:
                prev = ms
        s4 = next((r for r in rows if r["stage"] == "S4"), None)
        if s4 and s4.get("serial_ms"):
            print(f"       concurrency worth {s4['serial_ms'] - s4['ms']:+.3f} ms "
                  f"({s4['serial_ms']:.3f} serial -> {s4['ms']:.3f} overlapped)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
