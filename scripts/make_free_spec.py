#!/usr/bin/env python3
"""A placement-free chain spec at any camera rate, derived from one already in the tree.

`wh_chain30_free.json` and `wh_chain45_free.json` are the same spec at two camera rates: diffing them
shows exactly four fields move, the perception and navigation `period` and their `num_instances`, and
nothing else -- not the windows, not the deadline, not the scheduler block, and above all not the
placement, which these specs deliberately do not state at all. So a spec for a third rate is that
substitution and nothing more; writing one by hand invites a fifth field to drift silently.

`num_instances` is how many releases fit the horizon, floor(horizon_ms / period), which is what both
existing specs carry (500/33.33 -> 15, 500/22.22 -> 22).

    scripts/make_free_spec.py 36 [--from data/toplevel/wh_chain30_free.json] [--out <path>]

Prints the fields it changed. Refuses to overwrite an existing spec unless --force, because a spec
that has been solved and flown is an input to figures already rendered.
"""
from __future__ import annotations
import argparse, json, math, os, sys

CHAIN_NETS = ("yolov8_nano_64x96", "fused_full")   # the camera-driven stages; control keeps its own tick


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rate_hz", type=float)
    ap.add_argument("--from", dest="src", default="data/toplevel/wh_chain30_free.json")
    ap.add_argument("--out")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    d = json.load(open(a.src))
    for k in ("allowed_machines", "machine_width", "preferred_hw", "pin", "affinity", "cores", "hart"):
        bad = [n for n, v in d["networks"].items() if k in v]
        if bad:
            return print(f"{a.src} constrains placement ({k} on {bad}); not a free spec", file=sys.stderr) or 2

    period = 1000.0 / a.rate_hz
    horizon = float(d["horizon_ms"])
    changed = []
    for n in CHAIN_NETS:
        if n not in d["networks"]:
            return print(f"{a.src} has no network {n}", file=sys.stderr) or 2
        net = d["networks"][n]
        inst = int(math.floor(horizon / period))
        changed += [(f"{n}.period", net["period"], period), (f"{n}.num_instances", net["num_instances"], inst)]
        net["period"], net["num_instances"] = period, inst

    src_rate = round(1000.0 / json.loads(json.dumps(json.load(open(a.src))))["networks"][CHAIN_NETS[0]]["period"])
    d["_comment"] = (d.get("_comment", "") +
                     f"\n\nDerived from {os.path.basename(a.src)} ({src_rate} Hz) by scripts/make_free_spec.py at "
                     f"{a.rate_hz:g} Hz: perception and navigation period {period:.4f} ms and num_instances "
                     f"{int(math.floor(horizon / period))} over the {horizon:g} ms horizon. Every other field, the "
                     f"windows and the deadline included, is the source spec's, and placement is stated nowhere.")

    out = a.out or f"data/toplevel/wh_chain{a.rate_hz:g}_free.json"
    if os.path.exists(out) and not a.force:
        return print(f"{out} exists; --force to overwrite (it may already be solved and flown)", file=sys.stderr) or 2
    json.dump(d, open(out, "w"), indent=2)
    print(f"wrote {out}")
    for f, was, now in changed:
        print(f"   {f:38s} {was} -> {now}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
