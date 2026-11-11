#!/usr/bin/env python3
"""The `3net` arm: the RoSE 3-network configs, pinned on the QRB5165.

`/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/data/toplevel/networks_3net_*.json`
and `3net_pairs/*.json` are fifteen FireSim workloads over the three base
networks the RISC-V flow runs -- `mlp_control`, `dronet`, `yolov8_nano` -- plus
`fused_full`. They cannot be run as written on this board: their
`profile_hw` is `gemmini_q31` / `V256D128_rvv` and their `profile.target` is a
FireSim bitstream. What ports is the WORKLOAD SHAPE -- which networks, at what
periods, windows and instance counts -- which is what makes a scheduling
problem hard and is the thing worth carrying over. That is the same
substitution `mk_workloads_qrb5165.py` made for sched_algo_sweep10, and it is
made explicit here rather than implied.

Two collapses happen on the way, and both are reported rather than hidden:

* **The machine axis collapses.** FireSim's `gempair` (2 P), `rvvpair` (2 E)
  and `hetero` (1+1) are counts of identical cores. This board has exactly one
  CPU, one cDSP, one HTA, so a whole-model pin always chooses from the same
  three lanes whatever the source config said. `gempair`, `hetero`, `milpfair`
  and `rvvpair` are therefore ONE workload here, and `armA`/`armB` (which
  differ only in FireSim bitstream) are another.
* **`yolov8_nano` becomes `yolov8n`.** The network that exists on this board is
  the 640x640 ultralytics export behind the QNN converter, not modelblaster's
  64x64 `yolov8_nano` -- the substitution `workloads/dronet_mlp_yolo.json`
  already documents. Absolute latencies are therefore NOT comparable with the
  FireSim numbers, and nothing here rests on such a comparison.

Costs come from this repo's own measured profile data, resolved exactly as
`xpu-rt/profile_loader.py:find_profile_csv` resolves it --
`gen/profile/<HW>/qrb5165_flowc/<net>/<net>.<quant>/*/topo_0/results.csv`,
summed over its dispatch rows, which for a whole-model pin is the whole model.
LEGALITY is decided by the binding manifest, never by the CSV: the artifact
emitter writes an exclusion cost for a lane a tile cannot use, and those are
large but finite (mlp_control 68.5 ms on HTA against 0.066 ms on CPU), so a
cost table alone would happily "pin" a network to a lane it cannot compose on.

    python3 pin3net.py shapes | enumerate | table
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
FLOWC = os.path.abspath(os.path.join(SWEEP, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))

sys.path.insert(0, HERE)
import pinsweep  # noqa: E402  (simulate/build_plan/ranking are shared)

ROSE = ("/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/data/toplevel")
TARGET = "qrb5165_flowc"
QUANT = "int8"
HW_LABEL = {"cpu": "CPU", "dsp": "DSP", "hta": "HTA", "gpu": "GPU"}

#: FireSim network name -> the network that exists on this board
NET_MAP = {"yolov8_nano": "yolov8n"}

PLANS = os.path.join(SWEEP, "plans3net")


# ------------------------------------------------------------------ inputs
def source_configs():
    fs = sorted(glob.glob(os.path.join(ROSE, "networks_3net_*.json")))
    fs += sorted(glob.glob(os.path.join(ROSE, "3net_pairs", "*.json")))
    out = {}
    for f in fs:
        with open(f) as fh:
            out[os.path.relpath(f, ROSE)] = json.load(fh)
    return out


def profile_cost_ms(net, backend):
    """Whole-model cost from this repo's profile tree, find_profile_csv rules."""
    hw = HW_LABEL[backend]
    base = f"{net}.{QUANT}"
    pats = [os.path.join(REPO, "gen", "profile", hw, TARGET, net, base, "*",
                         "topo_0", "results.csv"),
            os.path.join(REPO, "gen", "profile", hw, TARGET, net, base,
                         "topo_0", "results.csv")]
    hits = []
    for p in pats:
        hits += glob.glob(p)
    if not hits:
        return None, None
    p = max(hits, key=os.path.getmtime)
    rows = list(csv.DictReader(open(p)))
    tot = sum(float(r["mean_time_ns"]) for r in rows if r.get("mean_time_ns"))
    return round(tot / 1e6, 4), os.path.relpath(p, REPO)


def base_costs():
    """{net: {backend: {ms, tiles:[...]}}} for the four base networks.

    A backend is present only if the binding manifest gives EVERY tile of the
    network a context on it -- whole-model pinning has nowhere else to put a
    tile that will not compose.
    """
    out, prov = {}, {}
    for net in ("dronet", "mlp_control", "yolov8n", "fused_full"):
        man = json.load(open(os.path.join(FLOWC, "bindings", f"{net}.json")))
        per = {}
        for be in pinsweep.PIN_BACKENDS:
            tiles = []
            ok = True
            for b in man["bindings"]:
                bb = (b.get("backends") or {}).get(be)
                if not bb:
                    ok = False
                    break
                tiles.append({"tile": b["name"], "ctx": bb["ctx"],
                              "graph": bb["graph"],
                              "precision": bb.get("precision")})
            if not ok:
                continue
            ms, src = profile_cost_ms(net, be)
            if ms is None:
                continue
            # the whole-model cost is the network's total, so attribute it to
            # the tiles proportionally only for display; the pin pays the total
            for t in tiles:
                t["us"] = ms * 1000.0 / len(tiles)
            per[be] = {"ms": ms, "n_tiles": len(tiles), "tiles": tiles}
            prov[f"{net}/{be}"] = src
        out[net] = per
    return out, prov


# ------------------------------------------------------------------ shapes
def shape_key(d):
    nets = []
    for k, v in d["networks"].items():
        nets.append((NET_MAP.get(k, k), v.get("period"), v.get("window_duration"),
                     v.get("num_instances") or 1))
    return tuple(sorted(nets))


def shape_name(key):
    parts = []
    for n, per, win, inst in key:
        short = {"mlp_control": "mlp", "dronet": "dronet",
                 "yolov8n": "yolo", "fused_full": "fused"}.get(n, n)
        parts.append(f"{short}{inst}")
    return "3net_" + "_".join(parts)


def shapes():
    src = source_configs()
    groups = collections.defaultdict(list)
    for name, d in src.items():
        groups[shape_key(d)].append(name)
    out = []
    for key, names in sorted(groups.items(), key=lambda kv: shape_name(kv[0])):
        d = src[sorted(names)[0]]
        nets = {}
        for i, (orig, v) in enumerate(d["networks"].items()):
            n = NET_MAP.get(orig, orig)
            nets[n] = {"id": i, "identifier": n,
                       "num_instances": v.get("num_instances") or 1}
            if v.get("period"):
                nets[n]["period"] = v["period"]
                nets[n]["window_duration"] = v.get("window_duration", v["period"])
        out.append(dict(name=shape_name(key), sources=sorted(names),
                        networks=nets, edges=d.get("edges") or []))
    return out


# --------------------------------------------------------------- enumerate
def cmd_shapes(args):
    sh = shapes()
    costs, prov = base_costs()
    print(f'{"network":14s} ' + " ".join(f"{b:>9s}" for b in pinsweep.PIN_BACKENDS))
    for n, per in costs.items():
        print(f"{n:14s} " + " ".join(
            f'{per[b]["ms"]:9.3f}' if b in per else f'{"--":>9s}'
            for b in pinsweep.PIN_BACKENDS))
    print()
    for s in sh:
        legal = 1
        for n in s["networks"]:
            legal *= len([b for b in pinsweep.PIN_BACKENDS if b in costs.get(n, {})])
        print(f'{s["name"]:34s} n_legal={legal:3d}  '
              f'sources={",".join(x.replace(".json","") for x in s["sources"])}')
    return 0


def cmd_enumerate(args):
    sh = shapes()
    costs, prov = base_costs()
    os.makedirs(PLANS, exist_ok=True)
    index, total = [], 0
    for s in sh:
        wl = {"networks": s["networks"], "edges": s["edges"]}
        legal_per_net = {n: [b for b in pinsweep.PIN_BACKENDS if b in costs.get(n, {})]
                         for n in s["networks"]}
        row = dict(cell=s["name"], family="3net", config="qrb5165",
                   lanes=list(pinsweep.PIN_BACKENDS),
                   pin_lanes=list(pinsweep.PIN_BACKENDS),
                   verdict="OK", notes=["3net-arm", "machine-axis-collapsed"],
                   n_legal=1, legal_per_net=legal_per_net, blocked={},
                   edges=s["edges"], n_networks=len(s["networks"]))
        for n in s["networks"]:
            row["n_legal"] *= len(legal_per_net[n])
        scored = pinsweep.enumerate_cell(s["name"], wl, costs, row)
        # this arm is small enough to measure in full everywhere
        ids = [x["id"] for x in scored]
        plan = pinsweep.build_plan(s["name"], wl, costs, row, scored, set(ids))
        plan["family"] = "3net"
        plan["config"] = "qrb5165"
        plan["measure_mode"] = "full"
        plan["sources"] = s["sources"]
        plan["cost_provenance"] = prov
        plan["_comment"] = (
            "ROS 2 whole-network-pinning plan for one RoSE 3net workload shape, "
            "ported to the QRB5165. The FireSim machine axis collapses here "
            "(one core of each kind), so several source configs map to this one "
            "shape -- see `sources`. Costs are from this repo's gen/profile "
            "tree, resolved as find_profile_csv resolves it; legality is from "
            "the binding manifests. @generated by scripts/pin3net.py.")
        with open(os.path.join(PLANS, s["name"] + ".json"), "w") as f:
            json.dump(plan, f, indent=1)
        total += len(ids)
        index.append(dict(cell=s["name"], sources=s["sources"],
                          n_legal=len(scored), measured=len(ids), mode="full",
                          best=scored[0]["label"],
                          best_ms=scored[0]["makespan_ms"],
                          best_np_ms=scored[0]["np_makespan_ms"],
                          worst=scored[-1]["label"],
                          worst_ms=scored[-1]["makespan_ms"],
                          spread=round(scored[-1]["makespan_ms"]
                                       / scored[0]["makespan_ms"], 4)))
    p = os.path.join(SWEEP, "results", "enumeration_3net.json")
    with open(p, "w") as f:
        json.dump({"_comment":
                   "The 3net arm: RoSE's fifteen 3-network FireSim configs "
                   "collapse to these distinct workload shapes on this board, "
                   "because the machine axis (gempair/rvvpair/hetero) is a "
                   "count of identical cores and this board has one of each "
                   "kind. Full enumeration everywhere.",
                   "source_dir": ROSE, "cost_provenance": prov,
                   "cells": index}, f, indent=1)
    print(f"wrote {p}")
    print(f"{len(index)} shapes, {total} assignments to measure "
          f"({total * 3} board runs at 3 reps)")
    return 0


def cmd_table(args):
    idx = json.load(open(os.path.join(SWEEP, "results",
                                      "enumeration_3net.json")))["cells"]
    print(f'{"shape":34s} {"n":>4s} {"best ms":>9s} {"worst ms":>9s} '
          f'{"spread":>7s}  sources')
    for r in idx:
        print(f'{r["cell"]:34s} {r["n_legal"]:4d} {r["best_ms"]:9.3f} '
              f'{r["worst_ms"]:9.3f} {r["spread"]:7.3f}  '
              + ",".join(x.replace(".json", "").replace("3net_pairs/", "")
                         for x in r["sources"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("shapes", cmd_shapes), ("enumerate", cmd_enumerate),
                     ("table", cmd_table)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
