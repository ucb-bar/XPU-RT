#!/usr/bin/env python3
"""ROS 2 whole-network-pinning baseline over `sched_algo_sweep10`'s QRB5165 port.

Expressibility classifier + placement enumerator + whole-model scorer, ported
from RoSE/experiments/microros_pinsweep/pinsweep.py onto this target.

The deployment model being scored: **one ROS 2 node per network, its own
executor, the whole dispatch graph on one backend.** No per-op placement, no
tile splitting, no sharding. The only free variable is the map
`network -> backend`.

Unlike the FireSim reference, where a network's backend followed from which
hart its node sat on and hart relabellings had to be canonicalised, here the
backends (CPU / DSP / HTA) are genuinely distinguishable hardware and there is
nothing to canonicalise. Two networks pinned to the same backend contend for
it, and that contention is what the score models.

Subcommands
    classify   expressibility verdict per cell (the table SETUP.md quotes)
    costs      emit model_costs.json from the frozen cost model
    enumerate  emit plans/<cell>.json -- every legal assignment, scored+ranked
    table      compact per-cell summary of the enumeration
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
FLOWC = os.path.abspath(os.path.join(SWEEP, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))

#: the XPU-RT sweep this baseline is measured against
XSWEEP = os.path.join(FLOWC, "sweeps", "qrb5165_sched_algo_sweep10_20260908-210226")
ARM = "s10port"
TOPLEVEL = os.path.join(REPO, "data", "toplevel", ARM)

#: The pinning model's backend set. GPU is excluded: its measured per-dispatch
#: floor is 2307.9 us (qnn_models/opsweep/README.md, warm) and it is not the
#: fastest lane for a single network in the whole zoo, so no team pinning a
#: whole model would choose it. It stays in the *cells* -- the `cg` and `quad`
#: configs declare it -- but it is never a pinning candidate. See SETUP.md.
PIN_BACKENDS = ("cpu", "dsp", "hta")

#: registry kind -> QNN backend .so on the board
BACKEND_LIB = {
    "cpu": "libQnnCpu.so",
    "dsp": "libQnnDsp.so",
    "hta": "libQnnHta.so",
    "gpu": "libQnnGpu.so",
}
#: xpu-rt profile_hw label -> registry kind
HW_KIND = {"HTA": "hta", "DSP": "dsp", "CPU": "cpu", "GPU": "gpu"}

CTX_DIR = "/root/qnn_runtime_ctx"

#: where emitted artifacts go. Overridable with --out so reproduce.py can
#: re-emit into a scratch tree and diff, without touching the committed one.
OUT = SWEEP

# Full enumeration is measured on the board up to this many legal assignments;
# above it, only the ranked head plus contrast placements are measured.
FULL_ENUM_MAX = 8
#: how many ranked candidates to measure when the legal set is larger
SAMPLE_HEAD = 3


# ---------------------------------------------------------------- inputs
def bindings_path(net: str) -> str:
    if net == "vint":
        return os.path.join(FLOWC, "bindings", "vint.json")
    return os.path.join(XSWEEP, "bindings", f"{net}.json")


def load_cost_model():
    with open(os.path.join(XSWEEP, "cost_model.json")) as f:
        return json.load(f)


def load_bindings(net):
    p = bindings_path(net)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_cells():
    """{cell_name: workload dict} for the 42 ported cells."""
    out = {}
    for fn in sorted(os.listdir(TOPLEVEL)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(TOPLEVEL, fn)) as f:
            out[fn[: -len(".json")]] = json.load(f)
    return out


def cell_family_config(cell):
    """`networks_depth_chain_dc` -> ("depth_chain", "dc").

    Names that do not follow the sweep10 `networks_<family>_<config>` form --
    the 3net arm's shape names -- get their whole name as the family and no
    config; pin3net.py overrides both anyway.
    """
    if not cell.startswith("networks_"):
        return cell, ""
    base = cell[len("networks_"):]
    if "_" not in base:
        return base, ""
    cfg = base.rsplit("_", 1)[1]
    return base[: -(len(cfg) + 1)], cfg


def cell_lanes(wl):
    """Registry kinds the cell's machine declares, in slot order."""
    return [HW_KIND[v] for v in wl["hardware"]["profile_hw"].values()]


# ------------------------------------------------------------ model costs
def model_costs(cm=None):
    """{net: {backend: {"ms":.., "tiles":[(tile, ctx, graph, ms)]}}}.

    A whole-model pin pays the SUM of that model's tile costs on the one
    backend it is pinned to -- which for the fifteen single-tile networks is
    the tile, and for `vint` is its two tiles run back to back. A backend that
    cannot compose EVERY tile of a network cannot host that network at all:
    the point of whole-model pinning is that there is nowhere else to put the
    tile it cannot run.

    Costs come from this repo's own measured profile data, resolved exactly as
    `xpu-rt/profile_loader.py:find_profile_csv` resolves them -- which for this
    target is the frozen `cost_model.json` the XPU-RT sweep solved against, the
    same file `flow_c.py artifacts` re-emits the gen/profile/<HW>/... CSVs from.
    Both sides therefore read the same numbers.
    """
    cm = cm or load_cost_model()
    cells = cm["cells"]
    out = {}
    for net in sorted({n for wl in load_cells().values() for n in wl["networks"]}):
        man = load_bindings(net)
        if man is None:
            out[net] = {}
            continue
        per = {}
        for be in PIN_BACKENDS + ("gpu",):
            tiles = []
            ok = True
            for b in man["bindings"]:
                key = f'{man["network"]}/{b["name"]}'
                cost = (cells.get(key) or {}).get(be)
                bind = (b.get("backends") or {}).get(be)
                if cost is None or bind is None:
                    ok = False
                    break
                tiles.append({"tile": b["name"], "ctx": bind["ctx"],
                              "graph": bind["graph"], "us": cost,
                              "precision": bind.get("precision")})
            if ok and tiles:
                per[be] = {"ms": round(sum(t["us"] for t in tiles) / 1000.0, 4),
                           "n_tiles": len(tiles), "tiles": tiles}
        out[net] = per
    return out


# --------------------------------------------------------- expressibility
#: Harness limits, stated once and reported per cell by `classify`.
#:
#: L1  ITERATION SEMANTICS. Every network runs exactly its declared
#:     `num_instances`; the taskset is finite and the run ends when the last
#:     instance of the last network completes. There is no iteration cap
#:     (the reference's NET_A_MAX_ITERS=30) and no one-shot rewrite (the
#:     reference's "a one-shot network runs exactly once"): a cell declaring
#:     `yolov8_nano_se` at num_instances=2 gets two. This is the same
#:     semantics XPU-RT's runtime uses ("N/N entries executed"), which is why
#:     the two makespans are comparable at all.
#: L2  RELEASE SEMANTICS. Instance k of a periodic network is released at
#:     t0 + k*period and executes as soon as its node's executor is free after
#:     that. Releases are NEVER skipped. rcl's own wall timer skips to the next
#:     period boundary after an overrun, which would idle a busy node for up to
#:     a period per late instance and make the baseline look worse for a reason
#:     that has nothing to do with pinning; the node therefore drains all due
#:     releases in one callback. XPU-RT gates each entry on its release the same
#:     way, so this is the matched rule, not a favour.
#: L3  EXECUTOR FLOOR. One rclcpp SingleThreadedExecutor per node. Measured
#:     empty-callback floor is recorded per campaign in results/floor.json; a
#:     declared period below it is reported per cell rather than silently
#:     absorbed.
#: L4  ONE PROCESS. All of a cell's nodes live in one process, each on its own
#:     executor and its own thread, sharing one QNN backend handle per .so --
#:     which is exactly what XPU-RT's emitted runtime does (`SharedBackend` in
#:     runtime_main.cpp), and what makes two contexts on one backend legal.
#: L5  NO AFFINITY. Nodes are not pinned to cores and get no real-time
#:     priority. XPU-RT pins each lane to a core and asks for SCHED_FIFO. This
#:     is a deliberate asymmetry: an ordinary ROS 2 deployment does neither,
#:     and it is reported rather than corrected.
HARNESS_LIMITS = ["L1", "L2", "L3", "L4", "L5"]


def classify(cells, costs):
    """Expressibility verdict per cell. Ported from the reference's section 1.

    OK                      this deployment model expresses the cell as written
    INEXPRESSIBLE (NETS)    a network in the cell has no legal backend at all
    DEGENERATE (1-LANE)     exactly one legal assignment; no placement decision
    EDGE-WIRED              carries a cross-network edge, which ROS 2 CAN
                            express and this harness does; called out because
                            the reference could not and dropped it
    """
    rows = []
    for cell, wl in sorted(cells.items()):
        fam, cfg = cell_family_config(cell)
        lanes = cell_lanes(wl)
        pin_lanes = [l for l in lanes if l in PIN_BACKENDS]
        nets = wl["networks"]
        legal = {}
        blocked = {}
        for n in nets:
            ok = [b for b in pin_lanes if b in costs.get(n, {})]
            legal[n] = ok
            if not ok:
                blocked[n] = sorted(costs.get(n, {}))
        n_legal = 1
        for n in nets:
            n_legal *= max(len(legal[n]), 0)
        if blocked:
            verdict = "INEXPRESSIBLE (NETS)"
            n_legal = 0
        elif n_legal == 1:
            verdict = "DEGENERATE (1-LANE)"
        else:
            verdict = "OK"
        notes = []
        if wl.get("edges"):
            notes.append("EDGE-WIRED")
        dropped = [l for l in lanes if l not in PIN_BACKENDS]
        if dropped:
            notes.append("gpu-not-a-candidate" if "gpu" in dropped else
                         "lane-dropped:" + ",".join(dropped))
        rows.append(dict(cell=cell, family=fam, config=cfg,
                         lanes=lanes, pin_lanes=pin_lanes,
                         n_networks=len(nets), n_legal=n_legal,
                         verdict=verdict, notes=notes,
                         legal_per_net={n: legal[n] for n in nets},
                         blocked=blocked,
                         edges=wl.get("edges") or []))
    return rows


# ------------------------------------------------------------- the scorer
def simulate(wl, assign, costs, edges=None):
    """Whole-model fixed-taskset simulation of one `network -> backend` map.

    Every network is a node with its own executor; a node's instances run
    strictly in order.  Networks sharing a backend contend for it: the QNN
    backend serialises, so a backend is one FIFO server and the job that
    became ready first gets it.

    Instance k of a network is released at k*period (0 for an aperiodic
    network, whose instances are released together at t0) and additionally,
    if the network has an incoming edge, no earlier than the completion of
    instance k of its upstream -- the same instance-to-instance rule XPU-RT's
    scheduler applies to `edges`.

    Returns makespan (ms), per-network finish times, and the deadline misses,
    which are reported SEPARATELY and never folded into the makespan.
    """
    nets = wl["networks"]
    edges = edges if edges is not None else (wl.get("edges") or [])
    upstream = {e["to"]: e["from"] for e in edges}

    dur = {n: costs[n][assign[n]]["ms"] for n in nets}
    ninst = {n: int(nets[n].get("num_instances", 1)) for n in nets}
    period = {n: nets[n].get("period") for n in nets}
    window = {n: nets[n].get("window_duration") for n in nets}

    # per-network job queues, and per-backend server clocks
    nxt = {n: 0 for n in nets}                 # next instance to schedule
    node_free = {n: 0.0 for n in nets}         # that node's executor
    be_free = collections.defaultdict(float)   # the backend's FIFO server
    finish = {n: [] for n in nets}
    done = {n: [] for n in nets}               # completion of each instance

    def release(n, k):
        r = (period[n] or 0.0) * k if period[n] else 0.0
        up = upstream.get(n)
        if up is not None:
            if k >= len(done[up]):
                return None                    # upstream instance not done yet
            r = max(r, done[up][k])
        return r

    total = sum(ninst.values())
    guard = 0
    while sum(len(v) for v in done.values()) < total:
        guard += 1
        if guard > total * 8 + 64:
            raise RuntimeError("simulate() did not converge")
        # every network whose next instance has a known release
        cand = []
        for n in nets:
            if nxt[n] >= ninst[n]:
                continue
            r = release(n, nxt[n])
            if r is None:
                continue
            ready = max(r, node_free[n], be_free[assign[n]])
            cand.append((ready, r, nets[n]["id"], n))
        if not cand:
            raise RuntimeError("simulate() deadlocked on an edge")
        cand.sort()
        ready, r, _, n = cand[0]
        end = ready + dur[n]
        node_free[n] = end
        be_free[assign[n]] = end
        done[n].append(end)
        finish[n].append((nxt[n], r, ready, end))
        nxt[n] += 1

    makespan = max(max(v) for v in done.values())
    # The quantity the XPU-RT sweep RANKS on is not the wall clock: its
    # `evaluate(..., True)` objective is the makespan over NON-PERIODIC
    # operations only, because periodic ops carry their own windows. The wall
    # clock is usually pinned by the last periodic RELEASE -- (n-1)*period --
    # and is therefore nearly identical whatever the placement. Both are
    # computed here, and drive.py extracts both from the board, so the
    # comparison can be made against the one the ranking is about.
    np_ends = [max(done[n]) for n in nets if not period[n]]
    np_makespan = max(np_ends) if np_ends else makespan
    misses = {}
    for n in nets:
        if window[n] is None:
            continue
        m = sum(1 for (_k, r, _s, e) in finish[n] if e > r + window[n] + 1e-9)
        if m:
            misses[n] = m
    util = {}
    for be in set(assign.values()):
        busy = sum(dur[n] * ninst[n] for n in nets if assign[n] == be)
        util[be] = round(busy / makespan, 4) if makespan else None
    # a network that cannot keep its declared cadence on its pinned backend:
    # the live analogue of the reference's "starved to zero"
    overrun = [n for n in nets if period[n] and dur[n] > period[n]]
    return dict(makespan_ms=round(makespan, 4),
                np_makespan_ms=round(np_makespan, 4),
                np_degenerate=not np_ends,
                misses=misses, n_missed_nets=len(misses),
                n_missed_instances=sum(misses.values()),
                overrun_nets=overrun,
                backend_util=util,
                per_net_finish={n: round(max(done[n]), 4) for n in nets},
                per_net_instances={n: len(done[n]) for n in nets})


def assignment_id(assign, nets):
    return "_".join(f"{n}@{assign[n]}" for n in nets)


def short_id(idx):
    return f"a{idx}"


def enumerate_cell(cell, wl, costs, row):
    """Every legal assignment for one cell, scored and ranked.

    Ranking is `(networks starved to zero, then makespan)` -- deliberately not
    makespan alone.  In this harness the taskset is finite and every instance
    is run to completion, so `starved` is 0 on every completed run by
    construction and the key reduces to makespan; that is a real structural
    difference from the reference (whose window was closed by a one-shot and
    could therefore drop a network entirely) and it is reported, not hidden.
    Deadline misses stay OUT of the key and are reported separately.
    """
    nets = list(wl["networks"])
    options = [row["legal_per_net"][n] for n in nets]
    scored = []
    for combo in itertools.product(*options):
        assign = dict(zip(nets, combo))
        try:
            s = simulate(wl, assign, costs)
        except RuntimeError as exc:
            s = {"error": str(exc), "makespan_ms": float("inf"),
                 "np_makespan_ms": float("inf"), "np_degenerate": False,
                 "misses": {}, "n_missed_nets": 0, "n_missed_instances": 0}
        starved = sum(1 for n in nets
                      if s.get("per_net_instances", {}).get(n, 0) == 0)
        scored.append(dict(assign=assign, starved=starved, **s))
    scored.sort(key=lambda d: (d["starved"], d["makespan_ms"]))
    for i, s in enumerate(scored):
        s["rank"] = i
        s["id"] = short_id(i)
        s["label"] = assignment_id(s["assign"], nets)
    for i, s in enumerate(sorted(scored, key=lambda d: (d["starved"],
                                                        d["np_makespan_ms"]))):
        s["np_rank"] = i
    return scored


def pick_to_measure(scored, nets):
    """Which assignments actually go to the board.

    Full enumeration where the legal set is small; the ranked head plus
    contrast placements where it is large.  The contrasts are the two
    all-on-one-backend placements that bracket the space (best single lane and
    worst single lane) and the bottom-ranked assignment, so the measured set
    always spans the predicted range rather than only its optimistic end.
    """
    if len(scored) <= FULL_ENUM_MAX:
        return [s["id"] for s in scored], "full"
    want = [s["id"] for s in scored[:SAMPLE_HEAD]]
    # also the head of the NON-PERIODIC ranking, which is the objective the
    # XPU-RT sweep ranks solvers on and which the wall clock can hide
    for s in sorted(scored, key=lambda d: d["np_rank"])[:SAMPLE_HEAD]:
        if s["id"] not in want:
            want.append(s["id"])
    # the uniform placements, which are what a team does when it does not think
    uniform = [s for s in scored if len(set(s["assign"].values())) == 1]
    for s in (uniform[:1] + uniform[-1:] if uniform else []):
        if s["id"] not in want:
            want.append(s["id"])
    if scored[-1]["id"] not in want:
        want.append(scored[-1]["id"])
    return want, "sampled"


# ------------------------------------------------------------------ plans
def build_plan(cell, wl, costs, row, scored, measure_ids):
    nets = list(wl["networks"])
    fam, cfg = cell_family_config(cell)
    out = {
        "_comment": (
            "ROS 2 whole-network-pinning plan for one sched_algo_sweep10 cell. "
            "One node per network, one SingleThreadedExecutor per node, the "
            "whole model on one backend. @generated by scripts/pinsweep.py "
            "enumerate; do not hand-edit."),
        "cell": cell, "family": fam, "config": cfg,
        "lanes_declared": row["lanes"], "pin_lanes": row["pin_lanes"],
        "verdict": row["verdict"], "notes": row["notes"],
        "n_legal": row["n_legal"],
        "measure_mode": "full" if len(scored) <= FULL_ENUM_MAX else "sampled",
        "edges": row["edges"],
        "networks": {
            n: {"id": wl["networks"][n]["id"],
                "num_instances": int(wl["networks"][n].get("num_instances", 1)),
                "period_ms": wl["networks"][n].get("period"),
                "window_ms": wl["networks"][n].get("window_duration"),
                "legal_backends": row["legal_per_net"][n]}
            for n in nets},
        "assignments": [],
    }
    for s in scored:
        rec = {"id": s["id"], "rank": s["rank"], "label": s["label"],
               "assign": s["assign"],
               "predicted_makespan_ms": s["makespan_ms"],
               "predicted_np_makespan_ms": s["np_makespan_ms"],
               "np_degenerate": s.get("np_degenerate", False),
               "np_rank": s["np_rank"],
               "starved": s["starved"],
               "predicted_missed_nets": s["n_missed_nets"],
               "predicted_missed_instances": s["n_missed_instances"],
               "predicted_misses": s["misses"],
               "overrun_nets": s.get("overrun_nets", []),
               "backend_util": s.get("backend_util", {}),
               "measure": s["id"] in measure_ids}
        if s["id"] in measure_ids:
            rec["nodes"] = [
                {"network": n,
                 "backend": s["assign"][n],
                 "backend_lib": "/root/qairt/lib/target/" + BACKEND_LIB[s["assign"][n]],
                 "num_instances": int(wl["networks"][n].get("num_instances", 1)),
                 "period_ms": wl["networks"][n].get("period"),
                 "window_ms": wl["networks"][n].get("window_duration"),
                 "whole_model_ms": costs[n][s["assign"][n]]["ms"],
                 "tiles": [{"tile": t["tile"],
                            "ctx": os.path.join(CTX_DIR, t["ctx"]),
                            "graph": t["graph"],
                            "precision": t["precision"],
                            "predicted_ms": round(t["us"] / 1000.0, 4)}
                           for t in costs[n][s["assign"][n]]["tiles"]]}
                for n in nets]
        out["assignments"].append(rec)
    return out


# ------------------------------------------------------------------ mains
def cmd_costs(args):
    costs = model_costs()
    doc = {
        "_comment": (
            "Whole-model cost per (network, backend) for the ROS 2 pinning "
            "baseline: the SUM of that network's tile costs on the one backend "
            "it is pinned to, which is what whole-model pinning actually pays. "
            "Source is the FROZEN cost model the XPU-RT sweep solved against, "
            "qnn_models/flow_c/sweeps/qrb5165_sched_algo_sweep10_20260908-"
            "210226/cost_model.json, resolved through each network's binding "
            "manifest the way xpu-rt/profile_loader.py:find_profile_csv "
            "resolves gen/profile/<HW>/qrb5165_flowc/.../topo_0/results.csv -- "
            "the same file flow_c.py artifacts re-emits those CSVs from, so "
            "both sides of the comparison read identical numbers. A backend "
            "absent for a network is a backend on which some tile of it does "
            "not compose; whole-model pinning cannot put that tile elsewhere, "
            "so the network cannot be pinned there at all."),
        "unit": "ms",
        "source_cost_model": os.path.relpath(
            os.path.join(XSWEEP, "cost_model.json"), REPO),
        "pin_backends": list(PIN_BACKENDS),
        "networks": costs,
    }
    p = os.path.join(OUT, "model_costs.json")
    with open(p, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {p}")
    print(f'{"network":18s} ' + " ".join(f"{b:>9s}" for b in PIN_BACKENDS)
          + f'{"gpu":>10s}   best')
    for n, per in costs.items():
        cells = " ".join(f'{per[b]["ms"]:9.3f}' if b in per else f'{"--":>9s}'
                         for b in PIN_BACKENDS)
        g = f'{per["gpu"]["ms"]:10.3f}' if "gpu" in per else f'{"--":>10s}'
        pin = {b: per[b]["ms"] for b in PIN_BACKENDS if b in per}
        best = min(pin, key=pin.get) if pin else "NONE"
        ratio = ""
        if len(pin) > 1:
            ratio = f'  ({max(pin.values()) / min(pin.values()):.2f}x spread)'
        print(f"{n:18s} {cells}{g}   {best}{ratio}")
    return 0


def cmd_classify(args):
    cells = load_cells()
    costs = model_costs()
    rows = classify(cells, costs)
    p = os.path.join(OUT, "results", "expressibility.json")
    with open(p, "w") as f:
        json.dump({"harness_limits": HARNESS_LIMITS, "cells": rows}, f, indent=1)
    print(f"wrote {p}\n")
    print(f'{"cell":34s} {"lanes":12s} {"nets":>4s} {"n_legal":>7s}  verdict')
    for r in rows:
        print(f'{r["cell"]:34s} {"+".join(r["pin_lanes"]):12s} '
              f'{r["n_networks"]:4d} {r["n_legal"]:7d}  {r["verdict"]}'
              + (f'  [{",".join(r["notes"])}]' if r["notes"] else ""))
    c = collections.Counter(r["verdict"] for r in rows)
    print("\nverdicts:", dict(c))
    dist = collections.Counter(r["n_legal"] for r in rows)
    print("n_legal distribution:", dict(sorted(dist.items())))
    return 0


def cmd_enumerate(args):
    cells = load_cells()
    costs = model_costs()
    rows = {r["cell"]: r for r in classify(cells, costs)}
    os.makedirs(os.path.join(OUT, "plans"), exist_ok=True)
    index, n_meas = [], 0
    for cell, wl in sorted(cells.items()):
        row = rows[cell]
        if row["verdict"].startswith("INEXPRESSIBLE"):
            index.append(dict(cell=cell, verdict=row["verdict"], n_legal=0,
                              measured=0))
            continue
        scored = enumerate_cell(cell, wl, costs, row)
        measure_ids, mode = pick_to_measure(scored, list(wl["networks"]))
        plan = build_plan(cell, wl, costs, row, scored, set(measure_ids))
        p = os.path.join(OUT, "plans", f"{cell}.json")
        with open(p, "w") as f:
            json.dump(plan, f, indent=1)
        n_meas += len(measure_ids)
        best, worst = scored[0], scored[-1]
        index.append(dict(cell=cell, verdict=row["verdict"],
                          n_legal=len(scored), measured=len(measure_ids),
                          mode=mode,
                          best=best["label"], best_ms=best["makespan_ms"],
                          best_np_ms=best["np_makespan_ms"],
                          np_best=min(scored, key=lambda d: d["np_rank"])["label"],
                          np_best_ms=min(s2["np_makespan_ms"] for s2 in scored),
                          np_worst_ms=max(s2["np_makespan_ms"] for s2 in scored),
                          best_missed=best["n_missed_nets"],
                          runner_up=(scored[1]["label"] if len(scored) > 1 else None),
                          runner_up_ms=(scored[1]["makespan_ms"] if len(scored) > 1 else None),
                          delta=(round(scored[1]["makespan_ms"] / best["makespan_ms"], 4)
                                 if len(scored) > 1 and best["makespan_ms"] else None),
                          worst=worst["label"], worst_ms=worst["makespan_ms"],
                          spread=(round(worst["makespan_ms"] / best["makespan_ms"], 4)
                                  if best["makespan_ms"] else None)))
    p = os.path.join(OUT, "results", "enumeration.json")
    with open(p, "w") as f:
        json.dump({"full_enum_max": FULL_ENUM_MAX, "sample_head": SAMPLE_HEAD,
                   "cells": index}, f, indent=1)
    print(f"wrote {p}")
    print(f"{len(index)} cells, {n_meas} assignments to measure "
          f"({n_meas * 3} board runs at 3 reps)")
    return 0


def cmd_table(args):
    with open(os.path.join(SWEEP, "results", "enumeration.json")) as f:
        idx = json.load(f)["cells"]
    hdr = (f'{"cell":34s} {"n":>4s} {"meas":>4s} {"mode":8s} '
           f'{"best ms":>9s} {"miss":>4s} {"r-up ms":>9s} {"delta":>6s} '
           f'{"worst ms":>9s} {"spread":>7s}')
    print(hdr)
    for r in idx:
        if r.get("n_legal", 0) == 0:
            print(f'{r["cell"]:34s} {"--":>4s} {"--":>4s} {r["verdict"]}')
            continue
        print(f'{r["cell"]:34s} {r["n_legal"]:4d} {r["measured"]:4d} '
              f'{r["mode"]:8s} {r["best_ms"]:9.3f} {r["best_missed"]:4d} '
              f'{(r["runner_up_ms"] or 0):9.3f} '
              f'{(r["delta"] or 0):6.3f} {r["worst_ms"]:9.3f} '
              f'{(r["spread"] or 0):7.3f}')
    ns = [r["n_legal"] for r in idx if r.get("n_legal", 0)]
    print(f"\nn_legal: n={len(ns)} min={min(ns)} median={statistics.median(ns)} "
          f"max={max(ns)} total={sum(ns)}")
    print(f"to measure: {sum(r.get('measured', 0) for r in idx)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="emit into this directory instead of the sweep root")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("classify", cmd_classify), ("costs", cmd_costs),
                     ("enumerate", cmd_enumerate), ("table", cmd_table)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
    args = ap.parse_args()
    if args.out:
        global OUT
        OUT = os.path.abspath(args.out)
        os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
        os.makedirs(os.path.join(OUT, "plans"), exist_ok=True)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
