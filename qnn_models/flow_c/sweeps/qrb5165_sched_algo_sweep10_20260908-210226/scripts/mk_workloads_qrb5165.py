#!/usr/bin/env python3
"""Phase 2 — the 11 sched_algo_sweep10 workload families, re-expressed on the
QRB5165's own machine model.

Ported from RoSE/experiments/workload_gen/mk_workloads.py. The family
definitions -- which networks, how many instances, and the period MULTIPLE --
are carried over VERBATIM, because those are what make a scheduling problem
hard and they are the thing being replicated. Three things change, and nothing
else:

1. THE MACHINE MODEL. FireSim's config axis is two machine kinds at two counts
   each (rvvpair 0+2, gempair 2+0, hetero 1+1, quad 2+2). That does not port:
   `cores/qrb5165_qnn.json` has exactly ONE core of each kind, and
   `flowc/mb.py::install_slot_map` refuses a slot index past the number of
   cores of that kind, so `cpu_x: 2` is not a thing this flow can emit a
   runtime for. What this board offers instead is a CHOICE OF LANES, so the
   config axis becomes which lane subset is available:

       hd     hta + dsp                  two accelerator lanes  (gempair's role)
       dc     dsp + cpu                  accelerator + general  (hetero's role)
       cg     cpu + gpu                  general + slow accel   (rvvpair's role)
       quad   hta + dsp + cpu + gpu      the whole part         (quad's role)

   Three two-lane configs -- a fast pair, a mixed pair and a slow pair -- plus
   one four-lane config, which is the same shape as the reference's three
   two-machine configs plus `quad`.

   HTA is a lane because Phase 1R MADE it one. Phase 1 measured that the
   reference's networks, as the zoo exports them, compose on HTA for exactly
   one of sixteen (fastdepth); the other fifteen were rejected on Batchnorm,
   Elu or StridedSlice. Eleven of those were then unlocked by numerics-
   preserving graph rewrites, so the cost model now carries twelve HTA cells
   instead of one. The four that are not unlocked are the mlp_control rungs,
   where Elu has no algebraic equivalent and the numerics-CHANGING probe
   measured 2.186 ms on HTA against 0.176 ms on the CPU -- 12x worse, so
   there is nothing there to want. A network with no cell on a config's lanes
   makes that cell fail predicate 5 and it is not generated.

   The arms are matched WITHIN THIS TARGET ONLY. No absolute latency in this
   sweep is comparable with the reference's FireSim numbers.

2. PERIODS ARE RE-DERIVED FROM THIS BOARD'S MEASUREMENTS. The reference
   anchors an analytic MAC estimate to measured single-hart FireSim runtimes.
   Those numbers are wrong here by two orders of magnitude in both directions,
   so every period comes from Phase 1's own gap-phase cells: for each network,
   the slowest and fastest lane cost IN THIS CONFIG, exactly as the reference
   does over its own two backends. A family is REJECTED for a config in which
   some network has no measured cell on any of its lanes.

   The consequence is that a family's periods differ between configs, which is
   also true in the reference (its `cost` dict is per-pair). Within a config
   every solver sees identical durations, which is the comparison the sweep10
   ranking claim is actually about.

3. THE DISPATCH SPACE IS TILES, NOT OPERATORS. `dispatch_deps_path` points at
   the coarse dispatch graphs flowc/artifacts.py emits, one dispatch per
   binding tile. Phase 1 built each network as ONE tile, so op counts here are
   far smaller than the reference's 126-801.

   That is also why THE SHARD ARM DOES NOT PORT. `wl_sweep_shard` is the same
   networks with their operators sharded across two harts, which shows up as
   more dispatches. This target has no such primitive: the work inside a QNN
   dispatch belongs to HVX, the tensor accelerator or the CPU op package, and
   the host cannot subdivide it (qnn_models/flow_c/README.md, "Threading and
   tiling -- why not modelblaster's"). A granularity arm here would mean
   re-slicing and recompiling every network, which is a different experiment.
   This sweep therefore has ONE arm, and says so.

    python3 mk_workloads_qrb5165.py [--emit] [--configs dc,dg,cg,dcg]
"""
from __future__ import annotations

import argparse, collections, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
FLOWC = os.path.abspath(os.path.join(SWEEP, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))

ARM = "s10port"                      # the single arm; see docstring point 3
TARGET = "qrb5165_flowc"
GEN_ROOT = "gen"

#: lane subset -> {xpu-rt machine slot: registry core kind}. Slot names are
#: what `build_machine_combinations` keys on and what `install_slot_map`
#: resolves; CPU_P/CPU_E/CPU_X are the three the ingest already knows.
CONFIGS = {
    # two accelerator lanes -- the "both fast" pair, gempair's role
    "hd":   {"CPU_P": "hta", "CPU_E": "dsp"},
    # accelerator + general purpose -- hetero's role
    "dc":   {"CPU_P": "dsp", "CPU_E": "cpu"},
    # general purpose + the slow accelerator -- the "both slow" pair,
    # rvvpair's role
    "cg":   {"CPU_P": "cpu", "CPU_E": "gpu"},
    # the whole part -- quad's role
    "quad": {"CPU_P": "hta", "CPU_E": "dsp", "CPU_X": "cpu", "CPU_G": "gpu"},
}
#: registry kind -> the profile_hw label artifacts.py writes profiles under
HW_LABEL = {"hta": "HTA", "dsp": "DSP", "cpu": "CPU", "gpu": "GPU"}

#: Family definitions, VERBATIM from mk_workloads.py's build(): each entry is
#: network -> (num_instances, period as a MULTIPLE of that network's own
#: worst-lane cost; None = aperiodic/one-shot). `edges` are network-level
#: precedence pairs that workload_factory turns into real dispatch precedence.
FAMILIES = [
    ("tight_loop", {"mlp_control_sa": (16, 3.0), "mlp_control_sb": (8, 4.0),
                    "dronet_sa": (4, 6.0)}, []),
    ("control_mix", {"mlp_control_sd": (8, 4.0), "dronet_se": (4, 5.0),
                     "yolov8_nano_sc": (1, None)}, []),
    ("perception_heavy", {"yolov8_nano_sf": (1, None),
                          "mlp_control_sd": (4, 8.0)}, []),
    ("scale_ladder", {f"dronet_s{k}": (1, None) for k in "bcdefg"}, []),
    ("bimodal", {"yolov8_nano_sh": (1, None), "mlp_control_sa": (32, 2.0)}, []),
    ("vint_intro", {"vint": (1, None), "mlp_control_sf": (8, 6.0)}, []),
    ("vint_multi", {"vint": (1, None), "dronet_se": (4, 8.0),
                    "mlp_control_sd": (8, 6.0)}, []),
    ("saturation", {"yolov8_nano_se": (2, None), "dronet_sf": (6, 2.0),
                    "mlp_control_sf": (16, 1.5)}, []),
    ("depth_chain", {"fastdepth": (2, 1.5), "dronet_sf": (2, 1.5)},
     [{"from": "fastdepth", "to": "dronet_sf"}]),
    ("depth_nav", {"fastdepth": (2, 1.5), "dronet_sf": (2, 1.5),
                   "mlp_control_sd": (16, 4.0)},
     [{"from": "fastdepth", "to": "dronet_sf"}]),
    ("depth_contended", {"fastdepth": (2, 1.5), "dronet_sf": (2, 1.5),
                         "yolov8_nano_sc": (1, None)},
     [{"from": "fastdepth", "to": "dronet_sf"}]),
]

#: Flow C binding manifest per network, relative to qnn_models/flow_c/.
#: Every entry except vint is written by Phase 1 into this sweep's own
#: bindings/ dir; vint reuses the shipped two-tile manifest and its existing
#: measured cells (see SETUP.md).
def bindings_rel(net: str) -> str:
    if net == "vint":
        return "bindings/vint.json"
    return os.path.join(os.path.relpath(SWEEP, FLOWC), "bindings", f"{net}.json")


def load_cost_model(path):
    with open(path) as f:
        return json.load(f)


def tile_costs(cost_model, bindings_dir, net):
    """[(tile_name, {kind: us})] for one network, in binding order."""
    man_path = os.path.join(FLOWC, bindings_rel(net))
    if not os.path.exists(man_path):
        return None                      # network not built — caller rejects
    with open(man_path) as f:
        man = json.load(f)
    cells = cost_model.get("cells", {})
    out = []
    for b in man["bindings"]:
        key = f'{man["network"]}/{b["name"]}'
        measured = {k: v for k, v in (cells.get(key) or {}).items()
                    if v is not None and "@" not in k}
        out.append((b["name"], measured))
    return out


def net_cost_ms(cost_model, net, kinds):
    """(slowest_ms, fastest_ms) for one network restricted to `kinds`.

    A network is the SUM of its tiles: with one tile per network that is the
    tile, and with vint's two tiles it is the serial critical path, which is
    what a period has to cover. Per lane first, then min/max across lanes, so
    a lane that cannot run one tile disqualifies that lane for the network
    rather than being silently mixed in per tile.
    """
    tiles = tile_costs(cost_model, None, net)
    if tiles is None:
        return None, None, None
    per_lane = {}
    for k in kinds:
        if all(k in m for _, m in tiles):
            per_lane[k] = sum(m[k] for _, m in tiles) / 1000.0
    if not per_lane:
        return None, None, {}
    return max(per_lane.values()), min(per_lane.values()), per_lane


def chain_groups(spec, edges):
    """{name: [names in its dependency chain]} -- union-find over the edges.
    Verbatim from mk_workloads.py."""
    parent = {n: n for n in spec}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        a, b = e.get("from"), e.get("to")
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    grp = collections.defaultdict(list)
    for n in spec:
        grp[find(n)].append(n)
    return {n: grp[find(n)] for n in spec}


def dispatch_deps_path(net, hw_label):
    man_path = os.path.join(FLOWC, bindings_rel(net))
    with open(man_path) as f:
        man = json.load(f)
    q = man.get("ir", {}).get("quant", "int8")
    n = man["network"]
    return (f"{GEN_ROOT}/qnn_vmfb/{n}/{TARGET}/{hw_label}/{n}.{q}/"
            f"{n}.{q}_dispatch_graph.json")


def build_one(fam, spec, edges, cfg, cost_model):
    kinds = list(CONFIGS[cfg].values())
    cost, lanes = {}, {}
    problems = []
    for name in spec:
        worst, best, per_lane = net_cost_ms(cost_model, name, kinds)
        if per_lane is None:
            problems.append(
                f"{name}: no binding manifest — Phase 1 did not build this "
                f"network, so the cell cannot be generated")
            continue
        if worst is None:
            problems.append(
                f"{name}: no lane in config {cfg} ({','.join(kinds)}) has a "
                f"measured cell for every tile — the solver would only ever "
                f"see exclusion costs")
            continue
        cost[name] = (worst, best)
        lanes[name] = {k: round(v, 4) for k, v in per_lane.items()}
    if problems:
        return None, problems, None
    chains = chain_groups(spec, edges)
    nets, nid, busy = {}, 0, 0.0
    # deterministic id order: the family's declaration order
    for name, (inst, pmult) in spec.items():
        worst, best = cost[name]
        grp = chains[name]
        if len(grp) > 1 and pmult is not None:
            # A pipeline ticks at ONE rate: the period covers the whole
            # chain's latency, not this member's share of it.
            worst = sum(cost[m][0] for m in grp)
            pmult = max(spec[m][1] for m in grp if spec[m][1] is not None)
        per = worst * pmult if pmult else None
        hw0 = sorted(HW_LABEL[k] for k in kinds)[0]
        e = {"id": nid, "identifier": name,
             "dispatch_deps_path": dispatch_deps_path(name, hw0)}
        if per is not None:
            e["period"] = round(per, 3)
            e["window_duration"] = round(per, 3)
        e["num_instances"] = inst
        nets[name] = e
        busy += best * inst
        nid += 1
    machines = {s.lower(): 1 for s in CONFIGS[cfg]}
    profile_hw = {s.lower(): HW_LABEL[k] for s, k in CONFIGS[cfg].items()}
    doc = {
        "_comment": (
            f"workload family '{fam}' on the QRB5165 lane configuration "
            f"'{cfg}' ({' + '.join(CONFIGS[cfg].values())}). @generated by "
            f"qnn_models/flow_c/sweeps/<this sweep>/scripts/"
            f"mk_workloads_qrb5165.py, a port of "
            f"RoSE/experiments/workload_gen/mk_workloads.py. Family shape "
            f"(networks, instance counts, period multiples) is verbatim from "
            f"the reference; periods are re-derived from THIS BOARD's "
            f"gap-phase measured cells, so they are tight but satisfiable "
            f"here and are NOT comparable with the reference's FireSim "
            f"periods."
            + (" Networks joined by `edges` form a dependency chain and share "
               "one derived period covering the whole chain's latency."
               if edges else "")),
        "hardware": {
            "machines": machines,
            "profile_hw": profile_hw,
            "profile": {"target": TARGET, "topo_tag": "topo_0",
                        "topo_tag_override": False, "gen_root": GEN_ROOT},
            "p_core_speedup": 1.0},
        # Verbatim from the reference's specs so the solve is method-comparable.
        "scheduler": {"random_seed": 42, "solver_verbosity": 2,
                      "time_limit": 120, "use_profiled": True,
                      "prune_periodic": True,
                      "restrict_makespan_to_nonperiodic": False},
        "networks": nets,
    }
    if edges:
        doc["edges"] = edges
    meta = dict(kinds=kinds, est_busy_ms=round(busy, 3),
                lane_costs_ms=lanes,
                periods={k: v.get("period") for k, v in nets.items()},
                instances={k: v["num_instances"] for k, v in nets.items()})
    return doc, [], meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--cost-model",
                    default=os.path.join(SWEEP, "cost_model.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "toplevel", ARM))
    a = ap.parse_args()
    cfgs = [c for c in a.configs.split(",") if c in CONFIGS]
    cm = load_cost_model(a.cost_model)
    rows, n = [], 0
    print(f"  cost model: {a.cost_model} "
          f"(statistic={cm.get('statistic')}, {len(cm.get('cells', {}))} cells)")
    print(f"\n  {'family':<20}{'cfg':<6}{'status':<10}{'tiles':>7}"
          f"{'busy ms':>10}  periods (ms)")
    print("  " + "-" * 124)
    for fam, spec, edges in FAMILIES:
        for cfg in cfgs:
            doc, problems, meta = build_one(fam, spec, edges, cfg, cm)
            name = f"networks_{fam}_{cfg}"
            if problems:
                rows.append(dict(family=fam, config=cfg, workload=name,
                                 status="REJECTED", problems=problems))
                print(f"  {fam:<20}{cfg:<6}{'REJECTED':<10}")
                for p in problems:
                    print(f"      - {p}")
                continue
            ntiles = sum(len(tile_costs(cm, None, k) or []) * v["num_instances"]
                         for k, v in doc["networks"].items())
            per = " ".join(f"{k}={v['period']}" if v.get("period")
                           else f"{k}=np"
                           for k, v in doc["networks"].items())
            print(f"  {fam:<20}{cfg:<6}{'ok':<10}{ntiles:>8}"
                  f"{meta['est_busy_ms']:>10.1f}  {per[:74]}")
            rows.append(dict(family=fam, config=cfg, workload=name,
                             status="ok", **meta))
            if a.emit:
                os.makedirs(a.out, exist_ok=True)
                with open(os.path.join(a.out, f"{name}.json"), "w") as f:
                    json.dump(doc, f, indent=1)
                n += 1
    out_meta = os.path.join(SWEEP, "results", "phase2_generated.json")
    if a.emit:
        os.makedirs(os.path.dirname(out_meta), exist_ok=True)
        json.dump(rows, open(out_meta, "w"), indent=1)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n  {ok}/{len(rows)} (family, config) cells built"
          + (f" -> {a.out} and {out_meta}" if a.emit
             else "  (dry run -- pass --emit)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
