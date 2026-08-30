#!/usr/bin/env python3
"""Full characterization of transfer costs and placement. Read-only.

Sources
  sweeps/qrb5165_20260829-200620/          12 points x 3 reps  (placement)
  sweeps/.../addendum_periodic_only/        6 points x 3 reps  (placement)
  costmodel_validation/                     3 points x 3 reps  (placement)
  transfer_study/runs/                      instrumented reruns (transfer)
  cores/qrb5165_qnn.json + bindings/*.json  capability ground truth

Emits a machine-readable log (transfer_study/data_log.json + .csv) and the
tables the writeup quotes.
"""
import csv, glob, json, os, statistics as st, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FLOWC = os.path.abspath(os.path.join(HERE, ".."))
S = os.path.join(FLOWC, "sweeps", "qrb5165_20260829-200620")

PLACEMENT_SRC = [
    ("sweep",      os.path.join(S, "results.json"), S),
    ("addendum",   os.path.join(S, "addendum_periodic_only", "results.json"),
                   os.path.join(S, "addendum_periodic_only")),
    ("validation", os.path.join(FLOWC, "costmodel_validation", "results.json"),
                   os.path.join(FLOWC, "costmodel_validation")),
]


def parse_trace(path):
    """Tolerant trace reader. Handles the pre-instrumentation 20-column
    layout and the 27-column layout, and repairs rows split by the
    stdout/stderr interleaving defect fixed earlier."""
    rows, on, pending, ncol = [], False, "", None
    for ln in open(path):
        if "TRACE_BEGIN" in ln: on = True; continue
        if "TRACE_END" in ln: break
        if not on: continue
        ln = ln.rstrip("\n")
        cut = ln.find("[main] ")
        if cut >= 0:
            ln = ln[:cut]
            if not ln: continue
        if ncol is None and ln.startswith("entry_id,"):
            ncol = ln.count(",") + 1
            rows.append(ln); continue
        if ncol is None: continue
        cand = pending + ln
        if cand.count(",") == ncol - 1: rows.append(cand); pending = ""
        elif cand.count(",") < ncol - 1: pending = cand
        else: rows.append(cand); pending = ""
    if len(rows) < 2: return []
    return list(csv.DictReader(rows[1:], fieldnames=rows[0].split(",")))


def ms(r, k):
    v = float(r.get(k) or 0)
    return v / 1000.0 if r.get("time_unit") == "us" else v


def fnum(r, k, d=0.0):
    try: return float(r.get(k) or d)
    except (TypeError, ValueError): return d


def inum(r, k, d=0):
    try: return int(float(r.get(k) or d))
    except (TypeError, ValueError): return d


# ======================================================================
# 1. TRANSFER -- measured boundary handoff cost
# ======================================================================
def collect_transfers():
    recs = []
    for p in sorted(glob.glob(os.path.join(HERE, "runs", "*", "rep*", "run.log"))):
        pt = p.split(os.sep)[-3]; rep = p.split(os.sep)[-2]
        t = parse_trace(p)
        if not t or "hin_ms" not in (t[0] or {}):
            continue
        # lane timeline, to attribute each handoff to a producer lane
        by_end = sorted(t, key=lambda r: ms(r, "actual_end_cycles"))
        for r in t:
            recs.append({
                "point": pt, "rep": rep,
                "entry": inum(r, "entry_id"), "network": r["network"],
                "tile": r["name"], "lane": r["backend"].lower(),
                "exec_ms": ms(r, "actual_end_cycles") - ms(r, "actual_start_cycles"),
                "hin_ms": fnum(r, "hin_ms"), "hout_ms": fnum(r, "hout_ms"),
                "hin_bytes": inum(r, "hin_bytes"), "hout_bytes": inum(r, "hout_bytes"),
                "hin_hits": inum(r, "hin_hits"), "hin_tensors": inum(r, "hin_tensors"),
                "hout_tensors": inum(r, "hout_tensors"),
                "n_ir_ops": inum(r, "n_ir_ops"),
            })
    return recs


def transfer_tables(recs, out):
    if not recs:
        print("  (no instrumented traces yet)"); return
    print("=" * 104)
    print("1. TRANSFER COST -- measured boundary handoff, per entry")
    print("=" * 104)
    print("  The runtime memcpy's every output tensor into a mutex-guarded cache after")
    print("  execute (hout), and searches that cache for every input tensor before")
    print("  execute (hin). Both sit OUTSIDE actual_start..actual_end, so neither the")
    print("  tile cost cells nor the MILP (transfer_times = zeros) account for them.")
    print()
    tot_in = sum(r["hin_ms"] for r in recs)
    tot_out = sum(r["hout_ms"] for r in recs)
    tot_exec = sum(r["exec_ms"] for r in recs)
    print(f"  entries instrumented   {len(recs)}")
    print(f"  total execute time     {tot_exec:10.3f} ms")
    print(f"  total handoff-in       {tot_in:10.3f} ms   ({100*tot_in/tot_exec:.2f}% of execute)")
    print(f"  total handoff-out      {tot_out:10.3f} ms   ({100*tot_out/tot_exec:.2f}% of execute)")
    print(f"  total handoff          {tot_in+tot_out:10.3f} ms   "
          f"({100*(tot_in+tot_out)/tot_exec:.2f}% of execute)")

    print("\n  --- per tile: what it copies and what that costs ---")
    agg = defaultdict(list)
    for r in recs: agg[(r["network"], r["tile"], r["lane"])].append(r)
    print(f"{'network/tile@lane':<44} {'n':>4} {'exec p50':>9} {'hin p50':>9} "
          f"{'hout p50':>9} {'in KiB':>9} {'out KiB':>9} {'h/exec':>8}")
    rows = []
    for k in sorted(agg, key=lambda k: -st.median(x["hout_ms"] for x in agg[k])):
        v = agg[k]
        ex = st.median(x["exec_ms"] for x in v)
        hi = st.median(x["hin_ms"] for x in v)
        ho = st.median(x["hout_ms"] for x in v)
        ib = st.median(x["hin_bytes"] for x in v) / 1024.0
        ob = st.median(x["hout_bytes"] for x in v) / 1024.0
        frac = (hi + ho) / ex if ex else 0
        rows.append(dict(zip(("network","tile","lane"), k)) |
                    {"n": len(v), "exec_p50_ms": ex, "hin_p50_ms": hi,
                     "hout_p50_ms": ho, "hin_kib": ib, "hout_kib": ob,
                     "handoff_frac_of_exec": frac})
        print(f"{k[0]+'/'+k[1]+'@'+k[2]:<44} {len(v):>4} {ex:>9.3f} {hi:>9.4f} "
              f"{ho:>9.4f} {ib:>9.1f} {ob:>9.1f} {100*frac:>7.1f}%")
    out["transfer_per_tile"] = rows

    print("\n  --- cost vs bytes: is the copy bandwidth-bound? ---")
    pts = [(r["hout_bytes"], r["hout_ms"]) for r in recs if r["hout_bytes"] > 0 and r["hout_ms"] > 0]
    if pts:
        by = defaultdict(list)
        for b, t in pts:
            dec = 1 << (int(b).bit_length() - 1)
            by[dec].append(t)
        print(f"{'output bytes (pow2 bucket)':<30} {'n':>5} {'hout p50 ms':>13} {'GiB/s':>9}")
        bw_rows = []
        for b in sorted(by):
            v = by[b]; m = st.median(v)
            bw = (b / (1024**3)) / (m / 1000.0) if m > 0 else 0
            bw_rows.append({"bucket_bytes": b, "n": len(v), "p50_ms": m, "gib_s": bw})
            print(f"{b:>26,} B {len(v):>5} {m:>13.4f} {bw:>9.2f}")
        out["transfer_bandwidth"] = bw_rows
    out["transfer_totals"] = {"entries": len(recs), "exec_ms": tot_exec,
                              "hin_ms": tot_in, "hout_ms": tot_out}
    out["transfer_raw"] = recs


# ======================================================================
# 2. TRANSFER -- what the scheduler models (nothing) and what it costs
# ======================================================================
def transfer_vs_model(recs, out):
    print()
    print("=" * 104)
    print("2. WHAT THE SCHEDULER MODELS FOR TRANSFER")
    print("=" * 104)
    print("  scripts/run_xpurt_schedule.py:244")
    print("      transfer_times = np.zeros((n_cores, n_cores))")
    print("  Every cross-lane edge therefore contributes exactly 0 to a predicted start.")
    print("  scheduler.py does consume the matrix -- constraint (3) adds")
    print("      t[i] >= t[pred] + dur[pred] + max_transfer_time")
    print("  so the mechanism exists and is wired; it is only ever fed zeros.")
    if not recs: return
    per_point = defaultdict(lambda: {"h": 0.0, "e": 0.0})
    for r in recs:
        k = (r["point"], r["rep"])
        per_point[k]["h"] += r["hin_ms"] + r["hout_ms"]
        per_point[k]["e"] += r["exec_ms"]
    print(f"\n{'point':<22} {'rep':<6} {'handoff ms':>11} {'execute ms':>11} {'handoff/exec':>13}")
    agg = defaultdict(list)
    for (pt, rep), v in sorted(per_point.items()):
        print(f"{pt:<22} {rep:<6} {v['h']:>11.3f} {v['e']:>11.3f} "
              f"{100*v['h']/v['e'] if v['e'] else 0:>12.2f}%")
        agg[pt].append(v["h"])
    out["transfer_per_run"] = [{"point": pt, "rep": rep, "handoff_ms": v["h"],
                                "exec_ms": v["e"]} for (pt, rep), v in sorted(per_point.items())]


# ======================================================================
# 3. PLACEMENT
# ======================================================================
def collect_placement(out):
    print()
    print("=" * 104)
    print("3. PLACEMENT -- where every tile actually ran")
    print("=" * 104)
    place = defaultdict(lambda: defaultdict(int))
    per_point, rows = {}, []
    for tag, res, root in PLACEMENT_SRC:
        if not os.path.exists(res): continue
        for r in json.load(open(res)):
            if r.get("status") != "run": continue
            lp = r.get("lane_placement") or {}
            per_point[(tag, r["point"])] = {"lanes": r.get("lane_entry_counts"),
                                            "placement": lp, "arm": r.get("arm")}
            for job_tile, lanes in lp.items():
                net = job_tile.split("/", 1)[0]
                tile = job_tile.split("/", 1)[1] if "/" in job_tile else job_tile
                for lane, n in lanes.items():
                    place[tile][lane] += n
                    rows.append({"source": tag, "point": r["point"], "arm": r.get("arm"),
                                 "job": net, "tile": tile, "lane": lane, "count": n})
    print(f"{'tile':<26} {'hta':>6} {'dsp':>6} {'cpu':>6} {'total':>7} {'lanes used':>11}  distribution")
    tab = []
    for tile in sorted(place, key=lambda t: -sum(place[t].values())):
        d = place[tile]; tot = sum(d.values())
        share = "  ".join(f"{k}={100*v/tot:.0f}%" for k, v in sorted(d.items(), key=lambda x: -x[1]))
        tab.append({"tile": tile, "hta": d.get("hta", 0), "dsp": d.get("dsp", 0),
                    "cpu": d.get("cpu", 0), "total": tot, "lanes_used": len(d)})
        print(f"{tile:<26} {d.get('hta',0):>6} {d.get('dsp',0):>6} {d.get('cpu',0):>6} "
              f"{tot:>7} {len(d):>11}  {share}")
    out["placement_by_tile"] = tab
    out["placement_rows"] = rows
    out["placement_per_point"] = {f"{k[0]}/{k[1]}": v for k, v in per_point.items()}
    return place, per_point


# ======================================================================
# 4. PLACEMENT -- capability envelope: where COULD each tile have run?
# ======================================================================
def capability(place, out):
    print()
    print("=" * 104)
    print("4. CAPABILITY ENVELOPE -- eligible lanes vs lanes actually used")
    print("=" * 104)
    elig = {}
    for fn in sorted(os.listdir(os.path.join(FLOWC, "bindings"))):
        if not fn.endswith(".json"): continue
        man = json.load(open(os.path.join(FLOWC, "bindings", fn)))
        for b in man["bindings"]:
            elig[b["name"]] = sorted((b.get("backends") or {}).keys())
    print(f"{'tile':<26} {'eligible lanes':<22} {'used':<22} {'unused capability'}")
    rows = []
    for tile in sorted(place, key=lambda t: -sum(place[t].values())):
        e = elig.get(tile, [])
        u = sorted(place[tile])
        unused = [x for x in e if x not in u]
        rows.append({"tile": tile, "eligible": e, "used": u, "unused": unused})
        print(f"{tile:<26} {','.join(e) or '--':<22} {','.join(u):<22} "
              f"{','.join(unused) if unused else '(all used)'}")
    out["capability"] = rows


# ======================================================================
# 5. PLACEMENT -- stability and migration
# ======================================================================
def stability(out):
    print()
    print("=" * 104)
    print("5. PLACEMENT STABILITY -- does a tile keep its lane across workloads?")
    print("=" * 104)
    per_tile_point = defaultdict(lambda: defaultdict(int))
    for tag, res, root in PLACEMENT_SRC:
        if not os.path.exists(res): continue
        for r in json.load(open(res)):
            if r.get("status") != "run": continue
            for job_tile, lanes in (r.get("lane_placement") or {}).items():
                tile = job_tile.split("/", 1)[1] if "/" in job_tile else job_tile
                dom = max(lanes.items(), key=lambda x: x[1])[0] if lanes else None
                if dom: per_tile_point[tile][dom] += 1
    print(f"{'tile':<26} {'points':>7} {'modal lane':>11} {'modal share':>12}  per-point majority lane")
    rows = []
    for tile, d in sorted(per_tile_point.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(d.values()); mode, mc = max(d.items(), key=lambda x: x[1])
        rows.append({"tile": tile, "points": tot, "modal_lane": mode,
                     "modal_share": mc / tot, "distribution": dict(d)})
        print(f"{tile:<26} {tot:>7} {mode:>11} {100*mc/tot:>11.0f}%  "
              + "  ".join(f"{k}:{v}" for k, v in sorted(d.items(), key=lambda x: -x[1])))
    out["placement_stability"] = rows
    split = [r for r in rows if r["modal_share"] < 0.9]
    print(f"\n  {len(rows)-len(split)}/{len(rows)} tiles keep one lane in >=90% of points;"
          f" {len(split)} migrate")
    for r in split:
        print(f"    {r['tile']:<24} {r['distribution']}")


# ======================================================================
# 6. PLACEMENT -- was the chosen lane the fastest available?
# ======================================================================
def placement_quality(out):
    print()
    print("=" * 104)
    print("6. PLACEMENT QUALITY -- chosen lane vs the fastest eligible lane")
    print("=" * 104)
    meas = json.load(open(os.path.join(FLOWC, "measurements", "qrb5165_v66.json")))
    cells = meas["cells"]
    counts = defaultdict(lambda: defaultdict(int))
    for tag, res, root in PLACEMENT_SRC:
        if not os.path.exists(res): continue
        for r in json.load(open(res)):
            if r.get("status") != "run": continue
            for job_tile, lanes in (r.get("lane_placement") or {}).items():
                net = job_tile.split("/", 1)[0]
                tile = job_tile.split("/", 1)[1] if "/" in job_tile else job_tile
                base = net.rstrip("_abcdefgh") if len(net) > 2 and net[-2] == "_" else net
                for lane, n in lanes.items():
                    counts[(base, tile)][lane] += n
    print(f"{'cell':<40} {'fastest lane':>14} {'cost ms':>9} | {'placed on':<26} {'excess'}")
    rows = []
    for (net, tile), lanes in sorted(counts.items()):
        cell = cells.get(f"{net}/{tile}")
        if not cell: continue
        best = min(cell.items(), key=lambda x: x[1])
        placed = sorted(lanes.items(), key=lambda x: -x[1])
        tot = sum(lanes.values())
        excess = 0.0
        for lane, n in lanes.items():
            c = cell.get(lane)
            if c is not None: excess += n * (c - best[1]) / 1000.0
        rows.append({"cell": f"{net}/{tile}", "fastest_lane": best[0],
                     "fastest_ms": best[1] / 1000.0, "placements": dict(lanes),
                     "excess_ms_vs_fastest": excess,
                     "share_on_fastest": lanes.get(best[0], 0) / tot})
        pl = " ".join(f"{k}x{v}" for k, v in placed)
        print(f"{net+'/'+tile:<40} {best[0]:>14} {best[1]/1000:>9.3f} | {pl:<26} "
              f"{excess:>+8.1f} ms")
    out["placement_quality"] = rows
    tot_ex = sum(r["excess_ms_vs_fastest"] for r in rows)
    print(f"\n  total cost of not always using the fastest lane: {tot_ex:+.1f} ms across all placements")
    print("  (this is the price of parallelism -- a lane can only run one tile at a time,")
    print("   so spilling to a slower lane is often correct)")


# ======================================================================
# 7. TRANSFER -- how much of the copying is wasted?
# ======================================================================
def waste(recs, out):
    """Classify each tile by its role in the binding DAG, then price the copies.

    cache_put runs unconditionally for EVERY output tensor of EVERY tile,
    whether or not anything downstream will ever ask for it. A terminal tile --
    one no other tile depends on -- therefore pays a full memcpy of its output
    for nothing.
    """
    print()
    print("=" * 104)
    print("7. WASTED TRANSFER -- copies nothing consumes")
    print("=" * 104)
    role = {}
    for fn in sorted(os.listdir(os.path.join(FLOWC, "bindings"))):
        if not fn.endswith(".json"): continue
        man = json.load(open(os.path.join(FLOWC, "bindings", fn)))
        bl = man["bindings"]
        consumed = set()
        for b in bl:
            for i in (b.get("depends_on") or []):
                if 0 <= i < len(bl): consumed.add(bl[i]["name"])
        for b in bl:
            has_dep = bool(b.get("depends_on"))
            role[b["name"]] = ("intermediate" if (b["name"] in consumed and has_dep)
                               else "source" if b["name"] in consumed
                               else "terminal" if has_dep else "isolated")
    agg = defaultdict(lambda: {"n": 0, "out_b": 0, "hout_ms": 0.0,
                               "hits": 0, "in_b": 0, "hin_ms": 0.0})
    for r in recs:
        a = agg[r["tile"]]
        a["n"] += 1; a["out_b"] += r["hout_bytes"]; a["hout_ms"] += r["hout_ms"]
        a["hits"] += r["hin_hits"]; a["in_b"] += r["hin_bytes"]; a["hin_ms"] += r["hin_ms"]
    print(f"{'tile':<22} {'role':<13} {'n':>4} {'cached MiB':>11} {'read MiB':>9} "
          f"{'put ms':>8} {'get ms':>8}  output ever read?")
    wasted_ms = wasted_b = 0.0
    rows = []
    for tile, a in sorted(agg.items(), key=lambda kv: -kv[1]["hout_ms"]):
        rl = role.get(tile, "?")
        # a tile's OUTPUT is read only if it is a source or intermediate
        out_read = rl in ("source", "intermediate")
        if not out_read:
            wasted_ms += a["hout_ms"]; wasted_b += a["out_b"]
        rows.append({"tile": tile, "role": rl, "n": a["n"],
                     "cached_bytes": a["out_b"], "read_bytes": a["in_b"],
                     "put_ms": a["hout_ms"], "get_ms": a["hin_ms"],
                     "output_ever_read": out_read})
        print(f"{tile:<22} {rl:<13} {a['n']:>4} {a['out_b']/1048576:>11.3f} "
              f"{a['in_b']/1048576:>9.3f} {a['hout_ms']:>8.3f} {a['hin_ms']:>8.3f}  "
              f"{'yes' if out_read else 'NO -- pure waste'}")
    tot = sum(r["hin_ms"] + r["hout_ms"] for r in recs)
    tot_b = sum(r["hout_bytes"] for r in recs)
    read_b = sum(r["hin_bytes"] for r in recs)
    hits = sum(r["hin_hits"] for r in recs)
    look = sum(r["hin_tensors"] for r in recs)
    print(f"\n  cache lookups            {look:6d}, of which {hits} hit  "
          f"({100*hits/look:.1f}%)")
    print(f"  bytes cached             {tot_b/1048576:8.2f} MiB")
    print(f"  bytes ever read back     {read_b/1048576:8.2f} MiB  "
          f"({100*read_b/tot_b:.2f}% of cached)")
    print(f"  copies for terminal tiles{wasted_ms:8.3f} ms of {tot:.3f} ms "
          f"({100*wasted_ms/tot:.1f}%) -- structurally unreadable, {wasted_b/1048576:.1f} MiB")
    out["transfer_waste"] = {"rows": rows, "lookups": look, "hits": hits,
                             "cached_bytes": tot_b, "read_bytes": read_b,
                             "terminal_waste_ms": wasted_ms,
                             "terminal_waste_bytes": wasted_b,
                             "total_handoff_ms": tot}


def main():
    out = {"_comment": ("Machine-readable log for the transfer-cost and placement "
                        "characterization. Generated by transfer_study/characterize.py; "
                        "see transfer_study/REPORT.md for the writeup."),
           "generated": "2026-08-30", "target": "qrb5165_v66"}
    recs = collect_transfers()
    transfer_tables(recs, out)
    transfer_vs_model(recs, out)
    place, per_point = collect_placement(out)
    capability(place, out)
    stability(out)
    placement_quality(out)
    waste(recs, out)

    dl = os.path.join(HERE, "data_log.json")
    with open(dl, "w") as f:
        json.dump(out, f, indent=1)
    # flat CSV of every instrumented entry, for spreadsheet work
    if recs:
        cl = os.path.join(HERE, "data_log_entries.csv")
        with open(cl, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader(); w.writerows(recs)
    pl = os.path.join(HERE, "data_log_placements.csv")
    with open(pl, "w", newline="") as f:
        rows = out["placement_rows"]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n  data log  -> {dl}")
    if recs: print(f"  entries   -> {os.path.join(HERE,'data_log_entries.csv')} ({len(recs)} rows)")
    print(f"  placements-> {pl} ({len(out['placement_rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
