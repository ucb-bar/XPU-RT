#!/usr/bin/env python3
"""Deeper reads on the trace blocks: punctuality, lane load, critical path."""
from __future__ import annotations
import csv, io, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PTS = [r["point"] for r in json.load(open(os.path.join(HERE, "results.json")))
       if r["status"] == "run"]


def rows_of(pt, rep):
    text = open(os.path.join(HERE, "runs", pt, rep, "run.log"), errors="replace").read()
    block = text.split("MODELBLASTER_XPURT_TRACE_BEGIN ===")[1] \
                .split("=== MODELBLASTER_XPURT_TRACE_END")[0].strip()
    out = []
    for r in csv.DictReader(io.StringIO(block)):
        if not r.get("actual_end_cycles"):
            continue
        r["a_start"] = int(r["actual_start_cycles"]) / 1000.0
        r["a_end"] = int(r["actual_end_cycles"]) / 1000.0
        r["a_dur"] = r["a_end"] - r["a_start"]
        r["p_start"] = float(r["predicted_start_ms"])
        r["p_dur"] = float(r["predicted_duration_ms"])
        r["late"] = r["a_start"] - r["p_start"]
        r["model"] = r["network"].rsplit("_", 1)[0] if r["network"].rsplit("_", 1)[-1] in "abcde" \
            and len(r["network"].rsplit("_", 1)[-1]) == 1 else r["network"]
        out.append(r)
    return out


def pooled(pt):
    out = []
    for rep in sorted(os.listdir(os.path.join(HERE, "runs", pt))):
        if os.path.exists(os.path.join(HERE, "runs", pt, rep, "run.log")):
            out.append(rows_of(pt, rep))
    return out


def q(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    i = min(len(v) - 1, int(round(p * (len(v) - 1))))
    return v[i]


print("### mlp_control punctuality and CPU-lane load, by arm (medians over 3 reps)\n")
print(f"{'point':<20} {'mlp n':>6} {'exec p50':>9} {'exec p90':>9} {'exec max':>9} "
      f"{'late p50':>9} {'late max':>9} | {'cpu busy ms':>12} {'dsp busy':>9} {'hta busy':>9} {'wall':>8}")
per_point = {}
for pt in PTS:
    reps = pooled(pt)
    e50, e90, emx, l50, lmx, cpub, dspb, htab, wall = [], [], [], [], [], [], [], [], []
    for rows in reps:
        m = [r for r in rows if r["network"] == "mlp_control"]
        if m:
            e50.append(q([r["a_dur"] for r in m], .5)); e90.append(q([r["a_dur"] for r in m], .9))
            emx.append(max(r["a_dur"] for r in m))
            l50.append(q([r["late"] for r in m], .5)); lmx.append(max(r["late"] for r in m))
        cpub.append(sum(r["a_dur"] for r in rows if r["core_kind"] == "cpu"))
        dspb.append(sum(r["a_dur"] for r in rows if r["core_kind"] == "dsp"))
        htab.append(sum(r["a_dur"] for r in rows if r["core_kind"] == "hta"))
        wall.append(max(r["a_end"] for r in rows))
    md = lambda v: statistics.median(v) if v else float("nan")
    per_point[pt] = dict(mlp_n=len([r for r in reps[0] if r["network"] == "mlp_control"]),
                         e50=md(e50), e90=md(e90), emx=md(emx), l50=md(l50), lmx=md(lmx),
                         cpu=md(cpub), dsp=md(dspb), hta=md(htab), wall=md(wall))
    p = per_point[pt]
    print(f"{pt:<20} {p['mlp_n']:>6} {p['e50']:>9.3f} {p['e90']:>9.3f} {p['emx']:>9.3f} "
          f"{p['l50']:>9.3f} {p['lmx']:>9.3f} | {p['cpu']:>12.2f} {p['dsp']:>9.2f} "
          f"{p['hta']:>9.2f} {p['wall']:>8.2f}")

print("\n### matched arms: what the fused_split substitution actually buys\n")
print(f"{'seed':<5} {'mlp exec p50 b/f':<22} {'mlp late max b/f':<22} {'cpu busy b/f':<20} {'wall b/f':<20}")
for s in (0, 1, 2, 3, 6):
    b, f_ = per_point.get(f"baseline_seed{s}"), per_point.get(f"fused_seed{s}")
    if not b or not f_:
        continue
    print(f"{s:<5} {b['e50']:>9.3f} /{f_['e50']:>9.3f}   {b['lmx']:>9.3f} /{f_['lmx']:>9.3f}   "
          f"{b['cpu']:>8.2f} /{f_['cpu']:>8.2f}  {b['wall']:>8.2f} /{f_['wall']:>8.2f}")

print("\n### what bounds the makespan (last entry to finish, median rep)\n")
for pt in PTS:
    reps = pooled(pt)
    rows = reps[len(reps) // 2]
    last = max(rows, key=lambda r: r["a_end"])
    # how much of the critical path is ViNT?
    vint = [r for r in rows if r["network"] == "vint"]
    vs = f"  vint busy {sum(r['a_dur'] for r in vint):7.2f} ms on {sorted({r['core_kind'] for r in vint})}" if vint else ""
    print(f"{pt:<20} last = {last['network']}/{last['name']}@{last['core_kind']} "
          f"ends {last['a_end']:7.2f} ms (pred end {last['p_start']+last['p_dur']:7.2f}){vs}")

print("\n### ViNT arm vs its matched fused arm: where the extra time goes\n")
for s in (0, 1):
    a, v = per_point.get(f"fused_seed{s}"), per_point.get(f"fused_vint_seed{s}")
    if not a or not v:
        continue
    print(f"seed {s}:  wall {a['wall']:.2f} -> {v['wall']:.2f} ms  (+{v['wall']-a['wall']:.2f})")
    print(f"          cpu busy {a['cpu']:7.2f} -> {v['cpu']:7.2f}   dsp {a['dsp']:7.2f} -> {v['dsp']:7.2f}"
          f"   hta {a['hta']:7.2f} -> {v['hta']:7.2f}")
    print(f"          mlp exec p50 {a['e50']:.3f} -> {v['e50']:.3f} ms, late max "
          f"{a['lmx']:.3f} -> {v['lmx']:.3f} ms")

print("\n### per-lane over/under-prediction, summed over every entry of every run point\n")
agg = {}
for pt in PTS:
    for rows in pooled(pt):
        for r in rows:
            k = r["core_kind"]
            a = agg.setdefault(k, {"pred": 0.0, "act": 0.0, "n": 0})
            a["pred"] += r["p_dur"]; a["act"] += r["a_dur"]; a["n"] += 1
for k in sorted(agg):
    a = agg[k]
    print(f"  {k:<4} {a['n']:>5} entries  predicted {a['pred']:9.1f} ms  actual {a['act']:9.1f} ms  "
          f"ratio {a['act']/a['pred']:.3f}")

print("\n### placement of the yolov8n head, by point (which lane the solver chose)\n")
res = {r["point"]: r for r in json.load(open(os.path.join(HERE, "results.json")))}
for pt in PTS:
    pl = res[pt]["lane_placement"]
    heads = {k: v for k, v in pl.items() if k.endswith("yolov8n_head")}
    tails = {k: v for k, v in pl.items() if k.endswith("fused_tail")}
    n_cpu = sum(v.get("cpu", 0) for v in heads.values())
    n_dsp = sum(v.get("dsp", 0) for v in heads.values())
    tcpu = sum(v.get("cpu", 0) for v in tails.values())
    tdsp = sum(v.get("dsp", 0) for v in tails.values())
    print(f"{pt:<20} yolov8n_head: {n_dsp} on dsp, {n_cpu} on cpu"
          + (f"   |  fused_tail: {tdsp} dsp, {tcpu} cpu" if tails else ""))

print("\n### mlp_control deadline behaviour (instance i due by start+i*period+window)\n")
gen = {r["point"]: r for r in json.load(open(os.path.join(HERE, "generated.json")))}
print(f"{'point':<20} {'period':>7} {'window':>7} {'n':>4} {'misses (median rep)':>21} {'worst overrun ms':>18}")
for pt in PTS:
    net = gen[pt]["networks"].get("mlp_control")
    if not net or not net.get("period"):
        continue
    per, win = net["period"], net["window_duration"]
    miss, worst = [], []
    for rows in pooled(pt):
        m = [r for r in rows if r["network"] == "mlp_control"]
        over = [r["a_end"] - (int(r["instance"]) * per + win) for r in m]
        miss.append(sum(1 for o in over if o > 0))
        worst.append(max(over) if over else 0.0)
    print(f"{pt:<20} {per:>7} {win:>7} {len(m):>4} "
          f"{statistics.median(miss):>13.0f} / {len(m):<5} {statistics.median(worst):>18.3f}")
