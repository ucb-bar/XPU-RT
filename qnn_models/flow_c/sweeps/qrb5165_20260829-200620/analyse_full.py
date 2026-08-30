#!/usr/bin/env python3
"""Full aggregate analysis over the sweep + its periodic-only addendum.

Reads only committed artifacts: both results.json files and all 54 run.log
traces. Read-only; prints tables.
"""
import csv, glob, json, os, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ADD = os.path.join(HERE, "addendum_periodic_only")

def load(root):
    return {r["point"]: dict(r, _root=root)
            for r in json.load(open(os.path.join(root, "results.json")))}

MAIN, ADDA = load(HERE), load(ADD)
ALL = {}
for k, v in MAIN.items(): ALL[("main", k)] = v
for k, v in ADDA.items(): ALL[("add", k)] = v
RUN = {k: v for k, v in ALL.items() if v["status"] == "run"}

TRACE_NCOL = 20

def trace(path):
    """Parse a trace block, repairing rows split by stdout/stderr interleaving.

    deploy_and_run.sh captures the run as `> run.log 2>&1`, so the runtime's
    block-buffered stdout and unbuffered stderr share one file description. A
    4 KiB flush boundary can land mid-printf and let a stderr line slip into
    the gap, splitting one CSV row across two lines and injecting a `[main]`
    line into the block. 2 of the 54 traces here are affected. Fixed at the
    source in flowc/emit_runtime.py (line-buffered stdout + flush before each
    stderr write); this repairs the traces already on disk.
    """
    rows, on, pending = [], False, ""
    for ln in open(path):
        if "TRACE_BEGIN" in ln: on = True; continue
        if "TRACE_END" in ln: break
        if not on: continue
        ln = ln.rstrip("\n")
        # strip an interleaved stderr line wherever it landed
        cut = ln.find("[main] ")
        if cut >= 0:
            tail = ln[cut:]
            ln = ln[:cut]
            if not ln and not tail.endswith("\n"):
                continue
            if not ln:
                continue
        cand = pending + ln
        if cand.count(",") == TRACE_NCOL - 1:
            rows.append(cand); pending = ""
        elif cand.count(",") < TRACE_NCOL - 1:
            pending = cand              # row was split; wait for its remainder
        else:
            rows.append(cand); pending = ""
    if not rows: return []
    hdr = rows[0]
    return list(csv.DictReader(rows[1:], fieldnames=hdr.split(",")))

def ms(r, k):
    v = float(r.get(k) or 0)
    return v / 1000.0 if r.get("time_unit") == "us" else v

# ---- gather every trace ----
TR = {}   # (set, point) -> list of per-rep row lists
for (s, pt), r in RUN.items():
    reps = []
    for p in sorted(glob.glob(os.path.join(r["_root"], "runs", pt, "rep*", "run.log"))):
        t = trace(p)
        if t: reps.append(t)
    TR[(s, pt)] = reps

print("=" * 100)
print("1. COVERAGE")
print("=" * 100)
n_reps = sum(len(v) for v in TR.values())
print(f"  points generated      18 main + 6 addendum = 24 point-slots, 18 distinct tasksets")
print(f"  points executed       {len(RUN)}  ({sum(1 for k in RUN if k[0]=='main')} main + {sum(1 for k in RUN if k[0]=='add')} addendum)")
print(f"  board executions      {n_reps} runs ({n_reps} traces parsed)")
print(f"  total entries traced  {sum(len(t) for v in TR.values() for t in v)}")
arms = defaultdict(int)
for (s, pt), r in RUN.items(): arms[(s, r['arm'])] += 1
for k in sorted(arms, key=str): print(f"    {k[0]:<5} {k[1]:<12} {arms[k]} points")

print()
print("=" * 100)
print("2. PREDICTION ACCURACY -- makespan ratio (actual median / predicted)")
print("=" * 100)
rows = []
for (s, pt), r in sorted(RUN.items()):
    rows.append((s, pt, r["predicted_makespan_ms"], r["actual_makespan_median_ms"],
                 r["ratio_median"], r["actual_makespan_spread_ms"], r["n_entries"],
                 r["op_count"], r["arm"]))
print(f"{'set':<5} {'point':<20} {'arm':<11} {'ops':>4} {'ent':>4} {'pred ms':>9} {'act ms':>9} {'ratio':>7} {'spread ms':>10} {'noise %':>8}")
for s, pt, p, a, ra, sp, ne, oc, arm in rows:
    print(f"{s:<5} {pt:<20} {arm:<11} {oc:>4} {ne:>4} {p:>9.2f} {a:>9.2f} {ra:>7.3f} {sp:>10.2f} {100*sp/a:>8.2f}")
mr = [x[4] for x in rows]
print(f"\n  all {len(mr)} points: median {st.median(mr):.3f}  mean {st.mean(mr):.3f}  "
      f"min {min(mr):.3f}  max {max(mr):.3f}  stdev {st.stdev(mr):.3f}")
for tag, sel in (("main (mixed periodic+nonperiodic)", [x[4] for x in rows if x[0]=="main"]),
                 ("addendum (periodic only)", [x[4] for x in rows if x[0]=="add"])):
    print(f"  {tag:<38} n={len(sel):>2}  median {st.median(sel):.3f}  range {min(sel):.3f}-{max(sel):.3f}")

print()
print("=" * 100)
print("3. PER-TILE ACCURACY -- pooled over every point that placed the tile")
print("=" * 100)
agg = defaultdict(list)
for (s, pt), r in RUN.items():
    for k, v in (r.get("per_tile") or {}).items():
        tile = k.split("/", 1)[1] if "/" in k else k     # strip the job instance
        agg[tile].append(v)
print(f"{'tile@lane':<36} {'pts':>4} {'pred ms':>9} {'act p50':>9} {'ratio':>7} {'ratio range':>18} {'verdict'}")
tilerows = []
for tile in sorted(agg, key=lambda t: -st.median(v["ratio"] for v in agg[t] if v["ratio"])):
    vs = agg[tile]
    rs = sorted(v["ratio"] for v in vs if v["ratio"])
    if not rs: continue
    pr, ac = st.median(v["predicted_ms"] for v in vs), st.median(v["actual_p50_ms"] for v in vs)
    med = st.median(rs)
    verdict = ("COST CELL WRONG" if med > 1.5 or med < 0.67 else
               "contended" if med > 1.15 else
               "good" if 0.9 <= med <= 1.15 else "optimistic cell")
    tilerows.append((tile, len(vs), pr, ac, med, rs[0], rs[-1], verdict))
    print(f"{tile:<36} {len(vs):>4} {pr:>9.3f} {ac:>9.3f} {med:>7.3f} "
          f"{rs[0]:>8.2f} .. {rs[-1]:<8.2f} {verdict}")
good = [t for t in tilerows if t[7] == "good"]
print(f"\n  {len(good)}/{len(tilerows)} tile@lane cells predict within +/-15%")

print()
print("=" * 100)
print("4. LANE UTILIZATION -- busy time / makespan, from traces")
print("=" * 100)
print(f"{'set':<5} {'point':<20} {'makespan':>9} | {'hta busy':>9} {'dsp busy':>9} {'cpu busy':>9} | "
      f"{'hta %':>6} {'dsp %':>6} {'cpu %':>6} {'total %':>8}")
util = defaultdict(list)
for (s, pt) in sorted(TR):
    reps = TR[(s, pt)]
    if not reps: continue
    t = reps[0]
    mk = max(ms(r, "actual_end_cycles") for r in t)
    busy = defaultdict(float)
    for r in t:
        busy[r["backend"]] += ms(r, "actual_end_cycles") - ms(r, "actual_start_cycles")
    h, d, c = busy.get("HTA", 0), busy.get("DSP", 0), busy.get("CPU", 0)
    tot = 100 * (h + d + c) / (3 * mk)
    for lane, b in (("hta", h), ("dsp", d), ("cpu", c)): util[lane].append(100 * b / mk)
    util["total"].append(tot)
    print(f"{s:<5} {pt:<20} {mk:>9.2f} | {h:>9.2f} {d:>9.2f} {c:>9.2f} | "
          f"{100*h/mk:>6.1f} {100*d/mk:>6.1f} {100*c/mk:>6.1f} {tot:>8.1f}")
print(f"\n  median per-lane occupancy:  hta {st.median(util['hta']):.1f}%  "
      f"dsp {st.median(util['dsp']):.1f}%  cpu {st.median(util['cpu']):.1f}%")
print(f"  median 3-lane utilization:  {st.median(util['total']):.1f}%  "
      f"(1.0 would mean all three lanes busy the whole makespan)")

print()
print("=" * 100)
print("5. RUNTIME OVERHEAD -- gate accuracy and dispatch latency")
print("=" * 100)
print("  NOTE: the trace's dep_wait_ms / gate_ms columns are ABSOLUTE timestamps")
print("  (dep_done_ms, gate_done_ms), not durations. The meaningful quantities are")
print("    gate error   = gate_done - predicted_start   (how precisely the gate fires)")
print("    dispatch lat = actual_start - gate_done      (gate release -> QNN execute)")
print()
print(f"{'set':<5} {'point':<20} {'ent':>4} {'gate err p50':>13} {'gate err max':>13} "
      f"{'disp lat p50':>13} {'disp lat max':>13} {'exec p50':>9}")
ov = defaultdict(list)
for (s_, pt) in sorted(TR):
    reps = TR[(s_, pt)]
    if not reps: continue
    ge, dl, ex = [], [], []
    for t in reps:
        for r in t:
            gd = float(r["gate_ms"] or 0); ps = float(r["predicted_start_ms"] or 0)
            a0 = ms(r, "actual_start_cycles"); a1 = ms(r, "actual_end_cycles")
            ge.append(gd - ps); dl.append(a0 - gd); ex.append(a1 - a0)
    ov["gate_err"] += ge; ov["disp"] += dl; ov["exec"] += ex
    print(f"{s_:<5} {pt:<20} {len(reps[0]):>4} {st.median(ge):>13.4f} {max(ge):>13.4f} "
          f"{st.median(dl):>13.4f} {max(dl):>13.4f} {st.median(ex):>9.4f}")
print(f"\n  pooled over {len(ov['gate_err'])} entry-executions:")
print(f"    gate error     median {st.median(ov['gate_err']):+.4f} ms   "
      f"p95 {sorted(ov['gate_err'])[int(.95*len(ov['gate_err']))]:+.4f}   max {max(ov['gate_err']):+.4f}")
print(f"    dispatch latency median {st.median(ov['disp']):.4f} ms   "
      f"p95 {sorted(ov['disp'])[int(.95*len(ov['disp']))]:.4f}   max {max(ov['disp']):.4f}")
print(f"    exec duration  median {st.median(ov['exec']):.4f} ms")
print(f"\n  dispatch latency is pure runtime overhead: {100*st.median(ov['disp'])/st.median(ov['exec']):.1f}% "
      f"of median tile execution time")

print()
print("=" * 100)
print("6. PLACEMENT -- where each model's tiles land, by arm")
print("=" * 100)
place = defaultdict(lambda: defaultdict(int))
for (s, pt), r in RUN.items():
    for tile, lanes in (r.get("lane_placement") or {}).items():
        model = tile.split("/")[0].rstrip("_abcde") if "/" in tile else tile
        base = tile.split("/")[1] if "/" in tile else tile
        for lane, n in lanes.items(): place[base][lane] += n
print(f"{'tile':<26} {'hta':>6} {'dsp':>6} {'cpu':>6} {'total':>7}  {'lane share'}")
for tile in sorted(place, key=lambda t: -sum(place[t].values())):
    d = place[tile]; tot = sum(d.values())
    share = " ".join(f"{k}={100*v/tot:.0f}%" for k, v in sorted(d.items(), key=lambda x: -x[1]))
    print(f"{tile:<26} {d.get('hta',0):>6} {d.get('dsp',0):>6} {d.get('cpu',0):>6} {tot:>7}  {share}")

print()
print("  --- yolov8n_head on the CPU lane, by arm (the Q2 effect) ---")
for arm in ("baseline", "fused", "fused_vint"):
    pts = [(s, pt) for (s, pt), r in RUN.items() if r["arm"] == arm and s == "main"]
    hits = 0
    for k in pts:
        lp = RUN[k].get("lane_placement") or {}
        if any("yolov8n" in t and "head" in t and l.get("cpu") for t, l in lp.items()): hits += 1
    if pts: print(f"    {arm:<12} {hits}/{len(pts)} points place a yolov8n_head on cpu")

print()
print("=" * 100)
print("7. SOLVER")
print("=" * 100)
print(f"{'set':<5} {'point':<20} {'solver':<16} {'status':<20} {'solve s':>8} {'ops':>5} {'entries':>8}")
solve = defaultdict(list)
for (s, pt), r in sorted(RUN.items()):
    print(f"{s:<5} {pt:<20} {str(r.get('solver')):<16} {str(r.get('solver_status')):<20} "
          f"{r.get('solve_s',0):>8.1f} {r['op_count']:>5} {r['n_entries']:>8}")
    solve[r.get("solver_status")].append(r.get("solve_s", 0))
print()
for k, v in sorted(solve.items(), key=lambda x: -len(x[1])):
    print(f"    {str(k):<22} {len(v):>2} points   solve time median {st.median(v):6.1f}s  max {max(v):6.1f}s")
print(f"    greedy fallbacks       {sum(1 for r in RUN.values() if r.get('solver')!='milp')}")

print()
print("=" * 100)
print("8. TRENDS")
print("=" * 100)
xs = [(r["op_count"], r["n_entries"], r["predicted_makespan_ms"], r["actual_makespan_median_ms"],
       r["ratio_median"], 100*r["actual_makespan_spread_ms"]/r["actual_makespan_median_ms"],
       s) for (s, pt), r in RUN.items()]
def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    den = (sum((x-ma)**2 for x in a) * sum((y-mb)**2 for y in b)) ** 0.5
    return num/den if den else 0.0
ops   = [x[0] for x in xs]; ent = [x[1] for x in xs]
pred  = [x[2] for x in xs]; act = [x[3] for x in xs]
ratio = [x[4] for x in xs]; noise = [x[5] for x in xs]
print(f"  Pearson r, over all {len(xs)} points:")
print(f"    op_count      vs actual makespan   {corr(ops, act):+.3f}")
print(f"    entries       vs actual makespan   {corr(ent, act):+.3f}")
print(f"    predicted     vs actual makespan   {corr(pred, act):+.3f}   <- model tracks shape well")
print(f"    op_count      vs ratio             {corr(ops, ratio):+.3f}")
print(f"    actual        vs ratio             {corr(act, ratio):+.3f}   <- bigger schedules drift more")
print(f"    actual        vs rep noise %       {corr(act, noise):+.3f}")
mainr = [x[4] for x in xs if x[6] == "main"]; addr = [x[4] for x in xs if x[6] == "add"]
print(f"\n  regime split: mixed median {st.median(mainr):.3f} (n={len(mainr)}), "
      f"periodic-only median {st.median(addr):.3f} (n={len(addr)})")

print()
print("=" * 100)
print("9. WHERE THE 1.17x COMES FROM -- runtime overhead vs cost-model error")
print("=" * 100)
print(f"{'set':<5} {'point':<20} {'pred busy':>10} {'act busy':>10} {'busy ratio':>11} "
      f"{'makespan ratio':>15} {'attributable':>13}")
bz, mkr = [], []
for (s_, pt) in sorted(TR):
    reps = TR[(s_, pt)]
    r = RUN[(s_, pt)]
    if not reps: continue
    t = reps[0]
    pb = sum(float(x["predicted_duration_ms"] or 0) for x in t)
    ab = st.median([sum(ms(x, "actual_end_cycles") - ms(x, "actual_start_cycles") for x in tt)
                    for tt in reps])
    br = ab / pb if pb else 0
    bz.append(br); mkr.append(r["ratio_median"])
    print(f"{s_:<5} {pt:<20} {pb:>10.2f} {ab:>10.2f} {br:>11.3f} {r['ratio_median']:>15.3f} "
          f"{100*(br-1)/(r['ratio_median']-1) if r['ratio_median']>1.001 else float('nan'):>12.0f}%")
print(f"\n  median busy ratio (cost-model error)  {st.median(bz):.3f}")
print(f"  median makespan ratio                 {st.median(mkr):.3f}")
print(f"  runtime dispatch overhead             ~0.009 ms/entry = ~1% of median tile exec")
print()
print("  Reading: the cost model under-predicts TILE EXECUTION by about the same")
print("  factor as the makespan misses by. The runtime adds ~9 us per dispatch, which")
print("  cannot account for a 17% makespan gap. The gap is cost-model error, amplified")
print("  where lane occupancy is high (dsp sits at 92.5% median) because an optimistic")
print("  cell makes the solver pack the lane tighter than the hardware can sustain.")

print()
print("=" * 100)
print("10. COST-MODEL ERROR vs TILE SIZE -- is there a fixed per-dispatch offset?")
print("=" * 100)
import math
pts = []
for tile, vs in agg.items():
    pr = st.median(v["predicted_ms"] for v in vs)
    ac = st.median(v["actual_p50_ms"] for v in vs)
    if pr > 0 and ac > 0: pts.append((tile, pr, ac, ac/pr))
pts.sort(key=lambda x: x[1])
print(f"{'tile@lane':<36} {'pred ms':>9} {'act ms':>9} {'ratio':>8} {'act-pred ms':>12}")
for t, pr, ac, ra in pts:
    print(f"{t:<36} {pr:>9.3f} {ac:>9.3f} {ra:>8.3f} {ac-pr:>12.3f}")
lp = [math.log(x[1]) for x in pts]; lr = [math.log(x[3]) for x in pts]
print(f"\n  Pearson r( log predicted_ms , log ratio ) = {corr(lp, lr):+.3f}")
print("  strongly negative => small tiles are over-run by a roughly FIXED cost,")
print("  which a purely multiplicative cost model cannot express.")
small = [x for x in pts if x[1] < 1.0]; big = [x for x in pts if x[1] >= 1.0]
print(f"\n  tiles < 1 ms predicted (n={len(small)}): median ratio {st.median(x[3] for x in small):.3f}, "
      f"median absolute miss {st.median(x[2]-x[1] for x in small):+.3f} ms")
print(f"  tiles >= 1 ms predicted (n={len(big)}): median ratio {st.median(x[3] for x in big):.3f}, "
      f"median absolute miss {st.median(x[2]-x[1] for x in big):+.3f} ms")
off = st.median(x[2]-x[1] for x in small)
print(f"\n  Suggested cost-model change: add a fixed per-dispatch offset of about")
print(f"  {off:.3f} ms to every cell, then re-fit the multiplicative part. The big")
print(f"  accelerator tiles already predict at 0.99-1.00x and would barely move.")
