#!/usr/bin/env python3
"""Write a profile tree whose per-width tables carry the board's measured shard costs.

WHY. `gen/mb/profile/.../yolov8_nano_64x96/.../topo_0`, `topo_0_1` and `topo_0_1_2_3` hold the
same numbers: 55.7, 55.6 and 55.7 ms summed over the 90 dispatches. The topo tag names the harts
a measurement held, and the scheduler reads the table whose tag matches a machine combination's
size -- so with those tables a four-hart combination is costed exactly like one hart, sharding can
never shorten anything, and CP-SAT is right to place every dispatch on one hart. The board says a
four-hart pool runs the same 90 dispatches in 24.24 ms against 47.13 on one.

WHAT IT READS. `results/codesign_feedback/ros_traced/yolo_standalone/<name>.txt` -- the standalone
harness with `MODELBLASTER_CPU` set to the harts it may use, which prints one
`MODELBLASTER_ITER_PROFILE` block per iteration: `dispatch_id,name,op,shape,cycles`. The first
iteration of each dispatch is dropped (cold) and the rest taken at the median, so this is the same
statistic the tables already hold.

WHAT IT WRITES. A copy of the profile tree under `--out-root` with the per-width tables of the
named network replaced, every other file byte-identical. `source` becomes `k1_pool_w<N>` so a row's
origin is visible, and the run each number came from is recorded in `provenance.json` next to it.

    scripts/emit_shard_profile.py --widths 1:1core_w 2:2core 4:4core_w 8:8core_w \
        --out-root gen/mb_shard
"""
import argparse, collections, csv, json, os, shutil, statistics as st, sys

RT_DEFAULT = "results/codesign_feedback/ros_traced/yolo_standalone"
TOPO = {1: "topo_0", 2: "topo_0_1", 4: "topo_0_1_2_3", 8: "topo_0_1_2_3_4_5_6_7"}
RDTIME_HZ = 24_000_000.0


def read_records(path):
    """{dispatch_id: median cycles over the warm iterations}, and the op/shape seen."""
    per = collections.defaultdict(list); meta = {}
    inside = False
    for ln in open(path):
        if "ITER_PROFILE_BEGIN" in ln: inside = True; continue
        if "ITER_PROFILE_END" in ln: inside = False; continue
        if not inside or ln.startswith("dispatch_id"): continue
        p = ln.rstrip("\n").split(",")
        if len(p) < 5 or not p[0].isdigit(): continue
        did = int(p[0]); per[did].append(int(p[-1])); meta[did] = (p[1], p[2])
    return {d: st.median(v[1:] or v) for d, v in per.items()}, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="yolov8_nano_64x96")
    ap.add_argument("--impl", default="rvv_x60")
    ap.add_argument("--target", default="spacemit_x60")
    ap.add_argument("--gen-root", default="gen/mb")
    ap.add_argument("--out-root", default="gen/mb_shard")
    ap.add_argument("--runs-dir", default=RT_DEFAULT,
                    help="directory of standalone harness logs the widths name; one per network")
    ap.add_argument("--widths", nargs="+", required=True,
                    help="N:<standalone log basename>, e.g. 4:4core_w")
    a = ap.parse_args()

    widths = {}
    for w in a.widths:
        n, name = w.split(":"); widths[int(n)] = name

    src = os.path.join(a.gen_root, "profile")
    dst = os.path.join(a.out_root, "profile")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copytree(src, dst, symlinks=False)

    base = os.path.join(dst, a.impl, a.target, a.net)
    hits = [os.path.join(r, "results.csv") for r, _, fs in os.walk(base) if "results.csv" in fs]
    by_topo = {}
    for h in hits:
        by_topo.setdefault(os.path.basename(os.path.dirname(h)), []).append(h)

    prov = {"script": os.path.basename(__file__), "net": a.net, "runs_dir": a.runs_dir, "widths": {}}
    for n, name in sorted(widths.items()):
        log = os.path.join(a.runs_dir, f"{name}.txt")
        if not os.path.exists(log):
            print(f"  width {n}: no {log}; left as it was"); continue
        cyc, meta = read_records(log)
        topo = TOPO.get(n)
        targets = by_topo.get(topo, [])
        if not targets:
            print(f"  width {n}: no {topo} table under {base}"); continue
        for t in targets:
            rows = list(csv.DictReader(open(t)))
            hdr = list(rows[0].keys())
            miss = 0
            for r in rows:
                d = int(r["dispatch_id"])
                if d not in cyc:
                    miss += 1; continue
                ms = cyc[d] / RDTIME_HZ * 1000.0
                r["cycles"] = str(int(cyc[d])); r["cycles_n"] = "1"
                r["mean_time"] = f"{ms:.6f}"; r["mean_unit"] = "ms"
                r["mean_time_ns"] = f"{ms * 1e6:.6f}"
                r["source"] = f"k1_pool_w{n}"
            with open(t, "w", newline="") as fh:
                wtr = csv.DictWriter(fh, fieldnames=hdr); wtr.writeheader(); wtr.writerows(rows)
            tot = sum(cyc[int(r["dispatch_id"])] for r in rows
                      if int(r["dispatch_id"]) in cyc) / RDTIME_HZ * 1000.0
            print(f"  width {n} -> {topo}: {len(rows) - miss}/{len(rows)} dispatches, "
                  f"sum {tot:.2f} ms  ({os.path.relpath(t, dst)})")
            prov["widths"][n] = {"log": log, "topo": topo, "table": os.path.relpath(t, dst),
                                 "dispatches": len(rows) - miss, "sum_ms": round(tot, 3)}
    json.dump(prov, open(os.path.join(dst, "provenance.json"), "w"), indent=2)
    print(f"wrote {dst}")


if __name__ == "__main__":
    sys.exit(main())
