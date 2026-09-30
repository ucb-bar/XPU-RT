#!/usr/bin/env python3
"""Board-measured multi-hart profile rows for YOLO, so the solver can value sharding.

The profile database's `topo_0`, `topo_0_1` and `topo_0_1_2_3` rows for `yolov8_nano_64x96`
carry the same per-dispatch time (the shard harness timed each shard's work, not the sharded
dispatch's wall time), so a solver sees no benefit in a wider placement. The executed traces do
carry it: a dispatch sharded over two or four harts is one trace row whose span covers all its
shards, and the schedule the trace executed says how wide it ran. This writes, for each width,
the median wall time per dispatch over the warm instances of the trace(s) named, into a copy of
the profile database (`--gen-root`, default `gen/mb_cal`: YOLO's rows calibrated, every other
network's rows linked to the originals), keeping the original rows next to them as
`results.orig.csv` and recording the source traces in `results.provenance.json`. A spec that
sets `hardware.profile.gen_root` to that copy solves against the board-measured widths.

    scripts/calibrate_yolo_shard_profile.py \
        --w1 results/codesign_feedback/xpurt_long/trace_acpsat_hardr1_other_run1.csv \
        --w2 results/codesign_feedback/xpurt_long/trace_best45alt2_other_run1.csv \
        --w4 results/codesign_feedback/xpurt_long/trace_best45p4_other_run1.csv
"""
from __future__ import annotations
import argparse, collections, csv, glob, json, os, shutil, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from make_measured_gantt_pair import read_trace   # noqa: E402

NET = "yolov8_nano_64x96"
TOPO = {1: "topo_0", 2: "topo_0_1", 4: "topo_0_1_2_3"}


def make_profile_copy(gen_root):
    """gen/mb_cal/profile: YOLO's profile directory copied, every other model's linked."""
    src = os.path.join(REPO, "gen/mb", "profile"); dst = os.path.join(REPO, gen_root, "profile")
    for hw in os.listdir(src):
        for target in os.listdir(os.path.join(src, hw)):
            for model in os.listdir(os.path.join(src, hw, target)):
                s_ = os.path.join(src, hw, target, model); d_ = os.path.join(dst, hw, target, model)
                if os.path.exists(d_) or os.path.islink(d_):
                    continue
                os.makedirs(os.path.dirname(d_), exist_ok=True)
                if model == NET:
                    shutil.copytree(s_, d_)
                else:
                    os.symlink(s_, d_)
    return glob.glob(f"{dst}/rvv_x60/spacemit_x60/{NET}/*/*/topo_0/results.csv")


def widths_from_schedule(sched_path):
    """{(instance, dispatch_id): (width, on P cluster)} of the YOLO dispatches the trace executed."""
    sys.path.insert(0, os.path.join(REPO, "xpu-rt")); from job_names import split_job_name   # noqa: E402
    known = {NET, "fused_full", "mlp_control", "ffn_block", "dronet"}; out = {}
    for v in json.load(open(sched_path))["dispatches"].values():
        net, inst = split_job_name(v["job_name"], known)
        if net == NET:
            tgt = v["hardware_target"].split("+")
            out[(int(inst or 0), int(v["id"]))] = (len(tgt), all(t.startswith("CPU_P") for t in tgt))
    return out


def wall_per_dispatch(traces, width, p_only=True):
    """{dispatch_id: median wall ms} for YOLO dispatches executed `width` harts wide. A sharded
    dispatch is one trace row whose span covers all its shards; its width comes from the schedule
    the trace executed (manifest -> schedule)."""
    per = collections.defaultdict(list)
    for t in traces:
        man = json.load(open(t.replace("trace_", "manifest_").replace(".csv", ".json")))
        wmap = widths_from_schedule(man["schedule"])
        for r in read_trace(t):
            if r["net"] != NET or r["did"] is None or r["inst"] < 1:
                continue
            w, on_p = wmap.get((r["inst"], r["did"]), (1, r["hart"] < 4))
            if w != width or (p_only and width > 1 and not on_p):
                continue                      # E-cluster placements are slower; the row is the P-cluster figure
            per[r["did"]].append(r["e"] - r["s"])
    return {d: statistics.median(v) for d, v in per.items()}, {d: len(v) for d, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", nargs="+", required=True); ap.add_argument("--w2", nargs="+", required=True); ap.add_argument("--w4", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--gen-root", default="gen/mb_cal")
    a = ap.parse_args()
    DB = glob.glob(f"{REPO}/gen/mb/profile/rvv_x60/spacemit_x60/{NET}/*/*/topo_0/results.csv") if a.dry_run else make_profile_copy(a.gen_root)
    if not DB:
        raise SystemExit("no yolov8 topo_0 profile row")
    base_dir = os.path.dirname(os.path.dirname(DB[0]))
    w = {1: wall_per_dispatch(a.w1, 1, p_only=False), 2: wall_per_dispatch(a.w2, 2), 4: wall_per_dispatch(a.w4, 4)}
    prov = {"note": "per-dispatch wall time (first shard start to last shard end), median over warm instances, P-cluster placements",
            "sources": {"w1": a.w1, "w2": a.w2, "w4": a.w4}}
    for width, topo in TOPO.items():
        path = os.path.join(base_dir, topo, "results.csv")
        rows = list(csv.DictReader(open(path))); fields = list(rows[0].keys())
        meas, n = w[width]
        tot_before = sum(float(r["mean_time"]) for r in rows); changed = 0
        for r in rows:
            did = int(r["dispatch_id"])
            if did in meas:
                r["mean_time"] = f"{meas[did]:.6f}"; r["mean_time_ns"] = f"{meas[did] * 1e6:.6f}"; r["mean_unit"] = "ms"
                r["source"] = "k1_trace"; r["cycles_n"] = str(n[did]); changed += 1
            elif width > 1 and did in w[1][0]:
                r["mean_time"] = f"{w[1][0][did]:.6f}"; r["mean_time_ns"] = f"{w[1][0][did] * 1e6:.6f}"; r["source"] = "k1_trace"   # not sharded at this width: its one-hart time
        tot_after = sum(float(r["mean_time"]) for r in rows)
        print(f"{topo}: {changed} dispatches measured at width {width}; sum of row times {tot_before:.2f} -> {tot_after:.2f} ms")
        prov[topo] = {"dispatches_measured": changed, "sum_ms_before": round(tot_before, 3), "sum_ms_after": round(tot_after, 3)}
        if a.dry_run:
            continue
        orig = os.path.join(base_dir, topo, "results.orig.csv")
        if not os.path.exists(orig):
            shutil.copy(path, orig)
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows(rows)
    if not a.dry_run:
        json.dump(prov, open(os.path.join(base_dir, "results.provenance.json"), "w"), indent=2)
        print("wrote", os.path.join(base_dir, "results.provenance.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
