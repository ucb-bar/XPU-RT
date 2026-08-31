#!/usr/bin/env python3
"""Calibrate the cell measurement methodology against in-situ ground truth.

The sweep gives a strong reference: for each (tile, lane) it recorded the p50
execution actually observed in a schedule, pooled over up to 41 placements.
A cell measured standalone should reproduce that. This sweeps the measurement
parameters -- idle gap before each execute, and CPU affinity -- and reports
which setting reproduces in-situ best, instead of picking one by assumption.

    python3 calibrate_cells.py [--iters 30]
"""
import argparse, json, os, statistics as st, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD = os.environ.get("QNN_BOARD_HOST", "root@10.44.120.201")
CTX = "/root/qnn_runtime_ctx"
BD = "/root/flowc_remeasure"
SDK = "/root/qairt"
LIB = {"hta": "libQnnHta.so", "dsp": "libQnnDsp.so", "cpu": "libQnnCpu.so"}
GAPS = [0, 250, 1000, 3000, 10000]

# (tile, lane) -> ctx, chosen for a strong in-situ reference across sizes/lanes
PROBES = [
    ("yolov8n/yolov8n_backbone", "hta", "ctx_yolov8n_backbone__Hta.bin"),
    ("yolov8n/yolov8n_head",     "dsp", "ctx_yolov8n_head__Dsp.bin"),
    ("dronet/dronet_full",       "dsp", "ctx_dronet_full_hta__Dsp.bin"),
    ("dronet/dronet_full",       "hta", "ctx_dronet_full_hta__Hta.bin"),
    ("fused_split/fused_vision_conv", "hta", "ctx_fused_vision_conv__Hta.bin"),
    ("fused_split/fused_depth_conv",  "dsp", "ctx_fused_depth_conv__Dsp.bin"),
    ("mlp_control/mlp_control_full",  "dsp", "ctx_mlp_control__Dsp.bin"),
    ("mlp_control/mlp_control_full",  "cpu", "ctx_mlp_control_fp32__Cpu.bin"),
]


def board(script, timeout=900):
    q = script.replace("'", "'\\''")
    return subprocess.run(
        ["timeout", "-s", "KILL", str(timeout + 60), "ssh", "-o", "ConnectTimeout=20",
         "-o", "BatchMode=yes", BOARD, f"flock -w 900 /tmp/qnn_board.lock -c '{q}'"],
        capture_output=True, text=True, timeout=timeout + 120)


def in_situ():
    agg = {}
    S = os.path.join(HERE, "sweeps", "qrb5165_20260829-200620")
    for rel in ("results.json", "addendum_periodic_only/results.json"):
        p = os.path.join(S, rel)
        if not os.path.exists(p): continue
        for r in json.load(open(p)):
            if r.get("status") != "run": continue
            for k, v in (r.get("per_tile") or {}).items():
                tl = k.split("/", 1)[1] if "/" in k else k
                tile, lane = tl.rsplit("@", 1)
                agg.setdefault((tile, lane), []).append(v["actual_p50_ms"])
    return {k: (st.median(v), len(v)) for k, v in agg.items()}


def run(ctx, lib, iters, gap, mask):
    env = (f'LD_LIBRARY_PATH={SDK}/lib/target '
           f'ADSP_LIBRARY_PATH="{SDK}/lib/hexagon-v66/unsigned;{SDK}/lib/hexagon-v66;'
           f'/dsp/cdsp;/dsp" ')
    cmd = (f"cd {BD} && {env}{mask}./profile_seg {CTX}/{ctx} {lib} {iters} --gap-us {gap}")
    r = board(cmd, timeout=600)
    for ln in (r.stdout or "").splitlines():
        if ln.strip().startswith("{"):
            try: return json.loads(ln.strip())
            except json.JSONDecodeError: pass
    return {"status": "failed", "err": (r.stderr or "")[-160:]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    situ = in_situ()
    out = []
    print(f"{'tile@lane':<38} {'in-situ':>9} {'n':>3} | " +
          " ".join(f"{'gap '+str(g):>9}" for g in GAPS) + f" {'loop':>9}")
    for cell, lane, ctx in PROBES:
        tile = cell.split("/", 1)[1]
        ref = situ.get((tile, lane))
        cells = []
        loops = []
        for g in GAPS:
            js = run(ctx, LIB[lane], a.iters, g, "")
            if js.get("status") != "ok":
                cells.append(None); loops.append(None); continue
            cells.append(js["gap_median_us"] / 1000.0)
            loops.append(js["median_us"] / 1000.0)
            out.append({"cell": cell, "lane": lane, "gap_us": g, "mask": "none",
                        "gap_median_ms": js["gap_median_us"] / 1000.0,
                        "loop_median_ms": js["median_us"] / 1000.0,
                        "in_situ_ms": ref[0] if ref else None})
        f = lambda v: f"{v:9.3f}" if v is not None else "       --"
        lm = st.median([x for x in loops if x is not None]) if any(loops) else None
        print(f"{tile+'@'+lane:<38} {f(ref[0]) if ref else '       --'} "
              f"{ref[1] if ref else 0:>3} | " + " ".join(f(c) for c in cells) + f" {f(lm)}")

    # which gap best reproduces in-situ?
    print()
    for g in GAPS + ["loop"]:
        errs = []
        for r in out:
            if r["in_situ_ms"] is None: continue
            v = r["loop_median_ms"] if g == "loop" else (r["gap_median_ms"] if r["gap_us"] == g else None)
            if g == "loop" and r["gap_us"] != GAPS[0]: continue
            if v is None: continue
            errs.append(abs(v - r["in_situ_ms"]) / r["in_situ_ms"])
        if errs:
            print(f"  gap {str(g):>6}: median |error| vs in-situ {100*st.median(errs):6.1f}%  (n={len(errs)})")
    with open(os.path.join(HERE, "measurements", "calibration_gap_sweep.json"), "w") as fh:
        json.dump({"iters": a.iters, "gaps_us": GAPS, "probes": out}, fh, indent=1)
    print(f"\n  -> measurements/calibration_gap_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
