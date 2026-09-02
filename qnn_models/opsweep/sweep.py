#!/usr/bin/env python3
"""Operator sweep across backends, precisions and problem sizes on QRB5165.

Produces one row per (op, size, precision, backend) with enough structure to
separate the two things that decide every placement on this board:

    system overhead  the per-dispatch cost, which on HTA is ~543 us warm and
                     ~2470 us cold and is INDEPENDENT of the work
    compute          the marginal term, recovered as the slope of the size ladder

Both are measured, not modelled. Each point is profiled twice -- `--gap-us 0`
(back to back, accelerator stays clocked) and `--gap-us 3000` (idle long enough
to power down) -- so the cold/warm difference isolates the DVFS component from
the dispatch cost proper. `fit` then regresses `t = overhead + macs/throughput`
per (op, backend, precision) series.

Failures are data. A backend that refuses an op is recorded with its verbatim
validator message rather than left blank: "HTA has no two-dynamic-operand MatMul
at any rank" is a result, and it is why attention cannot be placed there.

Resumable: every finished row is appended to results.jsonl and skipped on a
re-run, so the sweep can be interrupted and restarted, and partial results plot.

    python3 sweep.py --all                      # gen, build, measure, fit
    python3 sweep.py --only measure --ops conv2d,linear
    python3 sweep.py --only fit                 # re-fit without touching the board
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "gen")
RESULTS = os.path.join(HERE, "results.jsonl")
QNN_SDK = os.environ.get("QNN_SDK", "/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326")
DOCKER_IMG = os.environ.get("QNN_DOCKER_IMG", "qnn-convert")
BOARD = os.environ.get("QNN_BOARD", "root@10.44.120.201")
LOCK = "/tmp/qnn_board.lock"
BDIR = "/data/opsweep"
PROFILE_SEG = "/root/models/smolvlm_vision_v3/profile_seg"
LIB = "/root/qairt/lib/target/libQnn%s.so"
GAPS = (0, 3000)


def tag(op, prm):
    return f"{op}__" + "_".join(str(x) for x in prm)


def sh(cmd, timeout=9000):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def board(script, timeout=3000):
    inner = f"timeout -s KILL {timeout-120} ssh -o ConnectTimeout=10 {BOARD} bash -s"
    cmd = f"timeout {timeout} flock -w {timeout} {LOCK} -c {shlex.quote(inner)}"
    r = subprocess.run(cmd, shell=True, input=script, capture_output=True,
                       text=True, timeout=timeout + 180)
    return r.stdout + r.stderr


def done_keys():
    keys = set()
    if os.path.exists(RESULTS):
        for ln in open(RESULTS):
            try:
                d = json.loads(ln)
                keys.add((d["op"], tuple(d["params"]), d["precision"], d["backend"]))
            except Exception:
                pass
    return keys


def emit(row):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------- stages ---
def stage_gen(pts):
    """Build the ONNX models in the converter container (host needs no onnx)."""
    os.makedirs(WORK, exist_ok=True)
    plan = [[op, list(prm), axis, val, tag(op, prm)] for op, prm, axis, val in pts]
    todo = [r for r in plan
            if not os.path.exists(os.path.join(WORK, r[4], "m.onnx"))]
    if not todo:
        print(f"[gen]   0 new models, {len(pts)} total"); return
    json.dump(todo, open(os.path.join(WORK, "_plan.json"), "w"))
    script = ('pip install -q "numpy<2" onnx >/dev/null 2>&1\n'
              'python3.10 /src/genmodels.py /workspace/_plan.json /workspace\n')
    out = sh(f"sudo docker run --rm -v {shlex.quote(HERE)}:/src:ro "
             f"-v {shlex.quote(WORK)}:/workspace {DOCKER_IMG} bash -c {shlex.quote(script)}",
             timeout=3600)
    sh(f"sudo chown -R {os.getuid()}:{os.getgid()} {shlex.quote(WORK)}")
    made = next((l.split()[1] for l in out.splitlines() if l.startswith("GENERATED")), "?")
    print(f"[gen]   {made} new models, {len(pts)} total", flush=True)


def stage_build(pts):
    """Convert to fp32 DLC and quantize to int8, batched into one container."""
    todo = [tag(op, prm) for op, prm, _, _ in pts
            if not os.path.exists(os.path.join(WORK, tag(op, prm), "m_q.dlc"))]
    if not todo:
        print("[build] nothing to do"); return
    print(f"[build] {len(todo)} models", flush=True)
    for i in range(0, len(todo), 25):
        chunk = todo[i:i + 25]
        script = 'pip install -q "numpy<2" >/dev/null 2>&1\n'
        for t in chunk:
            script += (
                f'C=/qnn/bin/x86_64-linux-clang\n'
                f'python3.10 $C/snpe-onnx-to-dlc --input_network /workspace/{t}/m.onnx '
                f'--output_path /workspace/{t}/m.dlc >/workspace/{t}/conv.log 2>&1\n'
                f'python3.10 $C/qairt-quantizer --input_dlc /workspace/{t}/m.dlc '
                f'--output_dlc /workspace/{t}/m_q.dlc --input_list /workspace/{t}/list.txt '
                f'--act_bitwidth 8 --weights_bitwidth 8 >/workspace/{t}/quant.log 2>&1\n'
                f'echo "BUILT {t} $([ -f /workspace/{t}/m.dlc ] && echo fp32) '
                f'$([ -f /workspace/{t}/m_q.dlc ] && echo int8)"\n')
        out = sh(f"sudo docker run --rm -v {shlex.quote(QNN_SDK)}:/qnn:ro "
                 f"-v {shlex.quote(WORK)}:/workspace {DOCKER_IMG} bash -c {shlex.quote(script)}",
                 timeout=9000)
        ok = sum(1 for l in out.splitlines() if l.startswith("BUILT") and "int8" in l)
        print(f"[build]   chunk {i//25+1}: {ok}/{len(chunk)} quantized", flush=True)
    sh(f"sudo chown -R {os.getuid()}:{os.getgid()} {shlex.quote(WORK)}")


def stage_measure(pts, iters, chunk_n):
    from grid import PRECISION_BACKENDS
    have = done_keys()
    work = []
    for op, prm, axis, val in pts:
        t = tag(op, prm)
        for prec, bks in PRECISION_BACKENDS.items():
            dlc = "m_q.dlc" if prec == "int8" else "m.dlc"
            if not os.path.exists(os.path.join(WORK, t, dlc)):
                continue
            for bk in bks:
                if (op, tuple(prm), prec, bk) not in have:
                    work.append((op, prm, axis, val, t, prec, bk, dlc))
    print(f"[measure] {len(work)} pending of {len(pts)*7} max", flush=True)
    if not work:
        return
    sh(f"timeout 300 flock -w 900 {LOCK} -c {shlex.quote(f'timeout -s KILL 200 ssh -n {BOARD} mkdir -p {BDIR}')}")
    pushed = set()
    for i in range(0, len(work), chunk_n):
        ch = work[i:i + chunk_n]
        for _, _, _, _, t, _, _, dlc in ch:
            if (t, dlc) in pushed:
                continue
            pushed.add((t, dlc))
            sh(f"timeout 900 flock -w 1200 {LOCK} -c "
               f"{shlex.quote(f'timeout -s KILL 800 scp {os.path.join(WORK,t,dlc)} {BOARD}:{BDIR}/{t}__{dlc}')}")
        lines = "\n".join(
            f'run {t}__{dlc} {bk} {t}_{prec}_{bk}' for _, _, _, _, t, prec, bk, dlc in ch)
        script = f"""set -u
cd {BDIR}
export LD_LIBRARY_PATH=/root/qairt/lib/target:${{LD_LIBRARY_PATH:-}}
export ADSP_LIBRARY_PATH=/root/qairt/lib/target
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $c; done
run() {{
  D=ctx_$3; rm -rf $D; mkdir -p $D
  timeout -s KILL 400 /root/qairt/bin/target/qnn-context-binary-generator \\
    --model /root/qairt/lib/target/libQnnModelDlc.so --backend /root/qairt/lib/target/libQnn$2.so \\
    --dlc_path {BDIR}/$1 --binary_file b --output_dir {BDIR}/$D > $D.log 2>&1
  if [ ! -f $D/b.bin ]; then
    E=$(grep -E "\\[ *ERROR *\\]" $D.log | grep -viE "Failed to (successfully compose|create|initialize)|Exception encountered" \\
        | head -1 | sed -E 's/^ *[0-9.]+m?s *\\[ *ERROR *\\] *//' | cut -c1-120 | tr -d '\\n"')
    echo "ROW $3 COMPOSE_FAIL ${{E:-no_error_line}}"
    rm -rf $D $D.log; return
  fi
  # Discard run: the first dispatch on a fresh context pays backend bringup,
  # which otherwise lands on whichever phase happens to run first.
  timeout -s KILL 300 {PROFILE_SEG} $D/b.bin /root/qairt/lib/target/libQnn$2.so 4 --gap-us 0 >/dev/null 2>&1
  # ONE call gives both phases: the loop phase is back-to-back (warm), the gap
  # phase inserts 3 ms so the accelerator power-collapses between calls (cold).
  R=$(timeout -s KILL 300 {PROFILE_SEG} $D/b.bin /root/qairt/lib/target/libQnn$2.so {iters} --gap-us 3000 2>&1 | grep -E '^{{' | head -1)
  echo "ROW $3 OK $R"   # bare: a ${{R:-...}} default would close on the JSON's own brace
  rm -rf $D $D.log
}}
{lines}
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo $GOV > $c; done
"""
        out = board(script, timeout=min(3000, 300 + 90 * len(ch)))
        idx = {f"{t}_{prec}_{bk}": (op, prm, axis, val, prec, bk)
               for op, prm, axis, val, t, prec, bk, _ in ch}
        got = 0
        for ln in out.splitlines():
            if not ln.startswith("ROW "):
                continue
            f = ln.split(None, 3)
            key = f[1]
            if key not in idx:
                continue
            op, prm, axis, val, prec, bk = idx.pop(key)
            meta = json.load(open(os.path.join(WORK, tag(op, prm), "meta.json")))
            row = {"op": op, "params": list(prm), "axis": axis, "value": val,
                   "precision": prec, "backend": bk.lower(), "macs": meta["macs"],
                   "shapes": meta["shapes"]}
            if f[2] == "OK":
                try:
                    j = json.loads(f[3]) if len(f) > 3 else {}
                except ValueError:
                    j = {}
                # Back-to-back is the condition a packed pipeline actually
                # runs in, so the median of the loop phase is the headline.
                # HTA and GPU are heavy-tailed here (they power-collapse part
                # way through even a zero-gap loop, sd ~900 us), so keep the
                # min as the achievable floor and the sd as the evidence.
                row["warm_us"] = j.get("median_us")
                row["warm_min_us"] = j.get("min_us")
                row["warm_std_us"] = j.get("std_us")
                row["cold_us"] = j.get("gap_median_us")
                row["init_us"] = j.get("init_us")
                row["iters"] = j.get("iters")
                row["status"] = "ok" if row["warm_us"] else "no_timing"
            else:
                row["status"] = "compose_fail"
                row["error"] = (f[3] if len(f) > 3 else "").strip()[:180]
            emit(row); got += 1
        for key, (op, prm, axis, val, prec, bk) in idx.items():
            emit({"op": op, "params": list(prm), "axis": axis, "value": val,
                  "precision": prec, "backend": bk.lower(), "status": "no_result"})
        print(f"[measure]   {i+len(ch)}/{len(work)}  (+{got})", flush=True)


def stage_fit():
    rows = [json.loads(l) for l in open(RESULTS)] if os.path.exists(RESULTS) else []
    ok = [r for r in rows if r.get("status") == "ok" and r.get("macs")]
    print(f"[fit] {len(rows)} rows, {len(ok)} timed", flush=True)
    series = {}
    for r in ok:
        series.setdefault((r["op"], r["backend"], r["precision"]), []).append(r)
    out = []
    for (op, bk, prec), rs in sorted(series.items()):
        if len(rs) < 3:
            continue
        m = np.array([r["macs"] for r in rs], float) / 1e6
        for phase in ("warm_us", "cold_us"):
            y = np.array([r.get(phase) or np.nan for r in rs], float)
            g = ~np.isnan(y)
            if g.sum() < 3:
                continue
            A = np.vstack([np.ones(g.sum()), m[g]]).T
            (a, s), *_ = np.linalg.lstsq(A, y[g], rcond=None)
            # A negative intercept is not a negative dispatch cost, it means
            # this series is not linear in MACs (the op is memory- or
            # element-bound).  Refit through the origin and say so, rather than
            # clamping it out of sight at plot time.
            clamped = bool(a < 0)
            if clamped:
                a = 0.0
                s = float(np.linalg.lstsq(m[g][:, None], y[g], rcond=None)[0][0])
                A = np.vstack([np.zeros(g.sum()), m[g]]).T
            pred = A @ np.array([a, s])
            r2 = 1 - ((y[g]-pred)**2).sum()/max(((y[g]-y[g].mean())**2).sum(), 1e-9)
            out.append({"op": op, "backend": bk, "precision": prec, "phase": phase,
                        "overhead_us": round(float(a), 2), "origin_fit": clamped,
                        # 4 significant digits, not 1 decimal: a slow low-MAC op
                        # (softmax on DSP) rounds to 0.0 and its compute
                        # term silently vanishes from the decomposition.
                        "gmac_per_s": float(f"{1e3/s:.4g}") if s > 0 else None,
                        "r2": round(float(r2), 3), "n": int(g.sum())})
    json.dump(out, open(os.path.join(HERE, "fits.json"), "w"), indent=1)
    print(f"  {'op':<18}{'bk':<5}{'prec':<6}{'phase':<9}{'overhead_us':>12}{'GMAC/s':>10}{'R2':>7}")
    for f in out:
        if f["phase"] != "warm_us":
            continue
        print(f"  {f['op']:<18}{f['backend']:<5}{f['precision']:<6}{f['phase']:<9}"
              f"{f['overhead_us']:>12.1f}{(f['gmac_per_s'] or 0):>10.1f}{f['r2']:>7.3f}")
    print(f"  -> {os.path.join(HERE,'fits.json')}")


def main():
    sys.path.insert(0, HERE)
    from grid import points
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append",
                    choices=["gen", "build", "measure", "fit"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ops", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=12)
    a = ap.parse_args()
    pts = points()
    if a.ops:
        keep = set(a.ops.split(","))
        pts = [p for p in pts if p[0] in keep]
    if a.limit:
        pts = pts[:a.limit]
    stages = a.only or (["gen", "build", "measure", "fit"] if a.all else None)
    if not stages:
        ap.error("pass --all or --only STAGE")
    for s in stages:
        if s == "gen":     stage_gen(pts)
        elif s == "build": stage_build(pts)
        elif s == "measure": stage_measure(pts, a.iters, a.chunk)
        elif s == "fit":   stage_fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
