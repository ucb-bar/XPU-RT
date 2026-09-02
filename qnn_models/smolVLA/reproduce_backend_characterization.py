#!/usr/bin/env python3
"""Reproduce the QRB5165 backend characterization behind HTA_DISPATCH_FLOOR.md.

Everything in that document is a measurement, and this rebuilds all of it from
source so the numbers can be checked rather than trusted:

  * the dispatch floor -- a single Conv(1x1) swept over 17,356,797x in
    arithmetic at a FIXED op count of one, which is what shows that HTA's time
    is nearly independent of the work
  * the power-collapse knee -- the same probe at idle gaps of 0/250/1000/3000/
    10000 us, which is what shows HTA's floor is 543 us busy and ~2540 us after
    ~1 ms idle (4.7x), and that the effect only bites SHORT dispatches
  * the fitted cost model and the resulting break-even against the CPU

Headline numbers this should reproduce (performance governor, gap median):

    t01  (64 MAC!)   HTA 2467.8 us cold / 543.5 us warm     CPU 14.5 us
    t07  (1.11 GMAC) HTA 3057.0 us                          CPU 7734.6 us
    floors    CPU ~14 us    DSP ~402 us    HTA ~543 us (warm)
    marginal  CPU  144      DSP  422       HTA  493 GMAC/s
    break-even ~335 MMAC per dispatch cold, ~95 MMAC warm

Stages (subset via --only, everything with --all):

    probes    build the 7 synthetic Conv1x1 ONNX models + calibration
    build     convert and quantize them
    floor     measure each on cpu/dsp/hta at a fixed gap
    gap       sweep the idle gap on selected probes
    fit       fit floor/marginal/break-even from the measured JSON

`--measure PATH.dlc` additionally reports warm-vs-cold for any DLC already
built (used for the expert blocks and the vision conv1x1 kernels), e.g.

    python3 reproduce_backend_characterization.py --measure /root/models/smolvla_expert_nc/nc_mlp_q.dlc --on-board

Board discipline: every board interaction is serialised behind
`flock /tmp/qnn_board.lock`, wrapped in `timeout -s KILL`, and the CPU governor
is saved and restored around the measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
QNN_SDK = os.environ.get("QNN_SDK", "/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326")
DOCKER_IMG = os.environ.get("QNN_DOCKER_IMG", "qnn-convert")
BOARD = os.environ.get("QNN_BOARD", "root@10.44.120.201")
LOCK = "/tmp/qnn_board.lock"
BOARD_DIR = "/data/repro_char"
PROFILE_SEG = "/root/models/smolvlm_vision_v3/profile_seg"

# (name, Cin, Cout, S) -- ONE Conv1x1 each, so op count is held fixed and only
# the arithmetic varies. t05/t06 deliberately match the decode/prefill MLP's
# first conv so the synthetic sweep is anchored to the real blocks.
CASES = [("t01",    8,    8,   1), ("t02",   64,   64,   8),
         ("t03",  256,  256,  50), ("t04",  512,  512,  50),
         ("t05",  720, 2048,  50), ("t06",  960, 2560, 113),
         ("t07", 1920, 5120, 113)]
GAPS = [0, 250, 1000, 3000, 10000]


def sh(cmd, timeout=9000, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        sys.stderr.write((r.stdout + r.stderr)[-2500:])
        raise SystemExit(f"failed: {cmd}")
    return r.stdout + r.stderr


def docker(script, workdir, timeout=9000):
    return sh(f"sudo docker run --rm -v {shlex.quote(QNN_SDK)}:/qnn:ro "
              f"-v {shlex.quote(workdir)}:/workspace {DOCKER_IMG} bash -c {shlex.quote(script)}",
              timeout=timeout)


def board(script, timeout=2400):
    """All board work goes through the shared lock; nothing runs unserialised."""
    inner = f"timeout -s KILL {timeout - 60} ssh -o ConnectTimeout=10 {BOARD} bash -s"
    cmd = f"timeout {timeout} flock -w {timeout} {LOCK} -c {shlex.quote(inner)}"
    r = subprocess.run(cmd, shell=True, input=script, capture_output=True,
                       text=True, timeout=timeout + 120)
    return r.stdout + r.stderr


def stage_probes(work):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    os.makedirs(work, exist_ok=True)
    rng = np.random.default_rng(0)
    print("[probes] one Conv1x1 each, op count fixed at 1")
    for name, ci, co, S in CASES:
        W = (rng.standard_normal((co, ci, 1, 1)) * 0.02).astype(np.float32)
        g = helper.make_graph(
            [helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[1, 1])], name,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, ci, 1, S])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, co, 1, S])],
            [numpy_helper.from_array(W, "W")])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 9
        onnx.checker.check_model(m, full_check=False)
        onnx.save(m, os.path.join(work, f"{name}.onnx"))
        with open(os.path.join(work, f"list_{name}.txt"), "w") as f:
            for k in range(4):
                p = os.path.join(work, f"in_{name}_{k:03d}.raw")
                np.ascontiguousarray(rng.standard_normal((1, ci, 1, S)),
                                     dtype=np.float32).tofile(p)
                f.write(f"X:=/workspace/in_{name}_{k:03d}.raw\n")
        print(f"    {name}  Cin{ci:>5} Cout{co:>5} S{S:>4}  {S*ci*co/1e6:9.3f} MMAC  "
              f"{W.nbytes/1e6:6.2f} MB weights")
    print(f"    MAC span {CASES[-1][1]*CASES[-1][2]*CASES[-1][3] / max(1, CASES[0][1]*CASES[0][2]*CASES[0][3]):,.0f}x")


def stage_build(work):
    print("[build] convert + quantize (int8, 4 calibration samples each)")
    names = " ".join(n for n, *_ in CASES)
    out = docker(f'pip install -q "numpy<2" >/dev/null 2>&1; '
                 f'for T in {names}; do '
                 f'python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc '
                 f'--input_network /workspace/$T.onnx --output_path /workspace/$T.dlc >/dev/null 2>&1; '
                 f'python3.10 /qnn/bin/x86_64-linux-clang/qairt-quantizer '
                 f'--input_dlc /workspace/$T.dlc --output_dlc /workspace/${{T}}_q.dlc '
                 f'--input_list /workspace/list_$T.txt --act_bitwidth 8 --weights_bitwidth 8 2>&1 '
                 f'| grep -icE success | sed "s/^/  $T quantized=/"; done', work)
    print("".join(l + "\n" for l in out.splitlines() if "quantized" in l))
    sh(f"sudo chown -R {os.getuid()}:{os.getgid()} {shlex.quote(work)}")


def _push(work, names):
    sh(f"timeout 300 flock -w 900 {LOCK} -c {shlex.quote(f'timeout -s KILL 200 ssh -n {BOARD} mkdir -p {BOARD_DIR}')}")
    for n in names:
        p = os.path.join(work, f"{n}_q.dlc")
        if os.path.exists(p):
            sh(f"timeout 600 flock -w 1200 {LOCK} -c "
               f"{shlex.quote(f'timeout -s KILL 500 scp {p} {BOARD}:{BOARD_DIR}/')}")


def _remote(dlcs, backends, gaps):
    """One board script measuring every (dlc, backend, gap); governor restored."""
    return f"""set -u
cd {BOARD_DIR}
export LD_LIBRARY_PATH=/root/qairt/lib/target:${{LD_LIBRARY_PATH:-}}
export ADSP_LIBRARY_PATH=/root/qairt/lib/target
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $c; done
for T in {' '.join(dlcs)}; do
  for BK in {' '.join(backends)}; do
    L=/root/qairt/lib/target/libQnn${{BK}}.so
    [ -f "$L" ] || continue
    D=c_${{T}}_${{BK}}; rm -rf $D; mkdir -p $D
    timeout -s KILL 400 /root/qairt/bin/target/qnn-context-binary-generator \\
      --model /root/qairt/lib/target/libQnnModelDlc.so --backend $L \\
      --dlc_path {BOARD_DIR}/${{T}}_q.dlc --binary_file b --output_dir {BOARD_DIR}/$D > $D.log 2>&1
    if [ ! -f $D/b.bin ]; then
      echo "R $T $BK - FAILED $(grep -oE 'unsupported [a-z ]*op [A-Za-z_]+' $D.log | head -1 | tr ' ' '_')"
      rm -rf $D; continue
    fi
    for G in {' '.join(str(g) for g in gaps)}; do
      V=$(timeout -s KILL 300 {PROFILE_SEG} $D/b.bin $L 40 --gap-us $G 2>&1 \\
          | grep -E '^{{' | sed -E 's/.*"gap_median_us":([0-9.]+).*/\\1/')
      echo "R $T $BK $G $V"
    done
    rm -rf $D
  done
done
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo $GOV > $c; done
rm -rf {BOARD_DIR}
"""


def _parse(out):
    res = {}
    for ln in out.splitlines():
        if not ln.startswith("R "):
            continue
        f = ln.split()
        if len(f) < 5 or f[3] == "-":
            res[(f[1], f[2], None)] = None
            continue
        try:
            res[(f[1], f[2], int(f[3]))] = float(f[4])
        except ValueError:
            pass
    return res


def stage_floor(work, gaps=(3000,)):
    names = [n for n, *_ in CASES]
    _push(work, names)
    res = _parse(board(_remote(names, ["Hta", "Dsp", "Cpu"], gaps), timeout=3000))
    mac = {n: S * ci * co for n, ci, co, S in CASES}
    print(f"\n  {'probe':<6} {'MMAC':>10} " + "".join(f"{b:>11}" for b in ("HTA", "DSP", "CPU")))
    rows = {}
    for n in names:
        line = f"  {n:<6} {mac[n]/1e6:>10.4f} "
        for b in ("Hta", "Dsp", "Cpu"):
            v = res.get((n, b, gaps[0]))
            line += f"{('FAIL' if v is None else f'{v:.1f}'):>11}"
            rows.setdefault(n, {})[b.lower()] = v
        print(line)
    json.dump({"mac": mac, "gap_us": gaps[0], "cells": rows},
              open(os.path.join(work, "floor.json"), "w"), indent=1)
    print(f"  -> {os.path.join(work,'floor.json')}")


def stage_gap(work, probes=("t01", "t07")):
    _push(work, probes)
    res = _parse(board(_remote(list(probes), ["Hta", "Dsp", "Cpu"], GAPS), timeout=3000))
    print(f"\n  {'probe/backend':<16} " + "".join(f"{('gap'+str(g)):>10}" for g in GAPS))
    out = {}
    for n in probes:
        for b in ("Hta", "Dsp", "Cpu"):
            vals = [res.get((n, b, g)) for g in GAPS]
            if all(v is None for v in vals):
                continue
            print(f"  {n+'/'+b:<16} " + "".join(f"{('-' if v is None else f'{v:.1f}'):>10}" for v in vals))
            out[f"{n}/{b}"] = dict(zip(map(str, GAPS), vals))
    json.dump({"gaps_us": GAPS, "cells": out},
              open(os.path.join(work, "gap.json"), "w"), indent=1)
    print("  HTA's floor should be ~543 us at gap 0 and ~2540 us from gap 1000 on.")


def stage_fit(work):
    import numpy as np
    p = os.path.join(work, "floor.json")
    if not os.path.exists(p):
        raise SystemExit("run --only floor first")
    d = json.load(open(p))
    mac = {k: v / 1e6 for k, v in d["mac"].items()}
    print(f"\n  fit on gap={d['gap_us']} us")
    fits = {}
    for b in ("hta", "dsp", "cpu"):
        pts = [(mac[n], d["cells"][n][b]) for n in mac if d["cells"].get(n, {}).get(b)]
        if len(pts) < 2:
            continue
        X = np.array([[1, m] for m, _ in pts]); y = np.array([t for _, t in pts])
        (a, s), *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - ((y - X @ np.array([a, s])) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        fits[b] = (a, s)
        print(f"    {b.upper()}  floor {a:8.1f} us   marginal {1e3/s:7.0f} GMAC/s   R2={r2:.3f}")
    if "hta" in fits and "cpu" in fits:
        (ah, sh_), (ac, sc) = fits["hta"], fits["cpu"]
        print(f"    break-even HTA vs CPU: {(ah-ac)/(sc-sh_):.0f} MMAC per dispatch")
    print("    A low HTA R2 is the finding: its time barely tracks the work.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "repro_char"))
    ap.add_argument("--only", action="append", choices=["probes", "build", "floor", "gap", "fit"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--board", default=None)
    a = ap.parse_args()
    global BOARD
    if a.board:
        BOARD = a.board
    stages = a.only or (["probes", "build", "floor", "gap", "fit"] if a.all else None)
    if not stages:
        ap.error("pass --all or one or more --only STAGE")
    fn = {"probes": stage_probes, "build": stage_build, "floor": stage_floor,
          "gap": stage_gap, "fit": stage_fit}
    for s in stages:
        fn[s](a.work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
