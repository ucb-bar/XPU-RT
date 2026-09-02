#!/usr/bin/env python3
"""Quantize the 12 lone-Tanh vision trampolines to int8.

The odd `vision_slices_v3/cpu_seg_{01,03,...,23}.onnx` are a single `Tanh` at
`[1,1024,3072]`, carved out of each SigLIP GELU. They ship as **fp32** and cost
9.171 ms each -- 110.0 ms over the 12. TANH_PROBE.md measured the same block at
**2.408 ms in int8**, a 4.4x recovery worth 81.2 ms, because most of the cost is
fp32 element traffic rather than tanh math.

The whole difficulty is calibration. `profile_inputs/cpu_seg_NN_*.raw` is
unrepresentative -- range +-0.505, std 0.100 -- while the real distribution
entering the Tanh has range ~+-2.7 and std ~0.500. Quantizing against the
former wastes most of the int8 range on values that never occur.

Rather than substitute a nearby tensor, this derives the exact input. Each
`dsp_seg_NN` ends in the GELU pre-tanh chain

    val_364 = fc1(x)                        <- real calibration exists for this
    val_366 = val_364 ^ 3                       (trampolines/calibration/
    val_368 = 0.044715 * val_366                 dsp_seg_NN_tramp_p2/)
    val_369 = val_364 + val_368
    val_371 = sqrt(2/pi) * val_369          <- the Tanh input we need

so the sub-model from the fc1 output to the Tanh input is extracted and run on
the real fc1 activations. That is exact, and it does not depend on reading the
GELU constants correctly by hand.

    python3 quantize_tanh_trampolines.py --calib      # derive Tanh inputs
    python3 quantize_tanh_trampolines.py --build      # convert + quantize int8
"""
from __future__ import annotations

import argparse
import glob
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SLICES = os.path.join(HERE, "vision_slices_v3")
TRAMP_CAL = os.path.join(SLICES, "trampolines", "calibration")
OUT = os.path.join(SLICES, "tanh_int8")
QNN_SDK = os.environ.get("QNN_SDK", "/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326")
DOCKER_IMG = os.environ.get("QNN_DOCKER_IMG", "qnn-convert")


def odd_segs():
    out = []
    for p in sorted(glob.glob(os.path.join(SLICES, "cpu_seg_*.onnx"))):
        n = int(os.path.basename(p).split("_")[2].split(".")[0])
        if n % 2 == 1:
            out.append((n, p))
    return out


def stage_calib(nsamp):
    import numpy as np
    import onnx
    from onnx.utils import extract_model
    import onnxruntime as ort
    os.makedirs(OUT, exist_ok=True)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    made = 0
    for n, path in odd_segs():
        m = onnx.load(path, load_external_data=False)
        tanh_in = m.graph.input[0].name
        seg = os.path.join(SLICES, f"dsp_seg_{n:02d}.onnx")
        if not os.path.exists(seg):
            print(f"  cpu_seg_{n:02d}: no producing dsp_seg_{n:02d}, skipped")
            continue
        d = onnx.load(seg, load_external_data=False)
        outs = [o.name for o in d.graph.output]
        if tanh_in not in outs:
            print(f"  cpu_seg_{n:02d}: {tanh_in} not an output of dsp_seg_{n:02d}, skipped")
            continue
        # the fc1 output is the other 3072-wide output of the same segment
        wide = [o.name for o in d.graph.output
                if [x.dim_value for x in o.type.tensor_type.shape.dim][-1:] == [3072]]
        fc1 = next((w for w in wide if w != tanh_in), None)
        if fc1 is None:
            print(f"  cpu_seg_{n:02d}: could not identify the fc1 output, skipped")
            continue
        sub = os.path.join(OUT, f"gelu_pre_{n:02d}.onnx")
        extract_model(seg, sub, [fc1], [tanh_in])
        raws = sorted(glob.glob(os.path.join(TRAMP_CAL, f"dsp_seg_{n:02d}_tramp_p2", "*.raw")))
        if not raws:
            print(f"  cpu_seg_{n:02d}: no tramp_p2 calibration, skipped")
            continue
        s = ort.InferenceSession(sub, so, providers=["CPUExecutionProvider"])
        shp = [x.dim_value or 1 for x in
               onnx.load(sub, load_external_data=False).graph.input[0].type.tensor_type.shape.dim]
        lines = []
        for k, r in enumerate(raws[:nsamp]):
            x = np.fromfile(r, np.float32).reshape(shp)
            y = s.run(None, {fc1: x})[0]
            p = os.path.join(OUT, f"in_cpu_seg_{n:02d}_{k:03d}.raw")
            np.ascontiguousarray(y, dtype=np.float32).tofile(p)
            lines.append(f"{tanh_in}:=/workspace/tanh_int8/in_cpu_seg_{n:02d}_{k:03d}.raw")
            if k == 0:
                print(f"  cpu_seg_{n:02d}  tanh_in={tanh_in:<10} "
                      f"derived range [{y.min():+.3f},{y.max():+.3f}] std {y.std():.3f}  "
                      f"(fc1 [{x.min():+.2f},{x.max():+.2f}])")
        with open(os.path.join(OUT, f"list_cpu_seg_{n:02d}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        del s
        made += 1
    print(f"  derived calibration for {made} trampolines -> {OUT}")
    print("  compare: profile_inputs raws are range +-0.505 std 0.100 -- do not use them")


def stage_build():
    segs = [n for n, _ in odd_segs()]
    for n, p in odd_segs():
        dst = os.path.join(OUT, f"cpu_seg_{n:02d}.onnx")
        if not os.path.exists(dst):
            subprocess.run(["cp", p, dst], check=True)
    names = " ".join(f"cpu_seg_{n:02d}" for n in segs)
    script = (f'pip install -q "numpy<2" >/dev/null 2>&1; '
              f'for T in {names}; do '
              f'python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc '
              f'--input_network /workspace/tanh_int8/$T.onnx '
              f'--output_path /workspace/tanh_int8/$T.dlc >/dev/null 2>&1; '
              f'python3.10 /qnn/bin/x86_64-linux-clang/qairt-quantizer '
              f'--input_dlc /workspace/tanh_int8/$T.dlc '
              f'--output_dlc /workspace/tanh_int8/${{T}}_q.dlc '
              f'--input_list /workspace/tanh_int8/list_$T.txt '
              f'--act_bitwidth 8 --weights_bitwidth 8 2>&1 '
              f'| grep -icE success | sed "s/^/  $T int8=/"; done')
    r = subprocess.run(f"sudo docker run --rm -v {shlex.quote(QNN_SDK)}:/qnn:ro "
                       f"-v {shlex.quote(SLICES)}:/workspace/tanh_int8_parent "
                       f"-v {shlex.quote(OUT)}:/workspace/tanh_int8 {DOCKER_IMG} bash -c {shlex.quote(script)}",
                       shell=True, capture_output=True, text=True, timeout=7000)
    print("".join(l + "\n" for l in (r.stdout + r.stderr).splitlines() if "int8=" in l))
    subprocess.run(f"sudo chown -R {os.getuid()}:{os.getgid()} {shlex.quote(OUT)}", shell=True)
    got = len(glob.glob(os.path.join(OUT, "*_q.dlc")))
    print(f"  {got}/{len(segs)} trampolines quantized to int8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    if not (a.calib or a.build):
        ap.error("pass --calib and/or --build")
    if a.calib:
        stage_calib(a.n)
    if a.build:
        stage_build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
