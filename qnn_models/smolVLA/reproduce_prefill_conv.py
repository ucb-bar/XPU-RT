#!/usr/bin/env python3
"""Reproduce the expert-prefill Conv1x1 result from scratch.

Rebuilds every artifact behind PREFILL_CONV1X1.md and re-measures on the board,
so the headline number can be checked rather than taken on trust:

    prefill trunk, CPU int8   384.5 ms  (MatMul form)
                          ->  297.6 ms  (Conv1x1 + RmsNorm scale barrier)   1.29x

and the native-conv extraction that follows it:

    nc_mlp   cpu 4414.9   hta 2452.2 us   -> HTA wins by 1.80x
    nc_qkv   cpu 1193.1   hta 2455.3 us
    nc_oproj cpu  925.1   dsp  932.6 us

Stages (any subset via --only):

  calib     8 calibration samples, float32 raws in the converter's declared
            layout, from real vision statistics
  rewrite   conv1x1 + the RmsNorm scale barrier, with numeric checks
  build     convert + quantize both trunks and the three native-conv blocks
  measure   push to the board and profile every (tile, backend) that composes

Three things this encodes that cost a day to find:

  * the trunk DLC declares only TWO inputs -- the rotary fold left position_ids
    dead -- so the calibration list must be built from the DLC's own APP_WRITE
    tensors, never from the ONNX graph inputs. A 3-input list is rejected with
    "Graph contains 2 inputs, but only found input data for 3 inputs".
  * the conv rewrite re-triggers the RmsNorm fusion the trunk was built to
    avoid, and `--ir_optimizer_config` skipping RemoveNoOps/SquashConstantInput
    does NOT stop it. Only a barrier that cannot legally be deleted does:
    scale the variance by 4 and undo with 2 after the reciprocal, both exact
    powers of two.
  * the native-conv blocks must be converted WITHOUT --preserve_io layout, so
    the converter declares NHWC and emits no layout ops. With the layout dance
    the v66 DSP refuses to finalize at any graph size, down to one layer.

    python3 reproduce_prefill_conv.py --all
    python3 reproduce_prefill_conv.py --only measure --board root@10.44.120.201
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
QNN_SDK = os.environ.get("QNN_SDK", "/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326")
DOCKER_IMG = os.environ.get("QNN_DOCKER_IMG", "qnn-convert")
BOARD = os.environ.get("QNN_BOARD", "root@10.44.120.201")
LOCK = "/tmp/qnn_board.lock"
BOARD_DIR = "/data/repro_prefill"
CTX_DIR = "/root/models/smolvla_expert_nc"
PROFILE_SEG = "/root/models/smolvlm_vision_v3/profile_seg"

TRUNK = "smolvlm_expert_prefill_trunk.onnx"
CONV = "smolvlm_expert_prefill_trunk_conv.onnx"
CONVBAR = "smolvlm_expert_prefill_trunk_convbar.onnx"
NC = ("nc_qkv", "nc_oproj", "nc_mlp")


def sh(cmd, cwd=HERE, timeout=9000, check=True):
    print(f"    $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}"[:150])
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, timeout=timeout,
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise SystemExit(f"failed ({r.returncode}): {cmd}")
    return r.stdout + r.stderr


def docker(script, workdir, timeout=9000):
    cmd = (f"sudo docker run --rm -v {shlex.quote(QNN_SDK)}:/qnn:ro "
           f"-v {shlex.quote(workdir)}:/workspace {DOCKER_IMG} bash -c {shlex.quote(script)}")
    return sh(cmd, timeout=timeout)


def board(script, timeout=2400):
    """Every board interaction is serialised behind the shared flock."""
    inner = shlex.quote(script)
    cmd = (f"timeout {timeout} flock -w {timeout} {LOCK} -c "
           f"{shlex.quote(f'timeout -s KILL {timeout - 60} ssh -o ConnectTimeout=10 {BOARD} bash -s')}")
    print(f"    [board] {script.splitlines()[0][:110]}")
    r = subprocess.run(cmd, shell=True, input=script, capture_output=True,
                       text=True, timeout=timeout + 120)
    return r.stdout + r.stderr


def dlc_inputs(workdir, dlc):
    """Input list must come from the DLC, not the ONNX -- position_ids is dead."""
    out = docker(f"python3.10 /qnn/bin/x86_64-linux-clang/snpe-dlc-info "
                 f"-i /workspace/{dlc} 2>/dev/null", workdir, timeout=2400)
    names = re.findall(r"([A-Za-z_0-9.]+) \(data type: [A-Za-z_0-9]+; "
                       r"tensor dimension: \[[0-9,]+\]; tensor type: APP_WRITE\)", out)
    return sorted(set(names))


def stage_calib(work):
    print("[calib] 8 samples, float32 raws, converter-declared layout")
    sh([sys.executable, os.path.join(HERE, "gen_expert_calibration.py"),
        "--model", TRUNK, "--out", "expert_rewrite/prefill_calib", "--n", "8"],
       check=False)
    print("      NOTE: raws must be float32 for EVERY input regardless of the DLC's")
    print("      declared dtype; a uint8 mask is 1/4 the extent and is misreported")
    print("      as a batch-size mismatch.")


def stage_rewrite(work):
    print("[rewrite] conv1x1, then the non-removable RmsNorm scale barrier")
    for f in (CONV, CONV + ".data", CONVBAR, CONVBAR + ".data"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            os.remove(p)
    print(sh([sys.executable, "rewrite_matmul_to_conv1x1.py", TRUNK, "-o", CONV]).strip()[-400:])
    print(sh([sys.executable, "rewrite_rmsnorm_scale_barrier.py",
              "--in", CONV, "--out", CONVBAR]).strip()[-400:])
    print("      expect: 112 MatMul -> Conv1x1, 32 chains given a scale barrier")


def stage_build(work):
    print("[build] convert + quantize both trunks and the three native-conv blocks")
    os.makedirs(work, exist_ok=True)
    for f in (TRUNK, TRUNK + ".data", CONVBAR, CONVBAR + ".data"):
        s = os.path.join(HERE, f)
        if os.path.exists(s):
            sh(f"cp -n {shlex.quote(s)} {shlex.quote(work)}/", check=False)
    sh(f"cp -r {shlex.quote(os.path.join(HERE,'expert_rewrite/prefill_calib'))} {shlex.quote(work)}/", check=False)
    for name, src in (("pf_trunk", TRUNK), ("pf_convbar", CONVBAR)):
        docker(f'python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc '
               f'--input_network /workspace/{src} --output_path /workspace/{name}.dlc 2>&1 | tail -1', work)
        ins = dlc_inputs(work, f"{name}.dlc")
        print(f"      {name}.dlc declares {len(ins)} inputs: {ins}")
        lst = os.path.join(work, "prefill_calib", f"list_{name}.txt")
        with open(lst, "w") as fh:
            for k in range(8):
                fh.write(" ".join(f"{n}:=/workspace/prefill_calib/{n}_{k:03d}.raw" for n in ins) + "\n")
        docker(f'pip install -q "numpy<2" >/dev/null 2>&1; '
               f'python3.10 /qnn/bin/x86_64-linux-clang/qairt-quantizer '
               f'--input_dlc /workspace/{name}.dlc --output_dlc /workspace/{name}_q.dlc '
               f'--input_list /workspace/prefill_calib/list_{name}.txt '
               f'--act_bitwidth 8 --weights_bitwidth 8 2>&1 | grep -iE "error|success" | head -3', work)
    # native-conv blocks: NO --preserve_io layout, so the input is declared NHWC
    sh(f"cp {shlex.quote(os.path.join(HERE,'expert_rewrite/nativeconv'))}/nc_*.onnx {shlex.quote(work)}/", check=False)
    sh(f"cp {shlex.quote(os.path.join(HERE,'expert_rewrite/nativeconv'))}/in_*.raw {shlex.quote(work)}/", check=False)
    for b in NC:
        tag = b.split("_", 1)[1]
        with open(os.path.join(work, f"list_{b}.txt"), "w") as fh:
            for k in range(8):
                fh.write(f"X:=/workspace/in_{tag}_{k:03d}.raw\n")
        docker(f'pip install -q "numpy<2" >/dev/null 2>&1; '
               f'python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc '
               f'--input_network /workspace/{b}.onnx --output_path /workspace/{b}.dlc 2>&1 | tail -1; '
               f'python3.10 /qnn/bin/x86_64-linux-clang/qairt-quantizer '
               f'--input_dlc /workspace/{b}.dlc --output_dlc /workspace/{b}_q.dlc '
               f'--input_list /workspace/list_{b}.txt --act_bitwidth 8 --weights_bitwidth 8 2>&1 '
               f'| grep -iE "error|success" | head -2', work)
    sh(f"sudo chown -R {os.getuid()}:{os.getgid()} {shlex.quote(work)}", check=False)


def stage_measure(work):
    print("[measure] compose + profile every (tile, backend) that takes the graph")
    dlcs = [("pf_trunk_q", ("Cpu", "Dsp")), ("pf_convbar_q", ("Cpu", "Dsp"))]
    dlcs += [(f"{b}_q", ("Cpu", "Dsp", "Hta")) for b in NC]
    have = [d for d, _ in dlcs if os.path.exists(os.path.join(work, d + ".dlc"))]
    if not have:
        raise SystemExit("no quantized DLCs in the work dir -- run --only build first")
    sh(f"timeout 300 flock -w 900 {LOCK} -c "
       f"{shlex.quote(f'timeout -s KILL 200 ssh -n {BOARD} mkdir -p {BOARD_DIR}')}", check=False)
    for d in have:
        sh(f"timeout 900 flock -w 1800 {LOCK} -c "
           f"{shlex.quote(f'timeout -s KILL 800 scp {os.path.join(work, d)}.dlc {BOARD}:{BOARD_DIR}/')}",
           check=False)
    script = f"""set -u
cd {BOARD_DIR}
export LD_LIBRARY_PATH=/root/qairt/lib/target:${{LD_LIBRARY_PATH:-}}
export ADSP_LIBRARY_PATH=/root/qairt/lib/target
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $c; done
for V in {' '.join(have)}; do
  for BK in Cpu Dsp Hta; do
    L=/root/qairt/lib/target/libQnn${{BK}}.so
    D=r_${{V}}_${{BK}}; rm -rf $D; mkdir -p $D
    timeout -s KILL 900 /root/qairt/bin/target/qnn-context-binary-generator \\
      --model /root/qairt/lib/target/libQnnModelDlc.so --backend $L \\
      --dlc_path {BOARD_DIR}/$V.dlc --binary_file b --output_dir {BOARD_DIR}/$D > $D.log 2>&1
    if [ ! -f $D/b.bin ]; then
      echo "RESULT $V $BK FAILED $(grep -E '\\[ *ERROR *\\]' $D.log | head -1 | sed 's/.*ERROR *. //' | cut -c1-60)"
      rm -rf $D; continue
    fi
    R=""
    for i in 1 2 3; do
      R="$R $(timeout -s KILL 600 {PROFILE_SEG} $D/b.bin $L 20 --gap-us 3000 2>&1 \\
              | grep -E '^{{' | sed -E 's/.*"gap_median_us":([0-9.]+).*/\\1/')"
    done
    echo "RESULT $V $BK OK $R"
    rm -rf $D
  done
done
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo $GOV > $c; done
rm -rf {BOARD_DIR}
"""
    out = board(script, timeout=3000)
    rows = [l.split() for l in out.splitlines() if l.startswith("RESULT ")]
    print(f"\n  {'tile':<16} {'backend':<8} {'gap medians (us)'}")
    res = {}
    for r in rows:
        if r[3] == "OK":
            vals = [float(x) for x in r[4:]]
            vals.sort()
            res[(r[1], r[2])] = vals[len(vals) // 2]
            print(f"  {r[1]:<16} {r[2]:<8} {' '.join(f'{v:.1f}' for v in vals)}   median {vals[len(vals)//2]:.1f}")
        else:
            print(f"  {r[1]:<16} {r[2]:<8} FAILED {' '.join(r[4:])}")
    print("\n  expected (PREFILL_CONV1X1.md):")
    print("    pf_trunk_q   Cpu ~384500   Dsp ~1411000")
    print("    pf_convbar_q Cpu ~297600   Dsp FAILED (rpc/6022, conv layout dance)")
    print("    nc_mlp_q     Cpu ~4415  Hta ~2452  Dsp ~3820")
    print("    nc_qkv_q     Cpu ~1193  Hta ~2455  Dsp ~3076")
    print("    nc_oproj_q   Cpu ~925   Hta ~1773  Dsp ~933")
    json.dump({f"{k[0]}/{k[1]}": v for k, v in res.items()},
              open(os.path.join(work, "reproduced.json"), "w"), indent=1)
    print(f"\n  -> {os.path.join(work, 'reproduced.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "repro_prefill"))
    ap.add_argument("--only", action="append",
                    choices=["calib", "rewrite", "build", "measure"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--board", default=None)
    a = ap.parse_args()
    global BOARD
    if a.board:
        BOARD = a.board
    stages = a.only or (["calib", "rewrite", "build", "measure"] if a.all else None)
    if not stages:
        ap.error("pass --all or one or more --only STAGE")
    fn = {"calib": stage_calib, "rewrite": stage_rewrite,
          "build": stage_build, "measure": stage_measure}
    for s in stages:
        fn[s](a.work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
