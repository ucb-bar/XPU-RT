#!/usr/bin/env python3
"""What kernels each build carries, and whether the board is running the good ones.

Two silent failures motivate this. Generating kernels without CROSS in the environment makes the
curated RVV kernels fail their cross-compile verify, and the generator falls back to the scalar
reference -- with `check_kernel_coverage.py` still reporting OK, because every op does have a kernel.
For `fused_full` that fallback measures about 17x slower. Separately, a shard factor that does not
divide an op's output-channel count yields an unsharded build under a sharded name, also silently.

Neither is visible in a directory listing, so this reads the evidence: `kernel_picks.json` records a
provenance per op, a curated build's `kernels.c` is several times larger than the reference one, and
the count of `conv2d_s8_sharded(` in the generated `model.c` says whether a width took.

    scripts/kernel_inventory.py [--board] [--host k1]

`--board` additionally asks the K1 what is deployed under /root/ros_mb and greps each kernels.c for
vector and IME instructions, which is what decides whether the measured arms ran the good kernels.
"""
from __future__ import annotations
import argparse, collections, glob, json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def provenance(picks_path):
    try:
        d = json.load(open(picks_path))
    except Exception:
        return {}
    ks = d if isinstance(d, list) else list(d.values())
    c = collections.Counter()
    for k in ks:
        s = json.dumps(k).lower()
        c["curated" if "curated" in s else ("reference" if "reference" in s else "other")] += 1
    return dict(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", action="store_true", help="also inspect what is deployed on the K1")
    ap.add_argument("--host", default=os.environ.get("MODELBLASTER_K1_HOST", "k1"))
    a = ap.parse_args()

    print(f"{'build tree':<22}{'network':<26}{'provenance':<30}{'kernels.c':>10}{'sharded':>9}")
    flagged = []
    seen = set()
    for picks in sorted(glob.glob(os.path.join(REPO, "ModelBlaster/build/*/*/int8/generated/kernel_picks.json"))):
        parts = os.path.relpath(picks, REPO).split("/")
        tree, net = parts[2], parts[3]
        if (tree, net) in seen:
            continue
        seen.add((tree, net))
        gen = os.path.dirname(picks)
        prov = provenance(picks)
        kc = os.path.join(gen, "kernels.c")
        size = os.path.getsize(kc) // 1024 if os.path.exists(kc) else None
        mc = os.path.join(gen, "model.c")
        body = open(mc).read() if os.path.exists(mc) else ""
        nsh = body.count("conv2d_s8_sharded(")
        nconv = body.count("conv2d_s8")        # sharded and plain alike; 0 means the net has no convolution
        note = ""
        if prov.get("reference"):
            note = "   <-- scalar reference"
            flagged.append(f"{tree}/{net}: kernels are the scalar reference, not the curated RVV set")
        # only meaningful for a network that HAS convolutions: the split is over output channels, so a
        # net without one (mlp_control is linear + elu) is correctly unsharded whatever the tree is named
        if "shard" in tree and tree[-1].isdigit() and int(tree[-1]) > 1 and nsh == 0 and nconv:
            note += "   <-- named sharded but emits no sharded conv"
            flagged.append(f"{tree}/{net}: shard factor did not take ({nconv} convs, none sharded)")
        print(f"{tree:<22}{net:<26}{str(prov):<30}{(str(size)+'K') if size else '-':>10}{nsh:>9}{note}")

    if a.board:
        print(f"\ndeployed on {a.host}:/root/ros_mb")
        cmd = ('cd /root/ros_mb && for d in */; do d=${d%/}; [ -f $d/kernels.c ] || continue; '
               'printf "%s %s %s %s\\n" "$d" "$(wc -c < $d/kernels.c)" '
               '"$(grep -c \'vsetvl\\|__riscv_v\\|vle8\\|vfmacc\' $d/kernels.c)" '
               '"$(grep -c \'vmadot\\|0x71\' $d/kernels.c)"; done')
        try:
            out = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", a.host, cmd],
                                 capture_output=True, text=True, timeout=60).stdout
        except Exception as e:
            print(f"  board unreachable: {e}"); out = ""
        print(f"  {'dir':<18}{'kernels.c':>11}{'rvv lines':>11}{'ime lines':>11}")
        for ln in out.split("\n"):
            p = ln.split()
            if len(p) != 4:
                continue
            d, sz, rvv, ime = p[0], int(p[1]) // 1024, int(p[2]), int(p[3])
            note = "   <-- no vector code" if rvv == 0 else ""
            if rvv == 0:
                flagged.append(f"board {d}: kernels.c contains no vector intrinsics")
            print(f"  {d:<18}{str(sz)+'K':>11}{rvv:>11}{ime:>11}{note}")

    print()
    if flagged:
        print("flagged:")
        for f in flagged:
            print(f"  - {f}")
    else:
        print("nothing flagged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
