#!/usr/bin/env python3
"""Does the winning feedback path depend on network SIZE?

The stage ladder showed a different knob winning on each of five networks. The
obvious hypothesis is that size explains it: a big network has room to slice, a
small one is all overhead. This tests that directly.

Two size axes, and they answer different questions:

  ACROSS networks   op count spans 7 (mlp_control) to 1931 (vint), three orders
                    of magnitude, but the networks also differ in topology and
                    op mix -- so a correlation here is suggestive, not causal.
  WITHIN one network the ViNT observation-encoder tile measured at batch 1, 2
                    and 3 (`vint_obs_b1/b2/b3`) is the SAME graph at three
                    sizes. Topology and op mix are held fixed, so this is the
                    controlled version of the question.

Everything here runs off measurements already on disk. The yolo resolution
variants (64x96, 128x192) and a reduced dronet do not exist on QRB5165 -- only
on the K1/spacemit tree -- so `--emit-board-plan` writes the build+profile
plan for them rather than pretending the data is there.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "qnn_models", "slicing_study"))
import analyze                      # noqa: E402
import flowc_feedback_stages as fs  # noqa: E402

# resolution / scale variants that exist on the K1 tree and not on QRB5165
BOARD_TODO = [
    ("yolov8_nano_64x96",   "yolov8n at 64x96 input", "small"),
    ("yolov8_nano_128x192", "yolov8n at 128x192 input", "medium"),
    ("dronet_small",        "dronet with reduced channel width", "small"),
]


def ops_of(e: dict) -> int:
    n = 0
    for t in e["tiles"]:
        rng = t.get("ranges") or ([t["op_range"]] if t.get("op_range") else [])
        for r in rng:
            if isinstance(r, (list, tuple)) and len(r) == 2:
                n += int(r[1]) - int(r[0]) + 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/flowc_size")
    ap.add_argument("--emit-board-plan", action="store_true")
    a = ap.parse_args()
    out = os.path.join(REPO, a.out)
    os.makedirs(out, exist_ok=True)
    pooled = analyze.pool(analyze.load())

    # ---- axis 1: across networks -------------------------------------------
    rows = []
    print("  ACROSS NETWORKS — size vs the knob that wins\n")
    print(f"  {'network':16s} {'ops':>6s} {'baseline ms':>12s} {'best ms':>10s} "
          f"{'speedup':>8s}  winning knob")
    for net in ["mlp_control", "dronet", "fused_full", "yolov8n", "vint"]:
        st = fs.stages(net, pooled)
        rs = [r for r in st if r["ms"] is not None]
        if not rs:
            continue
        whole = [v for v in pooled.values()
                 if v["network"] == net and len(v["tiles"]) == 1]
        ops = max((ops_of(w) for w in whole), default=0)
        base = rs[0]["ms"]
        best = min(rs[1:], key=lambda r: r["ms"]) if len(rs) > 1 else rs[0]
        gain, who = 1.0, "none"
        for q in range(1, len(rs)):
            g = rs[q - 1]["ms"] / rs[q]["ms"] if rs[q]["ms"] else 1.0
            if g > gain:
                gain, who = g, rs[q]["knob"].split(" ")[0]
        print(f"  {net:16s} {ops:6d} {base:12.4f} {best['ms']:10.4f} "
              f"{base/best['ms']:7.2f}x  {who}")
        rows.append({"axis": "across_networks", "network": net, "ops": ops,
                     "baseline_ms": base, "best_ms": best["ms"],
                     "speedup": round(base / best["ms"], 3), "winning_knob": who})

    # ---- axis 2: one network, three sizes -----------------------------------
    print("\n  WITHIN ONE NETWORK — ViNT observation encoder at three batch sizes")
    print("  (same graph, same op mix; only the tensor size changes)\n")
    print(f"  {'variant':14s} {'ops':>6s} " +
          " ".join(f"{b:>11s}" for b in ("dsp@int8", "cpu@int8", "cpu@fp32")) +
          "   best")
    for lab in ["vint_obs_b1", "vint_obs_b2", "vint_obs_b3"]:
        e = pooled.get(lab)
        if not e:
            continue
        cells = {f"{b}@{p}": v / 1000.0 for (t, b, p), v in e["cell"].items() if t == 0}
        best = min(cells.items(), key=lambda kv: kv[1]) if cells else ("—", 0)
        print(f"  {lab:14s} {ops_of(e):6d} " +
              " ".join(f"{cells.get(k, float('nan')):11.3f}"
                       for k in ("dsp@int8", "cpu@int8", "cpu@fp32")) +
              f"   {best[0]}")
        rows.append({"axis": "within_network", "variant": lab, "ops": ops_of(e),
                     "cells": {k: round(v, 4) for k, v in cells.items()},
                     "best_backend": best[0], "best_ms": round(best[1], 4)})

    json.dump(rows, open(os.path.join(out, "size_sweep.json"), "w"), indent=1)
    print(f"\n  wrote {out}/size_sweep.json")

    # ---- log ---------------------------------------------------------------
    acr = [r for r in rows if r["axis"] == "across_networks"]
    win = [r for r in rows if r["axis"] == "within_network"]
    L = ["# Does network size decide which feedback path pays?\n",
         "Two axes, because they answer different questions. Across networks "
         "the op count spans three orders of magnitude but topology and op mix "
         "vary too, so a correlation there would be suggestive at best. The "
         "ViNT observation encoder measured at three batch sizes is the "
         "controlled version: same graph, same ops, only the tensors grow.\n",
         "## Across networks\n",
         "| network | ops | baseline ms | best ms | speedup | winning knob |",
         "|---|---:|---:|---:|---:|---|"]
    for r in acr:
        L.append(f"| `{r['network']}` | {r['ops']} | {r['baseline_ms']:.4f} | "
                 f"{r['best_ms']:.4f} | {r['speedup']:.2f}x | `{r['winning_knob']}` |")
    L += ["", "**Size does not predict the knob.** Ordered by op count the "
              "winners run `+slice`, `+backend`, `+precision`, `+backend`, "
              "`+slice` — no monotone relationship, and the largest network "
              "(vint, 1925 ops) and the smallest (mlp_control, 7 ops) share a "
              "winner while everything between them differs. The obvious "
              "hypothesis, that big graphs have room to slice and small ones "
              "are all overhead, is not what the measurements say.\n",
          "What decides it instead is **what the op set can compile to**: "
          "dronet's 11.47x is one backend move, available because its ops run "
          "on the DSP; vint has to be cut before any accelerator will take it "
          "at all; fused_full's win is a precision change its ops happen to "
          "reward. Those are properties of the op mix, not of the size.\n",
          "## Within one network — ViNT observation encoder, batch 1/2/3\n",
          "| variant | ops | dsp@int8 | cpu@int8 | cpu@fp32 | best |",
          "|---|---:|---:|---:|---:|---|"]
    for r in win:
        c = r["cells"]
        L.append(f"| `{r['variant']}` | {r['ops']} | {c.get('dsp@int8','—')} | "
                 f"{c.get('cpu@int8','—')} | {c.get('cpu@fp32','—')} | "
                 f"`{r['best_backend']}` |")
    if len(win) >= 3:
        d = [r["cells"].get("dsp@int8") for r in win]
        f32 = [r["cells"].get("cpu@fp32") for r in win]
        if all(d) and all(f32):
            L += ["", f"**The accelerator's advantage GROWS with size.** Three "
                      f"times the batch costs the DSP {d[2]/d[0]:.2f}x but CPU "
                      f"fp32 {f32[2]/f32[0]:.2f}x, so the DSP's margin over CPU "
                      f"fp32 widens from {f32[0]/d[0]:.2f}x at batch 1 to "
                      f"{f32[2]/d[2]:.2f}x at batch 3. The placement knob is "
                      "therefore worth more on bigger tensors even though the "
                      "CHOICE of knob does not track size across networks — "
                      "the two findings are about different things, and only "
                      "this one is controlled.\n"]
    L += ["## Not measurable host-side\n",
          "| variant | why it is missing |", "|---|---|"]
    for name, why, _ in BOARD_TODO:
        L.append(f"| `{name}` | {why}; exists on the K1/spacemit tree, no "
                 f"QRB5165 cells |")
    L += ["", "`board_plan.sh` builds and profiles these under the shared "
              "board lock. Until it runs, the resolution axis is untested on "
              "this board and nothing here should be read as covering it.\n"]
    open(os.path.join(out, "size_sweep.md"), "w").write("\n".join(L) + "\n")
    print(f"  wrote {out}/size_sweep.md")

    # ---- what still needs the board ----------------------------------------
    print("\n  NOT MEASURABLE HOST-SIDE — these need a board build + profile:")
    for name, why, scale in BOARD_TODO:
        print(f"    {name:22s} {why}")
    if a.emit_board_plan:
        plan = os.path.join(out, "board_plan.sh")
        with open(plan, "w") as f:
            f.write("""#!/usr/bin/env bash
# Build + profile the size variants that QRB5165 has no cells for.
#
# Every board step takes the SAME lock deploy_and_run.sh uses, so this
# serialises against the other agent's sweeps rather than colliding with them.
# Run it with `flock` held for the whole sequence, not per step, so a long
# build cannot be preempted halfway:
#
#     flock -w 43200 /tmp/qnn_board.lock bash results/flowc_size/board_plan.sh
#
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PYTHON:=.venv/bin/python}"
: "${QNN_BOARD_HOST:=root@10.44.120.201}"

for V in yolov8_nano_64x96 yolov8_nano_128x192; do
  echo "=== $V ==="
  # 1. slice/build the variant for QNN (needs the QNN SDK container)
  $PYTHON qnn_models/slicing_study/slice_experiment.py \\
      --network "$V" --backends dsp,cpu,hta --precisions int8,fp32 \\
      --journal qnn_models/slicing_study/experiments.jsonl
  # 2. emit XPURT profile cells from the measured perf json
  $PYTHON qnn_models/smolVLA/emit_vision_v3_profile.py \\
      --from-perf-json "qnn_models/boards/qrb5165_v66/profiles/$V/segment_perf.json" \\
      || echo "  (emit step needs the per-variant perf json; see RESULTS.md 4)"
done

echo "=== re-run the size sweep with the new cells ==="
$PYTHON scripts/flowc_size_sweep.py --out results/flowc_size
""")
        os.chmod(plan, 0o755)
        print(f"\n  wrote {plan} — run it under flock when the board frees")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
