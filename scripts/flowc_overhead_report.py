#!/usr/bin/env python3
"""Experiment log for the QRB5165 overhead ablation.

Writes <out>/overhead_ablation.md and overhead_ablation.jsonl: for every
network, which slice set wins at every point of a (call overhead x transfer
rate) grid, and how far the winning recommendation survives before it flips.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "qnn_models", "slicing_study"))
import analyze                       # noqa: E402
import flowc_overhead_ablation as OA  # noqa: E402

MEASURED_CALL_MS = 0.37   # RESULTS.md fit, DSP, idle board
MEASURED_NS = 5.4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", default="vint,yolov8n,fused_full,dronet,mlp_control")
    ap.add_argument("--call-ms", default="0,0.1,0.2,0.37,0.54,1,2,4,8,16,32")
    ap.add_argument("--ns-per-byte", default="0,5.4,20")
    ap.add_argument("--out", default="results/flowc_overhead")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    calls = [float(x) for x in a.call_ms.split(",")]
    rates = [float(x) for x in a.ns_per_byte.split(",")]
    pooled = analyze.pool(analyze.load())

    L = ["# QRB5165 overhead ablation — full experiment log\n",
         "## What is swept, and why\n",
         "A slicing recommendation is only worth acting on if it survives the "
         "overhead we cannot pin down. On K1 the per-dispatch cost is small, "
         "stable and visible, so a slice decision can be read straight off the "
         "compute cells. On QRB5165 it is none of those: a tile boundary "
         "crosses FastRPC, may re-quantize, and may move a large tensor between "
         "an accelerator and the CPU.\n",
         f"`qnn_models/slicing_study/RESULTS.md` fits the marginal cost of a cut "
         f"as **{MEASURED_CALL_MS} ms + {MEASURED_NS} ns x boundary_bytes** "
         "(DSP; >=0.5 ms fixed for HTA). That single fit was taken **on an idle "
         "board**, and it is exactly the quantity a busy board, a different "
         "governor or a context eviction would change. So rather than trust one "
         "number, this sweeps it and reports the range over which each "
         "network's best slice set stays best.\n",
         "Two knobs:\n",
         "* **call overhead (ms)** — charged once per tile: dispatch setup, "
         "context acquire.\n"
         "* **transfer rate (ns/byte)** — charged on every byte a tile must "
         "receive across a cut.\n",
         "The measured cells already contain whatever overhead each tile pays "
         "as a standalone graph. What is swept is the MARGINAL cost of having "
         "cut at all — which a finer slice set pays and a coarser one does not. "
         "So the overhead can only ever push the ranking toward coarser sets; "
         "the question is how hard it has to push.\n",
         "## How to read the notation\n",
         "**This is NOT the S0..S4 ladder.** That ladder (see "
         "`../flowc_stages/stage_ladder.md`) adds a degree of freedom per step. "
         "Here every cell is a *complete* re-ranking of every whole-network "
         "slice set for that network under one overhead assumption, and the "
         "cell names the winner:\n",
         "    <slice-set label>  k=<number of tiles>  <makespan in ms>\n",
         "`k=1` means the monolith won — i.e. at that overhead, cutting does "
         "not pay. A row where the winner changes is a **flip**: the overhead "
         "got large enough to reverse the recommendation.\n",
         "Partial-coverage subgraph probes are excluded from the ranking. "
         "`vint_obs_b*` covers ops 558-1069 of a 1931-op graph — it is cheaper "
         "than any full slice set for the trivial reason that it does less, and "
         "including it makes the partial set win every row.\n"]

    records = []
    envelope = []
    for net in [n.strip() for n in a.networks.split(",") if n.strip()]:
        exps = [v for v in pooled.values() if v["network"] == net]
        n_all = len(exps)
        exps = OA.full_coverage(exps)
        if not exps:
            continue
        L.append(f"\n## {net}\n")
        if len(exps) < n_all:
            L.append(f"*{n_all - len(exps)} partial-coverage probe(s) excluded.*\n")
        L.append("| call ms | " + " | ".join(f"@{r:g} ns/byte" for r in rates) + " |")
        L.append("|---:|" + "---|" * len(rates))
        first_k1 = None
        for c in calls:
            cells = []
            for r in rates:
                ranked = []
                for e in exps:
                    got = OA.makespan(e, c, r)
                    if got:
                        ranked.append((got[0], e["label"], len(e["tiles"])))
                if not ranked:
                    cells.append("—")
                    continue
                ranked.sort()
                mk, lab, k = ranked[0]
                mark = "**" if k > 1 else ""
                cells.append(f"{mark}`{lab}` k={k}{mark} · {mk:.1f} ms")
                records.append({"network": net, "call_ms": c, "ns_per_byte": r,
                                "winner": lab, "tiles": k, "ms": round(mk, 3)})
                if r == rates[0] and k == 1 and first_k1 is None:
                    first_k1 = c
            L.append(f"| {c:g} | " + " | ".join(cells) + " |")
        if first_k1 is None:
            verdict = (f"**slicing wins at every overhead tested** "
                       f"(up to {max(calls):g} ms/call, {max(rates):g} ns/byte)")
            envelope.append((net, ">%g" % max(calls), verdict))
        elif first_k1 == calls[0]:
            verdict = "**the monolith wins everywhere** — no cut ever repays itself"
            envelope.append((net, "never", verdict))
        else:
            below = [x for x in calls if x < first_k1]
            verdict = (f"slicing stops winning at **{first_k1:g} ms/call** "
                       f"(last held at {max(below):g} ms)")
            envelope.append((net, f"{max(below):g}", verdict))
        L.append(f"\n{verdict}.\n")

    L += ["\n## Robustness envelope\n",
          f"How much call overhead each recommendation survives, against the "
          f"measured {MEASURED_CALL_MS} ms:\n",
          "| network | slicing survives to | vs the measured 0.37 ms |",
          "|---|---|---|"]
    for net, surv, _ in envelope:
        if surv == "never":
            rel = "n/a — never slices"
        elif surv.startswith(">"):
            rel = f"~{float(surv[1:])/MEASURED_CALL_MS:.0f}x headroom"
        else:
            f = float(surv)
            rel = (f"**{f/MEASURED_CALL_MS:.2f}x — inside the uncertainty**"
                   if f <= MEASURED_CALL_MS else f"{f/MEASURED_CALL_MS:.1f}x headroom")
        L.append(f"| `{net}` | {surv} ms/call | {rel} |")
    L += ["", "**Four of five recommendations sit inside the measurement "
              "uncertainty.** Only ViNT's survives a plausible overhead range. "
              "fused_full flips one step above the measured value, and dronet "
              "and mlp_control flip *below* it. A current, measured overhead "
              "number is therefore a prerequisite for trusting any slicing "
              "advice except ViNT's.\n",
          "**The per-byte term barely matters.** Across the whole grid only the "
          "0 ms/call rows of dronet and mlp_control move between rate columns; "
          "everywhere else the columns are identical. The fixed per-cut cost "
          "dominates at these tensor sizes, so the harder-to-measure transfer "
          "rate is the term you can afford to be wrong about — and the "
          "FastRPC/context-acquire cost is the one that decides the outcome.\n"]

    md = os.path.join(a.out, "overhead_ablation.md")
    open(md, "w").write("\n".join(L) + "\n")
    jl = os.path.join(a.out, "overhead_ablation.jsonl")
    with open(jl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {md}")
    print(f"  wrote {jl}  ({len(records)} grid points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
