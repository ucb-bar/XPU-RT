#!/usr/bin/env python3
"""Phase 1G — the per-(network, rung) granularity decision, against the
break-even rule, including the rungs where the decision is NOT to cut.

The rule is `qnn_models/slicing_study/RESULTS.md` §7 conclusion 2, measured on
this board over five networks and four granularities:

    an extra DSP dispatch costs  ~ 0.37 ms + 5.4 ns x boundary_bytes
    an extra HTA dispatch costs  >= 0.50 ms + the same order
    an extra CPU dispatch costs  ~ 0.003 ms

so **a cut pays only when it moves more than that much accelerator work onto an
otherwise idle lane**, and `boundary_bytes` is every tensor crossing the cut at
1 byte per element, not just the one you named -- skip connections that jump a
tile are promoted to tile I/O and count.

Conclusion 1 of the same study is what makes this short: "slicing never buys
speed, it buys placement." Phase 1R bought the placement with numerics-
preserving graph rewrites instead, which PARTITIONING_GUIDE §5 says is the
cheaper move and which measured 2x better than the best cut on dronet. After
those rewrites every network in this set composes on every lane whole, so a cut
can no longer unlock anything -- it can only add dispatches.

This script computes, per network and rung: the whole-network cell on each
lane, the candidate cut points its architecture offers, the boundary bytes each
would cross, the resulting break-even threshold, and the decision.

    python3 granularity.py [--md]
"""
from __future__ import annotations

import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
DSP_FIXED_MS, HTA_FIXED_MS, NS_PER_BYTE = 0.37, 0.50, 5.4e-6   # ms, ms, ms/byte

#: candidate cut points per family, named the way the architecture offers them.
#: dronet cuts at the residual sums (the slicing study's own cut set); yolov8
#: at the FPN neck (the study's best HTA cut, 34 ops past the shipped one);
#: fastdepth at the encoder/decoder boundary; mlp_control has no interior.
CUTS = {
    "dronet": ("residual sums /Add_output_0, /Add_1_output_0, /Add_2_output_0",
               ["/Add_output_0", "/Add_1_output_0", "/Add_2_output_0"]),
    "yolov8_nano": ("FPN neck + detect-head split (the study's k=3 shape)",
                    ["/l15/Concat_output_0", "/l18/Concat_output_0"]),
    "fastdepth": ("encoder/decoder boundary and the decoder's upsample stages",
                  None),          # discovered below
    "mlp_control": ("none -- 7 ops, no interior activation worth a dispatch",
                    []),
}


def tensor_bytes(model, names, shapes):
    tot, detail = 0, []
    for n in names:
        s = shapes.get(n)
        if not s:
            continue
        b = 1
        for d in s:
            b *= int(d)
        tot += b
        detail.append((n, list(s), b))
    return tot, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="emit the markdown table")
    a = ap.parse_args()
    import onnx
    sys.path.insert(0, HERE)
    from onnx_rewrites import _shapes

    cm = json.load(open(os.path.join(SWEEP, "cost_model.json")))
    cells = cm["cells"]
    rows = json.load(open(os.path.join(SWEEP, "results", "phase1_compose.json")))
    out = []
    for r in rows:
        net, fam = r["id"], r["family"]
        cell = cells.get(f"{net}/{net}_full") or {}
        lanes = {k: v for k, v in cell.items() if "@" not in k and v}
        if not lanes:
            continue
        label, names = CUTS.get(fam, ("", []))
        onnx_path = os.path.join(SWEEP, r["onnx"])
        bb, detail = 0, []
        if names is None or names:
            m = onnx.load(onnx_path)
            sh = _shapes(m)
            if names is None:                     # fastdepth: widest Resize input
                cands = [n.input[0] for n in m.graph.node if n.op_type == "Resize"]
                names = sorted(cands, key=lambda t: -(
                    __import__("math").prod(sh.get(t, (0,)))))[:1]
                label = f"decoder upsample boundary ({names[0] if names else 'none'})"
            bb, detail = tensor_bytes(m, names, sh)
        thr_dsp = DSP_FIXED_MS + NS_PER_BYTE * bb
        thr_hta = HTA_FIXED_MS + NS_PER_BYTE * bb
        fastest = min(lanes.values()) / 1000.0
        n_cuts = max(1, len(names)) if names else 0
        # A cut can only pay if the tile it moves exceeds the threshold AND it
        # unlocks a lane. After Phase 1R nothing is locked, so the second
        # condition is false everywhere and the first is reported for the record.
        decision = "keep whole"
        why = []
        if not names:
            why.append("no interior boundary: the whole network is one dispatch "
                       "of work")
        else:
            why.append(f"{n_cuts} cut(s) would add "
                       f"{n_cuts * thr_dsp:.2f} ms of DSP dispatch "
                       f"({DSP_FIXED_MS} ms fixed + {bb} boundary bytes x "
                       f"{NS_PER_BYTE*1e6:.1f} ns) against a whole-network "
                       f"critical path of {fastest:.3f} ms on its fastest lane")
            if n_cuts * thr_dsp >= fastest:
                why.append("the added dispatch alone exceeds the whole network, "
                           "so no assignment of the pieces can win")
            else:
                why.append("the dispatch is affordable, but Phase 1R's rewrites "
                           "already made every lane reachable whole, so a cut "
                           "unlocks no placement it does not already have")
        out.append(dict(network=net, family=fam,
                        lanes_ms={k: round(v / 1000, 3) for k, v in sorted(lanes.items())},
                        fastest_lane=min(lanes, key=lanes.get),
                        fastest_ms=round(fastest, 3),
                        cut_candidates=label, n_cuts=n_cuts,
                        boundary_bytes=bb, boundary_detail=detail,
                        breakeven_dsp_ms=round(thr_dsp, 3),
                        breakeven_hta_ms=round(thr_hta, 3),
                        decision=decision, why=why))
    json.dump(out, open(os.path.join(SWEEP, "results", "granularity.json"), "w"),
              indent=1)
    if a.md:
        print("| network | lanes (ms) | fastest | candidate cut | boundary B | "
              "break-even/cut (DSP) | decision |")
        print("|---|---|---|---|---|---|---|")
        for r in out:
            print(f"| `{r['network']}` | "
                  + " ".join(f"{k} {v}" for k, v in r["lanes_ms"].items())
                  + f" | {r['fastest_lane']} {r['fastest_ms']} | "
                    f"{r['cut_candidates'][:44]} | {r['boundary_bytes']} | "
                    f"{r['breakeven_dsp_ms']} ms | **{r['decision']}** |")
    else:
        for r in out:
            print(f"{r['network']:<18} fastest {r['fastest_lane']}"
                  f" {r['fastest_ms']:>8.3f} ms   {r['n_cuts']} cut(s), "
                  f"{r['boundary_bytes']:>8} B, break-even "
                  f"{r['breakeven_dsp_ms']:.3f} ms/cut  -> {r['decision']}")
            for w in r["why"]:
                print(f"      - {w}")
    print(f"\n  -> {os.path.join(SWEEP, 'results', 'granularity.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
