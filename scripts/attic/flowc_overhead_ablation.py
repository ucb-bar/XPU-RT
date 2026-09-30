#!/usr/bin/env python3
"""How much does the QRB5165 slicing recommendation depend on runtime overhead?

WHY THIS EXISTS. On K1 the per-dispatch cost is small, visible and stable, so
a slicing decision can be read straight off the compute cells. On QRB5165 it
is neither small nor easy to attribute: a tile boundary crosses FastRPC, may
re-quantize, and may move a large tensor between an accelerator and the CPU.
`slicing_study/RESULTS.md` measures the marginal cost of a cut as

    0.37 ms + 5.4 ns x boundary_bytes          (DSP; >=0.5 ms fixed for HTA)

but that single fit is doing a lot of work: it was taken on an idle board,
and it is exactly the quantity that a busy board, a different governor or a
context-eviction would change. A recommendation that flips under a plausible
overhead is not a recommendation.

So instead of trusting one number, this sweeps the overhead model and reports
the RANGE over which each network's best slice set stays best -- the
robustness envelope. Two knobs:

  call overhead  ms added per tile (dispatch setup, context acquire)
  transfer rate  ns per boundary byte (the tensor crossing the cut)

The measured cells already contain whatever overhead each tile pays as a
standalone graph. What is swept here is the MARGINAL cost of having cut at
all, which is what a finer slice set pays and a coarser one does not.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(HERE, "..", "qnn_models", "slicing_study")
sys.path.insert(0, STUDY)
import analyze  # noqa: E402

KIND = {"hta": "hta", "dsp": "dsp", "cpu": "cpu"}


def covered_ops(e: dict) -> int:
    """How many source ops a slice set actually covers.

    The journal mixes whole-network slice sets with SUBGRAPH probes
    (`vint_obs_b*` is ops 558-1069 of a 1931-op graph, measured to price one
    encoder at three batch sizes). A probe is cheaper than any full slice set
    for the trivial reason that it does less, so ranking them together makes
    the partial set win every time. Only sets covering the same op span are
    comparable.
    """
    n = 0
    for t in e["tiles"]:
        rng = t.get("ranges") or ([t["op_range"]] if t.get("op_range") else [])
        for r in rng:
            if isinstance(r, (list, tuple)) and len(r) == 2:
                n += int(r[1]) - int(r[0]) + 1
    return n


def full_coverage(exps: list[dict]) -> list[dict]:
    """Keep only the slice sets that span the whole network."""
    if not exps:
        return []
    widest = max(covered_ops(e) for e in exps)
    # a full set is within 5% of the widest span seen for this network
    return [e for e in exps if covered_ops(e) >= 0.95 * widest]


def _elems(shape) -> int:
    n = 1
    for d in shape:
        n *= d if isinstance(d, int) and d > 0 else 1
    return n


def boundary_bytes(e: dict) -> dict[int, int]:
    """Bytes each tile must receive from another tile (int8 = 1 byte/elem)."""
    produced = {o["name"]: t["index"] for t in e["tiles"] for o in t["outputs"]}
    out: dict[int, int] = {t["index"]: 0 for t in e["tiles"]}
    for t in e["tiles"]:
        for i in t["inputs"]:
            src = produced.get(i["src_name"])
            if src is not None and src != t["index"]:
                out[t["index"]] += _elems(i.get("shape") or [])
    return out


def makespan(e: dict, call_ms: float, ns_per_byte: float) -> tuple[float, dict] | None:
    """One lane per backend kind, with overhead charged on every tile.

    A tile pays `call_ms` whenever it runs, and `ns_per_byte` on every byte it
    has to receive across a cut. A k=1 slice set pays one call and no
    transfer, which is why the overhead can only ever push the ranking toward
    coarser sets -- the question is how far it has to push.
    """
    d = analyze.deps(e["tiles"])
    bb = boundary_bytes(e)
    idx = sorted(t["index"] for t in e["tiles"])
    opts = {}
    for ti in idx:
        cands = [(v / 1000.0, f"{b}@{p}", b) for (t, b, p), v in e["cell"].items() if t == ti]
        if not cands:
            return None
        opts[ti] = cands
    best = None
    # brute force over assignments (tile counts here are <= 4)
    def rec(i, chosen):
        nonlocal best
        if i == len(idx):
            free: dict[str, float] = {}
            finish: dict[int, float] = {}
            for ti in idx:
                cost, label, backend = chosen[ti]
                over = call_ms + bb[ti] * ns_per_byte * 1e-6
                dep_ready = max((finish[p] for p in d[ti]), default=0.0)
                lane = KIND.get(backend, backend)
                start = max(dep_ready, free.get(lane, 0.0))
                finish[ti] = start + cost + over
                free[lane] = finish[ti]
            mk = max(finish.values())
            if best is None or mk < best[0]:
                best = (mk, {t: chosen[t][1] for t in idx})
            return
        for c in opts[idx[i]]:
            chosen[idx[i]] = c
            rec(i + 1, chosen)
    rec(0, {})
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--networks", default="vint,yolov8n,fused_full,dronet")
    ap.add_argument("--call-ms", default="0,0.37,0.54,1.0,2.0,4.0")
    ap.add_argument("--ns-per-byte", default="0,5.4,20")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    pooled = analyze.pool(analyze.load())
    calls = [float(x) for x in a.call_ms.split(",")]
    rates = [float(x) for x in a.ns_per_byte.split(",")]
    result = {}

    for net in a.networks.split(","):
        exps = [v for v in pooled.values() if v["network"] == net]
        alln = len(exps)
        exps = full_coverage(exps)
        if not exps:
            continue
        if len(exps) < alln:
            print(f"\n  (excluded {alln - len(exps)} partial-coverage probe(s) "
                  f"for {net}: not comparable with whole-network slice sets)")
        print(f"\n=== {net} — winning slice set vs overhead ===")
        print(f"  {'call ms':>8s} " + "".join(f"{('@%gns' % r):>30s}" for r in rates))
        rows = []
        for c in calls:
            cells = []
            for r in rates:
                ranked = []
                for e in exps:
                    got = makespan(e, c, r)
                    if got:
                        ranked.append((got[0], e["label"], len(e["tiles"])))
                if not ranked:
                    cells.append(f"{'-':>30s}")
                    continue
                ranked.sort()
                mk, lab, k = ranked[0]
                cells.append(f"{lab[:20]:>22s} k={k} {mk:5.1f}"[:30].rjust(30))
                rows.append({"call_ms": c, "ns_per_byte": r,
                             "winner": lab, "tiles": k, "ms": round(mk, 3)})
            print(f"  {c:8.2f} " + "".join(cells))
        result[net] = rows
        # robustness: largest call overhead at which the k>1 winner survives
        base = [x for x in rows if x["ns_per_byte"] == rates[0]]
        sliced = [x["call_ms"] for x in base if x["tiles"] > 1]
        if sliced and len(sliced) < len(base):
            print(f"  -> slicing stops winning above call overhead "
                  f"{max(sliced):.2f} ms (at {rates[0]:g} ns/byte)")
        elif sliced:
            print(f"  -> slicing still wins at every call overhead tested "
                  f"(<= {max(calls):.2f} ms)")
        else:
            print("  -> the monolith wins at every overhead tested")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
