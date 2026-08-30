#!/usr/bin/env python3
"""Derive the trade-off curves in RESULTS.md from experiments.jsonl.

Nothing here measures anything: it only reads the journal.  A tile's
dependencies are recovered from its boundary tensors (tile j depends on
tile i when one of j's inputs is one of i's outputs), so the critical
path is the longest path through that DAG with each tile costed at its
best composable backend.

    python3 analyze.py            # all networks
    python3 analyze.py dronet     # one
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
JOURNAL = os.path.join(HERE, "experiments.jsonl")

# Cells for the same (experiment, tile, backend) measured in more than one
# sweep are pooled; STAT picks which per-sweep statistic is pooled and
# POOL how.  The board has two other tenants, so a CPU sweep can catch a
# contention spike -- the median over sweeps is the robust choice.
STAT = "median_us"


def load() -> list[dict]:
    recs = []
    with open(JOURNAL) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def pool(recs: list[dict]) -> dict[str, dict]:
    """label -> {tiles, cells{tile,backend,precision -> [values]}, ...}"""
    out: dict[str, dict] = {}
    for r in recs:
        if r.get("network") in ("_synthetic", "_inventory") or "cut" not in r:
            continue  # probe / inventory records carry no slice set
        lab = r["label"]
        e = out.setdefault(lab, {
            "network": r["network"], "label": lab, "cut": r["cut"],
            "source_onnx": r["source_onnx"], "tiles": r["tiles"],
            "sweeps": 0, "vals": defaultdict(list), "fails": {},
            "artifacts_dir": r["artifacts_dir"], "cut_id": r["cut_id"],
        })
        m = r.get("measurements", {})
        if "cells" not in m:
            continue
        e["sweeps"] += 1
        for key, c in m["cells"].items():
            k = (c["tile"], c["backend"], c["precision"])
            if c["compose"] == "ok" and c.get("stats"):
                e["vals"][k].append(c["stats"][STAT])
            elif c["compose"] == "fail":
                e["fails"][k] = c.get("reason")
    for e in out.values():
        e["cell"] = {k: sorted(v)[len(v) // 2] for k, v in e["vals"].items() if v}
        e["spread"] = {k: (min(v), max(v)) for k, v in e["vals"].items() if len(v) > 1}
    return out


def deps(tiles: list[dict]) -> dict[int, set[int]]:
    prod = {}
    for t in tiles:
        for o in t["outputs"]:
            prod[o["name"]] = t["index"]
    d: dict[int, set[int]] = {t["index"]: set() for t in tiles}
    for t in tiles:
        for i in t["inputs"]:
            p = prod.get(i["src_name"])
            if p is not None and p != t["index"]:
                d[t["index"]].add(p)
    return d


def best(e: dict, ti: int) -> tuple[float | None, str | None]:
    cands = [(v, f"{b}@{p}") for (t, b, p), v in e["cell"].items() if t == ti]
    if not cands:
        return None, None
    return min(cands)


def critical_path(e: dict) -> tuple[float | None, list[str]]:
    tiles = e["tiles"]
    d = deps(tiles)
    finish: dict[int, float] = {}
    order = sorted(t["index"] for t in tiles)
    picks = []
    for ti in order:
        c, who = best(e, ti)
        if c is None:
            return None, []
        picks.append(f"t{ti}:{who}")
        start = max((finish[p] for p in d[ti]), default=0.0)
        finish[ti] = start + c
    return max(finish.values()), picks


def loads(e: dict) -> dict[str, float]:
    per: dict[str, float] = defaultdict(float)
    for t in e["tiles"]:
        c, who = best(e, t["index"])
        if c is None:
            continue
        per[who.split("@")[0]] += c
    return dict(per)


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    ex = pool(load())
    nets = sorted({e["network"] for e in ex.values()})
    for net in nets:
        if only and net != only:
            continue
        print(f"\n================ {net}")
        rows = [e for e in ex.values() if e["network"] == net]
        rows.sort(key=lambda e: (e["cut"]["n_tiles"], e["label"]))
        for e in rows:
            cp, picks = critical_path(e)
            l = loads(e)
            print(f"\n-- {e['label']}  k={e['cut']['n_tiles']}  "
                  f"sweeps={e['sweeps']}  cut={e['cut']['boundary_tensors']}")
            print(f"   op ranges {e['cut']['op_ranges']}")
            hdr = f"   {'tile':<6}{'ops':<14}"
            for b in ("hta@int8", "dsp@int8", "cpu@int8", "cpu@fp32"):
                hdr += f"{b:>14}"
            print(hdr)
            for t in e["tiles"]:
                ti = t["index"]
                line = f"   t{ti:<5}{str(t['op_range']):<14}"
                for b, p in (("hta", "int8"), ("dsp", "int8"),
                              ("cpu", "int8"), ("cpu", "fp32")):
                    v = e["cell"].get((ti, b, p))
                    if v is not None:
                        line += f"{v/1000:>14.3f}"
                    else:
                        r = e["fails"].get((ti, b, p))
                        line += f"{('x ' + (r or '?'))[:13]:>14}"
                print(line)
            print(f"   critical path {cp/1000:8.3f} ms   via {' -> '.join(picks)}")
            print(f"   per-lane load " + "  ".join(f"{k}={v/1000:.3f}ms"
                                                    for k, v in sorted(l.items())))
            if e["spread"]:
                worst = max(e["spread"].items(),
                            key=lambda kv: kv[1][1] / max(kv[1][0], 1e-9))
                (ti, b, p), (lo, hi) = worst
                print(f"   sweep spread (widest cell t{ti} {b}@{p}): "
                      f"{lo/1000:.3f}..{hi/1000:.3f} ms  ({hi/max(lo,1e-9):.2f}x)")


if __name__ == "__main__":
    main()
