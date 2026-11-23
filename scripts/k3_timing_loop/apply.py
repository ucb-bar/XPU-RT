#!/usr/bin/env python3
"""Build a corrected copy of the profile tree from the RAW tree and a corrections file (fit.py).
The planner is not changed.

    corrected = raw * ratio (raw >= 20 us)  or  raw + add (raw < 20 us)   + gap

Run time of an entry the board did not measure: the median of measured keys of the workload's models with
the same  1) op type, core kind, kernel type, cores   2) core kind, cores   3) core kind   4) all,
same mode. Gap: the entry's own, else the mean of its (network, core kind), else the mean of its core
kind over all models; with --flat-gap-ms one flat start cost on every row instead (gap rule "table").
Counts go to <out>/t023_tree_report.json, every entry to <out>/t023_tree_entries.csv. --identity: every
ratio 1, add 0, gap 0, files rewritten through the same code (must come out as a byte copy).

usage: apply.py --raw-tree <raw profile tree> --corrections corrections.json --out <new tree> [--identity] [--flat-gap-ms X]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loop_lib as T  # noqa: E402
from fit import SCHEMA, TINY_MS, key_str  # noqa: E402

LEVELS = ("measured", "level1 op+kind+kernel+cores", "level2 kind+cores", "level3 kind", "level4 all", "none (unchanged)")


class Corrections:
    def __init__(self, doc: dict):
        self.ops = {(o["network"], o["layer"], o["core_kind"], o["kernel_type"], o["cores"]): o for o in doc["ops"].values()}
        self.gap_by_key = dict(doc["gap_by_key_ms"])
        self.kind_mean = dict(doc["gap_kind_mean_ms"])
        pools: Dict[tuple, List[float]] = defaultdict(list)
        for (_, _, kind, impl, width), o in self.ops.items():
            m = o["mode"]
            pools[(m, 1, o["op"], kind, impl, width)].append(o["value"])
            pools[(m, 2, kind, width)].append(o["value"])
            pools[(m, 3, kind)].append(o["value"])
            pools[(m, 4)].append(o["value"])
        self.med = {k: statistics.median(v) for k, v in pools.items()}

    def value(self, key: T.Key, op: str, raw: float) -> Tuple[str, float, int]:
        mode = "ratio" if raw >= TINY_MS else "add"
        o = self.ops.get(key)
        if o is not None:
            if o["mode"] != mode:
                raise ValueError(f"{key}: fitted as {o['mode']} but raw {raw} ms says {mode}")
            return mode, o["value"], 0
        _, _, kind, impl, width = key
        for lvl, k in ((1, (mode, 1, op, kind, impl, width)), (2, (mode, 2, kind, width)), (3, (mode, 3, kind)), (4, (mode, 4))):
            if k in self.med:
                return mode, self.med[k], lvl
        return mode, (1.0 if mode == "ratio" else 0.0), 5

    def gap_for(self, key: T.Key) -> Tuple[float, str]:
        own = self.gap_by_key.get(key_str(key))
        if own is not None:
            return own, "own (ran in the latest round)"
        for k, where in ((f"{key[0]}|{key[2]}", "mean of its model and core kind"), (f"*|{key[2]}", "mean of its core kind, all models")):
            if k in self.kind_mean:
                return self.kind_mean[k], where
        return 0.0, "none"


def build(raw_tree: str, doc: dict, out: str, identity: bool, flat_gap_ms: float = -1.0) -> dict:
    if os.path.exists(out):
        raise FileExistsError(f"{out} exists; refusing to overwrite")
    corr = Corrections(doc)
    shutil.copytree(raw_tree, out)
    levels: Dict[str, Counter] = defaultdict(Counter)
    gaps: Dict[str, Counter] = defaultdict(Counter)
    entries: List[dict] = []
    files = 0
    for net in T.NETS:
        for (kind, impl) in T.BACKEND_DIR:
            for width in T.WIDTH_TOPO:
                path = T.profile_csv(out, net, kind, impl, width)
                if not os.path.exists(path):
                    continue
                with open(path, newline="") as f:
                    text = f.read()
                eol = "\r\n" if "\r\n" in text else "\n"
                rd = csv.DictReader(text.splitlines())
                cols, rows = list(rd.fieldnames or []), list(rd)
                touched = identity
                for row in rows:
                    raw = float(row["mean_time"])
                    if raw >= T.SENTINEL_MS or raw <= 0.0:
                        levels[net]["left alone (marker or zero)"] += 1
                        continue
                    key = (net, int(row["dispatch_id"]), kind, impl, width)
                    mode, val, lvl = corr.value(key, row.get("op", ""), raw)
                    gap, gsrc = corr.gap_for(key)
                    if flat_gap_ms >= 0:                 # gap rule "table": only the flat same-thread start cost is in the tree
                        gap, gsrc = flat_gap_ms, "flat same-thread start cost (the planner's hand-off table adds the wake-ups)"
                    if identity:
                        val, gap = (1.0 if mode == "ratio" else 0.0), 0.0
                    new = (raw * val if mode == "ratio" else raw + val) + gap
                    if new != raw:
                        touched = True
                        row["mean_time"] = f"{new:.6f}"
                        if row.get("mean_time_ns"):
                            row["mean_time_ns"] = f"{new * 1e6:.6f}"
                    levels[net][LEVELS[lvl].split()[0]] += 1
                    gaps[net][gsrc] += 1
                    entries.append(dict(network=net, layer=key[1], op=row.get("op", ""), core_kind=kind, kernel_type=impl, cores=width,
                                        raw_ms=f"{raw:.6f}", mode=mode, value=f"{val:.6f}", gap_ms=f"{gap:.6f}",
                                        corrected_ms=f"{new:.6f}", ratio_to_raw=f"{new / raw:.4f}", source=LEVELS[lvl], gap_source=gsrc))
                if touched:
                    with open(path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=cols, lineterminator=eol)
                        w.writeheader()
                        w.writerows(rows)
                    files += 1
    with open(os.path.join(out, T.TREE_ENTRIES), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
        w.writeheader()
        w.writerows(entries)
    report = dict(schema=SCHEMA, raw_tree=raw_tree, out=out, identity=identity, files_rewritten=files, entries=len(entries),
                  run_time_source={n: dict(c) for n, c in levels.items()}, gap_source={n: dict(c) for n, c in gaps.items()},
                  rounds=[r["name"] for r in doc.get("rounds", [])], gap_round=doc.get("gap_round"))
    with open(os.path.join(out, T.TREE_REPORT), "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-tree", required=True)
    ap.add_argument("--corrections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--flat-gap-ms", type=float, default=-1.0,
                    help="gap rule 'table': put this start cost on every row instead of any measured gap (handoff_table.py)")
    a = ap.parse_args()
    doc = json.load(open(a.corrections))
    if doc.get("schema") != SCHEMA:
        sys.exit(f"{a.corrections}: schema {doc.get('schema')!r}, expected {SCHEMA}")
    rep = build(a.raw_tree, doc, a.out, a.identity, a.flat_gap_ms)
    print(f"wrote {a.out}: {rep['files_rewritten']} files, {rep['entries']} entries" + (" (IDENTITY)" if a.identity else ""))
    for n in T.NETS:
        print(f"  {n:12s} run time: {rep['run_time_source'].get(n)}; gap: {rep['gap_source'].get(n)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
