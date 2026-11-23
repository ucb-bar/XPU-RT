#!/usr/bin/env python3
"""Checks on a corrected profile tree before the planner or the board sees it.

1. Structure against the raw tree: the same files; every file that is not a results.csv of one of the
   workload's models is a byte copy; every such results.csv has the same header, the same rows in the
   same order, and differs only in mean_time / mean_time_ns.
2. Optional (--profile-audit <script>): the profile tree's own audit (the script that wrote each
   folder's audit.json when the tree was measured) re-run on the corrected results.csv with the ORIGINAL
   profiling run's log and generated sources; it must still pass. Without the option this check is
   skipped and the output says so. (That audit does not look at mean_time itself; check 3 does.)
3. Values: every corrected mean_time is finite, > 0, and inside what the recipe allows:
       0.5 * raw  <=  corrected  <=  max(4 * raw, raw + largest additive) + largest gap
   (ratio clamp 0.5 .. 4; additive >= 0 for layers under 20 us; the gap is added to every layer). The
   largest additive and gap come from --corrections; without it the limits are corrected / raw in
   --min-ratio .. --max-ratio.
With --expect-identical the tree must be a byte copy of the raw tree's profile/ folder.

usage: tree_audit.py --raw-tree <raw profile tree> --tree <corrected tree> [--corrections corrections.json]
                     [--profile-audit <script>] [--expect-identical]
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import json
import math
import os
import subprocess
import sys

NETS = tuple(x for x in os.environ.get("K3TL_NETS", "mlp_control,dronet,yolov8_nano").split(",") if x)


def files_under(root: str) -> list:
    out = []
    for r, _, fs in os.walk(os.path.join(root, "profile")):
        out += [os.path.relpath(os.path.join(r, f), root) for f in fs]
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-tree", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--expect-identical", action="store_true")
    ap.add_argument("--min-ratio", type=float, default=0.4)
    ap.add_argument("--max-ratio", type=float, default=12.0)
    ap.add_argument("--corrections", help="corrections.json the tree was built from (gives the exact value limits)")
    ap.add_argument("--profile-audit", default=os.environ.get("K3TL_PROFILE_AUDIT", ""),
                    help="the audit script of the profiling campaign that measured the raw tree; re-run on every corrected folder")
    a = ap.parse_args()
    fails = []
    add_max = gap_max = None
    if a.corrections:
        doc = json.load(open(a.corrections))
        add_max = max([o["value"] for o in doc["ops"].values() if o["mode"] == "add"] or [0.0])
        gap_max = max(list(doc.get("gap_kind_mean_ms", {}).values()) + list(doc.get("gap_by_key_ms", {}).values()) or [0.0])
    raw_files, new_files = files_under(a.raw_tree), files_under(a.tree)
    if raw_files != new_files:
        fails.append(f"file sets differ: {len(raw_files)} raw vs {len(new_files)} corrected")
    changed = same = rows_changed = rows_total = 0
    worst = (1.0, 1.0)
    for rel in raw_files:
        r, n = os.path.join(a.raw_tree, rel), os.path.join(a.tree, rel)
        if not os.path.exists(n):
            continue
        if filecmp.cmp(r, n, shallow=False):
            same += 1
            continue
        changed += 1
        if not (rel.endswith("results.csv") and any(f"/{n}/" in rel for n in NETS)):
            fails.append(f"changed but not a results.csv of the workload's models: {rel}")
            continue
        ra, na = list(csv.DictReader(open(r, newline=""))), list(csv.DictReader(open(n, newline="")))
        if len(ra) != len(na) or (ra and list(ra[0]) != list(na[0])):
            fails.append(f"rows or header differ: {rel}")
            continue
        for x, y in zip(ra, na):
            rows_total += 1
            other = [c for c in x if c not in ("mean_time", "mean_time_ns") and x[c] != y[c]]
            if other:
                fails.append(f"{rel} dispatch {x['dispatch_id']}: columns {other} changed")
            v0, v1 = float(x["mean_time"]), float(y["mean_time"])
            if v0 != v1:
                rows_changed += 1
            if not (math.isfinite(v1) and v1 > 0):
                fails.append(f"{rel} dispatch {x['dispatch_id']}: mean_time {v1}")
            elif v0 > 0:
                q = v1 / v0
                worst = (min(worst[0], q), max(worst[1], q))
                if add_max is not None:
                    lo, hi = 0.5 * v0 - 1e-6, max(4.0 * v0, v0 + add_max) + gap_max + 1e-6
                    if not (lo <= v1 <= hi):
                        fails.append(f"{rel} dispatch {x['dispatch_id']}: corrected {v1} outside {lo:.6f} .. {hi:.6f}")
                elif not (a.min_ratio <= q <= a.max_ratio):
                    fails.append(f"{rel} dispatch {x['dispatch_id']}: corrected/raw = {q:.3f}")
            if y.get("mean_time_ns") and abs(float(y["mean_time_ns"]) - v1 * 1e6) > 1.0:
                fails.append(f"{rel} dispatch {x['dispatch_id']}: mean_time_ns does not match mean_time")
    if a.expect_identical and changed:
        fails.append(f"expected a byte copy, but {changed} files differ")

    # the tree's own audit, on the corrected CSVs, with the original run's log and sources
    audits = passed = 0
    for rel in (new_files if a.profile_audit else []):
        if not (rel.endswith("audit.json") and any(f"/{n}/" in rel for n in NETS)):
            continue
        j = json.load(open(os.path.join(a.tree, rel)))
        gen = os.path.dirname(j["checks"]["kernels"]["kernels_c"])
        log = gen[:-len(".gen")] + ".log" if gen.endswith(".gen") else ""
        res = os.path.join(a.tree, os.path.dirname(rel), "results.csv")
        if not (os.path.isdir(gen) and os.path.isfile(log)):
            fails.append(f"original profiling log or sources missing for {rel}")
            continue
        audits += 1
        p = subprocess.run([sys.executable, "-I", a.profile_audit, "--log", log, "--results", res, "--gen-dir", gen,
                            "--model", j["model"], "--backend", j["backend"],
                            "--cpus", ",".join(str(c) for c in j["cpus"]), "--flags", " ".join(j["flags"]),
                            "--iters", "50"], capture_output=True, text=True)
        line = (p.stdout.strip().splitlines() or ["no output"])[-1]
        if p.returncode == 0 and line.startswith("AUDIT PASS"):
            passed += 1
        else:
            fails.append(f"profile audit on {os.path.dirname(rel)}: {line}")

    print(f"tree {a.tree} vs raw {a.raw_tree}")
    print(f"  files: {len(new_files)} ({same} byte copies, {changed} changed); "
          f"rows changed {rows_changed} of {rows_total} in changed files; "
          f"corrected/raw between {worst[0]:.3f} and {worst[1]:.3f}")
    print(f"  profile audit re-run on {audits} folders: {passed} PASS" if a.profile_audit else "  profile audit: not re-run (no --profile-audit script given)")
    for f in fails[:20]:
        print("  FAIL:", f)
    print("RESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
