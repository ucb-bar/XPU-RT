#!/usr/bin/env python3
"""Freeze this sweep's cost model: gap-phase cells for the 16 networks Phase 1
built, plus the shipped cells for `vint`, which is reused as-is.

The result is written to `<sweep>/cost_model.json` and NOTHING downstream reads
`measurements/qrb5165_v66.json` again. That is the discipline the precedent
sweep established and its SETUP.md explains why: the shared measurements file
keeps being rebuilt (in-situ promotion, feedback runs), and it has already
moved a recorded makespan by 19% between two runs that were both correct for
their own cost model. Pinning here makes the predicted numbers in this sweep
reproducible from the committed record.

`vint` is the one network not rebuilt in Phase 1. Its manifest is a real
two-tile split (encoders / decoder) whose tiles were cut at the compress
projections after a documented GELU rewrite, and it has measured cells on dsp,
cpu and gpu. Rebuilding it as one tile would have made the two `vint_*`
families use a network that runs nowhere but the CPU, which is a different
workload from the reference's. Its cells are therefore copied verbatim from
measurements/qrb5165_v66.json, and this file records that they are
`in_situ_p50_pooled` while the 16 new ones are `gap_median` -- two provenances
in one model, which is stated rather than blended.

    python3 build_cost_model.py [--write]
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
FLOWC = os.path.abspath(os.path.join(SWEEP, "..", ".."))
RAW = os.path.join(SWEEP, "measurements", "qrb5165_v66_s10port_raw.json")
SHIPPED = os.path.join(FLOWC, "measurements", "qrb5165_v66.json")
OUT = os.path.join(SWEEP, "cost_model.json")

#: cells copied verbatim from the shipped model, with their own provenance
REUSED = {"vint/vint_encoders", "vint/vint_decoder"}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "-C", FLOWC, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    raw = json.load(open(RAW))
    shipped = json.load(open(SHIPPED))

    cells, prov = {}, {}
    # 1. the 16 networks Phase 1 built -- gap_median_us, declared lanes only
    for r in raw["results"]:
        if r.get("status") != "ok":
            continue
        cell, be = r["cell"], r["backend"]
        v = round(float(r["gap_median_us"]), 1)
        cells.setdefault(cell, {})[be] = v
        prov[f"{cell}@{be}"] = dict(
            statistic="gap_median", iters=raw["iters"], gap_us=raw["gap_us"],
            declared=r.get("declared", True), precision=r.get("precision"),
            loop_median_us=round(float(r["median_us"]), 1),
            gap_minus_loop_us=round(float(r["gap_minus_loop_us"]), 1),
            p99_us=round(float(r.get("gap_p99_us") or 0), 1))
    # 2. vint, verbatim
    for cell in sorted(REUSED):
        src = (shipped.get("cells") or {}).get(cell)
        if not src:
            print(f"  WARNING: reused cell {cell} not in the shipped model")
            continue
        cells[cell] = dict(src)
        for be in src:
            prov[f"{cell}@{be}"] = dict(
                statistic=shipped.get("statistic"),
                captured_at=shipped.get("captured_at"),
                source=os.path.relpath(SHIPPED, FLOWC), reused=True)

    fails = []
    fp = os.path.join(SWEEP, "results", "compose_failures.json")
    if os.path.exists(fp):
        fails = json.load(open(fp))
    # the shipped model's own vint failures stay attached to the reused cells
    for f in shipped.get("compose_failures", []):
        if str(f.get("cell", "")).startswith("vint/"):
            fails.append(dict(f, source=os.path.relpath(SHIPPED, FLOWC)))

    doc = {
        "_comment": (
            "FROZEN cost model for the sched_algo_sweep10 QRB5165 port. Cells "
            "for the 16 rebuilt networks are gap-phase medians "
            "(profile_segments.cpp, idle gap before each execute, which is how "
            "the scheduled runtime invokes a tile); vint's two tiles are copied "
            "verbatim from the shipped model and carry their own provenance. "
            "Every predicted number in this sweep is solved against THIS file "
            "and not against measurements/qrb5165_v66.json, which keeps being "
            "rebuilt."),
        "target": "qrb5165_v66",
        "captured_at": time.strftime("%Y-%m-%d", time.gmtime()),
        "unit": "us",
        "statistic": "gap_median (16 rebuilt networks) + "
                     "in_situ_p50_pooled (vint, reused)",
        "harness": "qnn_models/runtime/profile_segments.cpp",
        "iters": raw["iters"],
        "gap_us": raw["gap_us"],
        "conditions": raw["conditions"],
        "git_rev": git_rev(),
        "inputs": {
            "raw_measurements": os.path.relpath(RAW, SWEEP),
            "raw_measurements_sha256": sha256(RAW),
            "shipped_model": os.path.relpath(SHIPPED, FLOWC),
            "shipped_model_sha256": sha256(SHIPPED),
        },
        "cells": cells,
        "cell_provenance": prov,
        "compose_failures": fails,
    }
    n_pairs = sum(len(v) for v in cells.values())
    print(f"  {len(cells)} cells, {n_pairs} (cell, backend) pairs, "
          f"{len(fails)} compose failures")
    per_be = {}
    for v in cells.values():
        for k in v:
            per_be[k] = per_be.get(k, 0) + 1
    print(f"  per backend: {dict(sorted(per_be.items()))}")
    if a.write:
        json.dump(doc, open(OUT, "w"), indent=1)
        print(f"  -> {OUT}  (sha256 {sha256(OUT)[:16]})")
    else:
        print("  (dry run -- pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
