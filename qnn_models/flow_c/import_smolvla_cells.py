#!/usr/bin/env python3
"""Import measured smolVLA tile timings into a Flow C measurements file.

Reads the JSONL emitted by the board-side sweep (one line per
(context, backend) pair) and writes cells keyed `<network>/<tile>` the way
flowc/artifacts.py expects. Cells are the GAP-phase median where available --
that is the statistic this session established as matching in-situ invocation
(see cells_are_gap_phase / residual_makespan_gap_is_scheduling_not_cells in
measurements/qrb5165_v66.json) -- falling back to the loop median.

Writes a SEPARATE measurements file rather than editing qrb5165_v66.json, so
the smolVLA port cannot perturb the cost model the sweep and the cost-model
validation depend on.
"""
import argparse, json, re

BE = {"Cpu": "cpu", "Dsp": "dsp", "Hta": "hta"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--network", default="smolvlm_vision")
    ap.add_argument("--out", default="measurements/qrb5165_v66_smolvla.json")
    a = ap.parse_args()

    cells, failed, n = {}, [], 0
    for line in open(a.jsonl):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = re.match(r"ctx_(.+)__(Cpu|Dsp|Hta)$", rec["ctx"])
        if not m:
            continue
        tile, be = m.group(1), BE[m.group(2)]
        r = rec.get("r")
        if not r or r.get("status") != "ok":
            failed.append((tile, be)); continue
        us = r.get("gap_median_us") or r.get("median_us") or r.get("mean_us")
        if not us:
            failed.append((tile, be)); continue
        cells.setdefault(f"{a.network}/{tile}", {})[be] = round(float(us), 1)
        n += 1

    out = {
        "_comment": (
            "Measured cells for the smolVLA vision port. One profile_seg run per "
            "(context, backend) pair actually present on the board, 10 iters, gap "
            "phase 1000 us, all cores on the performance governor, behind the board "
            "lock. Cells are the gap-phase median. Kept OUT of qrb5165_v66.json so "
            "the port cannot perturb the cost model the sweep depends on. A backend "
            "absent from a cell means no context binary exists for it -- notably no "
            "whole dsp_seg_NN has an Hta context; HTA runs only the extracted "
            "Conv1x1 sub-models."),
        "target": "qrb5165_v66", "captured_at": "2026-08-31",
        "harness": "qnn_models/runtime/profile_segments.cpp (two-phase)",
        "statistic": "gap_median", "iters": 10, "unit": "us",
        "conditions": {"governor": "performance on all 8 cores",
                       "cpu_affinity": "unmasked"},
        "cells": cells,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    from collections import Counter
    c = Counter(",".join(sorted(v)) for v in cells.values())
    print(f"  {len(cells)} tiles, {n} cells -> {a.out}")
    for k, v in sorted(c.items()):
        print(f"    {v:>4} tiles with [{k}]")
    if failed:
        print(f"  {len(failed)} measurement(s) failed: {failed[:6]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
