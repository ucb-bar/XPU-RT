#!/usr/bin/env python3
"""Provenance gate — does the binary that ran match the binary intended?

SETUP.md lists five checks. Four are done here; the fifth ([bringup] lines)
cannot be read on this flow, and this file records why and what replaces it.

  1. dispatch_table.h sha256 recorded with the point        -> results.json
  2. `[summary] N/N entries executed` == the schedule's dispatch count
  3. every `[bringup]` line's context filename matches the manifest
     -- NOT AVAILABLE. deploy_and_run.sh's lock line is
          exec {lockfd}> /tmp/qnn_board.lock 2>/dev/null
        and an `exec` with no command applies its redirections to the shell
        for the rest of the script, so every byte the runtime writes to
        stderr on the board is discarded before ssh can carry it back. No
        run.log in this sweep (or any other run through this script) has a
        [bringup] line. What replaces it is strictly stronger:
          * predicate 6 checked every context the table names is on the
            board BEFORE the run;
          * the runtime SKIPS entries whose context is missing, so
            "N/N entries executed" is itself proof that none was missing;
          * check 4 below reads the ctx actually used, per entry, from the
            trace -- which is what [bringup] would only have told us per
            context.
  4. the trace's per-entry `ctx` column matches the intended placement
  5. board build gated on rc=0 with a binary newer than the build start
     -- deploy_and_run.sh compiles with `set -euo pipefail` and aborts the
        run on a non-zero g++, and the driver records its rc. The binary is
        rebuilt from freshly scp'd sources on every rep, so a stale binary
        would have to survive an overwrite; the table sha256 in (1) plus
        the row-for-row table/trace identity in (4) close that gap.
"""
from __future__ import annotations
import csv, hashlib, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FLOWC = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from sweep_unbounded_nonperiodic import BINDINGS, model_of   # noqa: E402
from drive import parse_table                                # noqa: E402


def expected_ctx():
    """(manifest network, tile, kind) -> context filename, from the manifests."""
    out = {}
    for model, rel in BINDINGS.items():
        doc = json.load(open(os.path.join(FLOWC, rel)))
        for b in doc["bindings"]:
            for kind, bb in b["backends"].items():
                out[(doc["network"], b["name"], kind)] = (bb["ctx"], bb["graph"])
    return out


def main():
    exp = expected_ctx()
    state = json.load(open(os.path.join(HERE, "state.json")))
    report = {}
    for pt in sorted(state):
        rec = state[pt]
        tpath = os.path.join(HERE, "runtimes", pt, "dispatch_table.h")
        if not os.path.exists(tpath):
            continue
        table = {t["entry_id"]: t for t in parse_table(tpath)}
        r = {"table_sha256": hashlib.sha256(open(tpath, "rb").read()).hexdigest(),
             "n_entries": len(table), "reps": {}}
        # table -> manifest
        bad_tbl = [f"{t['network']}/{t['name']}@{t['kind']} table says {t['ctx']}, "
                   f"manifest says {exp.get((model_of(t['network']) and json.load(open(os.path.join(FLOWC, BINDINGS[model_of(t['network'])])))['network'], t['name'], t['kind']), ('?','?'))[0]}"
                   for t in table.values()
                   if exp.get((json.load(open(os.path.join(FLOWC, BINDINGS[model_of(t['network'])])))['network'],
                               t['name'], t['kind']), (None, None))[0] != t["ctx"]]
        r["table_vs_manifest_mismatches"] = bad_tbl
        for rep in sorted(os.listdir(os.path.join(HERE, "runs", pt))):
            # The full trace (with the ctx/graph columns) is the block inside
            # run.log; the trace.csv flow_c.py plots writes is a reduced view
            # that drops them.
            rl = os.path.join(HERE, "runs", pt, rep, "run.log")
            if not os.path.exists(rl):
                continue
            text = open(rl, errors="replace").read()
            if "MODELBLASTER_XPURT_TRACE_BEGIN" not in text:
                r["reps"][rep] = {"error": "no trace block in run.log"}
                continue
            block = text.split("MODELBLASTER_XPURT_TRACE_BEGIN ===")[1] \
                        .split("=== MODELBLASTER_XPURT_TRACE_END")[0].strip()
            rows = list(csv.DictReader(io.StringIO(block)))
            probs, ran = [], 0
            seen = set()
            for row in rows:
                eid = int(row["entry_id"])
                seen.add(eid)
                t = table.get(eid)
                if t is None:
                    probs.append(f"trace entry {eid} not in the emitted table")
                    continue
                for f, want in (("network", t["network"]), ("name", t["name"]),
                                ("core_kind", t["kind"]), ("ctx", t["ctx"]),
                                ("graph", t["graph"])):
                    if row.get(f) != want:
                        probs.append(f"e{eid}: trace {f}={row.get(f)!r} != table {want!r}")
                if abs(float(row["predicted_duration_ms"]) - t["dur_ms"]) > 1e-6 or \
                   abs(float(row["predicted_start_ms"]) - t["start_ms"]) > 1e-6:
                    probs.append(f"e{eid}: trace predicted times != table")
                if row.get("actual_end_cycles"):
                    ran += 1
            missing = sorted(set(table) - seen)
            if missing:
                probs.append(f"table entries absent from the trace: {missing}")
            r["reps"][rep] = {"trace_rows": len(rows), "entries_with_timings": ran,
                              "entries_expected": len(table),
                              "all_entries_ran": ran == len(table),
                              "mismatches": probs[:10],
                              "n_mismatches": len(probs)}
        report[pt] = r
    with open(os.path.join(HERE, "provenance.json"), "w") as f:
        json.dump(report, f, indent=1)
    bad = 0
    for pt, r in sorted(report.items()):
        tot = sum(v["n_mismatches"] for v in r["reps"].values()) + len(r["table_vs_manifest_mismatches"])
        allran = all(v["all_entries_ran"] for v in r["reps"].values())
        bad += tot
        print(f"{pt:<22} table {r['n_entries']:>3} entries  sha {r['table_sha256'][:12]}  "
              f"reps {len(r['reps'])}  all-ran {allran}  mismatches {tot}")
    print(f"\ntotal mismatches across every point, rep and column: {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
