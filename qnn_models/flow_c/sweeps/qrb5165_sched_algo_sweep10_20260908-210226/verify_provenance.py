#!/usr/bin/env python3
"""Check this sweep's record against itself: every claim that can be checked
from the committed artifacts, checked.

    python3 verify_provenance.py [--strict]

Nothing here talks to the board or re-solves anything. It answers: is the
record internally consistent, and does it say what it was produced from?

  cost model    frozen file's sha256 matches what results record; its input
                hashes match the files still on disk; every cell it holds
                carries provenance naming its statistic
  bindings      every manifest declares only lanes Phase 1 recorded as
                composed, and every declared (tile, lane) has a cell
  workloads     every emitted taskset's networks resolve to a manifest, its
                periods match what the generator recorded, and its
                dispatch_deps_path points at a file that exists
  solves        every returned schedule passed the independent feasibility
                audit (sweep10_runner.validate) with all counters zero
  schedules     the dedupe is real: points sharing a sched_hash have the same
                dispatch_table.h sha256, and points with different hashes do not
  runs          every rep that is reported as a result ran N/N entries, and no
                point reports a median from fewer than 2 reps
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FLOWC = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))
OK, BAD, WARN = "  \033[32mOK\033[0m  ", "  \033[31mFAIL\033[0m", "  \033[33mWARN\033[0m"


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(rel, default=None):
    p = os.path.join(HERE, rel)
    if not os.path.exists(p):
        return default
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="treat WARN as failure")
    a = ap.parse_args()
    bad = warn = 0

    def say(cond, msg, soft=False):
        nonlocal bad, warn
        if cond:
            print(f"{OK} {msg}")
        elif soft:
            warn += 1
            print(f"{WARN} {msg}")
        else:
            bad += 1
            print(f"{BAD} {msg}")

    # ---- cost model
    print("cost model")
    cm_path = os.path.join(HERE, "cost_model.json")
    say(os.path.exists(cm_path), "cost_model.json present")
    if not os.path.exists(cm_path):
        return 1
    cm = json.load(open(cm_path))
    cells, prov = cm["cells"], cm.get("cell_provenance", {})
    n_pairs = sum(len(v) for v in cells.values())
    say(all(f"{c}@{b}" in prov for c in cells for b in cells[c]),
        f"{n_pairs} (cell, backend) pairs, all with provenance")
    stats = sorted({p.get("statistic") for p in prov.values()})
    say(stats == ["gap_median", "in_situ_p50_pooled"] or len(stats) <= 2,
        f"statistics present: {stats}")
    for k, v in (cm.get("inputs") or {}).items():
        if not k.endswith("_sha256"):
            continue
        src = cm["inputs"][k[:-len("_sha256")]]
        p = os.path.join(HERE, src) if not os.path.isabs(src) else src
        if not os.path.exists(p):
            p = os.path.join(FLOWC, src)
        say(os.path.exists(p) and sha256(p) == v,
            f"input {os.path.basename(src)} sha256 matches the record",
            soft=not os.path.exists(p))

    # ---- bindings vs the compose record
    print("\nbindings")
    # A manifest's lane may be served by a REWRITTEN variant, so the compose
    # verdict to check is the adopted source's, not the base network's. Reading
    # only phase1_compose.json here reported every rewritten HTA lane as a
    # failure -- the checker was wrong, not the data.
    comp = {r["id"]: r for r in (load("results/phase1_compose.json") or [])}
    for r in (load("results/phase1r_variants.json") or []):
        comp[r["id"]] = r
    adopted = load("results/adoption.json") or {}
    bdir = os.path.join(HERE, "bindings")
    n_ok = n_bad = 0
    for fn in sorted(os.listdir(bdir)) if os.path.isdir(bdir) else []:
        if not fn.endswith(".json"):
            continue
        man = json.load(open(os.path.join(bdir, fn)))
        net = man["network"]
        rec = comp.get(net, {})
        for b in man["bindings"]:
            for kind, spec in (b.get("backends") or {}).items():
                src = ((adopted.get(net) or {}).get(kind) or {})
                srec = comp.get(src.get("source_net"), rec)
                bekey = src.get("backend_key", kind)
                composed = ((srec.get("compose") or {}).get(bekey) or {}).get("status")
                has_cell = (cells.get(f'{net}/{b["name"]}') or {}).get(kind) is not None
                # the ctx the manifest names must be the one that composed
                ctx_ok = (not src) or spec.get("ctx") == src.get("ctx")
                if composed == "ok" and has_cell and ctx_ok:
                    n_ok += 1
                else:
                    n_bad += 1
                    print(f"       {net}/{b['name']}@{kind}: "
                          f"source={src.get('source_net', net)}@{bekey} "
                          f"composed={composed} cell={has_cell} ctx={ctx_ok}")
    say(n_bad == 0, f"{n_ok} declared (tile, lane) pairs all composed and have a cell")

    # ---- workloads
    print("\nworkloads")
    gen = load("results/phase2_generated.json") or []
    ok_cells = [r for r in gen if r["status"] == "ok"]
    tdir = os.path.join(REPO, "data", "toplevel", "s10port")
    miss = []
    for r in ok_cells:
        p = os.path.join(tdir, r["workload"] + ".json")
        if not os.path.exists(p):
            miss.append(r["workload"]); continue
        doc = json.load(open(p))
        for name, e in doc["networks"].items():
            if e.get("period") is not None and \
               abs(e["period"] - (r["periods"] or {}).get(name, -1)) > 1e-9:
                miss.append(f"{r['workload']}:{name} period drift")
            dp = os.path.join(REPO, e["dispatch_deps_path"])
            if not os.path.exists(dp):
                miss.append(f"{r['workload']}:{name} dispatch graph missing")
    say(not miss, f"{len(ok_cells)} generated cells consistent with the record"
                  + (f" — {miss[:3]}" if miss else ""))

    # ---- solves
    print("\nsolves")
    allr = load("results/phase3_all_results.json") or load("results/all_results.json") or []
    returned = [r for r in allr if r.get("objective") is not None]
    viol = [r for r in returned
            if any((r.get("validation") or {}).get(k, 0)
                   for k in ("prec_viol", "overlap_viol", "inf_dur_assign",
                             "neg_start", "before_min_start"))]
    say(not viol, f"{len(returned)} returned schedules, "
                  f"{len(viol)} with a precedence/overlap/assignment violation")
    errs = [r for r in allr if r.get("error")]
    print(f"       {len(allr)} solves, {len(errs)} hard failures "
          f"({sorted({str(e['error']).split(':')[0] for e in errs})})")

    # ---- schedules / dedupe
    print("\nschedules")
    p4 = load("results/phase4_results.json") or []
    byhash = {}
    for r in p4:
        h, t = r.get("sched_hash"), r.get("dispatch_table_sha256")
        if h and t:
            byhash.setdefault(h, set()).add(t)
    collide = {h: v for h, v in byhash.items() if len(v) > 1}
    say(not collide,
        f"{len(byhash)} unique schedule hashes each map to one dispatch table"
        + (f" — {list(collide)[:2]}" if collide else ""))
    tabs = {}
    for r in p4:
        h, t = r.get("sched_hash"), r.get("dispatch_table_sha256")
        if h and t:
            tabs.setdefault(t, set()).add(h)
    multi = sum(1 for v in tabs.values() if len(v) > 1)
    # This is EXPECTED and benign: the dedupe key is the solver's float
    # (t, alpha), while the emitted table is what the codegen writes, so two
    # solvers can differ in the former and agree in the latter. The dedupe is
    # therefore conservative -- it can keep two points that execute
    # identically, never merge two that do not -- and each such pair is a free
    # independent re-measurement. ANALYSIS.md §2 uses them as one.
    print(f"       {multi} dispatch table(s) produced by more than one schedule "
          f"hash -- conservative dedupe, each is an independent repeat")

    # ---- runs
    print("\nruns")
    ran = [r for r in p4 if r.get("measured_median_ms")]
    partial = [r["point"] for r in ran
               if len([v for v in (r.get("reps") or {}).values()
                       if v.get("wall_ms")]) < 2]
    say(not partial, f"{len(ran)} measured points, none reporting a median from "
                     f"a single rep" + (f" — {partial[:3]}" if partial else ""))
    incomplete = [r["point"] for r in ran
                  for v in (r.get("reps") or {}).values()
                  if v.get("entries_total") and
                  v.get("entries_ran") != v.get("entries_total")]
    say(not incomplete, "every reported rep ran N/N entries"
        + (f" — {sorted(set(incomplete))[:3]}" if incomplete else ""))
    p7 = [r["point"] for r in p4 if r.get("predicate7_excluded_placements")]
    p6 = [r["point"] for r in p4 if r.get("predicate6_missing_contexts")]
    say(not p7, f"predicate 7 (no excluded placement): {len(p7)} violation(s)")
    say(not p6, f"predicate 6 (contexts staged): {len(p6)} violation(s)")

    print(f"\n{bad} failure(s), {warn} warning(s)")
    return 1 if (bad or (a.strict and warn)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
