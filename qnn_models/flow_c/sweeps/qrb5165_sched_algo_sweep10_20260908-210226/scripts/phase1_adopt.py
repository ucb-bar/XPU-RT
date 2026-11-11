#!/usr/bin/env python3
"""Phase 1A — adopt a rewrite, or a precision, only where it MEASURES better.

Every candidate this sweep built is now measured: 16 base networks whole, 12
rewritten variants, and both CPU precisions for all of them. This step makes
the two decisions the study's methodology says are per-tile, and makes them on
the numbers:

  which GRAPH   base, or the rewritten variant that removed a backend blocker
  which CPU     fp32 or int8 -- "precision is a per-tile decision"; the
    PRECISION   slicing study found int8 beats fp32 by 4x on ViNT's encoders
                and loses by 5x on dronet, and this sweep's own audit cells
                reproduce that split on its own networks

Two rules that are not negotiable:

  * **A numerics-CHANGING variant is never adopted.** `mlp_control_sf_htaprobe`
    substitutes Relu for Elu, which is a different function, not an algebraic
    rewrite. It is built and measured so the question "would HTA ever be worth
    it for the control loop?" is answered by measurement, and its numbers are
    reported in the ledger under that label -- but it does not enter the cost
    model the schedules are solved against.
  * **A rewrite that composes somewhere new is not thereby better.** The
    per-lane winner is the smaller measured cell. A rewrite that unlocks HTA
    and is slower than the base everywhere still contributes exactly one thing:
    an HTA cell that did not exist.

Output:
  bindings/<net>.json     rewritten to name the winning context per lane
  results/adoption.json   per (network, lane): who won, by how much, and why
  results/REWRITE_LEDGER.md  the deliverable table

    python3 phase1_adopt.py [--write]
"""
from __future__ import annotations

import argparse, json, os, shutil, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(SWEEP, "measurements", "qrb5165_v66_s10port_raw.json")
BDIR = os.path.join(SWEEP, "bindings")
CAND = os.path.join(SWEEP, "bindings_candidates")
LANES = ["hta", "dsp", "cpu", "gpu"]


def cells_from_raw():
    """{(cell, backend_key): (median_us, [per-pass])} over the passes."""
    by = {}
    for r in json.load(open(RAW))["results"]:
        if r.get("status") == "ok":
            by.setdefault((r["cell"], r["backend"]), []).append(
                float(r["gap_median_us"]))
    return {k: (round(st.median(v), 1), [round(x, 1) for x in v])
            for k, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    base_rows = {r["id"]: r for r in json.load(open(
        os.path.join(SWEEP, "results", "phase1_compose.json")))}
    vp = os.path.join(SWEEP, "results", "phase1r_variants.json")
    variants = json.load(open(vp)) if os.path.exists(vp) else []
    by_base = {}
    for v in variants:
        by_base.setdefault(v["base"], []).append(v)
    meas = cells_from_raw()

    adoption, ledger = {}, []
    for net, brow in base_rows.items():
        sources = [("base", brow, "preserving")] + [
            (v["id"], v, v.get("numerics", "preserving")) for v in by_base.get(net, [])]
        per_lane = {}
        for lane in LANES:
            cands = []
            for sname, row, numerics in sources:
                sid = row["id"]
                cell = f"{sid}/{sid}_full"
                keys = [lane] + (["cpu@int8"] if lane == "cpu" else [])
                for k in keys:
                    c = (row.get("compose") or {}).get(k) or {}
                    if c.get("status") != "ok":
                        continue
                    m = meas.get((cell, k))
                    if not m:
                        continue
                    cands.append(dict(
                        source=sname, source_net=sid, backend_key=k,
                        precision=c.get("precision"), us=m[0], passes=m[1],
                        ctx=c["ctx"], graph=row["dlc_graph_name"],
                        numerics=numerics,
                        rewrite_chain=row.get("rewrite_chain")))
            usable = [c for c in cands if c["numerics"] == "preserving"]
            rejected = [c for c in cands if c["numerics"] != "preserving"]
            if usable:
                win = min(usable, key=lambda x: x["us"])
                per_lane[lane] = dict(win, candidates=cands)
            ledger.append(dict(
                network=net, lane=lane,
                blocker=((brow.get("compose") or {}).get(lane) or {}).get("reason")
                if ((brow.get("compose") or {}).get(lane) or {}).get("status") != "ok"
                else None,
                base_us=next((c["us"] for c in cands if c["source"] == "base"
                              and c["backend_key"] == lane), None),
                base_cpu_int8_us=next((c["us"] for c in cands
                                       if c["source"] == "base"
                                       and c["backend_key"] == "cpu@int8"), None),
                variant_us=next((c["us"] for c in cands if c["source"] != "base"
                                 and c["backend_key"] == lane
                                 and c["numerics"] == "preserving"), None),
                numerics_changing_us=next((c["us"] for c in rejected
                                           if c["backend_key"] == lane), None),
                adopted=(per_lane.get(lane) or {}).get("source"),
                adopted_precision=(per_lane.get(lane) or {}).get("precision"),
                adopted_us=(per_lane.get(lane) or {}).get("us")))
        adoption[net] = per_lane

    # --- rewrite the base manifests to name the winners
    if a.write:
        os.makedirs(CAND, exist_ok=True)
        for v in variants:
            src = os.path.join(BDIR, f"{v['id']}.json")
            if os.path.exists(src):
                shutil.move(src, os.path.join(CAND, f"{v['id']}.json"))
        for net, per_lane in adoption.items():
            p = os.path.join(BDIR, f"{net}.json")
            doc = json.load(open(p))
            b = doc["bindings"][0]
            b["backends"] = {
                lane: {"ctx": w["ctx"], "graph": w["graph"],
                       "precision": w["precision"], "from": w["source"],
                       "measured_us": w["us"]}
                for lane, w in sorted(per_lane.items())}
            adopted_variant = sorted({w["source"] for w in per_lane.values()
                                      if w["source"] != "base"})
            if adopted_variant:
                v = next(x for x in variants if x["id"] == adopted_variant[0])
                doc["ir"] = {"source": f"graph_json:{os.path.join(SWEEP, v['graph_json'])}",
                             "quant": "int8"}
                b["source_onnx"] = v["onnx"]
                doc["_rewrite_note"] = (
                    f"Lane(s) {[l for l, w in per_lane.items() if w['source'] != 'base']} "
                    f"come from the rewritten variant {adopted_variant[0]} "
                    f"(chain {v['rewrite_chain']}, NUMERICS-{v['numerics'].upper()}), "
                    f"adopted because it measured better on that lane. "
                    f"{v['rationale']}")
            doc["_precision_note"] = (
                "The CPU lane's precision is chosen per tile on measured "
                "evidence, not per network: " + ", ".join(
                    f"{l}={w['precision']}" for l, w in sorted(per_lane.items())))
            json.dump(doc, open(p, "w"), indent=1)
        json.dump(adoption, open(os.path.join(SWEEP, "results",
                                              "adoption.json"), "w"), indent=1)
        json.dump(ledger, open(os.path.join(SWEEP, "results",
                                            "rewrite_ledger.json"), "w"), indent=1)

    # --- report
    def ms(v):
        return f"{v/1000:.3f}" if v else "—"

    print(f"{'network':<20}{'lane':<5}{'base':>9}{'variant':>9}{'adopted':>9}"
          f"  {'from':<16}{'prec':<6} blocker the rewrite removed")
    print("-" * 112)
    for r in ledger:
        bl = (r["blocker"] or "").split("]")[-1].strip()[:32]
        print(f"{r['network']:<20}{r['lane']:<5}{ms(r['base_us']):>9}"
              f"{ms(r['variant_us']):>9}{ms(r['adopted_us']):>9}"
              f"  {str(r['adopted'] or '—'):<16}"
              f"{str(r['adopted_precision'] or '—'):<6} {bl}")
    n_new = sum(1 for r in ledger if r["base_us"] is None and r["adopted_us"])
    n_rw = sum(1 for r in ledger if r["adopted"] not in (None, "base"))
    print(f"\n  {n_new} lane(s) that did not exist before a rewrite; "
          f"{n_rw} lane(s) adopted from a rewritten variant")
    print("  " + ("written" if a.write else "dry run — pass --write"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
