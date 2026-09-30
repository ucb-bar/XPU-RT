#!/usr/bin/env python3
"""The cores x YOLO-service figures, re-derived from the schedules they name.

`fig_cores_yolo` is the paper's cores x YOLO figure. `scripts/cores_yolo_service.py` derives every point
from the schedules under `schedules/` and records in its sidecar where each point came from; this
checks that record against the schedules and against the renders.

What it asserts, per stem:

  * every sourced point equals the median per-frame YOLO response of the schedule it names, and
    that schedule's bytes are the ones the sidecar hashed;
  * the two published endpoint triples are reproduced -- 1.04x/0.90x/0.79x at the 24 ms anchor and
    1.07x/0.91x/0.80x at 24.5 -- so the derived figures are the published ones, not new claims;
  * every point with no schedule behind it is declared as such rather than drawn silently, and the
    count matches what the ROS mode implies;
  * the cruise anchor is recorded, because which curves fall inside the band follows from it;
  * the CP-SAT points carry the deadline their solve was given beside the value achieved, since the
    two coincide -- the solver packs to its budget -- and the curve would otherwise read as a
    measurement of how fast CP-SAT is rather than of where the deadline search stopped.

    scripts/verify_cores_yolo.py [--stem NAME ...] [-v]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

INFO = []


def info(msg):
    print("INFO  " + msg)
    INFO.append(msg)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, "results/codesign_feedback/refined")
sys.path.insert(0, os.path.join(REPO, "scripts"))
FAILED = 0

# stem -> (ros mode, anchor, the endpoint triple the render is expected to print, what it reproduces)
STEMS = {
    "cores_yolo_service_derived_published":
        ("published", 24.0, (1.04, 0.90, 0.79), "the paper's plots/fig_cores_yolo.png"),
    "cores_yolo_service_derived_published_a24p5":
        ("published", 24.5, (1.07, 0.91, 0.80), "refined/cores_yolo_service.png"),
    "cores_yolo_service_derived_flat":
        ("flat", 24.0, (1.04, 0.90, 0.79), "the same, with the one ROS measurement drawn as it is"),
    "cores_yolo_service_derived_flat_a22":
        ("flat", 22.0, (0.96, 0.82, 0.72), "the same at the 22 ms frame the paper's prose names"),
}


def check(ok, msg):
    global FAILED
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        FAILED += 1
    return ok


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", action="append", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    import cores_yolo_service as C

    for stem in (a.stem or sorted(STEMS)):
        mode, anchor, want, reproduces = STEMS[stem]
        side = os.path.join(REF, stem + "_metrics.json")
        print(f"\n=== {stem}  ({reproduces})")
        if not check(os.path.exists(side), "the figure's sidecar exists"):
            continue
        s = json.load(open(side))

        check(s.get("ros_mode") == mode, f"the ROS mode the sidecar records is {mode}")
        check(abs(float(s.get("cruise_anchor_ms", -1)) - anchor) < 1e-9,
              f"the cruise anchor is recorded, and is a choice not a measurement ({anchor:g} ms)")
        check(bool(s.get("cruise_anchor_note")),
              "the sidecar says in words that the anchor is chosen")
        check(s.get("cores") == C.K, f"the widths drawn are {C.K}")

        # every point, against the schedule it names
        prov = s.get("provenance") or []
        sourced = [p for p in prov if p.get("source")]
        unsourced = [p for p in prov if not p.get("source")]
        bad_sha = [p["source"] for p in sourced
                   if not os.path.exists(os.path.join(REPO, p["source"]))
                   or sha(os.path.join(REPO, p["source"])) != p.get("sha256")]
        check(not bad_sha, f"every schedule a point names is on disk with the bytes the sidecar "
                           f"hashed ({len(set(p['source'] for p in sourced))} schedules"
                           + (f"; {len(bad_sha)} differ: {bad_sha[0]}" if bad_sha else "") + ")")

        off = []
        for p in sourced:
            got, _, _ = C.median_response(p["source"])
            if abs(got - float(p["value_ms"])) > 0.001:
                off.append(f"{p['series']} K{p['cores']} drawn {p['value_ms']} re-derives {got:.4f}")
        check(not off, f"every sourced point is that schedule's median per-frame YOLO response "
                       f"({len(sourced)} points)" + (f" -- {off[0]}" if off else ""))

        want_unsourced = (2 if mode == "flat" else 10)
        check(len(unsourced) == want_unsourced and s.get("unsourced_points") == len(unsourced),
              f"every point with no schedule behind it is declared ({len(unsourced)}: "
              + "; ".join(sorted({p["kind"].split(";")[0] for p in unsourced})) + ")")

        # the CP-SAT rows are a deadline search, and the sidecar has to say so at every point
        cp = [p for p in sourced if p["series"].startswith("cpsat")]
        check(all("deadline_ms" in p for p in cp),
              f"every CP-SAT point carries the deadline its solve was given ({len(cp)} points)")
        packs = [p for p in cp if abs(float(p["value_ms"]) - float(p["deadline_ms"])) > 0.02]
        check(not packs, "each CP-SAT schedule packs to the deadline it was given, so the curve "
                         "traces where the search stopped"
                         + (f" ({len(packs)} do not, e.g. {packs[0]['series']} K{packs[0]['cores']})"
                            if packs else f" (within 0.02 ms at all {len(cp)})"))
        # "tightest" has to mean tightest attempted, because at some widths only one was tried
        not_tightest = [p for p in cp if p.get("tightest_attempted") is not True]
        check(not not_tightest,
              "each plotted deadline is the tightest one attempted at that width"
              + (f" ({len(not_tightest)} are not)" if not_tightest else ""))
        lone = [p for p in cp if len(p.get("deadlines_attempted") or []) < 2]
        if lone:
            info("at " + ", ".join(f"{p['series'].split('_')[1]} K{p['cores']}" for p in lone)
                 + " only one deadline was ever attempted, so nothing here says a tighter one "
                   "would have failed")

        # The series is what the figure draws; provenance is what it says each point came from.
        # Nothing tied the two together, so a render could plot one curve and record another.
        per = {}
        for e in prov:
            per.setdefault(e["series"], {})[e["cores"]] = float(e["value_ms"])
        drift = []
        for name, vals in (s.get("series") or {}).items():
            for k, v in zip(s["cores"], vals):
                recorded = per.get(name, {}).get(k)
                if recorded is None:
                    drift.append(f"{name} K{k} is drawn but has no provenance entry")
                elif abs(float(v) - recorded) > 0.001:
                    drift.append(f"{name} K{k} drawn {v} against provenance {recorded}")
        check(not drift, f"every drawn point is the value its provenance entry records "
                         f"({sum(len(v) for v in (s.get('series') or {}).values())} points)"
                         + (f" -- {drift[0]}" if drift else ""))

        # the per-point record: the width it belongs to, and how many instances the median is over
        badk = [e for e in prov if e["cores"] not in C.K]
        check(not badk, f"every point's width is one the figure draws ({len(prov)} points)")
        ordered = all(list(per[n].keys()) == sorted(per[n]) or set(per[n]) == set(C.K) for n in per)
        check(ordered, "each series has one point per width")
        badn = []
        for e in sourced:
            _, n, _ = C.median_response(e["source"])
            if e.get("instances") != n:
                badn.append(f"{e['series']} K{e['cores']} records {e.get('instances')} instances, "
                            f"the schedule has {n}")
        check(not badn, f"every point's instance count is the schedule's ({len(sourced)} points)"
                        + (f" -- {badn[0]}" if badn else ""))

        # the deadline search each CP-SAT point is the tightest of
        badsearch = []
        for e in cp:
            cost = "AOT" if e["series"].endswith("sched") else "board"
            on_disk = C.searched(e["cores"], cost)
            if sorted(e.get("deadlines_attempted") or []) != on_disk:
                badsearch.append(f"{e['series']} K{e['cores']}: records "
                                 f"{e.get('deadlines_attempted')}, on disk {on_disk}")
        check(not badsearch, f"the recorded deadline search is the one on disk ({len(cp)} points)"
                             + (f" -- {badsearch[0]}" if badsearch else ""))

        # the endpoint triple the figure prints
        got3 = tuple(round(anchor / s["series"][k][-1], 2)
                     for k in ("cpsat_board", "greedy_board", "ros_board"))
        check(got3 == want, f"the cruise labels at 8 cores are the published ones "
                            f"({' / '.join(f'{v:.2f}×' for v in got3)})")

        rec_cruise = s.get("cruise_at_8_cores") or {}
        bad_c = [k for k, v in rec_cruise.items()
                 if abs(anchor / s["series"][k][-1] - float(v)) > 5e-4]
        check(not bad_c, f"the recorded cruise at 8 cores follows from the series and the anchor "
                         f"({len(rec_cruise)} arms)" + (f"; {bad_c[0]} differs" if bad_c else ""))
        bad_m = []
        for e in cp:
            mt = os.path.join(REPO, e["source"].replace(".json", "_metrics.json"))
            if os.path.exists(mt):
                want_m = json.load(open(mt)).get("deadline_miss_count")
                if e.get("solver_deadline_miss_count") != want_m:
                    bad_m.append(f"{e['series']} K{e['cores']}: records "
                                 f"{e.get('solver_deadline_miss_count')}, solve reports {want_m}")
        check(not bad_m, f"each point's recorded miss count is its solve's ({len(cp)} points)"
                         + (f" -- {bad_m[0]}" if bad_m else ""))

        # the ROS arm has one schedule, whichever mode is drawn
        ros = {p["source"] for p in sourced if p["series"].startswith("ros")}
        check(len(ros) == 2 and all("ros_pin" in r for r in ros),
              f"the ROS curve rests on the one 1-hart pin pair, not a sweep ({len(ros)} schedules)")
        if mode == "flat":
            flat = all(len(set(round(v, 4) for v in s["series"][k])) == 1
                       for k in ("ros_sched", "ros_board"))
            check(flat, f"drawn flat, it is the same value at every width "
                        f"({s['series']['ros_board'][-1]:.2f} ms)")

        for ext in (".png", ".pdf"):
            check(os.path.exists(os.path.join(REF, stem + ext)), f"the {ext[1:]} is on disk")

    print(f"\n{FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
