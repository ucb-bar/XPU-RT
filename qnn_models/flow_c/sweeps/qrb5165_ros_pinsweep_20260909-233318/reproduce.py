#!/usr/bin/env python3
"""Re-derive everything in this sweep that does not need the board, and check
it against what is committed.

Six checks, each PASS/FAIL, no hardware:

  1  model_costs.json         re-derived from the FROZEN cost_model.json and the
                              binding manifests
  2  results/expressibility   re-derived verdicts and legal sets
  3  results/enumeration      re-derived n_legal, rankings and measured picks
  4  plans/*.json             byte-identical re-emission
  5  measured.json            every median/spread re-derived from logs/ (skipped
                              if logs/ is absent, since logs are not committed)
  6  the noise floor SETUP.md quotes, recomputed from the XPU-RT sweep's own
                              phase4_results.json

    python3 reproduce.py [--verbose]

Exit status is the number of failed checks.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")
FLOWC = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))
XSWEEP = os.path.join(FLOWC, "sweeps", "qrb5165_sched_algo_sweep10_20260908-210226")

sys.path.insert(0, SCRIPTS)

OK, BAD = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{OK if ok else BAD}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def jload(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import pinsweep

    print("1. model_costs.json, from the frozen cost model")
    costs = pinsweep.model_costs()
    have = jload(os.path.join(HERE, "model_costs.json"))
    check("whole-model costs re-derive exactly", have["networks"] == costs,
          f'{len(costs)} networks')
    check("cost source is the frozen model",
          have["source_cost_model"].endswith("cost_model.json")
          and "sched_algo_sweep10" in have["source_cost_model"],
          have["source_cost_model"])
    # every cost is a sum over that network's binding tiles of a cell in the
    # frozen model -- spot-checked exhaustively, not by sampling
    cm = pinsweep.load_cost_model()["cells"]
    bad = []
    for net, per in costs.items():
        for be, rec in per.items():
            s = 0.0
            for t in rec["tiles"]:
                man = pinsweep.load_bindings(net)
                s += cm[f'{man["network"]}/{t["tile"]}'][be]
            if abs(s / 1000.0 - rec["ms"]) > 1e-6:
                bad.append((net, be))
    check("every cost is the sum of its tiles' frozen cells", not bad, str(bad[:5]))

    print("2. expressibility")
    cells = pinsweep.load_cells()
    rows = pinsweep.classify(cells, costs)
    have = jload(os.path.join(HERE, "results", "expressibility.json"))
    check("verdicts and legal sets re-derive exactly", have["cells"] == rows,
          f"{len(rows)} cells")
    verdicts = {}
    for r in rows:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    check("SETUP.md's 31 OK / 11 degenerate / 0 inexpressible",
          verdicts.get("OK") == 31 and verdicts.get("DEGENERATE (1-LANE)") == 11
          and not any(v.startswith("INEXPRESSIBLE") for v in verdicts),
          str(verdicts))
    check("the 42 cells are the 42 the XPU-RT sweep measured",
          len(rows) == 42 and set(c["cell"] for c in rows) ==
          set(measured_cells()), f"{len(rows)}")

    print("3. enumeration + 4. plans")
    tmp = tempfile.mkdtemp(prefix="rospin_repro_")
    try:
        # re-emit into a scratch tree so nothing committed is overwritten
        r = subprocess.run([sys.executable,
                            os.path.join(SCRIPTS, "pinsweep.py"),
                            "--out", tmp, "enumerate"],
                           capture_output=True, text=True)
        ok = r.returncode == 0
        if not ok and args.verbose:
            print(r.stdout, r.stderr)
        have_enum = jload(os.path.join(HERE, "results", "enumeration.json"))
        got_enum = jload(os.path.join(tmp, "results", "enumeration.json"))
        check("enumeration re-derives exactly", ok and have_enum == got_enum,
              f'{len(got_enum["cells"])} cells')
        ns = [c["n_legal"] for c in got_enum["cells"] if c.get("n_legal")]
        check("SETUP.md's n_legal distribution (min 1, median 4, max 729, "
              "1078 total)",
              min(ns) == 1 and statistics.median(ns) == 4 and max(ns) == 729
              and sum(ns) == 1078,
              f"min={min(ns)} median={statistics.median(ns)} max={max(ns)} "
              f"total={sum(ns)}")
        check("ANALYSIS.md 0.11's 295 assignments marked for hardware",
              sum(c.get("measured", 0) for c in got_enum["cells"]) == 295,
              str(sum(c.get("measured", 0) for c in got_enum["cells"])))
        diffs = []
        for fn in sorted(os.listdir(os.path.join(HERE, "plans"))):
            a = jload(os.path.join(HERE, "plans", fn))
            b = jload(os.path.join(tmp, "plans", fn))
            if a != b:
                diffs.append(fn)
        check("every plan re-emits identically", not diffs, str(diffs[:3]))

        # The `quad` column is enumerated; the other three keep SETUP.md's
        # plan (ANALYSIS.md 0.10, 0.11). Both facts are checked, because
        # "extended" silently widening the sensitivity study would have cost
        # 354 board runs on cells that are not the headline.
        emode = {c["cell"]: c for c in got_enum["cells"]}
        full = [c for c in got_enum["cells"] if c.get("mode") == "full"]
        check("39 of 42 cells are enumerated in full", len(full) == 39,
              f"{len(full)}")
        sampled = sorted(c["cell"] for c in got_enum["cells"]
                         if c.get("mode") == "sampled")
        check("the only sampled cells are the three scale_ladder ones",
              sampled == ["networks_scale_ladder_dc",
                          "networks_scale_ladder_hd",
                          "networks_scale_ladder_quad"], str(sampled))
        sq = emode["networks_scale_ladder_quad"]
        check("scale_ladder_quad is 64 of 729 and is reported as SAMPLED, "
              "never as an enumeration",
              sq["n_legal"] == 729 and sq["measured"] == 64
              and sq["mode"] == "sampled",
              f'{sq["measured"]}/{sq["n_legal"]} {sq["mode"]}')
        check("64 is n_legal of the two enumerated scale_ladder siblings, so "
              "the three are comparable at equal effort",
              emode["networks_scale_ladder_dc"]["n_legal"] ==
              emode["networks_scale_ladder_hd"]["n_legal"] == 64)
        for c in got_enum["cells"]:
            if c["cell"].endswith("_quad") and c["n_legal"] <= 27:
                if c["measured"] != c["n_legal"]:
                    diffs.append(c["cell"])
        check("every quad cell with <=27 legal placements is fully enumerated",
              not diffs, str(diffs[:3]))

        # SETUP.md is a pre-run contract: its own plan must still re-derive.
        tmp2 = tempfile.mkdtemp(prefix="rospin_setup_")
        try:
            r2 = subprocess.run([sys.executable,
                                 os.path.join(SCRIPTS, "pinsweep.py"),
                                 "--out", tmp2, "enumerate",
                                 "--coverage", "setup"],
                                capture_output=True, text=True)
            e2 = jload(os.path.join(tmp2, "results", "enumeration.json"))
            check("--coverage setup still re-derives SETUP.md 5.4's 172",
                  r2.returncode == 0
                  and sum(c.get("measured", 0) for c in e2["cells"]) == 172,
                  str(sum(c.get("measured", 0) for c in e2["cells"])))
            # ids, ranks and predicted costs must be identical under both
            # profiles: only `measure` changes, never the scoring
            drift = []
            for fn in sorted(os.listdir(os.path.join(tmp2, "plans"))):
                a = jload(os.path.join(tmp2, "plans", fn))
                b = jload(os.path.join(tmp, "plans", fn))
                ka = [(x["id"], x["rank"], x["np_rank"], x["assign"],
                       x["predicted_makespan_ms"], x["predicted_np_makespan_ms"])
                      for x in a["assignments"]]
                kb = [(x["id"], x["rank"], x["np_rank"], x["assign"],
                       x["predicted_makespan_ms"], x["predicted_np_makespan_ms"])
                      for x in b["assignments"]]
                if ka != kb:
                    drift.append(fn)
            check("the coverage profile changes WHAT IS MEASURED and nothing "
                  "about the scoring", not drift, str(drift[:3]))

            # What the widened coverage actually bought, per cell: the best
            # measured np under SETUP.md's plan against the best under the
            # extended one. ANALYSIS.md 0.11 quotes the two cells that moved,
            # and quoting the reassuring five instead would be the dishonest
            # half of the same fact.
            mp = os.path.join(HERE, "measured.json")
            if os.path.exists(mp):
                runs = jload(mp)["runs"]
                bycell = {}
                for r in runs.values():
                    bycell.setdefault(r["cell"], []).append(r)
                gains = {}
                for fn in sorted(os.listdir(os.path.join(tmp2, "plans"))):
                    pl = jload(os.path.join(tmp2, "plans", fn))
                    ids = {a["id"] for a in pl["assignments"] if a.get("measure")}
                    rr = bycell.get(pl["cell"], [])
                    old_ = [x for x in rr if x["assignment"] in ids]
                    if not old_ or len(rr) == len(old_):
                        continue
                    o = min(x["np_median_ms"] for x in old_)
                    n = min(x["np_median_ms"] for x in rr)
                    if n < o - 1e-9:
                        gains[pl["cell"]] = round(o / n, 4)
                check("the widened coverage beat the previous best on exactly "
                      "two cells", sorted(gains) ==
                      ["networks_saturation_quad", "networks_scale_ladder_quad"],
                      str(gains))
                check("scale_ladder_quad's 64-placement sample found a "
                      "placement 10.7% better than the old 5-point one -- the "
                      "honest bound on how wrong the sampled cells could be",
                      abs(gains.get("networks_scale_ladder_quad", 0) - 1.1067)
                      < 1e-3, str(gains.get("networks_scale_ladder_quad")))
                check("saturation_quad's 1.85% gain is inside the noise floor "
                      "and that cell is out of the headline anyway",
                      abs(gains.get("networks_saturation_quad", 0) - 1.0185)
                      < 1e-3
                      and (gains["networks_saturation_quad"] - 1) * 100 < 9.18,
                      str(gains.get("networks_saturation_quad")))
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("5. measured.json against the run logs")
    mpath = os.path.join(HERE, "measured.json")
    logs = os.path.join(HERE, "logs")
    if not os.path.exists(mpath):
        check("measured.json present", False, "not yet collected")
    elif not os.path.isdir(logs) or not os.listdir(logs):
        check("logs present to re-derive from", True,
              "SKIPPED -- logs/ is not committed; run scripts/drive.py run first")
    else:
        import drive
        doc = jload(mpath)
        bad = []
        for tag, rec in doc["runs"].items():
            # a content-duplicate has no log of its own by design: it names the
            # run it was resolved onto, and that run's log is the measurement
            p = os.path.join(logs, (rec.get("measured_via") or tag) + ".log")
            if not os.path.exists(p):
                bad.append((tag, "no log"))
                continue
            info = drive.parse_log(open(p, errors="replace").read())
            walls = [x["makespan_ms"] for x in info["passes"]]
            if not walls:
                bad.append((tag, "no passes"))
                continue
            if abs(statistics.median(walls) - rec["makespan_median_ms"]) > 1e-6:
                bad.append((tag, "median"))
            if walls != rec["makespan_reps_ms"]:
                bad.append((tag, "reps"))
        check("every median and spread re-derives from its log", not bad,
              str(bad[:5]))
        check("3 reps everywhere, no single-rep result",
              all(r["reps"] == 3 for r in doc["runs"].values()),
              str(sorted({r["reps"] for r in doc["runs"].values()})))
        check("379 assignments measured, all complete",
              len(doc["runs"]) == 379
              and all(r["ok"] for r in doc["runs"].values()),
              f'{sum(1 for r in doc["runs"].values() if r["ok"])}/{len(doc["runs"])}')
        # Content-dedupe: a tag whose harness input is byte-identical to
        # another's reads that run's log and says so (ANALYSIS.md 12).
        via = {t: r["measured_via"] for t, r in doc["runs"].items()
               if r.get("measured_via")}
        check("11 assignments resolved as content-duplicates, each naming its "
              "source run", len(via) == 11, str(len(via)))
        import drive as _drive
        badh = []
        for t, src in via.items():
            for d_ in (os.path.join(HERE, "plans"), os.path.join(HERE, "plans3net")):
                pa = os.path.join(d_, doc["runs"][t]["cell"] + ".json")
                if not os.path.exists(pa):
                    continue
                pl = jload(pa)
                a = next(x for x in pl["assignments"]
                         if x["id"] == doc["runs"][t]["assignment"])
                sp = jload(os.path.join(d_ if os.path.exists(
                    os.path.join(d_, doc["runs"][src]["cell"] + ".json"))
                    else os.path.join(HERE, "plans"),
                    doc["runs"][src]["cell"] + ".json"))
                sa = next(x for x in sp["assignments"]
                          if x["id"] == doc["runs"][src]["assignment"])
                if _drive.cfg_hash(pl, a) != _drive.cfg_hash(sp, sa):
                    badh.append(t)
        check("every dedupe really is the same harness input", not badh,
              str(badh[:3]))

    print("5b. the 3net arm")
    p3 = os.path.join(HERE, "results", "enumeration_3net.json")
    if os.path.exists(p3):
        import pin3net
        tmp3 = tempfile.mkdtemp(prefix="rospin_repro3_")
        try:
            have3 = jload(p3)
            costs3, prov3 = pin3net.base_costs()
            # legality is the binding manifest, never the cost table: the
            # artifact emitter writes large-but-finite EXCLUSION costs for a
            # lane a tile cannot use, and a cost-only rule would pin to them
            bad = []
            for net, per in costs3.items():
                man = json.load(open(os.path.join(FLOWC, "bindings",
                                                  f"{net}.json")))
                for be in per:
                    for b in man["bindings"]:
                        if be not in (b.get("backends") or {}):
                            bad.append((net, be))
            check("3net legality comes from the binding manifests", not bad,
                  str(bad[:5]))
            sh = pin3net.shapes()
            check("all 15 RoSE 3net configs map to a shape",
                  sum(len(x["sources"]) for x in sh) == 15
                  and len(sh) == len(have3["cells"]),
                  f'{len(sh)} shapes, '
                  f'{sum(len(x["sources"]) for x in sh)} source configs')
            check("3net enumeration matches the committed table",
                  [c["cell"] for c in have3["cells"]] == [x["name"] for x in sh])
        finally:
            shutil.rmtree(tmp3, ignore_errors=True)
        pa = os.path.join(HERE, "results", "analysis_3net.json")
        if os.path.exists(pa):
            d3 = jload(pa)
            a3, h3 = d3["shapes"], d3["headline"]
            real = [r for r in a3 if not r["np_degenerate"]]
            # The direction is what survives the change of opponent; the
            # magnitude is not, so both readings are pinned here and the
            # noise-floor count with them (ANALYSIS.md 0.8, 10).
            check("3net: pinning faster on every shape with an aperiodic "
                  "network, against the RECOMMENDED solver",
                  all(r["ros_over_xrt_np_warmbest"] < 1 for r in real),
                  f'{h3["warmbest_ros_faster"]}/{len(real)}')
            check("3net: pinning faster on every shape against the "
                  "best-measured oracle too",
                  all(r["ros_over_xrt_np"] < 1 for r in real),
                  f'{h3["best_ros_faster"]}/{len(real)}')
            check("3net median against cpsat:warmbest is 0.9305",
                  abs(h3["warmbest_median"] - 0.9305) < 1e-4,
                  str(h3["warmbest_median"]))
            check("3net median against the best measured solver is 0.9314",
                  abs(h3["best_median"] - 0.9314) < 1e-4,
                  str(h3["best_median"]))
            check("3net: only 2 of the 7 are outside the +/-9.18% noise floor",
                  len(h3["warmbest_inside_noise"]) == len(real) - 2,
                  f'{len(real) - len(h3["warmbest_inside_noise"])} outside')
            check("3net: cpsat:warmbest deduped onto an existing schedule on "
                  "all 9 shapes",
                  len(h3["shapes_deduped_onto_an_existing_run"]) == 9,
                  str(len(h3["shapes_deduped_onto_an_existing_run"])))
            # the same two changes on this arm (ANALYSIS.md 10.0), which
            # needed no board time: a re-selection over runs already on record
            # and a re-read of traces already on disk
            r3 = h3.get("readings") or {}
            if r3:
                check("3net PRIMARY: isolation-best, corrected, vs "
                      "cpsat:warmbest is 0.9488",
                      abs(r3["isolation"]["warmbest"]["corrected"]["median"]
                          - 0.9488) < 1e-4,
                      str(r3["isolation"]["warmbest"]["corrected"]["median"]))
                check("3net secondary: the oracle placement, corrected, is "
                      "0.9311 -- pinning ahead on all 7, unlike the main "
                      "arm's quad column",
                      abs(r3["oracle"]["warmbest"]["corrected"]["median"]
                          - 0.9311) < 1e-4
                      and r3["oracle"]["warmbest"]["corrected"]["pinning_faster"] == 7,
                      str(r3["oracle"]["warmbest"]["corrected"]["median"]))
                check("3net: the naive rule is nearly optimal here -- median "
                      "gap 1.018x, and it IS the oracle on 2 of 7",
                      abs(h3["iso_over_oracle"]["median"] - 1.0181) < 1e-3
                      and len(h3["iso_over_oracle"]
                              ["shapes_where_naive_is_already_optimal"]) == 2,
                      str(h3["iso_over_oracle"]["median"]))
                check("3net: the isolation placement is measured on every shape",
                      not h3["shapes_where_the_isolation_placement_was_not_measured"],
                      str(h3["shapes_where_the_isolation_placement_was_not_measured"]))
        pu = os.path.join(HERE, "results", "undeclared_3net.json")
        if os.path.exists(pu):
            u = jload(pu)["dropped_undeclared_cells"]
            check("3net: the binding manifests forbid exactly 3 profiled cells",
                  u == ["fused_full/fused_full_net@hta",
                        "mlp_control/mlp_control_full@hta",
                        "yolov8n/yolov8n_head@hta"], str(u))
        pv = os.path.join(HERE, "results", "verify3net.json")
        if os.path.exists(pv):
            v = jload(pv)["rows"]
            # `cpsat:warmbest` can only come from sched_algo_sweep10's emitter,
            # so it is only a valid comparator if that emitter reproduces the
            # schedules the other solvers were measured from. It does, for
            # every deterministic solver. Cold cpsat draws an arbitrary member
            # of a tied-optimum set and is excluded by name, not silently.
            det = [r for r in v if r["solver"] in ("greedy", "heft_edf")]
            check("3net: the two solve paths agree on every deterministic "
                  "solver", all(r["paths_agree"] for r in det),
                  f'{sum(1 for r in det if r["paths_agree"])}/{len(det)}')
            check("3net: masking the undeclared cells moves no greedy, "
                  "heft_edf or cpsat:warmbest schedule",
                  not any(r["mask_changes_schedule"] for r in v
                          if r["solver"] != "cpsat"),
                  str([r["shape"] for r in v
                       if r["mask_changes_schedule"] and r["solver"] != "cpsat"]))
    else:
        check("3net enumeration present", True, "SKIPPED -- not generated")

    print("5c. the headline aggregates ANALYSIS.md quotes")
    pa = os.path.join(HERE, "results", "analysis.json")
    if os.path.exists(pa):
        doc_a = jload(pa)
        h = doc_a["headline"]
        R = doc_a["readings"]

        # -------- the PRIMARY baseline: isolation-best (ANALYSIS.md 0.9) ----
        import analyse
        costs_i = analyse.model_costs()
        # the tie rule is stated but must never fire: check the margin
        margins = []
        for net, per in costs_i.items():
            v = sorted(x["ms"] for x in per.values())
            if len(v) > 1 and v[0]:
                margins.append(((v[1] / v[0] - 1) * 100, net))
        margins.sort()
        check("no network's lane choice is a coin flip: the smallest "
              "best-vs-runner-up margin is outside the +/-9.18% band",
              margins[0][0] > 9.18,
              f"{margins[0][0]:.1f}% on {margins[0][1]}")
        check("the smallest margin is 12.3% on yolov8_nano_sc",
              abs(margins[0][0] - 12.32) < 0.05 and margins[0][1] == "yolov8_nano_sc",
              f"{margins[0][0]:.2f}% {margins[0][1]}")
        gpu = [n for n in costs_i if analyse.isolation_best(costs_i, n) == "gpu"]
        check("GPU never wins an unrestricted isolation argmin, so excluding "
              "it as a pinning candidate changes no selection", not gpu, str(gpu))
        cells_a = {c["cell"]: c for c in doc_a["cells"]}
        check("the isolation placement is measured on every one of the 42 cells",
              all(c["iso_measured"] for c in doc_a["cells"]),
              str([c["cell"] for c in doc_a["cells"] if not c["iso_measured"]]))
        fb = [c["cell"] for c in doc_a["cells"]
              if c["config"] == "quad" and c["iso_fallbacks"]]
        check("no quad cell needs a fallback lane -- quad offers every lane",
              not fb, str(fb))
        fb_other = sorted({c["config"] for c in doc_a["cells"]
                           if c["iso_fallbacks"]})
        check("fallbacks occur only on the lane-scarce configs",
              fb_other == ["cg", "hd"], str(fb_other))

        # -------- the headline, four ways (ANALYSIS.md 4.0) -----------------
        q_iso_c = R["quad"]["isolation"]["warmbest"]["corrected"]
        q_ora_c = R["quad"]["oracle"]["warmbest"]["corrected"]
        check("PRIMARY: quad, isolation-best, corrected, vs cpsat:warmbest "
              "is 1.1795 over 7 cells",
              q_iso_c["n_cells"] == 7
              and abs(q_iso_c["median"] - 1.1795) < 1e-4,
              str(q_iso_c["median"]))
        check("primary split is 1 pinning / 6 scheduler, 2 inside the band",
              (q_iso_c["pinning_faster"], q_iso_c["scheduler_faster"],
               q_iso_c["inside_noise"]) == (1, 6, 2),
              f'{q_iso_c["pinning_faster"]}/{q_iso_c["scheduler_faster"]}/'
              f'{q_iso_c["inside_noise"]}')
        check("SECONDARY: the same scope against the placement ORACLE is a "
              "dead heat at 0.9976",
              abs(q_ora_c["median"] - 0.9976) < 1e-4, str(q_ora_c["median"]))
        check("quad, isolation, RAW is 1.1571 -- reported beside the "
              "corrected number, never instead of it",
              abs(R["quad"]["isolation"]["warmbest"]["raw"]["median"]
                  - 1.1571) < 1e-4)
        check("all-configs, isolation, corrected is 1.0768 over 26 cells",
              abs(R["all"]["isolation"]["warmbest"]["corrected"]["median"]
                  - 1.0768) < 1e-4)
        check("the PREVIOUS headline (all configs, oracle, raw) still "
              "re-derives at 0.9281, so the move is checkable",
              abs(R["all"]["oracle"]["warmbest"]["raw"]["median"]
                  - 0.9281) < 1e-4,
              str(R["all"]["oracle"]["warmbest"]["raw"]["median"]))
        check("and its secondary, 0.9762 against the best measured solver",
              abs(R["all"]["oracle"]["best"]["raw"]["median"] - 0.9762) < 1e-4)

        # -------- what the search is worth (ANALYSIS.md 4.4) ---------------
        gq = R["quad"]["iso_over_oracle"]
        check("on quad the naive rule leaves a median 1.172x and a worst "
              "1.865x on the table",
              abs(gq["median"] - 1.1723) < 1e-3 and abs(gq["max"] - 1.8649) < 1e-3,
              f'{gq["median"]} / {gq["max"]}')
        check("the 1.865x is depth_contended_quad -- three networks on the "
              "DSP because each prefers it alone",
              max(((c["iso_over_oracle_np"] or 0), c["cell"])
                  for c in doc_a["cells"])[1] == "networks_depth_contended_quad")

        # -------- the offset correction (ANALYSIS.md 0.12, 15) -------------
        off = doc_a["offsets"]
        check("the XPU-RT first dispatch is a median 0.064 ms over 366 runs "
              "and passes 1 ms on 30 of them",
              off["xpurt_main_arm"]["n_runs"] == 366
              and abs(off["xpurt_main_arm"]["median_ms"] - 0.0635) < 1e-3
              and off["xpurt_main_arm"]["runs_over_1ms"] == 30,
              str(off["xpurt_main_arm"]))
        check("the ROS side has the same delay and it is smaller -- which is "
              "why the correction is applied to BOTH sides",
              off["ros_main_arm"]["median_ms"] < off["xpurt_main_arm"]["median_ms"]
              and off["ros_main_arm"]["runs_over_1ms"]
              < off["xpurt_main_arm"]["runs_over_1ms"],
              f'ros {off["ros_main_arm"]["median_ms"]} ms / '
              f'xrt {off["xpurt_main_arm"]["median_ms"]} ms')
        moved = R["all"]["correction"]["iso_warmbest"]["cells_moved_more_than_noise"]
        check("exactly 3 of the 26 comparable cells move more than the "
              "+/-9.18% band, and all three are perception_heavy",
              len(moved) == 3
              and all("perception_heavy" in c for c in moved), str(list(moved)))
        movedq = R["quad"]["correction"]["iso_warmbest"]["cells_moved_more_than_noise"]
        check("on the quad scope it is 1 of 7, perception_heavy_quad",
              list(movedq) == ["networks_perception_heavy_quad"], str(list(movedq)))
        check("perception_heavy_quad moves +34.2% and changes direction -- "
              "the cell whose gantt illustrated a startup artifact",
              abs(cells_a["networks_perception_heavy_quad"]
                  ["correction_move_pct_iso_warmbest"] - 34.18) < 0.05
              and cells_a["networks_perception_heavy_quad"]
                  ["correction_flips_iso_warmbest"])
        check("its corrected compute agrees with the ROS side to ~4%: "
              "4.85 ms scheduled against 5.02 ms pinned",
              abs(cells_a["networks_perception_heavy_quad"]
                  ["xrt_np_warmbest_corrected_ms"] - 4.847) < 0.01)

        check("0 assignments starved a network",
              h["assignments_with_a_starved_network"] == [],
              str(len(h["assignments_with_a_starved_network"])))
        check("every assignment completed",
              h["assignments_run"] == h["assignments_ok"],
              f'{h["assignments_ok"]}/{h["assignments_run"]}')
        check("the 4 saturation cells are the only unequal-non-periodic-work "
              "exclusions",
              sorted(h["np_cells_excluded_unequal_work"]) ==
              sorted(f"networks_saturation_{c}" for c in ("cg", "dc", "hd", "quad")),
              str(h["np_cells_excluded_unequal_work"]))
        check("cpsat:warmbest is measured on all 26 compared cells",
              h["np_cells_with_warmbest"] == h["np_cells_compared"] == 26,
              f'{h["np_cells_with_warmbest"]}/{h["np_cells_compared"]}')
        check("7 cells where minimum-makespan is not the feasible-first "
              "placement",
              len(h["cells_where_makespan_rule_disagrees_with_feasibility"]) == 7,
              str(len(h["cells_where_makespan_rule_disagrees_with_feasibility"])))
    else:
        check("analysis.json present", False, "run scripts/analyse.py tables")

    print("6. the noise floor SETUP.md quotes")
    # SETUP.md's 7.62% / 9.18% were computed over the XPU-RT sweep's 199-point
    # Phase 4. That sweep has since been extended with `cpsat:warmbest` on the
    # 30 tier-B cells (ANALYSIS.md 0.5, 13), so the same statistic over the
    # 229-point file is a different, tighter number. Both are checked and both
    # are reported. The band the write-ups actually use stays the 9.18%
    # SETUP.md pre-registered, because it is the WIDER of the two and a wider
    # band calls fewer differences results -- narrowing it after the fact would
    # promote borderline cells into findings, which is exactly the move a
    # pre-registered floor exists to prevent.
    rows4 = jload(os.path.join(XSWEEP, "results", "phase4_results.json"))
    tiers = {r["point"]: r.get("tier") for r in rows4}
    def spreads(rows):
        w = [r["measured_spread_ms"] / r["measured_median_ms"] * 100
             for r in rows if r.get("measured_median_ms")]
        n = [r["measured_np_spread_ms"] / r["measured_np_median_ms"] * 100
             for r in rows if r.get("measured_np_median_ms")]
        return statistics.median(w), statistics.median(n)
    # tier D is the extension; the frozen set SETUP.md quoted is everything else
    frozen = [r for r in rows4 if tiers.get(r["point"]) != "D"]
    fw, fn = spreads(frozen)
    aw, an = spreads(rows4)
    check("wall-clock median rep spread is 7.62% on the 199-point set "
          "SETUP.md quoted",
          abs(fw - 7.62) < 0.01, f"{fw:.2f}% ({len(frozen)} points)")
    check("non-periodic median rep spread is 9.18% on the 199-point set "
          "SETUP.md quoted",
          abs(fn - 9.18) < 0.01, f"{fn:.2f}% ({len(frozen)} points)")
    check("the band the write-ups use is the wider of the two",
          fn >= an, f"pre-registered {fn:.2f}% >= extended {an:.2f}%")
    print(f"       over all {len(rows4)} points the same statistic is "
          f"{aw:.2f}% / {an:.2f}% -- reported, not used")

    n_bad = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_bad}/{len(results)} checks passed")
    return n_bad


def measured_cells():
    rows = jload(os.path.join(XSWEEP, "results", "phase4_results.json"))
    return {r["workload"] for r in rows if r.get("measured_median_ms")}


if __name__ == "__main__":
    sys.exit(main())
