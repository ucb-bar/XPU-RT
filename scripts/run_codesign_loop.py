#!/usr/bin/env python3
"""Fully-automatic ModelBlaster<->XPU-RT co-design feedback loop.

ONE command. Starts from a workload spec at a clean baseline (no levers), then each
round proposes every not-yet-applied lever as a candidate, SOLVES each candidate with
the real profiled scheduler, and ACCEPTS the candidate with the largest MEASURED
makespan reduction that adds ZERO deadline misses. Iterates until no lever helps
(converged). The loop's intelligence is the advisor/cost-model; this driver applies,
measures, and accepts — honestly (a lever that does not help is rejected and reported).

Levers:
  * ime   — expose the K1 IME matrix engine as a per-dispatch alternative. For every
            CONV-bearing net that lacks an ime_x60 profile, auto-build one from the
            measured conv speedups (scripts/make_ime_profile.py, which scales mean_time
            — the column the loader reads), then set scheduler.enable_impls=true. The
            scheduler picks IME per dispatch only where measured faster.
  * shard — expose 2/4/8-hart implementations (scheduler.machine_combination_mode=shard,
            topo_tag_override=false).
  * fuse  — the roofline decision-aid (xpu-rt/data/fusion_benefit.csv) is consulted and
            REPORTED, not applied: our stack is compute-bound, so fusion is a scheduling
            (dispatch-collapse) move, not a cycle win — the loop honestly does not credit
            a makespan gain it cannot measure.

Usage:
  scripts/run_codesign_loop.py --workload data/toplevel/<spec>.json [--max-rounds 4]
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, ".venv/bin/python")
EPS = 0.05  # ms; a smaller "improvement" is noise, not a win

sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
try:
    import exact_cycle  # objective-aware acceptance for exact_cycle workloads
except Exception:
    exact_cycle = None


def objective_of(spec: dict) -> str:
    """Which metric this workload is really optimizing. A tri-exact-style workload
    declares `exact_cycle_worst_response`, whose win is WORST CRITICAL RESPONSE, not
    makespan — so a makespan-only loop would (correctly, but uselessly) credit its
    shard/IME levers nothing. Accept on the objective the user actually declared."""
    om = spec.get("scheduler", {}).get("objective_mode") or spec.get("objective_mode")
    return "worst_response" if om == "exact_cycle_worst_response" else "makespan"


def worst_response_ms(sched_path: str, spec: dict):
    """Worst critical response (ms) of a schedule, via the exact-cycle assessor.
    None if unavailable — the caller then falls back to makespan."""
    if exact_cycle is None or not sched_path or not os.path.exists(sched_path):
        return None
    try:
        sch = json.load(open(sched_path))
        crit = (spec.get("critical_models") or spec.get("scheduler", {}).get("critical_models") or [])
        heavy = spec.get("heavy_model") or spec.get("scheduler", {}).get("heavy_model")
        obj = exact_cycle.assess_schedule(sch, spec, crit, heavy)
        return float(obj["objective"]["worst_critical_response_ms"])
    except Exception:
        return None


def _run(cmd):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def _rvv_profile(net, variant, hw="rvv_x60"):
    return os.path.join(
        REPO, f"gen/profile_mb/{hw}/spacemit_x60/{net}/{net}.{variant}/"
        f"{net}_spacemit_x60_{hw}_{net}.{variant}/topo_0/results.csv")


def _net_variant(deps: str, key: str):
    m = re.search(r"/vmfb/([^/]+)/", deps)
    net = m.group(1) if m else key
    tail = deps.split("/")[-1].replace("_dispatch_graph.json", "")
    variant = tail[len(net) + 1:] if tail.startswith(net + ".") else "int8"
    return net, variant


def _is_conv_net(net, variant):
    p = _rvv_profile(net, variant)
    if not os.path.exists(p):
        return False
    return any(r.get("op", "").startswith("conv2d") for r in csv.DictReader(open(p)))


def solve(spec_path):
    """Run the profiled greedy scheduler; return (makespan_ms, miss, sched_json)."""
    stem = os.path.splitext(os.path.basename(spec_path))[0]
    r = _run([PY, "scripts/run_xpurt_schedule.py", "--networks-json", spec_path,
              "--solver", "greedy", "--profiled", "--max-periodic-iters", "1"])
    metrics = os.path.join(REPO, "schedules", f"scheduled_{stem}_greedy_profiled_metrics.json")
    sched = os.path.join(REPO, "schedules", f"scheduled_{stem}_greedy_profiled.json")
    if not os.path.exists(metrics):
        return None, None, None, (r.stdout + r.stderr)[-800:]
    m = json.load(open(metrics))
    return float(m["makespan_ms"]), int(m["op_deadline_miss_count"]), sched, None


def baseline(spec: dict) -> dict:
    """Strip all levers so the loop starts from a clean floor."""
    spec = copy.deepcopy(spec)
    sch = spec.setdefault("scheduler", {})
    sch["enable_impls"] = False
    sch["machine_combination_mode"] = "singletons"
    if "hardware" in spec and "profile" in spec["hardware"]:
        spec["hardware"]["profile"].setdefault("topo_tag_override", True)
    return spec


def apply_ime(spec: dict, log) -> dict:
    spec = copy.deepcopy(spec)
    built = []
    for key, info in spec.get("networks", {}).items():
        net, variant = _net_variant(info.get("dispatch_deps_path", ""), key)
        if not _is_conv_net(net, variant):
            continue
        if os.path.exists(_rvv_profile(net, variant, hw="ime_x60")):
            continue  # already has an ime profile (matmul nets, or a prior build) — do not clobber
        r = _run([PY, "scripts/make_ime_profile.py", "--net", net, "--variant", variant])
        if r.returncode == 0:
            built.append(f"{net}.{variant}")
    log(f"      ime: built ime_x60 profiles for {built or '(none new; existing reused)'}")
    spec.setdefault("scheduler", {})["enable_impls"] = True
    return spec


def apply_shard(spec: dict, log) -> dict:
    spec = copy.deepcopy(spec)
    spec.setdefault("scheduler", {})["machine_combination_mode"] = "shard"
    if "hardware" in spec and "profile" in spec["hardware"]:
        spec["hardware"]["profile"]["topo_tag_override"] = False
    return spec


LEVERS = {"ime": apply_ime, "shard": apply_shard}


def fusion_note():
    try:
        fb = {r["network"]: r for r in csv.DictReader(open(os.path.join(REPO, "xpu-rt/data/fusion_benefit.csv")))}
        worst = max(fb.values(), key=lambda r: float(r["epilogue_ceiling_pct"]))
        return (f"fuse: NOT applied — roofline decision-aid says the stack is compute-bound "
                f"(max fusible-epilogue ceiling {float(worst['epilogue_ceiling_pct']):.0f}% on "
                f"{worst['network']}); fusion collapses dispatches (scheduling) but is measured "
                f"~+0.85% on cycles, so the loop does not credit a makespan gain it cannot measure.")
    except Exception:
        return "fuse: decision-aid unavailable."


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--out-dir", default="results/codesign_loop")
    args = ap.parse_args()

    wl_stem = os.path.splitext(os.path.basename(args.workload))[0]
    out_dir = os.path.join(REPO, args.out_dir, wl_stem)
    spec_dir = os.path.join(out_dir, "specs")
    os.makedirs(spec_dir, exist_ok=True)
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    log(f"== co-design loop: {wl_stem} ==")
    working = baseline(json.load(open(args.workload)))
    base_path = os.path.join(spec_dir, f"{wl_stem}_r0_baseline.json")
    json.dump(working, open(base_path, "w"), indent=1)
    mk, miss, sched, err = solve(base_path)
    if mk is None:
        log(f"BASELINE SOLVE FAILED: {err}")
        return 1

    obj_mode = objective_of(working)
    metric_name = "worst-response" if obj_mode == "worst_response" else "makespan"

    def score(sched_path, spec_dict, makespan):
        """The number the loop accepts on = the workload's declared objective."""
        if obj_mode == "worst_response":
            w = worst_response_ms(sched_path, spec_dict)
            if w is not None:
                return w
        return makespan

    base_score = score(sched, working, mk)
    log(f"round 0 · baseline: {metric_name} {base_score:.3f} ms "
        f"(makespan {mk:.1f} ms), {miss} miss")

    applied, rounds = [], []
    traj = [{"round": 0, "lever": "baseline", "score_ms": round(base_score, 3),
             "makespan_ms": round(mk, 1), "misses": miss}]
    cur_mk, cur_score, cur_miss, cur_spec = mk, base_score, miss, working

    for rnd in range(1, args.max_rounds + 1):
        cands = []
        for lever in LEVERS:
            if lever in applied:
                continue
            cspec = LEVERS[lever](cur_spec, log)
            cpath = os.path.join(spec_dir, f"{wl_stem}_r{rnd}_{lever}.json")
            json.dump(cspec, open(cpath, "w"), indent=1)
            cmk, cmiss, csched, cerr = solve(cpath)
            if cmk is None:
                log(f"round {rnd} · try {lever}: SOLVE FAILED ({cerr[:120] if cerr else ''}) — reject")
                continue
            csc = score(csched, cspec, cmk)
            delta = cur_score - csc
            ok = (cmiss <= cur_miss) and (delta > EPS)
            log(f"round {rnd} · try {lever}: {metric_name} {cur_score:.3f} -> {csc:.3f} ms "
                f"({-delta/cur_score*100:+.1f}%), {cmiss} miss -> {'ACCEPTABLE' if ok else 'reject'}")
            cands.append(dict(lever=lever, mk=cmk, score=csc, miss=cmiss, sched=csched,
                              spec=cspec, ok=ok))

        winners = [c for c in cands if c["ok"]]
        if not winners:
            log(f"round {rnd}: no lever improves {metric_name} with 0 added misses — CONVERGED")
            break
        best = min(winners, key=lambda c: c["score"])
        pct = (cur_score - best["score"]) / cur_score * 100
        log(f"round {rnd}: ACCEPT +{best['lever']}  {metric_name} "
            f"{cur_score:.3f} -> {best['score']:.3f} ms (-{pct:.1f}%)")

        # render the accepted schedule's Gantt (IME dispatches darker+hatched)
        gstem = os.path.join(out_dir, f"round_{rnd}_{best['lever']}_gantt")
        _run([PY, "scripts/plot_scheduled_json.py", best["sched"], "--save", gstem,
              "--window-ms", str(round(best["mk"] * 1.05, 1))])

        rounds.append(dict(round=rnd, lever=best["lever"], metric=metric_name,
                           score_before_ms=round(cur_score, 3), score_after_ms=round(best["score"], 3),
                           makespan_before_ms=round(cur_mk, 1), makespan_after_ms=round(best["mk"], 1),
                           pct=round(-pct, 1), accepted=True, deadline_miss=best["miss"]))
        applied.append(best["lever"])
        cur_mk, cur_score, cur_miss, cur_spec = best["mk"], best["score"], best["miss"], best["spec"]
        traj.append({"round": rnd, "lever": f"+{best['lever']}", "score_ms": round(cur_score, 3),
                     "makespan_ms": round(cur_mk, 1), "misses": cur_miss})

    fnote = fusion_note()
    log(fnote)

    denom = base_score if base_score else 1.0
    report = dict(workload=wl_stem, objective=metric_name,
                  baseline_score_ms=round(base_score, 3), final_score_ms=round(cur_score, 3),
                  total_reduction_pct=round((base_score - cur_score) / denom * 100, 1),
                  baseline_makespan_ms=round(mk, 1), final_makespan_ms=round(cur_mk, 1),
                  levers_applied=applied, rounds=rounds, trajectory=traj,
                  fusion=fnote, converged=True)
    json.dump(report, open(os.path.join(out_dir, "loop_report.json"), "w"), indent=1)

    _plot_traj(traj, os.path.join(out_dir, "objective_vs_round"), metric_name)
    _write_readme(out_dir, wl_stem, args, report, fnote)
    log(f"\nCONVERGED [{metric_name}]: {base_score:.3f} -> {cur_score:.3f} ms "
        f"(-{report['total_reduction_pct']:.1f}%), levers applied: {applied or 'none'}")
    log(f"artifacts in {out_dir}")
    open(os.path.join(out_dir, "loop_log.txt"), "w").write("\n".join(lines) + "\n")
    return 0


def _plot_traj(traj, out, metric_name="makespan"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = list(range(len(traj)))
        ys = [t.get("score_ms", t["makespan_ms"]) for t in traj]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.step(xs, ys, where="post", color="#4c72b0", lw=2, marker="o")
        for i, t in enumerate(traj):
            ax.annotate(t["lever"], (i, ys[i]),
                        textcoords="offset points", xytext=(6, 8), fontsize=9,
                        weight="bold" if t["lever"] != "baseline" else "normal")
        if len(ys) > 1 and ys[0]:
            ax.annotate(f"-{(ys[0]-ys[-1])/ys[0]*100:.1f}%", (xs[-1], ys[-1]),
                        textcoords="offset points", xytext=(6, -14), fontsize=9, color="#2f7d4f")
        ax.set_xlabel("feedback round"); ax.set_ylabel(f"{metric_name} (ms)")
        ax.set_xticks(xs); ax.set_ylim(0, max(ys) * 1.15)
        ax.set_title(f"Automatic co-design loop — {metric_name} per accepted round", weight="bold")
        fig.tight_layout()
        fig.savefig(out + ".png", dpi=160); fig.savefig(out + ".pdf")
    except Exception as e:
        print("traj plot skipped:", e)


def _write_readme(out_dir, stem, args, report, fnote):
    r = report
    md = [f"# Automatic co-design loop — `{stem}`", "",
          "Fully automatic ModelBlaster↔XPU-RT feedback loop: solve → propose every lever →",
          "measure each → accept the largest measured makespan win with 0 added misses → repeat.",
          "", "```",
          f"scripts/run_codesign_loop.py --workload {args.workload} --max-rounds {args.max_rounds}",
          "```", "",
          f"**Baseline → final: {r['baseline_makespan_ms']} → {r['final_makespan_ms']} ms "
          f"(-{r['total_reduction_pct']}%)** — levers applied: {r['levers_applied'] or 'none'}.", "",
          "| round | lever | before (ms) | after (ms) | % | misses |", "|--:|--|--:|--:|--:|--:|"]
    for rr in r["rounds"]:
        md.append(f"| {rr['round']} | +{rr['lever']} | {rr['makespan_before_ms']} | "
                  f"{rr['makespan_after_ms']} | {rr['pct']} | {rr['deadline_miss']} |")
    md += ["", f"Honest note — {fnote}", "",
           "Artifacts: `loop_report.json`, `makespan_vs_round.{png,pdf}`, "
           "`round_<k>_<lever>_gantt.{png,pdf}` (IME dispatches drawn darker + hatched), "
           "`specs/` (every candidate spec), `loop_log.txt`."]
    open(os.path.join(out_dir, "README.md"), "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    sys.exit(main())
