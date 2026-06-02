"""Iterative scheduling-improvement driver (predicted-only).

baseline run -> advisor diagnoses -> propose a bundle of candidates (axes A/B
xpurt-realizable, axis C fusion-hints for ModelBlaster) -> run the A/B
candidates predicted -> compare -> pick a winner. Emits a user-facing report.md,
iteration_result.json, firesim_batch.json (the candidate set the ModelBlaster
session should build+run in one batched FireSim session), and a before/after
composite Gantt.

    python3 scripts/iterate_firesim.py \
        --networks-json data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json \
        --baseline-solver decomposed --deadline-us auto --gantt

This is the inner, fast loop (predicted). The ModelBlaster session realizes axis
C (fusion) on spike + runs firesim_batch.json on FireSim for authoritative timing
(see docs/iterative_firesim_loop.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xpu-rt"))
import _sched_eval as ev
import bundle as bundle_mod


def _fixture_of(report_path: str) -> str:
    return report_path.replace("_report.json", ".json")


def _label(solver, scheduler):
    return solver + (f"/{scheduler}" if scheduler else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--networks-json", required=True)
    ap.add_argument("--baseline-solver", default="decomposed")
    ap.add_argument("--baseline-scheduler", default=None)
    ap.add_argument("--deadline-us", default="auto",
                    help="'auto' (between best and baseline) or a number")
    ap.add_argument("--out-dir", default="artifacts/iterate")
    ap.add_argument("--timeout", type=int, default=90,
                    help="per-candidate scheduling budget (s); slow rvv-heavy backends skip")
    ap.add_argument("--gantt", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(ev.REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(ev.REPO, args.networks_json) if not os.path.isabs(args.networks_json)
              else args.networks_json) as f:
        base_spec = json.load(f)
    base_profile_hw = base_spec.get("hardware", {}).get("profile_hw", {})
    avail = ev.available_backends()
    print(f"available backends: {avail}")

    # --- baseline ---
    print(f"\n=== baseline: {_label(args.baseline_solver, args.baseline_scheduler)} ===", flush=True)
    base_res = ev.run_candidate(args.networks_json, profile_hw=base_profile_hw,
                                solver=args.baseline_solver, scheduler=args.baseline_scheduler,
                                stem="_iter_baseline", timeout=args.timeout)
    if base_res["status"] != "ok":
        print(f"baseline failed: {base_res['status']}")
        return 1
    base_report = base_res["report"]
    base_makespan = float(base_report.get("makespan_cycles", 0.0))
    base_diag0 = ev.advise(base_report, None)  # granularity verdict (deadline-independent)
    baseline_cfg = {"solver": args.baseline_solver, "scheduler": args.baseline_scheduler,
                    "profile_hw": base_profile_hw}

    # --- propose bundle ---
    bundle = bundle_mod.propose_bundle(base_report, base_diag0, baseline=baseline_cfg,
                                       available_backends=avail)
    xpurt_cands = [c for c in bundle["candidates"] if c["realizable_by"] == "xpurt"]
    print(f"bundle: {bundle['counts']['xpurt']} xpurt candidates, "
          f"{bundle['counts']['modelblaster']} modelblaster (fusion) candidates")

    # --- run xpurt candidates (axes A/B) ---
    runs = [{"id": "baseline", "label": _label(args.baseline_solver, args.baseline_scheduler),
             "axis": "baseline", "profile_hw": base_profile_hw,
             "makespan_us": base_makespan, "report": base_report,
             "report_path": base_res["report_path"]}]
    skipped = []
    for c in xpurt_cands:
        print(f"=== {c['id']} [{c['axis']}]: {_label(c['solver'], c.get('scheduler'))} "
              f"hw={c['profile_hw']} ===", flush=True)
        res = ev.run_candidate(args.networks_json, profile_hw=c["profile_hw"],
                               solver=c["solver"], scheduler=c.get("scheduler"),
                               timeout=args.timeout)
        if res["status"] != "ok":
            print(f"  {c['id']}: {res['status']}")
            # rvv-heavy backends explode into many dispatches and exceed the
            # scheduling budget — itself a finding (fusion candidate, axis C).
            hint = " (rvv emits many more dispatches — a fusion/coarsen candidate)" \
                if "rvv" in "".join(c["profile_hw"].values()).lower() and res["status"].startswith("timeout") else ""
            skipped.append({"id": c["id"], "label": _label(c["solver"], c.get("scheduler")),
                            "axis": c["axis"], "profile_hw": c["profile_hw"],
                            "status": res["status"] + hint})
            continue
        runs.append({"id": c["id"], "label": _label(c["solver"], c.get("scheduler")),
                     "axis": c["axis"], "profile_hw": c["profile_hw"],
                     "makespan_us": float(res["report"].get("makespan_cycles", 0.0)),
                     "report": res["report"], "report_path": res["report_path"]})

    # --- deadline (auto: midpoint between best and baseline, so baseline misses
    #     and the best candidate meets; explicit otherwise) ---
    makespans = [r["makespan_us"] for r in runs]
    best_ms = min(makespans)
    if args.deadline_us == "auto":
        deadline_us = round((best_ms + base_makespan) / 2.0, 2) if best_ms < base_makespan \
            else round(base_makespan * 0.95, 2)
        deadline_src = "auto (midpoint best..baseline)"
    else:
        deadline_us = float(args.deadline_us)
        deadline_src = "explicit"

    # --- advise everything with the chosen deadline ---
    for r in runs:
        d = ev.advise(r["report"], deadline_us)
        r["meets_deadline"] = d.meets_deadline
        r["granularity"] = d.granularity_verdict
        r["bottleneck"] = d.bottleneck_backend
        r["diag"] = d

    # --- winner: prefer meets-deadline, then min makespan (exclude baseline ties) ---
    def key(r):
        return (0 if r["meets_deadline"] else 1, r["makespan_us"])
    winner = min(runs, key=key)
    base_row = runs[0]
    bundle["deadline_us"] = deadline_us

    # --- emit firesim_batch.json (what MB should build+run on FireSim) ---
    firesim_batch = {
        "networks_json": args.networks_json,
        "deadline_us": deadline_us,
        "note": "Build+run these candidates in one batched FireSim session; "
                "modelblaster candidates require applying fusion hints first.",
        "candidates": bundle["candidates"],
    }
    with open(os.path.join(out_dir, "firesim_batch.json"), "w") as f:
        json.dump(firesim_batch, f, indent=2)
    with open(os.path.join(out_dir, "iteration_result.json"), "w") as f:
        json.dump({"deadline_us": deadline_us, "deadline_src": deadline_src,
                   "winner": winner["id"],
                   "runs": [{k: r[k] for k in ("id", "label", "axis", "profile_hw",
                            "makespan_us", "meets_deadline", "granularity", "bottleneck")}
                            for r in runs],
                   "skipped": skipped}, f, indent=2)

    # --- before/after composite Gantt ---
    gantt_rel = None
    if args.gantt:
        try:
            import plot_gantt
            before_fix = _fixture_of(base_row["report_path"])
            after_fix = _fixture_of(winner["report_path"])
            gantt_png = os.path.join(out_dir, "before_after_gantt.png")
            plot_gantt.render_composite_gantt(
                before_fix, after_fix, gantt_png,
                titles=(f"BEFORE: {base_row['label']} ({base_makespan:.1f}us)",
                        f"AFTER: {winner['label']} ({winner['makespan_us']:.1f}us)"))
            gantt_rel = os.path.relpath(gantt_png, ev.REPO)
        except Exception as exc:
            print(f"[warn] composite gantt failed: {exc}")

    # --- report.md ---
    md = _render_report(runs, base_row, winner, deadline_us, deadline_src, bundle, gantt_rel,
                        base_diag0, skipped)
    report_md = os.path.join(out_dir, "report.md")
    with open(report_md, "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"\nwrote {os.path.relpath(report_md, ev.REPO)}, iteration_result.json, firesim_batch.json"
          + (f", {gantt_rel}" if gantt_rel else ""))
    return 0


def _render_report(runs, base_row, winner, deadline_us, deadline_src, bundle, gantt_rel,
                   base_diag0, skipped=None) -> str:
    L = []
    L.append("# Iterative scheduling improvement\n")
    L.append(f"Deadline budget: **{deadline_us} us** ({deadline_src}).\n")
    L.append("| id | config | axis | profile_hw | makespan_us | meets | granularity | bottleneck |")
    L.append("|----|--------|------|-----------|------------:|:-----:|-------------|-----------|")
    for r in sorted(runs, key=lambda r: r["makespan_us"]):
        mark = " 🏆" if r["id"] == winner["id"] else ""
        L.append(f"| {r['id']}{mark} | {r['label']} | {r['axis']} | "
                 f"{'+'.join(sorted(set(r['profile_hw'].values())))} | {r['makespan_us']:.2f} | "
                 f"{'✅' if r['meets_deadline'] else '⚠️'} | {r['granularity']} | {r['bottleneck']} |")
    improved = base_row["makespan_us"] - winner["makespan_us"]
    pct = (improved / base_row["makespan_us"] * 100.0) if base_row["makespan_us"] else 0.0
    if skipped:
        L.append("")
        L.append("Skipped (did not finish within the per-candidate budget):")
        for s in skipped:
            L.append(f"- `{s['id']}` {s['label']} on "
                     f"{'+'.join(sorted(set(s['profile_hw'].values())))} — {s['status']}")
    L.append("")
    if winner["id"] == "baseline":
        L.append("**Result:** no candidate beat the baseline.")
    else:
        verb = "now meets" if (winner["meets_deadline"] and not base_row["meets_deadline"]) else "is"
        L.append(f"**Winner: `{winner['id']}` ({winner['label']})** — makespan "
                 f"{winner['makespan_us']:.2f} us vs baseline {base_row['makespan_us']:.2f} us "
                 f"(**{pct:.1f}% lower**); {verb} within the deadline.")
    L.append("")
    L.append("## Advisor on baseline\n")
    import advisor
    L.append("```")
    L.append(advisor.render_text(base_row["diag"]))
    L.append("```")
    # axis-B note: group by the ORDERED profile_hw assignment so distinct
    # placements (e.g. gemmini-on-P vs rvv-on-P) compare separately.
    def _hw_label(ph):
        return ", ".join(f"{k}={v}" for k, v in sorted(ph.items()))
    bk = {}
    for r in runs:
        bk.setdefault(_hw_label(r["profile_hw"]), []).append(r)
    L.append("\n## Profiler/backend comparison (axis B)\n")
    for hw, rs in sorted(bk.items(), key=lambda kv: min(r["makespan_us"] for r in kv[1])):
        best = min(rs, key=lambda r: r["makespan_us"])
        L.append(f"- **{hw}**: best makespan {best['makespan_us']:.2f} us "
                 f"({best['label']}, {'meets' if best['meets_deadline'] else 'misses'}).")
    if skipped:
        homog = [s for s in skipped if len(set(s["profile_hw"].values())) == 1]
        if homog:
            L.append(f"- _Homogeneous backends "
                     f"({', '.join(sorted({list(set(s['profile_hw'].values()))[0] for s in homog}))}) "
                     "exceeded the predicted scheduling budget (dispatch-count explosion) — "
                     "compare them on the real FireSim batch, and/or apply axis-C fusion first._")
    # axis-C handoff
    mb = [c for c in bundle["candidates"] if c["realizable_by"] == "modelblaster"]
    L.append("\n## Granularity/fusion (axis C — ModelBlaster)\n")
    if mb:
        nets = mb[0]["hints"]["networks"]
        L.append(f"Advisor flagged `{base_diag0.granularity_verdict}`. Emitted fusion hints for "
                 f"{len(nets)} network(s): " +
                 ", ".join(f"{n['network']} ({len(n['fuse_groups'])} groups)" for n in nets) + ".")
        L.append("See `firesim_batch.json` candidate `C1` for the hint payload; ModelBlaster "
                 "realizes it (re-extract/re-gen kernels on spike, re-profile on FireSim).")
    else:
        L.append("Granularity is balanced; no fusion candidate proposed.")
    if gantt_rel:
        L.append(f"\n## Before/after Gantt\n\n![before/after]({os.path.basename(gantt_rel)})  \n`{gantt_rel}`")
    L.append("\n## Next: FireSim batch\n")
    L.append(f"`firesim_batch.json` lists {len(bundle['candidates'])} candidates for the ModelBlaster "
             "session to build + run in one batched FireSim session (see docs/iterative_firesim_loop.md).")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
