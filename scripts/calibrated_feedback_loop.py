#!/usr/bin/env python3
"""Runtime -> scheduler / graph-partitioner feedback loop, round by round.

Each round: solve (greedy) with the current calibration -> execute -> fold the
trace into the calibration (``xpu-rt/calibration_feedback.py``) -> next round.

Execution is either the walker simulator (``--execute sim``, offline: durations
and dispatch gaps are drawn from real board traces given with ``--truth``), or a
board trace you produced yourself for that round's schedule (``--execute trace``,
one round per invocation; the script prints the schedule to run and stops).

``--feedback`` picks the update rule so the two can be compared under the same
conditions: ``calibrated`` (this work) or ``ratio`` (mean measured / predicted per
op, ``xpu-rt/duration_feedback.py``).

Example (offline, 4 rounds, static walker)::

    python scripts/calibrated_feedback_loop.py --spec data/toplevel/k3fb_base.json \\
        --truth SCHED.json:TRACE1.csv,TRACE2.csv --rounds 4 --out out/calib_static
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xpu-rt"))
import calibration_feedback as cfb      # noqa: E402
import duration_feedback as dfb         # noqa: E402
import walker_sim as W                  # noqa: E402

K3_HARTS = {"cpu_p": list(range(8)), "cpu_e": list(range(8, 16))}
HART_NAMES = {h: f"CPU_P#{h}" for h in range(8)} | {h: f"CPU_E#{h - 8}" for h in range(8, 16)}


def solve(spec: Path, calib: Path | None, ratio: Path | None, log: Path) -> Path:
    cmd = [sys.executable, "scripts/run_xpurt_schedule.py", "--networks-json", str(spec),
           "--solver", "greedy"]
    if calib:
        cmd += ["--calibration", str(calib)]
    if ratio:
        cmd += ["--duration-feedback", str(ratio)]
    with open(log, "w") as fh:
        subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, check=True)
    m = re.search(r"greedy: .*", log.read_text())
    print("   ", m.group(0) if m else "(no greedy line)")
    return ROOT / "schedules" / f"scheduled_{spec.stem}_greedy_profiled.json"


def op_names_from(truth: list[tuple[str, list[str]]]) -> dict[str, str]:
    names = {}
    for sched, traces in truth:
        tab = W.load_table(sched)
        for r in csv.DictReader(open(traces[0])):
            e = tab[int(r["entry_id"])]
            if r["op"]:
                names[f"{e.net}|dispatch_{e.ir_did}"] = r["op"]
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--feedback", choices=["calibrated", "ratio"], default="calibrated")
    ap.add_argument("--execute", choices=["sim", "trace"], default="sim")
    ap.add_argument("--truth", action="append", default=[],
                    help="SCHEDULE.json:TRACE1.csv[,TRACE2.csv...] board runs for the simulator")
    ap.add_argument("--policy", choices=["static", "dynamic"], default="static")
    ap.add_argument("--safety", type=float, default=1.5)
    ap.add_argument("--seeds", type=int, default=3, help="simulated runs per round")
    ap.add_argument("--ablate", choices=["none", "no-reserve", "no-cost"], default="none",
                    help="calibrated only: drop the hart reservations, or the cost calibration")
    ap.add_argument("--trace", type=Path, help="--execute trace: the board trace of --round's schedule")
    ap.add_argument("--round", type=int, help="--execute trace: which round this trace is")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    windows, periods = W.spec_windows(args.spec)
    truth = [(t.split(":", 1)[0], t.split(":", 1)[1].split(",")) for t in args.truth]
    op_names = op_names_from(truth) if truth else {}
    model_S = model_G = None
    if args.execute == "sim":
        pairs = []
        for sched, traces in truth:
            tab = W.load_table(sched)
            pairs += [(tab, W.read_trace(tr, tab)) for tr in traces]
        model_S, model_G = W.op_samples(pairs), W.gap_samples(pairs)

    rounds = range(args.rounds) if args.execute == "sim" else [args.round]
    summary = []
    for r in rounds:
        rd = args.out / f"round{r}"
        rd.mkdir(exist_ok=True)
        prev = args.out / f"round{r - 1}" / "feedback_after.json" if r > 0 else None
        spec_r = ROOT / "data" / "toplevel" / f"{args.spec.stem}_{args.out.name}_r{r}.json"
        shutil.copy(args.spec, spec_r)
        used = rd / "feedback_used.json"
        if prev and prev.exists():
            shutil.copy(prev, used)
        print(f"== round {r} ({args.feedback}, {args.policy})")
        sched = solve(spec_r, used if (used.exists() and args.feedback == "calibrated") else None,
                      used if (used.exists() and args.feedback == "ratio") else None, rd / "solve.log")
        shutil.copy(sched, rd / sched.name)
        tab = W.load_table(rd / sched.name)
        if args.execute == "trace":
            if not args.trace:
                print(f"   run this schedule on the board, then rerun with --trace: {rd / sched.name}")
                return
            traces = [args.trace]
        else:
            traces = []
            for s in range(1, args.seeds + 1):
                res = W.simulate(tab, W.DurationModel(model_S, seed=1000 * r + s, gaps=model_G),
                                 policy=args.policy, safety=args.safety, all_harts=K3_HARTS)
                p = rd / f"sim_trace_s{s}.csv"
                W.write_trace(tab, res, p)
                traces.append(p)
        misses = []
        doc_in = (cfb.load(used) if used.exists() else cfb.empty()) if args.feedback == "calibrated" \
            else (dfb.load(used) if used.exists() else dfb.empty())
        doc = doc_in
        for i, p in enumerate(traces):
            tr = W.read_trace(p, tab)
            mm = W.instance_misses(tab, [x["f"] for x in tr], windows, periods)
            misses.append({k: v["misses"] for k, v in mm.items()})
            if i == 0:     # one trace per round feeds the update, as on the board
                if args.feedback == "calibrated":
                    doc = cfb.update(doc_in, tab, tr, f"round{r}", windows=windows, periods=periods,
                                     op_names=op_names, hart_names=HART_NAMES, all_harts=K3_HARTS)
                else:
                    obs = [dict(network=e.net, dispatch=f"dispatch_{e.ir_did}", kind=e.kind,
                                measured_ms=t["dur"], predicted_ms=e.pd)
                           for e, t in zip(tab, tr) if not t["zero"]]
                    doc = dfb.update(doc_in, obs, f"round{r}")
        if args.feedback == "calibrated" and args.ablate == "no-reserve":
            doc["reserve"] = {}
        if args.feedback == "calibrated" and args.ablate == "no-cost":
            doc["ops"], doc["gap_ms"] = {}, {}
        (cfb.save if args.feedback == "calibrated" else dfb.save)(doc, rd / "feedback_after.json")
        if args.feedback == "calibrated" and doc.get("split_hints"):
            (rd / "split_hints.json").write_text(json.dumps(cfb.split_hint_contract(doc), indent=1))
        row = {"round": r, "misses": misses,
               "reserve": doc.get("reserve") if args.feedback == "calibrated" else None,
               "split_hints": len(doc.get("split_hints", [])) if args.feedback == "calibrated" else 0}
        summary.append(row)
        print("   misses per run:", [(m.get("mlp_control"), m.get("dronet"), m.get("yolov8_nano")) for m in misses])
        if args.feedback == "calibrated":
            print("   reserve:", doc.get("reserve"), "| split hints:", row["split_hints"],
                  "| blame:", {k: {b: round(x, 2) for b, x in v["blocked_by_ms"].items()} for k, v in doc["blame"].items()})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
