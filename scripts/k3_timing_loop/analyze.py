#!/usr/bin/env python3
"""Analyse one round (N runs of one build).

Writes into --out:
  runs.csv           per run: checks, late frames per model, worst lateness, frame medians
  frames.csv.gz      per (run, model, frame): planned, board, late, sums, the chain split
  layers.csv         per (model, layer, core kind, kernel type, cores): planned, run, factor, gap, grab
  handoff.json       the start gap by hand-off type (pooled and per model)
  summary.json       per model: frame times, late frames, the split, hand-offs per frame; frame 0 on its own
Frames 1..N-1 of each model go into the timing numbers; frame 0 of each model is reported on its own.
Late frames are counted over ALL frames (frame 0 included).
Runs of a walker without the stamps load too (no grab, and "run" is start -> end).
Exit code 0 only if every run passes every check (output bit-exact, placement, exit code, stamps in order,
trace == dispatch table == schedule).

usage: analyze.py --raw <raw run folder> --reps 20 --out <results folder> --label "before" [--profile-root <tree>]
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import os
import statistics
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loop_lib as T  # noqa: E402

Q, med = T.quantile, T.med


def write_csv(path: str, rows: List[dict], digits: int = 6) -> None:
    if not rows:
        return
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (("" if v != v else f"{v:.{digits}f}") if isinstance(v, float) else v) for k, v in r.items()})


def layer_table(pieces: List[T.Piece], stamps: bool) -> List[dict]:
    rows = []
    for key, ps in sorted(T.group_by_key(pieces).items()):
        net, irid, kind, impl, width = key
        run = [(p.run if stamps else p.end - p.start) for p in ps]
        free = [p.gap for p in ps if not p.gated]
        pl = med(p.planned for p in ps)
        hand = collections.Counter(p.handoff for p in ps if p.handoff).most_common(1)
        rows.append(dict(
            network=net, layer=irid, name=ps[0].name, op=ps[0].op, core_kind=kind, kernel_type=impl, cores=width,
            planned_ms=pl, planned_values=len({round(p.planned, 6) for p in ps}),
            run_p50_ms=Q(run, .5), run_p75_ms=Q(run, .75), factor_p50=Q(run, .5) / pl if pl > 0 else float("nan"),
            gap_p50_ms=Q(free, .5), gap_p75_ms=Q(free, .75), gap_mean_ms=statistics.fmean(free) if free else float("nan"),
            grab_p75_ms=Q([p.grab for p in ps], .75) if stamps else float("nan"),
            grab_max_ms=max(p.grab for p in ps) if stamps else float("nan"),
            kernel_p50_ms=Q([p.kernel for p in ps], .5) if stamps else float("nan"),
            release_p50_ms=Q([p.release for p in ps], .5) if stamps else float("nan"),
            finish_error_p50_ms=Q([p.finish_error for p in ps], .5),
            handoff_mostly=hand[0][0] if hand else "frame start", handoff_share=hand[0][1] / len(free) if hand and free else 1.0,
            n=len(ps), n_gap=len(free)))
    return rows


def model_block(frames: List[T.FrameTotals], pieces: List[T.Piece], runs: int) -> dict:
    act = [f.actual_run_ms for f in frames]
    pln = [f.planned_run_ms for f in frames]
    left = collections.Counter(f.chain_left_to or "its own start" for f in frames)
    return dict(
        frames=len(frames), planned_frame_ms=med(pln), planned_frame_max_ms=max(pln),
        board_frame_p50_ms=med(act), board_frame_p75_ms=Q(act, .75), board_frame_max_ms=max(act),
        board_over_planned=med(act) / med(pln), board_minus_planned_p50_ms=med(f.extra for f in frames),
        finish_after_release_planned_p50_ms=med(f.planned_ms for f in frames),
        finish_after_release_board_p50_ms=med(f.actual_ms for f in frames),
        finish_after_release_board_max_ms=max(f.actual_ms for f in frames), deadline_ms=frames[0].deadline_ms,
        discrepancy_p50_ms=med(p.finish_error for p in pieces),
        layers_per_frame=med(float(f.layers) for f in frames),
        handoffs_per_frame=statistics.fmean(f.handoffs for f in frames),
        handoffs_from_other_model_per_frame=statistics.fmean(f.handoffs_other_model for f in frames),
        all_layers=dict(sum_gap_ms=med(f.sum_gap for f in frames), sum_grab_ms=med(f.sum_grab for f in frames),
                        sum_run_minus_planned_ms=med(f.sum_run_extra for f in frames)),
        chain=dict(gap_ms=med(f.chain_gap for f in frames), grab_ms=med(f.chain_grab for f in frames),
                   run_minus_planned_ms=med(f.chain_run_extra for f in frames),
                   planned_idle_ms=med(f.chain_planned_idle for f in frames),
                   inherited_from_other_frames_ms=med(f.chain_inherited for f in frames),
                   mean=dict(gap_ms=statistics.fmean(f.chain_gap for f in frames), grab_ms=statistics.fmean(f.chain_grab for f in frames),
                             run_minus_planned_ms=statistics.fmean(f.chain_run_extra for f in frames),
                             planned_idle_ms=statistics.fmean(f.chain_planned_idle for f in frames),
                             inherited_from_other_frames_ms=statistics.fmean(f.chain_inherited for f in frames),
                             board_minus_planned_ms=statistics.fmean(f.extra for f in frames)),
                   chain_started_at={k: v / len(frames) for k, v in left.items()},
                   max_abs_residual_ms=max(abs(f.residual) for f in frames)))


def handoff_stats(pieces: List[T.Piece], runs: int) -> dict:
    def block(ps: List[T.Piece]) -> dict:
        g: Dict[str, List[float]] = collections.defaultdict(list)
        for p in ps:
            if p.handoff:
                g[p.handoff].append(p.gap)
        return {k: dict(per_run=len(v) / runs, p50_us=Q(v, .5) * 1e3, p75_us=Q(v, .75) * 1e3, p99_us=Q(v, .99) * 1e3,
                        mean_us=statistics.fmean(v) * 1e3, n=len(v)) for k, v in sorted(g.items())}
    return dict(all_models=block(pieces), **{net: block([p for p in pieces if p.net == net]) for net in T.NETS})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--profile-root", help="the profile tree the plan was solved with: checks planned == tree")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = [T.load(a.raw, k) for k in range(1, a.reps + 1)]
    stamps = all(r.has_stamps for r in runs)
    ftot = {r.rep: T.frame_totals(r) for r in runs}
    steady = [p for r in runs for p in r.pieces if p.frame > 0]
    first = [p for r in runs for p in r.pieces if p.frame == 0]

    run_rows = []
    for r in runs:
        c, fs = r.checks, ftot[r.rep]
        row = dict(run=r.rep, ok=int(c["ok"]), placement=c["placement"].split(":")[0][:30], rc=c["rc"], fatal=int(c["output"]["fatal"]),
                   policy=c["output"]["policy_line"].replace("xpurt_runner: ", ""), stamps=int(r.has_stamps),
                   stamps_out_of_order=c["stamps_out_of_order"], layer_rows=c["layers"], name_mismatch=c["audit"]["name_mismatch"],
                   schedule_max_diff_us=c["audit"].get("schedule_max_diff_ms", 0) * 1e3,
                   late_frames=sum(f.late for f in fs), worst_lateness_ms=max(f.lateness for f in fs))
        for net in T.NETS:
            o = c["output"]["nets"][net]
            nf = [f for f in fs if f.net == net]
            row.update({f"{net}_output": int(o["ok"]), f"{net}_output_golden": o["used"], f"{net}_output_default_golden": int(o["default_ok"]),
                        f"{net}_late": sum(f.late for f in nf), f"{net}_late_frame0": int(any(f.late and f.frame == 0 for f in nf)),
                        f"{net}_worst_lateness_ms": max(f.lateness for f in nf),
                        f"{net}_frame_p50_ms": med(f.actual_run_ms for f in nf if f.frame > 0)})
        run_rows.append(row)
    write_csv(os.path.join(a.out, "runs.csv"), run_rows)
    write_csv(os.path.join(a.out, "frames.csv.gz"), [dict(
        run=f.rep, network=f.net, frame=f.frame, planned_ms=f.planned_run_ms, board_ms=f.actual_run_ms, start_offset_ms=f.gate_offset_ms,
        finish_after_release_ms=f.actual_ms, deadline_ms=f.deadline_ms, late=int(f.late), lateness_ms=f.lateness,
        planned_late=int(f.planned_late), sum_gap_ms=f.sum_gap, sum_grab_ms=f.sum_grab, sum_run_minus_planned_ms=f.sum_run_extra,
        handoffs=f.handoffs, handoffs_other_model=f.handoffs_other_model, chain_gap_ms=f.chain_gap, chain_grab_ms=f.chain_grab,
        chain_run_minus_planned_ms=f.chain_run_extra, chain_planned_idle_ms=f.chain_planned_idle,
        chain_inherited_ms=f.chain_inherited, chain_started_at=f.chain_left_to or "own start", chain_layers=f.chain_len,
        residual_ms=f.residual) for fs in ftot.values() for f in fs])
    table = layer_table(steady, stamps)
    write_csv(os.path.join(a.out, "layers.csv"), table)
    hand = handoff_stats(steady, len(runs))
    with open(os.path.join(a.out, "handoff.json"), "w") as f:
        json.dump(dict(raw=a.raw, runs=len(runs), frames="1..N-1 of each model", unit="us", note="gap = start - ready, by where the entry it waited for last ran", **hand), f, indent=1)

    tree_check = None
    if a.profile_root:
        prof = T.read_profile(a.profile_root)
        diffs = [abs(prof[(r["network"], r["layer"], r["core_kind"], r["kernel_type"], r["cores"])]["ms"] - r["planned_ms"]) for r in table]
        tree_check = dict(root=a.profile_root, layers=len(diffs), max_abs_diff_us=max(diffs) * 1e3)

    allf = [f for fs in ftot.values() for f in fs]
    models = {}
    for net in T.NETS:
        nf = [f for f in allf if f.net == net]
        late = [f for f in nf if f.late]
        b = model_block([f for f in nf if f.frame > 0], [p for p in steady if p.net == net], len(runs))
        b.update(frames_per_run=len(nf) // len(runs), late_frames_total=len(late), late_frames_frame0=sum(f.frame == 0 for f in late),
                 runs_with_a_late_frame=len({f.rep for f in late}), late_frames_per_run=[sum(1 for f in late if f.rep == r.rep) for r in runs],
                 worst_lateness_ms=max(f.lateness for f in nf), planned_late_frames_per_run=sum(f.planned_late for f in nf) // len(runs),
                 frame_0=dict(planned_frame_ms=med(f.planned_run_ms for f in nf if f.frame == 0),
                              board_frame_p50_ms=med(f.actual_run_ms for f in nf if f.frame == 0),
                              board_frame_max_ms=max(f.actual_run_ms for f in nf if f.frame == 0),
                              discrepancy_p50_ms=med(p.finish_error for p in first if p.net == net)),
                 layer_keys=sum(1 for r in table if r["network"] == net),
                 multi_core_layers=sum(1 for r in table if r["network"] == net and r["cores"] > 1),
                 sum_run_p50_over_sum_planned=sum(r["run_p50_ms"] for r in table if r["network"] == net) / sum(r["planned_ms"] for r in table if r["network"] == net))
        models[net] = b
    late_per_run = [r["late_frames"] for r in run_rows]
    summary = dict(label=a.label, raw=a.raw, runs=len(runs), stamps=stamps, runs_ok=sum(r["ok"] for r in run_rows),
                   flagged_runs=[r["run"] for r in run_rows if not r["ok"]],
                   output_check={net: dict(passed=sum(r[f"{net}_output"] for r in run_rows), golden_used=sorted({r[f"{net}_output_golden"] for r in run_rows}),
                                           build_golden_passed=sum(r[f"{net}_output_default_golden"] for r in run_rows)) for net in T.NETS},
                   placement_pass=sum(r["placement"].startswith("PASS") for r in run_rows),
                   runs_with_any_late_frame=sum(1 for x in late_per_run if x), late_frames_per_run=late_per_run,
                   late_frames_per_run_median=med(float(x) for x in late_per_run), worst_lateness_ms=max(r["worst_lateness_ms"] for r in run_rows),
                   plan_vs_profile_tree=tree_check, models=models, discrepancy_all_layers_p50_ms=med(p.finish_error for p in steady),
                   handoff_gap_us={k: dict(per_run=round(v["per_run"], 1), p50=round(v["p50_us"], 2), mean=round(v["mean_us"], 2)) for k, v in hand["all_models"].items()})
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)

    print(f"{a.label}: runs ok {summary['runs_ok']}/{summary['runs']} flagged {summary['flagged_runs']}; placement PASS {summary['placement_pass']}; "
          f"output " + ", ".join(f"{n} {v['passed']} ({'/'.join(v['golden_used'])}; build golden {v['build_golden_passed']})" for n, v in summary["output_check"].items()))
    print(f"  runs with a late frame: {summary['runs_with_any_late_frame']}/{summary['runs']}; late frames per run {late_per_run} (median {summary['late_frames_per_run_median']:.0f}); "
          f"worst lateness {summary['worst_lateness_ms']:+.3f} ms")
    for net, b in models.items():
        c = b["chain"]["mean"]
        print(f"  {net:12s} planned {b['planned_frame_ms']:.3f} board p50 {b['board_frame_p50_ms']:.3f} p75 {b['board_frame_p75_ms']:.3f} max {b['board_frame_max_ms']:.3f} "
              f"(x{b['board_over_planned']:.3f}); late {b['late_frames_total']} in {b['runs_with_a_late_frame']} runs (frame 0: {b['late_frames_frame0']}), worst {b['worst_lateness_ms']:+.3f}; "
              f"planned late/run {b['planned_late_frames_per_run']}; discrepancy {b['discrepancy_p50_ms']:+.3f}; hand-offs/frame {b['handoffs_per_frame']:.1f}")
        print(f"  {'':12s} mean split of board-plan {c['board_minus_planned_ms']:+.3f}: gaps {c['gap_ms']:.3f} + cores {c['grab_ms']:.3f} + run-planned {c['run_minus_planned_ms']:+.3f} "
              f"- planned idle {c['planned_idle_ms']:.3f} + inherited {c['inherited_from_other_frames_ms']:+.3f}; residual {b['chain']['max_abs_residual_ms']:.1e}; "
              f"frame 0: board {b['frame_0']['board_frame_p50_ms']:.3f}")
    if tree_check:
        print(f"  planned == profile tree: max diff {tree_check['max_abs_diff_us']:.4f} us over {tree_check['layers']} layer keys")
    return 0 if summary["runs_ok"] == summary["runs"] else 1


if __name__ == "__main__":
    sys.exit(main())
