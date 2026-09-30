#!/usr/bin/env python3
"""Turn measured K1 board traces into a board-calibration table.

WHY THIS EXISTS. `results/codesign_feedback/k1_board_calibration.json` is the input the
whole outer loop runs on -- `--board-calibration` scales every dispatch cost by it -- and
this script is what produces it, so the table is reproducible and can be regenerated for
a different workload. The committed table is fit to 60 runs of ONE taskset
(dronet+fused_full+ffn_block+mlp_control, no yolo) rather than to a measurement of the
schedule under test, which is why its own `fallback_key` states
"EXTRAPOLATED for nets not in calibration set, e.g. yolo".

WHAT IT MEASURES. Per dispatch, `actual / predicted` where
  * `predicted` is `predicted_duration_ms`, which the runner stamps into the trace from
    the schedule it is executing, and
  * `actual` is `(actual_end_cycles - actual_start_cycles) / 24 MHz` -- rdtime ticks, see
    `xpu-rt/k1_trace.K1_RDTIME_HZ`.
Both come from the trace itself, so this needs no schedule file, no IR and no join: it
runs on any workload's traces, which is the point.

This is EXECUTION inflation and nothing else. Queue delay is excluded (the trace carries
it separately): a dispatch that waited is not a dispatch that ran slowly, and folding
queueing into a per-op cost would charge the scheduler twice for its own placement
decisions.

THE STATISTIC, and why it is the mean. The v1 table used the arithmetic mean of
per-sample ratios; `--validate-against` confirms this reproduces it: the SAME 48
per-dispatch keys, 40 of them within 2%, and 7 of 11 op keys within 2%. The 8 that
differ are all `dronet`, all high by 2-14%, and no tested variant gets closer -- RT
traces only, dropping the cold first instance, and trimming the top of each key's
distribution all make the match markedly worse (25/48, 13/48 and 9/48 within 2%). So the
v1 fit did something to the dronet samples this reconstruction cannot recover, and the
two absent op keys are ones whose every sample sits under the 0.1 ms floor. Mean-of-ratios is outlier-sensitive --
a dispatch of a few hundred rdtime ticks can show 18x purely from timer granularity --
so the POOLED op tier takes a `--min-pred-ms` floor (default 0.1 ms = 2400 ticks) while
the per-dispatch tier keeps every sample, because there the samples are the same code on
the same core and are directly comparable. `--stat median` is available and is the more
robust choice; it is not the default only because it would change every existing
outer-loop number.

THE AGGREGATE. The v1 `aggregate_multiplier` of 1.2608 matches
no single statistic over these 60 traces (mean 1.3348 at the 0.1 ms floor, median 1.1647,
mean with no floor 1.6858).
The aggregate emitted here is the mean under the stated floor, and both statistics are
recorded so a reader can see the spread rather than trust one number.

Three tiers are emitted, matching the lookup order in
`profile_loader._board_calibration_mult`:
  1. `per_dispatch_multiplier["net/dispatch_id"]` -- exact, measured, for nets in the trace
  2. `per_op_multiplier["op_kind"]`               -- pooled fallback, EXTRAPOLATED off-workload
  3. `aggregate_multiplier`                       -- last resort
and `coverage.nets_exact` names which nets are measured, so a consumer can tell
measurement from extrapolation instead of discovering it in a footnote.

Usage:
  # regenerate the deployed table and check it against the committed one
  scripts/emit_board_calibration.py \\
      --trace-glob 'results/k1_feedback_exact/board_runs*/[of]*_trace.csv' \\
      --validate-against results/codesign_feedback/k1_board_calibration.json \\
      --out /tmp/cal.json

  # calibrate a NEW workload from its own board runs
  scripts/emit_board_calibration.py --trace-glob 'results/<run>/*_trace.csv' \\
      --workload 'w5 ladder rung' --out results/codesign_feedback/k1_cal_w5.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import glob
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))

try:
    from k1_trace import K1_RDTIME_HZ  # the one true tick rate
except Exception:  # pragma: no cover - keep the script usable standalone
    K1_RDTIME_HZ = 24_000_000.0

#: Columns this tool needs. A trace missing any of them is reported, not skipped
#: silently -- an absent trace and an unreadable one read very differently.
NEEDED = ("network", "dispatch_id", "op", "predicted_duration_ms",
          "actual_start_cycles", "actual_end_cycles")

STATS = {"mean": statistics.fmean, "median": statistics.median}


def read_trace(path):
    """`[(net, dispatch_id, op, pred_ms, actual_ms)]`, or `(None, reason)` if unusable."""
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        missing = [c for c in NEEDED if c not in (rd.fieldnames or [])]
        if missing:
            return None, f"missing columns {missing}"
        out = []
        for r in rd:
            try:
                pred = float(r["predicted_duration_ms"])
                ticks = int(r["actual_end_cycles"]) - int(r["actual_start_cycles"])
            except (TypeError, ValueError):
                continue
            actual = ticks / K1_RDTIME_HZ * 1e3
            if pred <= 0 or actual <= 0:
                continue
            # THE NETWORK COLUMN IS ALREADY THE NETWORK. The trace carries `instance`
            # separately, so there is no suffix to strip -- and stripping one is
            # actively wrong: `yolov8_nano_64x96` ends in a digit, so trimming trailing
            # digits produced `yolov8_nano_64x`. That does not fail; it files every
            # yolo multiplier under a name no consumer looks up, so the calibration
            # silently covers four of five networks and reports the fifth as
            # extrapolated while holding its measurements the whole time. A network
            # name may end in a digit -- the same hazard `generate_xpurt_main.py` and
            # `decision_loop.py` both carry warnings about.
            out.append((r["network"], int(r["dispatch_id"]), r["op"], pred, actual))
    return out, None


def schedule_real_ids(sched_path):
    """`{network: {dispatch_id}}` for the dispatches a schedule actually costs.

    Zero-cost dispatches are included here: whether they appear is exactly what the
    alignment check has to see.
    """
    sched = json.load(open(sched_path))
    per = collections.defaultdict(set)
    for e in (sched.get("dispatches") or {}).values():
        mod = str(e.get("module_name") or "")
        net = (mod.split("$", 1)[0] if "$" in mod
               else str(e.get("job_name", "")).rstrip("0123456789"))
        per[net].add(int(e["id"]))
    return per


def aligned_networks(rows, sched_path, log):
    """Networks whose TRACE dispatch ids match the SCHEDULE's, so a per-dispatch key means
    what `profile_loader._board_calibration_mult` will look up.

    WHY THIS IS NOT PARANOIA. The runner renumbers: it records zero-cost ops (`chunk`,
    `split`, `slice`) with `dispatch_id = -1` and numbers the REMAINING dispatches from
    zero, while the schedule numbers every dispatch including those. So on a network
    containing zero-cost ops the two numberings diverge after the first one --
    `yolov8_nano_64x96` traces 90 ids as 0..89 where the schedule spans 0..97.

    Nothing about that is visible in the emitted table. Every ratio is computed within one
    trace row and is correct; only the KEY is wrong, so the multipliers are plausible,
    the file validates, and the solver silently costs each yolo dispatch with a different
    dispatch's measurement. The op tier is unaffected -- it is keyed by the row's own `op`
    -- which is why a misaligned network is better served by dropping its per-dispatch
    keys than by keeping them.
    """
    trace_ids = collections.defaultdict(set)
    for net, did, _op, _p, _a in rows:
        trace_ids[net].add(did)
    sched_ids = schedule_real_ids(sched_path)
    ok, bad = set(), {}
    for net, tids in sorted(trace_ids.items()):
        sids = sched_ids.get(net)
        if sids is None:
            bad[net] = "the schedule has no dispatches for this network"
        elif tids == sids:
            ok.add(net)
        elif tids <= sids:
            bad[net] = (f"the trace numbers {len(tids)} dispatches 0..{max(tids)} while "
                        f"the schedule spans 0..{max(sids)} ({len(sids)} dispatches); "
                        f"the runner renumbers around zero-cost ops, so a per-dispatch "
                        f"key would be applied to a different dispatch")
        else:
            bad[net] = (f"the trace has ids the schedule does not: "
                        f"{sorted(tids - sids)[:6]}")
    for net, why in bad.items():
        log(f"  MISALIGNED {net}: {why}")
        log(f"    -> per-dispatch keys for {net} are OMITTED; it is costed by its op "
            f"kinds, which are keyed by the row's own op and are unaffected")
    return ok, bad


def compare(got, want, tol):
    """`(n_common, n_off, [(key, got, want)])` -- how well two tiers agree."""
    common = [k for k in want if k in got]
    off = [(k, got[k], want[k]) for k in common
           if want[k] and abs(got[k] - want[k]) / abs(want[k]) > tol]
    return len(common), len(off), off


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="append", default=[],
                    help="a measured board trace csv (repeatable)")
    ap.add_argument("--trace-glob", action="append", default=[],
                    help="glob for traces (repeatable)")
    ap.add_argument("--stat", choices=sorted(STATS), default="mean",
                    help="mean reproduces the v1 table; median is more robust")
    ap.add_argument("--min-pred-ms", type=float, default=0.1,
                    help="floor for the POOLED op tier and the aggregate; the "
                         "per-dispatch tier always keeps every sample")
    ap.add_argument("--min-samples", type=int, default=1,
                    help="a per-dispatch key needs this many measurements to be emitted; "
                         "below it the dispatch falls back to its op kind")
    ap.add_argument("--workload", default=None,
                    help="human description; defaults to the nets actually measured")
    ap.add_argument("--schedule", default=None,
                    help="the schedule these traces were run from. Supply it: the runner "
                         "renumbers dispatches around zero-cost ops, so without a "
                         "schedule to check against, a per-dispatch key can be applied "
                         "to a DIFFERENT dispatch and nothing about the table looks "
                         "wrong. Networks that fail the check keep their op-kind "
                         "multipliers and lose their per-dispatch ones.")
    ap.add_argument("--validate-against", default=None,
                    help="an existing calibration json to diff the result against")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="relative tolerance for --validate-against")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    def log(s):
        print(s, flush=True)

    paths = list(a.trace)
    for g in a.trace_glob:
        paths += glob.glob(g if os.path.isabs(g) else os.path.join(REPO, g))
    paths = sorted({p for p in paths if os.path.exists(p)})
    if not paths:
        log("no traces found")
        return 2

    rows, used, bad = [], [], []
    for p in paths:
        got, why = read_trace(p)
        if got is None or not got:
            bad.append((p, why or "no usable rows"))
            continue
        rows += got
        used.append(p)
    for p, why in bad:
        log(f"  SKIPPED {os.path.basename(p)}: {why}")
    if not rows:
        log("no usable dispatch samples; nothing to calibrate")
        return 1
    log(f"read {len(used)} trace(s), {len(rows)} dispatch samples")

    stat = STATS[a.stat]
    per_key = collections.defaultdict(list)
    per_op = collections.defaultdict(list)
    pooled = []
    nets = set()
    for net, did, op, pred, actual in rows:
        ratio = actual / pred
        nets.add(net)
        per_key[f"{net}/{did}"].append(ratio)
        if pred >= a.min_pred_ms:
            pooled.append(ratio)
            if op:
                per_op[op].append(ratio)

    if a.schedule:
        ok_nets, bad_nets = aligned_networks(rows, a.schedule, log)
        if not bad_nets:
            log(f"  dispatch ids align with {os.path.basename(a.schedule)} for every "
                f"network")
    else:
        ok_nets, bad_nets = set(nets), {}
        log("  no --schedule given: per-dispatch key alignment is UNVERIFIED. If any "
            "network here contains zero-cost ops (chunk/split/slice) its keys are "
            "probably offset -- pass --schedule.")
    exact = {k: round(stat(v), 4) for k, v in sorted(per_key.items())
             if len(v) >= a.min_samples and k.rsplit("/", 1)[0] in ok_nets}
    opk = {k: round(stat(v), 4) for k, v in sorted(per_op.items())}
    if not pooled:
        log(f"every sample is below --min-pred-ms {a.min_pred_ms}; "
            "the op tier and aggregate would be empty")
        return 1
    agg = round(stat(pooled), 4)

    cal = {
        "schema": "k1_board_calibration/v2",
        "source": (f"{len(used)} board trace(s), real SpaceMiT K1; "
                   f"actual=(end-start)ticks/{K1_RDTIME_HZ / 1e6:.0f}MHz vs the "
                   f"predicted_duration_ms the runner stamped from the schedule"),
        "workload": a.workload or "+".join(sorted(nets)),
        "generated_by": "scripts/emit_board_calibration.py",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "measures": ("EXECUTION inflation only. Queue delay is carried separately by "
                     "the trace and excluded: a dispatch that waited is not a dispatch "
                     "that ran slowly, and folding queueing in would charge the "
                     "scheduler twice for its own placement."),
        "statistic": (f"{a.stat} of per-sample ratios; pooled op tier and aggregate "
                      f"floored at predicted >= {a.min_pred_ms} ms "
                      f"({a.min_pred_ms * K1_RDTIME_HZ / 1e3:.0f} rdtime ticks) so timer "
                      f"granularity cannot inflate a pooled multiplier"),
        "n_dispatch_samples": len(rows),
        "n_pooled_samples": len(pooled),
        "aggregate_multiplier": agg,
        "aggregate_median": round(statistics.median(pooled), 4),
        "aggregate_mean": round(statistics.fmean(pooled), 4),
        "primary_key": "network/dispatch_id (exact, for the measured workload)",
        "fallback_key": "op (generalizing; EXTRAPOLATED for nets not in coverage.nets_exact)",
        "alignment": {
            "schedule": a.schedule,
            "verified": bool(a.schedule),
            "networks_aligned": sorted(ok_nets),
            "networks_omitted_from_exact_tier": bad_nets,
            "note": ("the runner records zero-cost ops with dispatch_id -1 and numbers "
                     "the rest from zero, while the schedule numbers all of them, so on "
                     "a network with zero-cost ops the two numberings diverge. Ratios "
                     "are computed within a row and are always right; only the KEY can "
                     "be wrong, which is invisible in the emitted table -- hence the "
                     "check, and hence dropping a misaligned network's per-dispatch "
                     "keys rather than keeping plausible-looking wrong ones."),
        },
        "coverage": {
            "nets_exact": sorted(ok_nets),
            "nets_measured": sorted(nets),
            "n_exact_dispatch_keys": len(exact),
            "n_op_kind_keys": len(opk),
            "note": ("a net absent from nets_exact is costed by its op kinds or by the "
                     "aggregate, which is a prediction about that net, not a "
                     "measurement of it"),
        },
        "traces": [os.path.relpath(p, REPO) if p.startswith(REPO) else p for p in used],
        "traces_skipped": [{"path": os.path.relpath(p, REPO) if p.startswith(REPO) else p,
                            "why": w} for p, w in bad],
        "per_dispatch_multiplier": exact,
        "per_op_multiplier": opk,
    }

    if a.validate_against:
        ref_p = (a.validate_against if os.path.isabs(a.validate_against)
                 else os.path.join(REPO, a.validate_against))
        ref = json.load(open(ref_p))
        log(f"\n=== validate against {os.path.relpath(ref_p, REPO)} ===")
        report = {}
        for tier, mine in (("per_dispatch_multiplier", exact),
                           ("per_op_multiplier", opk)):
            want = ref.get(tier) or {}
            n_common, n_off, off = compare(mine, want, a.tol)
            missing = sorted(set(want) - set(mine))
            extra = sorted(set(mine) - set(want))
            log(f"{tier}: {n_common - n_off}/{len(want)} within {a.tol * 100:.0f}%"
                f"   (mine {len(mine)}, reference {len(want)}"
                f"{f', {len(missing)} reference keys absent here' if missing else ''}"
                f"{f', {len(extra)} new keys' if extra else ''})")
            for k, g, w in off[:8]:
                log(f"    {k:26} got {g:8.4f}  reference {w:8.4f}"
                    f"  ({(g - w) / w * 100:+.1f}%)")
            report[tier] = {"n_reference": len(want), "n_within_tol": n_common - n_off,
                            "n_off": n_off, "tol": a.tol,
                            "reference_keys_absent_here": missing, "new_keys": extra,
                            "off": [{"key": k, "got": g, "reference": w}
                                    for k, g, w in off]}
        ra = ref.get("aggregate_multiplier")
        if isinstance(ra, (int, float)):
            log(f"aggregate: got {agg:.4f}  reference {ra:.4f} "
                f"({(agg - ra) / ra * 100:+.1f}%)")
            report["aggregate"] = {"got": agg, "reference": ra}
        cal["validation"] = {"reference": os.path.relpath(ref_p, REPO), **report}

    out = a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(cal, open(out, "w"), indent=1)
    log(f"\nwrote {os.path.relpath(out, REPO) if out.startswith(REPO) else out}")
    log(f"  aggregate x{agg} ({a.stat}; median {cal['aggregate_median']}) "
        f"from {len(pooled)} pooled samples")
    log(f"  exact keys {len(exact)}   op-kind keys {len(opk)}")
    log(f"  nets measured: {', '.join(sorted(nets))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
