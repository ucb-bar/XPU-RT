#!/usr/bin/env python3
"""The start-barrier offset, and the makespan corrected for it.

WHAT THIS IS, IN ONE SENTENCE: both runtimes are re-timed from their own first
dispatch, so that neither is charged for the other's startup. It is a HARNESS
CORRECTION, not a modelling choice -- no schedule, no placement and no cost
model is touched by it, and it is applied to both sides by the same rule.

WHY IT IS NEEDED. XPU-RT's measured non-periodic makespan is taken from t0 of
its run loop, and on some runs the run loop's gate releases late: the trace of
`perception_heavy_quad__greedy` rep1 has BOTH first entries scheduled at
`predicted_start_ms = 0.000`, on two different lanes, and both actually
starting at ~1.62 ms, with `dep_wait_ms` of 0.002 and 0.028 -- so nothing was
waiting on a dependency. `gate_ms` for those two entries is 1.606 and 1.617,
i.e. the wait is at the gate itself. yolov8's execution is
6.462 - 1.622 = 4.84 ms against the ROS baseline's ~4.73 ms, so the compute
agrees to ~2% and essentially the whole 1.35x "pinning win" on that cell was
the head start.

It is INTERMITTENT, which is worse than constant: the same point measures
0.025 ms on one rep and 1.638 ms on the next (`perception_heavy_dc__greedy`),
so it moves the median of 3 rather than shifting every number equally.

THE CORRECTION.

    np_corrected = (end of the last aperiodic operation)
                 - (start of the FIRST dispatch anywhere in that run)

per rep, then the median over reps as before. On a cell with no aperiodic
network the non-periodic objective already degenerates to the all-operations
makespan (exactly as `evaluate()` does) and the same subtraction is applied,
with the caveat in `release_bound` below.

BOTH SIDES, SAME RULE. The ROS harness has the same class of delay -- its
per-run offset is a median 0.044 ms, so it is smaller, but it is not zero and
correcting only the slower side would be the same error with the sign flipped.

WHERE THE CORRECTION IS NOT MEANINGFUL, STATED RATHER THAN HIDDEN. Subtracting
the head start is right when the timed interval ENDS on work that inherited it.
On a cell whose last timed operation is pinned by a periodic RELEASE, it does
not: the release clock is not delayed by the gate (in the trace above, entry 2
is scheduled at 4.189 ms and starts at 4.196 ms, on time, in the same run whose
first dispatch was 1.6 ms late), so the end does not move and subtracting the
offset flatters that run. `release_bound()` flags exactly those runs from the
trace -- the last timed operation started on schedule despite a head start --
and the analysis reports them separately instead of letting them into a
headline.

ROOT CAUSE IS OUT OF SCOPE HERE and is left as future work. The runtime does
two iterations (`FLOWC_ITERATIONS=2`) and the trace is the second, so
first-touch on the fastRPC path or SCHED_FIFO lane spin-up bleeding across the
iteration boundary are the likely candidates; the evidence in this module is
that the delay sits in `gate_ms` and not in `dep_wait_ms`.
"""
from __future__ import annotations

import csv
import io
import os
import re
import statistics

XPURT_TRACE = re.compile(
    r"MODELBLASTER_XPURT_TRACE_BEGIN[^\n]*\n(.*?)\n[^\n]*"
    r"MODELBLASTER_XPURT_TRACE_END", re.S)
ROS_TRACE = re.compile(
    r"ROS_PINSWEEP_TRACE_BEGIN[^\n]*\n(.*?)\n[^\n]*"
    r"ROS_PINSWEEP_TRACE_END", re.S)
UNIT_MS = {"us": 1e-3, "ms": 1.0, "ns": 1e-6}

#: below this the delay is ordinary dispatch jitter, not a barrier; used only
#: to decide whether a run HAD a head start worth asking about
HEADSTART_MS = 0.5


def _rows(path, pat):
    if not os.path.exists(path):
        return []
    m = pat.search(open(path, errors="replace").read())
    if not m:
        return []
    return list(csv.DictReader(io.StringIO(m.group(1).strip())))


def xpurt_rep(path, aperiodic):
    """One XPU-RT rep: raw np, offset, corrected np, and the release-bound flag.

    `aperiodic` is the set of non-periodic network names. Returns None if the
    log carries no usable trace.
    """
    rows = _rows(path, XPURT_TRACE)
    recs = []
    for r in rows:
        try:
            u = UNIT_MS.get((r.get("time_unit") or "us").strip(), 1e-3)
            recs.append(dict(
                net=(r.get("network") or "").strip(),
                start=float(r["actual_start_cycles"]) * u,
                end=float(r["actual_end_cycles"]) * u,
                pred=float(r.get("predicted_start_ms") or 0.0),
                gate=float(r.get("gate_ms") or 0.0),
                dep=float(r.get("dep_wait_ms") or 0.0)))
        except (TypeError, ValueError, KeyError):
            continue
    if not recs:
        return None
    timed = [x for x in recs if x["net"] in aperiodic] or recs
    off = min(x["start"] for x in recs)
    last = max(timed, key=lambda x: x["end"])
    return dict(raw_ms=round(last["end"], 4), offset_ms=round(off, 4),
                corrected_ms=round(last["end"] - off, 4),
                degenerate=not any(x["net"] in aperiodic for x in recs),
                release_bound=release_bound(off, last),
                last_pred_start_ms=round(last["pred"], 4),
                last_actual_start_ms=round(last["start"], 4),
                first_gate_ms=round(min(recs, key=lambda x: x["start"])["gate"], 4),
                first_dep_wait_ms=round(min(recs, key=lambda x: x["start"])["dep"], 4),
                n_rows=len(recs))


def release_bound(offset_ms, last):
    """Did the head start reach the end of the timed interval, or was it absorbed?

    True means it was absorbed: the run had a head start worth the name, but
    the operation that closes the timed interval still started on its
    scheduled release, so the interval did NOT move and subtracting the offset
    would credit the run with time it never lost. Those runs are reported
    apart from the headline rather than silently corrected.
    """
    if offset_ms < HEADSTART_MS:
        return False
    late = last["start"] - last["pred"]
    return late < offset_ms / 2.0


def ros_reps(path, aperiodic):
    """The ROS harness, per measured rep.

    The harness runs a discarded warm pass and a measured pass per rep; only
    `warm == 0` rows are the measurement, and each rep has its own t0, so the
    offset is taken per rep exactly as it is on the XPU-RT side.
    """
    rows = _rows(path, ROS_TRACE)
    per = {}
    for r in rows:
        if (r.get("warm") or "").strip() != "0":
            continue
        try:
            per.setdefault(r["rep"], []).append(dict(
                net=(r.get("network") or "").strip(),
                start=float(r["start_ms"]), end=float(r["end_ms"]),
                pred=float(r.get("release_ms") or 0.0)))
        except (TypeError, ValueError, KeyError):
            continue
    out = []
    for rep in sorted(per, key=lambda k: int(k)):
        recs = per[rep]
        timed = [x for x in recs if x["net"] in aperiodic] or recs
        off = min(x["start"] for x in recs)
        last = max(timed, key=lambda x: x["end"])
        out.append(dict(rep=int(rep), raw_ms=round(last["end"], 4),
                        offset_ms=round(off, 4),
                        corrected_ms=round(last["end"] - off, 4),
                        release_bound=release_bound(off, last)))
    return out


def summarize(reps, key):
    """Median / spread over reps, in the shape the rest of the pipeline uses."""
    v = [r[key] for r in reps if r.get(key) is not None]
    if not v:
        return None, None, []
    return round(statistics.median(v), 4), round(max(v) - min(v), 4), v
