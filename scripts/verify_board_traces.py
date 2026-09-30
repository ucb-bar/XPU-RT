#!/usr/bin/env python3
"""Every board trace a documented check opens is either tracked or in an archive, byte for byte.

`measured_timing.py --verify` re-derives each recorded board constant from the trace it came from,
and `make_measured_gantt_pair.py` draws each Gantt row from the trace its sidecar names. Those traces
are 3-12 MB each, so `.gitignore` keeps them out of the tree and `results/codesign_feedback/archive_v3/`
holds them with their sha256 in the tracked MANIFEST -- the same arrangement the display dumps use.

The arrangement only means something if every trace a check reads is actually in one of those two
places. Without this, a clean clone plus the archives runs `--verify` against files that are not
there and reports "no runs" instead of failing, so a constant with no evidence behind it looks the
same as one with evidence.

    scripts/verify_board_traces.py [--archives <dir>]

Exit 0 when every such trace is tracked, or archived with the bytes on disk. A trace that is on disk
but in neither place is a FAIL; one that is in an archive with different bytes is also a FAIL.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import archive_members as AM                                 # noqa: E402  plain and .tar.gz alike


def tracked(path):
    rel = os.path.relpath(path, REPO)
    return subprocess.run(["git", "-C", REPO, "ls-files", "--error-unmatch", rel],
                          capture_output=True, text=True).returncode == 0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# Absolute paths of the archive members that sit at a repo-relative path, filled by main(). A clone
# that has not unpacked the archives has none of the untracked traces on disk, so a glob of the tree
# alone would find nothing and this check would pass on an empty set; each glob below also matches
# the archive members, component by component as glob does.
ARCHIVED: set = set()


def _glob(pattern):
    q = pattern.split("/")
    return set(glob.glob(pattern)) | {m for m in ARCHIVED if len(m.split("/")) == len(q)
                                      and all(fnmatch.fnmatchcase(a, b) for a, b in zip(m.split("/"), q))}


def wanted():
    """The traces a documented check opens, as that check globs for them."""
    import measured_timing as MT
    out = set()
    for pre in MT.SOLVER_ARMS:                               # measured_timing.verify(), solver arms
        out |= set(_glob(f"{MT.RES}/xpurt_long/trace_{pre}[0-9]_other_run1.csv"))
    # XPURT_POINTS is the other registry of board-measured XPU-RT arms, keyed by the run label itself
    # rather than a replicate prefix, so no SOLVER_ARMS glob reaches it. figure_constants.
    # check_registry() re-derives from it and reported "no trace on disk" for xpu_w2pg36ime in a clean
    # clone -- a constant whose evidence this script was not asking about.
    for lab in getattr(MT, "XPURT_POINTS", {}):
        out |= set(_glob(f"{MT.RES}/xpurt_long/trace_{lab}_other_run*.csv"))
    ros_tags = set()
    for arm, hz in MT.ROS_VANILLA:                           # ... and the ROS arms, pooled per rate
        ros_tags |= {os.path.basename(os.path.dirname(t))
                     for t in _glob(f"{MT.RES}/ros_traced/{hz}_{arm}_r*/trace.csv")}
    # ROS_SENSITIVITY is keyed by the run tag itself and is never pooled, so it has no (arm, hz) to
    # glob for. Its arms reached this set only when some Gantt sidecar happened to name them, which
    # made the coverage of a verified constant depend on which figures exist.
    ros_tags |= {t for t in getattr(MT, "ROS_SENSITIVITY", {})
                 if os.path.isdir(f"{MT.RES}/ros_traced/{t}")}
    for tag in ros_tags:
        # trace.csv is what a Gantt row draws, ctrl_gaps.csv is what the replayed cadence is cut from,
        # and summary.csv is what derive() actually re-reads for every ROS constant -- all three are
        # opened by a documented check, so all three belong to the guarantee this script states.
        out |= set(_glob(f"{MT.RES}/ros_traced/{tag}/trace.csv"))
        out |= set(_glob(f"{MT.RES}/ros_traced/{tag}/ctrl_gaps.csv"))
    if os.path.exists(f"{MT.RES}/ros_traced/summary.csv"):
        out.add(f"{MT.RES}/ros_traced/summary.csv")
    for m in (glob.glob(f"{REPO}/results/codesign_feedback/**/measured_gantt_*_metrics.json", recursive=True)
              + glob.glob(f"{REPO}/schedules/measured_gantt_*_metrics.json")):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        for k in ("source", "cpu_source"):                   # the trace each drawn Gantt row came from
            v = d.get(k)
            if isinstance(v, str) and v.endswith(".csv"):
                out.add(v if os.path.isabs(v) else os.path.join(REPO, v))
    return {p for p in out if os.path.exists(p) or p in ARCHIVED}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", default=os.path.join(REPO, "results/codesign_feedback/archive_v3"))
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    # index the archives by the tail of each member's path, as a trace is addressed either by
    # <run dir>/trace.csv or by its own basename under xpurt_long/
    idx = {}
    for t in AM.archives(a.archives):
        try:
            for m in AM.members(t):
                if m.isfile():
                    parts = m.name.split("/")
                    for k in (1, 2, 3):
                        if k <= len(parts):
                            idx.setdefault("/".join(parts[-k:]), []).append((t, m.name))
        except Exception as e:
            print(f"  ?? {os.path.basename(t)}: cannot read ({e})")
    ARCHIVED.update(os.path.join(REPO, n) for c in idx.values() for _, n in c
                    if n.startswith(("results/", "schedules/")))

    npass = nfail = nunpacked = 0
    for p in sorted(wanted()):
        rel = os.path.relpath(p, REPO)
        if tracked(p):
            npass += 1
            if a.verbose:
                print(f"PASS  {rel}: tracked")
            continue
        parts = rel.split("/")
        cands = []
        for k in (3, 2, 1):
            cands = idx.get("/".join(parts[-k:]), [])
            if cands:
                break
        if not cands:
            print(f"FAIL  {rel}: neither tracked nor in any archive")
            nfail += 1
            continue
        if not os.path.exists(p):
            # archived, and not unpacked into this tree: the archive is the only copy there is, so
            # there are no bytes to compare it with; a check that opens it needs the unpack first
            npass += 1
            nunpacked += 1
            if a.verbose:
                print(f"PASS  {rel}: in {os.path.basename(cands[0][0])}, not unpacked here")
            continue
        want = sha256(p)
        ok = False
        for tar, member in cands:
            if AM.member_sha256(tar, member) == want:
                ok = True
                if a.verbose:
                    print(f"PASS  {rel}: in {os.path.basename(tar)}, sha256 matches")
                break
        if ok:
            npass += 1
        else:
            print(f"FAIL  {rel}: in {os.path.basename(cands[0][0])} but the bytes differ")
            nfail += 1

    if nunpacked:
        print(f"\n{nunpacked} of them are archived but not unpacked into this tree; the checks that open"
              f" them need `tar -xzf <archive> -C {REPO}` first (docs/Artifact/external_data.md)")
    print(f"\n{npass} tracked or archived and matching, {nfail} failed")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
