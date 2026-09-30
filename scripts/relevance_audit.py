#!/usr/bin/env python3
"""What the big directories under results/ are for, and which of them nothing needs.

`results/` is ~142 GB on disk of which about 5 GB is tracked and 26 GB is the `archive_v3` tars.
The rest is campaign and energy run output. Some of it is what the figures are derived from, some is
already inside an archive, and some is neither -- output from a run nothing ended up drawing. Until
now no check could tell those apart, because every check starts from a figure and walks to its
inputs; nothing walks the other way, from a directory to whoever wants it.

This does that walk. For every directory over a threshold it reports one of:

  input        a sidecar's `inputs` or a registry names a file inside it -- a figure reads it
  mentioned    only its name appears in a tracked doc or script -- prose, not a dependency
  orphaned     neither -- nothing in the repository asks for it

`input` and `mentioned` are kept apart deliberately. Nearly every directory here is named in some
page, so folding prose into "referenced" reports 141 GB as needed and hides exactly what this is
for. Only `input` means a figure would break.

    scripts/relevance_audit.py [--min-gb 0.1] [--out attic/disk_relevance.md] [--verify DIR]

Archive membership is **not** reported per directory, because the cheap test for it does not work.
Indexing tar members by the last two or three path components collides on the basenames these runs
all share -- `figure_data.npz`, `manifest.json` -- so a name match says almost nothing. Measured:
`results/codesign_feedback/midground` matches 96% by name and, byte-verified, has **1 of 88 files**
in any archive. So the report says "not established" and `--verify DIR` does the real check, by
sha256, for one directory at a time. Run it before deleting anything.

This script only reads and reports. It deletes nothing: an orphaned directory may be the only copy
of a run somebody still wants, and that call is not this script's to make.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
ARCH = os.path.join(RES, "codesign_feedback/archive_v3")
sys.path.insert(0, os.path.join(REPO, "scripts"))


STOP = {"results", "codesign", "feedback", "runs", "run", "data", "out", "csv", "json", "v2", "v3"}


def summaries_for(rel, tracked_all):
    """Tracked files that look like the summary this run produced.

    A run directory is usually reduced to one small tracked table -- the thing every check actually
    reads -- and the two share the name of the sweep. Matching on shared name tokens finds that
    pairing without knowing anything about which sweeps exist, so it keeps working for the next one.
    """
    want = {t for t in re.split(r"[_.\-]", os.path.basename(rel)) if t and t not in STOP}
    out = []
    for f in tracked_all:
        toks = {t for t in re.split(r"[_.\-]", os.path.basename(f)) if t and t not in STOP}
        if len(want & toks) >= 2 and not f.startswith(rel + "/"):
            out.append(f)
    return sorted(out)[:4]


def tracked_in(rel):
    """(files, bytes) that git tracks inside a directory.

    This is the signal that matters and it needs no per-directory knowledge: whatever a check reads
    has to be tracked or archived, so the tracked part of a directory is its durable content and
    the rest is the run output it was derived from.
    """
    out = subprocess.run(["git", "-C", REPO, "ls-files", "-z", rel],
                         capture_output=True, text=True).stdout.split("\0")
    n = b = 0
    for f in out:
        p = os.path.join(REPO, f) if f else None
        if p and os.path.exists(p):
            n += 1
            b += os.path.getsize(p)
    return n, b


def du(path):
    r = subprocess.run(["du", "-sb", path], capture_output=True, text=True)
    try:
        return int(r.stdout.split()[0])
    except Exception:
        return 0


def big_dirs(min_bytes):
    """Every directory worth asking about: the top level of results/ and of codesign_feedback/."""
    out = {}
    for parent in (RES, os.path.join(RES, "codesign_feedback")):
        for name in sorted(os.listdir(parent)):
            p = os.path.join(parent, name)
            if not os.path.isdir(p) or os.path.islink(p):
                continue
            if p == os.path.join(RES, "codesign_feedback"):
                continue                              # reported through its children
            if p == ARCH:
                continue                              # the archive is the thing being restored FROM
            n = du(p)
            if n >= min_bytes:
                out[os.path.relpath(p, REPO)] = n
    return out


def referenced_paths():
    """Every path the repository asks for, from the three places that name one."""
    named = set()
    # 1. every sidecar's recorded inputs, at any depth
    for side in glob.glob(os.path.join(RES, "**", "*_metrics.json"), recursive=True):
        try:
            j = json.load(open(side))
        except Exception:
            continue
        # some sidecars are a list of per-round records (hil_feedback), not a mapping
        if isinstance(j, list):
            for e in j:
                if isinstance(e, dict):
                    for kk, vv in e.items():
                        if isinstance(vv, str) and "/" in vv:
                            named.add(vv)
            continue
        if not isinstance(j, dict):
            continue
        for k in ("inputs", "sources"):
            v = j.get(k)
            if isinstance(v, dict):
                named |= set(v)
            elif isinstance(v, list):
                named |= {x for x in v if isinstance(x, str)}
        for key in ("xpu_dir", "ros_dir", "source", "cpu", "manifest", "trace"):
            v = j.get(key)
            if isinstance(v, str):
                named.add(v)
    # 2. what the board-trace check already enumerates
    try:
        import verify_board_traces as VB
        w = VB.wanted()
        named |= set(w if isinstance(w, (set, list)) else w.keys())
    except Exception as e:
        print(f"  (verify_board_traces.wanted() unavailable: {e})", file=sys.stderr)
    return {n.replace(REPO + "/", "") for n in named}


def text_mentions(dirnames):
    """Directory basenames named anywhere in a tracked doc, script or registry."""
    files = subprocess.run(["git", "-C", REPO, "ls-files", "docs", "scripts", "artifact"],
                           capture_output=True, text=True).stdout.split()
    hay = []
    for f in files:
        try:
            hay.append(open(os.path.join(REPO, f), errors="ignore").read())
        except Exception:
            pass
    blob = "\n".join(hay)
    return {d for d in dirnames if d in blob}


def archive_names():
    """Every member name in every archive tar, by basename and by last two components."""
    import verify_archived_dumps as VA
    idx = VA.archive_index(ARCH)
    keys = set(idx)
    return keys, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gb", type=float, default=0.1)
    ap.add_argument("--out", default=os.path.join(REPO, "attic/disk_relevance.md"))
    ap.add_argument("--verify", help="byte-verify one directory against the archives and exit")
    a = ap.parse_args()

    if a.verify:
        return verify_one(a.verify)

    print("sizing directories ...", flush=True)
    dirs = big_dirs(int(a.min_gb * 2**30))
    print(f"  {len(dirs)} directories at or above {a.min_gb} GB", flush=True)

    print("collecting what the repository asks for ...", flush=True)
    refs = referenced_paths()
    mentioned = text_mentions({os.path.basename(d) for d in dirs})
    print(f"  {len(refs)} paths named by a sidecar or registry; "
          f"{len(mentioned)} directory names appear in a tracked file", flush=True)

    tracked_all = [f for f in subprocess.run(["git", "-C", REPO, "ls-files", "results"],
                                             capture_output=True, text=True).stdout.split() if f]
    rows = []
    for d, n in sorted(dirs.items(), key=lambda kv: -kv[1]):
        base = os.path.basename(d)
        is_input = any(r.startswith(d + "/") or r == d for r in refs)
        is_mentioned = base in mentioned
        files = sum(len(fs) for _, _, fs in os.walk(os.path.join(REPO, d)))
        tn, tb = tracked_in(d)
        rows.append({"dir": d, "bytes": n, "files": files,
                     "tracked_files": tn, "tracked_bytes": tb,
                     "input": is_input, "mentioned": is_mentioned,
                     "summaries": summaries_for(d, tracked_all),
                     "class": ("input" if is_input else "mentioned" if is_mentioned else "orphaned")})
        print(f"  {d:<56} {n/2**30:7.2f} GB  {rows[-1]['class']}", flush=True)

    by = {}
    for r in rows:
        by.setdefault(r["class"], []).append(r)
    write_doc(a.out, rows, by)
    print(f"\nwrote {os.path.relpath(a.out, REPO)}")
    for k in ("input", "mentioned", "orphaned"):
        v = by.get(k, [])
        print(f"  {k:<11} {len(v):>3} dirs, {sum(x['bytes'] for x in v)/2**30:8.2f} GB")
    return 0


def verify_one(rel):
    """Byte-verify every file of one directory against the archives. The check before a delete.

    Grouped by tar and hashed in one pass per archive. The obvious shape -- for each file, open the
    tar it might be in -- reopens and rescans a multi-GB archive once per file, which on a
    2779-file directory is thousands of full scans and does not finish. Here each tar is opened
    once, streamed through, and every member that some file wants is hashed as it goes.
    """
    import verify_archived_dumps as VA
    import tarfile
    import collections
    idx = VA.archive_index(ARCH)
    root = os.path.join(REPO, rel)

    # local file -> its sha256, and which (tar, member) candidates could hold it
    want, by_tar, miss = {}, collections.defaultdict(set), 0
    for dirpath, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dirpath, f)
            parts = os.path.relpath(p, REPO).split("/")
            cands = []
            for k in (3, 2):
                cands += idx.get("/".join(parts[-k:]), [])
            if not cands:
                miss += 1
                continue
            want[p] = (hashlib.sha256(open(p, "rb").read()).hexdigest(), cands)
            for tar, member in cands:
                by_tar[tar].add(member)

    # one pass per archive, hashing only the members something asks for
    have = {}
    for tar, members in by_tar.items():
        try:
            with tarfile.open(tar) as tf:
                for m in tf:
                    if m.name in members and m.isfile():
                        fo = tf.extractfile(m)
                        if fo:
                            have[(tar, m.name)] = hashlib.sha256(fo.read()).hexdigest()
        except Exception as e:
            print(f"  ?? {os.path.basename(tar)}: {e}")

    ok = bad = 0
    for p, (sha_, cands) in want.items():
        if any(have.get((t, m)) == sha_ for t, m in cands):
            ok += 1
        else:
            bad += 1
    print(f"{rel}: {ok} byte-identical in an archive, {bad} present but differing, "
          f"{miss} not in any archive")
    print("safe to delete and restore from the archives" if (bad == 0 and miss == 0)
          else "NOT fully recoverable from the archives -- do not delete")
    return 0 if (bad == 0 and miss == 0) else 1


def write_doc(path, rows, by):
    L = ["# What the big directories under `results/` are for", "",
         "Generated by `scripts/relevance_audit.py`. It walks from each directory to whoever asks",
         "for it, which is the opposite direction from every other check here: those start at a",
         "figure and reach its inputs, so a directory nothing draws from is invisible to them.", "",
         "| class | meaning |", "|---|---|",
         "| `input` | a sidecar's `inputs` or a registry names a file inside it — **a figure reads it** |",
         "| `mentioned` | only its name appears in a tracked doc or script — prose, not a dependency |",
         "| `orphaned` | neither — nothing in the repository asks for it |", "",
         "`input` and `mentioned` are kept apart deliberately. Nearly every directory here is named",
         "in some page, so folding prose into one \"referenced\" class reports the whole tree as",
         "needed and hides what this report is for. Only `input` means a figure would break.", "",
         "**Whether a directory is recoverable from an archive is not reported here.** The cheap",
         "test for it does not work: indexing tar members by their last path components collides",
         "on the basenames these runs all share, so a name match says almost nothing. Measured,",
         "`results/codesign_feedback/midground` matches 96% by name and has **1 of 88 files** in",
         "any archive once the bytes are checked. Run `scripts/relevance_audit.py --verify <dir>`",
         "for the real answer, one directory at a time, before acting on any row.", "",
         "Nothing here has been deleted: an orphaned directory may be the only copy of a run",
         "somebody still wants, and that is not a script's call.", "",
         "## Totals", "", "| class | directories | size |", "|---|---|---|"]
    for k in ("input", "mentioned", "orphaned"):
        v = by.get(k, [])
        L.append(f"| `{k}` | {len(v)} | {sum(x['bytes'] for x in v)/2**30:.1f} GB |")
    raw = sum(r["bytes"] - r["tracked_bytes"] for r in rows)
    trk = sum(r["tracked_bytes"] for r in rows)
    L += ["", "## Raw output against what is kept", "",
          f"Across these {len(rows)} directories: **{raw/2**30:.1f} GB** of untracked run output",
          f"behind **{trk/2**20:.1f} MB** of tracked files. The tracked part is what every check",
          "reads — a sidecar can only name an input that a clone will have — so the ratio is the",
          "answer to what the tree is carrying. It is not waste by itself: the raw output is what",
          "the tracked summaries were derived from, and deleting it costs the ability to re-derive",
          "them, not the ability to redraw a figure.", "",
          "| directory | size | files | tracked | class |", "|---|---|---|---|---|"]
    for r in rows:
        t = f"{r['tracked_files']} files, {r['tracked_bytes']/1024:.0f} KB" if r["tracked_files"] else "—"
        L.append(f"| `{r['dir']}` | {r['bytes']/2**30:.2f} GB | {r['files']} | {t} | "
                 f"`{r['class']}` |")
    orph = by.get("orphaned", []) + by.get("mentioned", [])
    if orph:
        L += ["", "## The orphans", "",
              f"{len(orph)} directories totalling **{sum(x['bytes'] for x in orph)/2**30:.1f} GB** have",
              "no figure reading anything inside them. `mentioned` ones are",
              "named in prose only; `orphaned` ones are not named at all. That is the reclaimable",
              "space, and it is also the list to read before reclaiming it — a run that produced no",
              "figure is still a run that was made, and deleting it is not reversible.", ""]
        L += ["A run directory is usually reduced to one small tracked table, and that table is what",
              "every check reads. Where one was found by name, it is given below: deleting the",
              "directory then costs the ability to re-derive that table, not the ability to redraw",
              "anything.", ""]
        for r in sorted(orph, key=lambda x: -x["bytes"]):
            sm = (" → " + ", ".join(f"`{os.path.basename(x)}`" for x in r["summaries"])
                  if r["summaries"] else "")
            L.append(f"* `{r['dir']}` — {r['bytes']/2**30:.2f} GB, {r['files']} files, "
                     f"`{r['class']}`{sm}")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
