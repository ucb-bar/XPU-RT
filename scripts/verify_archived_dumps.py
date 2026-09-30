#!/usr/bin/env python3
"""Every display dump a figure read is in an archive, byte for byte.

Panels A, a-d and the telemetry row are drawn from `figure_data.npz` dumps that are too large to
track (100-200 MB each), so they are ignored by pattern and archived as tars under
`results/codesign_feedback/archive_v3/`, with the tars' own sha256 in the tracked
`archive_v3/MANIFEST.sha256`.

That arrangement is only worth anything if the archived copy is the file the figure actually read.
This checks exactly that: for every figure sidecar, every recorded input that is not tracked by git
is looked up in the archives and its sha256 compared with the one the render recorded.

    scripts/verify_archived_dumps.py [--refined <dir>] [--archives <dir>]

Exit 0 when every untracked input is archived and matches. A tracked input needs no archive and is
not reported; an input that is neither tracked nor archived is a FAIL, because a clean clone plus the
archives could not redraw that figure.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import archive_members as AM                     # noqa: E402  plain .tar and the raw traces' .tar.gz


def tracked(path):
    rel = os.path.relpath(path, REPO)
    r = subprocess.run(["git", "-C", REPO, "ls-files", "--error-unmatch", rel],
                       capture_output=True, text=True)
    return r.returncode == 0


def symlink_members(archdir):
    """Every symlink an archive stores, as (tar, member, target).

    An archive is unpacked over a checkout, and a stored symlink replaces the one git put there.
    energy_dumps_v1.tar carried eighteen links naming an absolute path in the machine that packed
    it; extracting it into a clone replaced the relative links with dangling ones and then aborted
    on the first file that landed on one, so the clone got no energy dumps at all and nothing said
    why. An archive should carry files at real paths and leave the tree's own links alone.
    """
    out = []
    for t in AM.archives(archdir):
        try:
            for m in AM.members(t):
                if m.issym() or m.islnk():
                    out.append((os.path.basename(t), m.name, m.linkname))
        except Exception:
            pass
    return out


def archive_index(archdir):
    """member name -> (tar path, member) for every tar in the archive directory, indexed by the last
    three path components, which is how a dump is addressed (<search>/<arm>_figdata/<file>)."""
    idx = {}
    for t in AM.archives(archdir):
        try:
            for m in AM.members(t):
                if not m.isfile():
                    continue
                parts = m.name.split("/")
                # archives are rooted differently (some at the search directory, some at the arm's
                # own directory), so index every tail of the path and match on the longest that hits
                # a bare basename is too ambiguous to address a dump by, so index from two
                # components up; several archives can share a key and all candidates are kept
                for k in range(2, min(4, len(parts)) + 1):
                    idx.setdefault("/".join(parts[-k:]), []).append((t, m.name))
        except Exception as e:                       # a truncated or unreadable tar is reported, not fatal
            print(f"  ?? {os.path.basename(t)}: cannot read ({e})")
    return idx


def frames_index(archdir):
    """dump-directory name -> the number of frames/*.npz an archive holds for it.

    Panels a-d draw frames/frame_NNN.npz, but a sidecar records only the figure_data.npz it hashes,
    so the input walk below cannot see them: a tar of npz files alone passed every check while a
    clean clone could not render the figure at all (scripts/repro_clean_clone.sh, which died on
    frames/frame_061.npz). This counts them so that hole is a FAIL rather than a silence.
    """
    n = {}
    for t in AM.archives(archdir):
        try:
            for m in AM.members(t):
                if not m.isfile() or "/frames/" not in m.name:
                    continue
                dump = m.name.split("/frames/")[0].split("/")[-1]
                n[dump] = n.get(dump, 0) + 1
        except Exception:
            pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined", default=os.path.join(REPO, "results", "codesign_feedback", "refined"))
    ap.add_argument("--archives", default=os.path.join(REPO, "results", "codesign_feedback", "archive_v3"))
    a = ap.parse_args()

    if not os.path.isdir(a.archives):
        print(f"FAIL  no archive directory at {a.archives}")
        return 1
    idx = archive_index(a.archives)
    fidx = frames_index(a.archives)
    print(f"{len(idx)} files indexed across {len(AM.archives(a.archives))} archives\n")

    npass = nfail = nskip = nstale = nunpacked = 0
    seen_frames = set()
    for side in sorted(glob.glob(os.path.join(a.refined, "*_metrics.json"))):
        try:
            m = json.load(open(side))
        except Exception:
            continue
        if not isinstance(m, dict) or not isinstance(m.get("inputs"), dict):
            continue
        stem = os.path.basename(side).replace("_metrics.json", "")
        for p, sha in sorted(m["inputs"].items()):
            f = p if os.path.isabs(p) else os.path.join(REPO, p)
            if tracked(f):
                continue                             # tracked inputs travel with the clone
            absent = not os.path.exists(f)           # e.g. a clone that has not unpacked the archives
            parts = p.split("/")
            # Every tail length, pooled -- not the longest that hits. A three-component tail such as
            # search_c1.4/xpu_s1007_figdata/figure_data.npz is not unique: two campaigns use the same
            # search directory and seed. Stopping at the first tail that matched picked whichever tar
            # happened to hold that name and then failed on the sha256, reporting a missing dump that
            # was in fact archived under a longer path. Pooling the candidates lets the hash decide.
            cands, key = [], "/".join(parts[-2:])
            for k in range(min(5, len(parts)), 1, -1):
                for c in idx.get("/".join(parts[-k:]), []):
                    if c not in cands:
                        cands.append(c)
                if cands and k > 2:
                    key = "/".join(parts[-k:])
            hit = cands or None
            if absent:
                # nothing on disk to require, as before; but an archived copy of the recorded bytes is
                # counted, so a clone that has not unpacked reports what the archives hold for it
                if hit and any(AM.member_sha256(t, mb) == sha for t, mb in hit):
                    npass += 1
                    nunpacked += 1
                continue
            if hit is None:
                print(f"FAIL  {stem}: {key} is neither tracked nor in any archive")
                nfail += 1
                continue
            found = None
            for tar, member in hit:
                if AM.member_sha256(tar, member) == sha:
                    found = tar
                    break
            # the strips beside it: an npz alone is not a renderable dump
            if p.endswith("figure_data.npz"):
                dump = os.path.basename(os.path.dirname(f))
                have = fidx.get(dump, 0)
                on_disk = len(glob.glob(os.path.join(os.path.dirname(f), "frames", "*.npz")))
                if (dump, on_disk) not in seen_frames:
                    seen_frames.add((dump, on_disk))
                    if on_disk and have < on_disk:
                        print(f"FAIL  {stem}: {dump}/frames/ has {on_disk} frames on disk but "
                              f"{have} in any archive — panels a-d cannot be drawn from a clean clone")
                        nfail += 1
            if found:
                print(f"PASS  {stem}: {key} in {os.path.basename(found)}, sha256 matches the render")
                npass += 1
                continue
            where = ", ".join(sorted({os.path.basename(t) for t, _ in hit}))
            # The file in this tree is no longer the one the render read. That is the render's
            # staleness, which this tree has with or without the archives (a tracked input in the same
            # state is not compared at all); what this check owns is that the archive holds the bytes
            # the tree holds. So an archive copy of the current bytes is reported as STALE, not FAIL.
            hd = hashlib.sha256()
            with open(f, "rb") as fo:
                for chunk in iter(lambda: fo.read(1 << 20), b""):
                    hd.update(chunk)
            if any(AM.member_sha256(t, mb) == hd.hexdigest() for t, mb in hit):
                print(f"STALE {stem}: {key} is archived ({where}) with the bytes on disk, which the "
                      f"render predates (it recorded {sha[:12]}, the file is {hd.hexdigest()[:12]})")
                nstale += 1
            else:
                print(f"FAIL  {stem}: {key} is archived ({where}) but no copy matches what the render read")
                nfail += 1

    # Panel D's bars are labelled with each arm's control rate, and hil_story_figure._cond_rate_hz
    # reads that out of energy_runs*/<cond>_s*/figure_data.npz -- a render input that no sidecar
    # hashes, so the walk above cannot see it. Without this, untracking those dumps (which the
    # repository's own policy in docs/Artifact/artifact_checklist.md §1 calls for) would leave panel D
    # unlabelable from a clean clone with every check still green.
    for side in sorted(glob.glob(os.path.join(a.refined, "*_metrics.json"))):
        try:
            m = json.load(open(side))
        except Exception:
            continue
        D = m.get("D") if isinstance(m, dict) else None
        if not isinstance(D, dict) or not isinstance(D.get("conditions"), dict):
            continue
        ecsv = D.get("energy_csv") or ""
        base = os.path.dirname(ecsv if os.path.isabs(ecsv) else os.path.join(REPO, ecsv))
        stem = os.path.basename(side).replace("_metrics.json", "")
        for cond in sorted(D["conditions"]):
            hits = sorted(glob.glob(os.path.join(base, "energy_runs*", f"{cond}_s*", "figure_data.npz")))
            if not hits:
                continue                             # nothing on disk to require
            f = hits[0]                              # _cond_rate_hz takes the first sorted
            if tracked(f):
                continue
            rel = os.path.relpath(f, REPO)
            cands = idx.get("/".join(rel.split("/")[-3:]), []) + idx.get("/".join(rel.split("/")[-2:]), [])
            sha = hashlib.sha256()
            with open(f, "rb") as fo:
                for chunk in iter(lambda: fo.read(1 << 20), b""):
                    sha.update(chunk)
            ok = False
            for tar, member in cands:
                if AM.member_sha256(tar, member) == sha.hexdigest():
                    ok = True
                    break
            if ok:
                npass += 1
            else:
                print(f"FAIL  {stem}: panel D's rate label for {cond} reads "
                      f"{'/'.join(rel.split('/')[-3:])}, which is neither tracked nor archived")
                nfail += 1

    # The manifest is what a reviewer checks a downloaded tar against, and nothing checked the
    # manifest itself: a tar added without a line, or a line whose size no longer matches its file,
    # is invisible until someone tries to use it. Sizes are read from the directory entry, so this
    # costs nothing; the sha256 of 26 GB of tars is a deliberate omission, not an oversight.
    for tn, mn, tgt in symlink_members(a.archives):
        print(f"FAIL  {tn} stores a symlink, {mn} -> {tgt}; unpacking it would replace the "
              f"checkout's own link")
        nfail += 1

    mp = os.path.join(a.archives, "MANIFEST.sha256")
    if os.path.exists(mp):
        listed = {}
        for line in open(mp):
            f = line.split()
            if len(f) >= 3:
                listed[f[2]] = int(f[1])
        present = {f for f in os.listdir(a.archives) if f != "MANIFEST.sha256"}
        for name in sorted(present - set(listed)):
            print(f"FAIL  {name} is in the archive directory and not in MANIFEST.sha256")
            nfail += 1
        for name in sorted(set(listed) - present):
            print(f"FAIL  MANIFEST.sha256 lists {name}, which is not in the archive directory")
            nfail += 1
        for name in sorted(present & set(listed)):
            sz = os.path.getsize(os.path.join(a.archives, name))
            if sz != listed[name]:
                print(f"FAIL  {name} is {sz} bytes; MANIFEST.sha256 says {listed[name]}")
                nfail += 1
            else:
                npass += 1

    print(f"\n{npass} archived and matching, {nfail} failed, {nskip} skipped, "
          f"{nstale} stale renders (input archived, render predates it), "
          f"{nunpacked} archived and not unpacked here")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
