#!/usr/bin/env python3
"""Everything a flight needs, besides Isaac itself, is in the repository or an archive.

Isaac Sim is the reviewer's to install (`docs/Artifact/environment.md` pins the version and
`requirements-isaac.txt` the interpreter). Everything the flight drivers *load* is ours to supply:
the guidance and controller weights, the YOLO weights, the replayed control-cadence traces, and the
IsaacLab task package that builds the scene. A missing one of those is not a broken command, it is a
flight nobody else can fly -- and no other check looks at them, because a flight is not a figure and
leaves no sidecar.

    scripts/verify_flight_inputs.py [-v]

What it checks, per flight driver under sims/scripts/ and per driver script under scripts/:
  * every default file path in a driver's argparse (weights, models) exists and is tracked
  * every ctrl_traces/*.csv a driver names exists and is tracked
  * the IsaacLab submodule is pinned to a commit and the task package is tracked
  * the Isaac interpreter's requirements file is present

Exit 0 when a reviewer with Isaac has everything else.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILED = 0


def tracked(rel):
    return subprocess.run(["git", "-C", REPO, "ls-files", "--error-unmatch", rel],
                          capture_output=True, text=True).returncode == 0


def check(ok, msg):
    global FAILED
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        FAILED += 1


def model_defaults():
    """(path, the driver naming it) for every model file a driver defaults to or is documented with."""
    out = {}
    pat = re.compile(r"[\"']((?:sims/models|sims/assets)/[A-Za-z0-9_/.-]+\.(?:pt|pth|onnx))[\"']")
    srcs = glob.glob(os.path.join(REPO, "sims/scripts/*.py")) + glob.glob(os.path.join(REPO, "scripts/*.sh"))
    srcs += glob.glob(os.path.join(REPO, "docs/*.md"))
    for s in srcs:
        try:
            text = open(s, errors="ignore").read()
        except Exception:
            continue
        for m in pat.findall(text):
            out.setdefault(m, set()).add(os.path.relpath(s, REPO))
    # argparse defaults built with os.path.join(root, "sims/models/...") do not match the quoted form
    pat2 = re.compile(r"os\.path\.join\([^)]*?[\"'](sims/models/[A-Za-z0-9_/.-]+\.(?:pt|pth|onnx))[\"']")
    for s in glob.glob(os.path.join(REPO, "sims/scripts/*.py")):
        for m in pat2.findall(open(s, errors="ignore").read()):
            out.setdefault(m, set()).add(os.path.relpath(s, REPO))
    return out


# a verbatim archive of past commit messages, not a recipe: it quotes filenames that a change
# removed, and reading it as a driver would require keeping retired names alive forever. Matched on
# basename, so it stays held out now that the file lives in attic/ rather than docs/.
RECORDS = {"session_log.md"}


def ctrl_traces_named():
    out = {}
    pat = re.compile(r"ctrl_traces/([A-Za-z0-9_.]+\.csv)")
    docs = [d for d in glob.glob(os.path.join(REPO, "docs/*.md"))
            if os.path.basename(d) not in RECORDS]
    for s in glob.glob(os.path.join(REPO, "scripts/*.sh")) + docs:
        for m in pat.findall(open(s, errors="ignore").read()):
            out.setdefault(f"results/codesign_feedback/ctrl_traces/{m}", set()).add(os.path.relpath(s, REPO))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    print("=== the models a flight loads")
    for rel, who in sorted(model_defaults().items()):
        p = os.path.join(REPO, rel)
        if not os.path.exists(p):
            check(False, f"{rel} is named by {len(who)} file(s) and is not on disk")
        else:
            check(tracked(rel), f"{rel} ({os.path.getsize(p) / 1048576:.1f} MB)"
                                + ("" if tracked(rel) else " exists but is NOT tracked"))

    print("\n=== the cadence traces a flight replays")
    named = ctrl_traces_named()
    miss = [r for r in named if not os.path.exists(os.path.join(REPO, r))]
    untr = [r for r in named if os.path.exists(os.path.join(REPO, r)) and not tracked(r)]
    check(not miss, f"every ctrl_trace a driver names is on disk ({len(named)} named"
                    + (f"; missing {', '.join(os.path.basename(m) for m in sorted(miss)[:4])}" if miss else "") + ")")
    check(not untr, f"every ctrl_trace a driver names is tracked"
                    + (f" ({len(untr)} are not: {', '.join(os.path.basename(u) for u in sorted(untr)[:4])})" if untr else ""))

    print("\n=== the scene")
    r = subprocess.run(["git", "-C", REPO, "ls-tree", "HEAD", "sims/IsaacLab"], capture_output=True, text=True)
    check(r.stdout.strip().startswith("160000 commit"),
          f"IsaacLab is pinned as a submodule commit ({r.stdout.split()[2][:12] if len(r.stdout.split()) > 2 else 'absent'})")
    n_task = len(subprocess.run(["git", "-C", REPO, "ls-files", "sims/isaaclab_tasks"],
                                capture_output=True, text=True).stdout.split())
    check(n_task > 0, f"the IsaacLab task package that builds the scene is tracked ({n_task} files)")
    usd = [os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "sims/isaaclab_tasks/**/*.usd*"), recursive=True)]
    bad = [u for u in usd if not tracked(u)]
    check(not bad, f"every USD asset in the task package is tracked ({len(usd)} found"
                   + (f"; untracked: {', '.join(bad[:3])}" if bad else "") + ")")

    print("\n=== the interpreter")
    check(os.path.exists(os.path.join(REPO, "requirements-isaac.txt")),
          "requirements-isaac.txt pins the Isaac interpreter")
    env = os.path.join(REPO, "docs/Artifact/environment.md")
    check(os.path.exists(env) and "Isaac Sim" in open(env).read(),
          "docs/Artifact/environment.md names the Isaac Sim version to install")

    print(f"\n{FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
