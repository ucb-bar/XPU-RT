#!/usr/bin/env python3
"""Which recorded numbers are load-bearing: perturb each one and see whether a check notices.

`artifact/verify_no_hardware.sh` passing says the checks agree with the figures. It does not say the
checks *could* disagree -- a check that reads a field and compares it to itself passes forever, and
a number recorded in a sidecar that nothing reads looks exactly like a number that is verified.
This separates the two by mutation: for every numeric leaf of every verified sidecar, change it by
more than any tolerance, re-run the verifier that owns that figure, and record whether the verifier
failed.

  detected    perturbing the value makes a check fail -- the number is verified
  undetected  perturbing it changes nothing -- the number is recorded, not checked

Undetected is not automatically a defect. A sidecar legitimately records provenance, geometry and
counts kept for the record. The point is that the two populations should be *named* rather than
assumed identical, so a reader knows which numbers a green gate actually stands behind.

    scripts/mutation_audit.py [--stem NAME ...] [--jobs N] [--check] [-v]

`--check` re-runs the audit and exits non-zero if any stem's detected/undetected split differs from
the recorded `results/codesign_feedback/mutation_audit.json`, so the result cannot rot as the
verifiers change.

Safety: each sidecar's original bytes are held in memory and rewritten after every mutation,
including on exception, SIGINT and SIGTERM. The harness verifies with `git status` at the end that
no sidecar was left modified, and refuses to start if `refined/` is already dirty -- because after
a hard kill a leftover mutation is indistinguishable from an edit somebody meant. A SIGKILL cannot
be caught, so if that happens the sidecars are all committed and

    git checkout -- results/codesign_feedback/refined/

is the complete restore. The refusal message says so.

Each verifier pulls in numpy and scipy, whose BLAS spawns a thread per core: six of them at once
took the machine to a load of 50. The subprocesses are therefore run with the threading environment
pinned to one thread, which is what makes --jobs mean what it says.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile

# sidecar path -> its original bytes, for every mutation currently on disk
_LIVE = {}


def _restore_all(signum=None, frame=None):
    for path, blob in list(_LIVE.items()):
        try:
            with open(path, "wb") as f:
                f.write(blob)
        except Exception:
            pass
    _LIVE.clear()
    if signum is not None:
        print(f"\ncaught signal {signum}: restored every mutated sidecar", flush=True)
        os._exit(130)


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, "results/codesign_feedback/refined")
OUT = os.path.join(REPO, "results/codesign_feedback/mutation_audit.json")
DOC = os.path.join(REPO, "docs/Artifact/mutation_audit.md")
PY = sys.executable

# Fields that name a thing rather than measure one: mutating a sha256's numeric-looking parts or a
# timestamp says nothing about whether a check works, and a figure's own byte size is not a claim.
SKIP_KEYS = {"written", "sha256", "figure", "script"}


def owner(stem):
    """(argv, needs_stem) for the verifier that owns this figure.

    Read from VERIFIED_ELSEWHERE in verify_showdown_figure.py -- the same table the gate uses -- so
    this harness cannot drift from which script actually checks a figure.
    """
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import verify_showdown_figure as V
    who = V.VERIFIED_ELSEWHERE.get(stem)
    if who is None:
        return [PY, os.path.join(REPO, "scripts/verify_showdown_figure.py"),
                "--metrics", os.path.join(REF, stem + "_metrics.json")]
    script = who.split()[0]
    if not script.endswith(".py"):
        return None
    argv = [PY, os.path.join(REPO, "scripts", script)]
    # pass --stem where the verifier takes one, so a 6-figure verifier does not run six times per leaf
    if "--stem" in open(os.path.join(REPO, "scripts", script)).read():
        argv += ["--stem", stem]
    return argv


def leaves(o, path=""):
    """(json path, value) for every number that is not a label or a hash."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k in SKIP_KEYS:
                continue
            yield from leaves(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(o, bool):
        pass
    elif isinstance(o, (int, float)):
        yield path, o


def put(o, path, val):
    """Set the leaf at a path produced by leaves()."""
    cur, tok = o, []
    i = 0
    while i < len(path):
        c = path[i]
        if c == ".":
            j = i + 1
            while j < len(path) and path[j] not in ".[":
                j += 1
            tok.append(path[i + 1:j]); i = j
        elif c == "[":
            j = path.index("]", i)
            tok.append(int(path[i + 1:j])); i = j + 1
        else:
            raise ValueError(path)
    for t in tok[:-1]:
        cur = cur[t]
    cur[tok[-1]] = val


def perturb(v):
    """A change larger than any tolerance a check uses: 5% or 1.0, whichever is bigger."""
    if isinstance(v, int):
        return v + 1
    if v == 0:
        return 1.0
    return v + max(abs(v) * 0.05, 1.0)


# a verifier is arithmetic on small arrays; letting BLAS take a thread per core buys nothing and
# multiplies the load by the core count
ONE_THREAD = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def run(argv, timeout=900):
    env = dict(os.environ, **ONE_THREAD)
    try:
        r = subprocess.run(argv, capture_output=True, text=True, cwd=REPO, timeout=timeout, env=env)
        return r.returncode
    except subprocess.TimeoutExpired:
        return -1


def audit_stem(stem, verbose=False):
    side = os.path.join(REF, stem + "_metrics.json")
    argv = owner(stem)
    if argv is None or not os.path.exists(side):
        return None
    original = open(side, "rb").read()
    _LIVE[side] = original                        # so a signal handler can put it back
    base = run(argv)
    if base != 0:
        return {"stem": stem, "error": f"verifier does not pass before mutation (exit {base})",
                "owner": os.path.basename(argv[1])}

    doc = json.loads(original)
    items = list(leaves(doc))
    detected, undetected = [], []
    try:
        for path, val in items:
            mutated = json.loads(original)
            put(mutated, path, perturb(val))
            open(side, "w").write(json.dumps(mutated, indent=1))
            rc = run(argv)
            (detected if rc != 0 else undetected).append(path)
            if verbose:
                print(f"  {'detected  ' if rc != 0 else 'UNDETECTED'} {stem}{path}", flush=True)
    finally:
        with open(side, "wb") as f:                # always, including on exception or signal
            f.write(original)
        _LIVE.pop(side, None)
    return {"stem": stem, "owner": os.path.basename(argv[1]), "leaves": len(items),
            "detected": len(detected), "undetected": sorted(undetected)}


def write_doc():
    """The report, generated from the JSON so the prose cannot drift from the measurement."""
    import collections, re as _re
    if not os.path.exists(OUT):
        print(f"{os.path.relpath(OUT, REPO)} does not exist; run the audit first")
        return 1
    j = json.load(open(OUT))
    rows = [r for r in j["figures"] if "error" not in r]
    skipped = [r for r in j["figures"] if "error" in r]
    tot_l = sum(r["leaves"] for r in rows)
    tot_d = sum(r["detected"] for r in rows)
    field = collections.Counter()
    for r in rows:
        for pth in r["undetected"]:
            field[_re.sub(r"\[\d+\]", "[]", pth)] += 1

    L = ["# Which recorded numbers are load-bearing", "",
         "`artifact/verify_no_hardware.sh` passing says the checks agree with the figures. It does",
         "not say the checks *could* disagree: a check that reads a field and compares it to itself",
         "passes forever, and a number nothing reads looks exactly like a number that is verified.",
         "", "`scripts/mutation_audit.py` separates the two. For every numeric leaf of every verified",
         "sidecar it changes the value by more than any tolerance, re-runs the verifier that owns",
         "that figure, and records whether the verifier failed.", "",
         "```bash", "scripts/mutation_audit.py            # the sweep, ~1-2 h",
         "scripts/mutation_audit.py --check    # fails if any figure's split has changed",
         "scripts/mutation_audit.py --doc      # regenerate this page from the recording", "```", "",
         "## The measurement", "",
         f"**{tot_d} of {tot_l}** numbers across **{len(rows)} figures** are load-bearing: perturbing",
         f"one makes a check fail. **{tot_l - tot_d}** are recorded and unchecked, and",
         f"**{sum(1 for r in rows if r['detected'] == r['leaves'])} of {len(rows)}** figures verify",
         "every number they record.", "",
         "One entry dominates the residue and is not a figure: `audit_showdown_claims` is the output",
         "of a script that recomputes it, so perturbing a value and re-running simply regenerates",
         "the file. Mutation cannot say anything about a self-regenerating output, and its numbers",
         "are checked by the script's own Fisher tests rather than by comparison. Excluding it,",
         f"**{(tot_l - tot_d) - next((r['leaves'] - r['detected'] for r in rows if r['stem'] == 'audit_showdown_claims'), 0)}**",
         "numbers across the real figures remain unchecked.", "",
         f"{len(skipped)} further sidecars were skipped because their verifier already fails — the",
         "stale-input renders of `figure_verification_inventory.md` §2. A check that is already",
         "failing cannot be mutation-tested.", "",
         "Undetected is not automatically a defect. A sidecar legitimately records provenance and",
         "counts kept for the record. What matters is that the two populations are named, so a",
         "reader knows which numbers a green gate stands behind.", "",
         "## Per figure", "", "| figure | verifier | load-bearing | unchecked |", "|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["leaves"] - x["detected"]) / max(x["leaves"], 1), reverse=True):
        L.append(f"| `{r['stem']}` | `{r['owner']}` | {r['detected']}/{r['leaves']} | "
                 f"{r['leaves'] - r['detected']} |")
    # Which stems the residue actually belongs to, so a reader is not left to infer from the
    # counts whether a row sits on a figure or on the self-regenerating output.
    owners = sorted({r["stem"] for r in rows if r.get("undetected")})
    L += ["", "## What is still unchecked, by field", "",
          "Grouped across figures, so a gap that repeats in 38 renders reads as one gap.", ""]
    if owners == ["audit_showdown_claims"]:
        L += ["**Every row below belongs to `audit_showdown_claims`**, which mutation cannot speak",
              "about: it is a script's own regenerated output, so perturbing a value and re-running",
              "rewrites the file. The `figures` column counts entries inside that one file, not",
              "figures. No number a figure records is unchecked.", ""]
    elif owners:
        L += ["Carried by: " + ", ".join(f"`{o}`" for o in owners) + ".", ""]
    L += ["| field | figures | ", "|---|---|"]
    for k, n in field.most_common(24):
        L.append(f"| `{k}` | {n} |")
    tail = ["", "The sweep is what found them. Every field above was recorded by a render and read by",
            "nothing; where one carried a claim, a check was added to the owning verifier and the",
            "figure re-measured."]
    if owners != ["audit_showdown_claims"]:
        tail += ["What remains here is what could not be re-derived from the inputs the",
                 "render already read, and adding a check that merely re-reads the sidecar would put the",
                 "number back in the first column without verifying anything."]
    L += tail + [""]
    open(DOC, "w").write("\n".join(L) + "\n")
    print(f"wrote {os.path.relpath(DOC, REPO)}: {tot_d}/{tot_l} load-bearing across {len(rows)} figures")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", action="append", default=None)
    ap.add_argument("--jobs", type=int, default=4,
                    help="stems audited in parallel; each mutates its own sidecar, so only "
                         "distinct stems may run together")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--doc", action="store_true",
                    help="write docs/Artifact/mutation_audit.md from the recorded JSON and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if a.doc:
        return write_doc()

    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain",
                            "results/codesign_feedback/refined"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        print("refusing to start: refined/ already has uncommitted changes, so a restore could not "
              "be told from a pre-existing edit.\nIf a previous run was killed with SIGKILL, "
              "`git checkout -- results/codesign_feedback/refined/` is the complete restore.\n"
              + dirty[:400])
        return 2
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _restore_all)

    stems = a.stem
    if not stems:
        stems = sorted(os.path.basename(p)[:-len("_metrics.json")]
                       for p in os.listdir(REF) if p.endswith("_metrics.json"))
    todo = [s for s in stems if owner(s) is not None]
    print(f"auditing {len(todo)} stems, {a.jobs} at a time\n")

    rows = []
    if a.jobs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
            for r in ex.map(lambda s: audit_stem(s, a.verbose), todo):
                if r:
                    rows.append(r)
                    print(f"{r['stem']:<50} " + (r.get("error") or
                          f"{r['detected']}/{r['leaves']} detected, {len(r['undetected'])} not"), flush=True)
    else:
        for s in todo:
            r = audit_stem(s, a.verbose)
            if r:
                rows.append(r)
                print(f"{r['stem']:<50} " + (r.get("error") or
                      f"{r['detected']}/{r['leaves']} detected, {len(r['undetected'])} not"), flush=True)

    left = subprocess.run(["git", "-C", REPO, "status", "--porcelain",
                           "results/codesign_feedback/refined"],
                          capture_output=True, text=True).stdout.strip()
    ok_clean = not left
    print(f"\nno mutated sidecar left on disk: {'yes' if ok_clean else 'NO -- ' + left[:300]}")

    tot_l = sum(r.get("leaves", 0) for r in rows)
    tot_d = sum(r.get("detected", 0) for r in rows)
    print(f"{tot_d}/{tot_l} numbers across {len(rows)} figures are load-bearing "
          f"({tot_l - tot_d} recorded but unchecked)")

    if a.check:
        if not os.path.exists(OUT):
            print(f"FAIL  {os.path.relpath(OUT, REPO)} does not exist; run without --check first")
            return 1
        was = {r["stem"]: r for r in json.load(open(OUT))["figures"]}
        bad = []
        for r in rows:
            w = was.get(r["stem"])
            if not w or w.get("detected") != r.get("detected") or \
               sorted(w.get("undetected", [])) != sorted(r.get("undetected", [])):
                bad.append(r["stem"])
        print(("FAIL  " if bad else "PASS  ") +
              f"the recorded split still holds for every figure"
              + (f" ({len(bad)} differ: {', '.join(bad[:4])})" if bad else f" ({len(rows)} figures)"))
        return 1 if (bad or not ok_clean) else 0

    json.dump({"generated_by": "scripts/mutation_audit.py",
               "method": "each numeric leaf perturbed by max(5%, 1.0) (ints by 1), the owning "
                         "verifier re-run, detected when it fails",
               "totals": {"figures": len(rows), "leaves": tot_l, "detected": tot_d,
                          "undetected": tot_l - tot_d},
               "figures": rows}, open(OUT, "w"), indent=1)
    print(f"wrote {os.path.relpath(OUT, REPO)}")
    return 0 if ok_clean else 1


if __name__ == "__main__":
    sys.exit(main())
