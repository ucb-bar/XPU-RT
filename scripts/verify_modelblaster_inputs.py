#!/usr/bin/env python3
"""Everything the ModelBlaster side of the artifact needs is pinned, reachable and re-derivable.

The figure side of this artifact has a sidecar per render and a checker that re-derives it. The
ModelBlaster side has neither, and it is the half that carries the compiler, the IME kernels and the
measured IME-vs-RVV tables the kernel picker reads. Three things can silently rot there, and no
other check looks at any of them:

  * **The pin.** The superproject pins a ModelBlaster commit that is NOT on `origin`
    (`git ls-remote` does not have it), so `git clone --recurse-submodules` cannot fetch it and a
    reviewer gets an EMPTY `ModelBlaster/`. The commit is carried offline by
    `artifact/history/modelblaster.bundle` instead. That arrangement only means something if the
    bundle's tip is still the commit the superproject pins -- re-cut the bundle, or move the
    pointer, and the two drift apart with nothing to say so.
  * **What the docs name.** `docs/K1/ime_kernel_reproduction.md` and `docs/Baselines/ros_baseline_reproduction.md`
    §2b walk a reviewer through files inside the submodule. A path that does not exist at the PINNED
    commit is a command nobody else can run, however well it works in this working tree.
  * **The measured tables.** `pipeline/ime_cost.py` is the only-if-better rule, and it is only a rule
    because it reads a MEASURED table: an (op, shape) reaches the matrix engine where
    `artifacts/ime_conv*/` says the engine is faster than the RVV kernel that would otherwise run.
    Without the table the rule has no inputs and every shape stays on RVV.

    scripts/verify_modelblaster_inputs.py [-v] [--bundles <dir>]

Exit 0 when a reviewer can get the pinned submodule, run the commands the docs name against it, and
re-derive the IME numbers the docs state. The submodule's CONTENT checks need the objects: in a
clean clone before the bundle is fetched they report SKIP rather than PASS, and the count is printed
-- a skipped check is not a passed one.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(REPO, "ModelBlaster")
FAILED = 0
SKIPPED = 0
VERBOSE = False


def check(ok, msg):
    global FAILED
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        FAILED += 1
    return ok


def skip(msg):
    global SKIPPED
    print("SKIP  " + msg)
    SKIPPED += 1


def git(*args, cwd=REPO):
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True)


def tracked(rel, cwd=REPO):
    return git("ls-files", "--error-unmatch", rel, cwd=cwd).returncode == 0


def archived(rel, archdir):
    """rel is a member of a tar under archdir and, when it is also on disk, holds the same bytes.

    The raw board traces are untracked and archived (docs/Artifact/external_data.md, "Raw traces"),
    so "tracked or archived" is the guarantee a trace can carry; a clone that has not unpacked the
    archives has only the archive copy, and that is accepted as it stands."""
    import hashlib
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import archive_members as AM
    path = os.path.join(REPO, rel)
    disk = None
    if os.path.exists(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        disk = h.hexdigest()
    for t in AM.archives(archdir):
        try:
            for m in AM.members(t):
                if m.isfile() and m.name == rel and (disk is None or AM.member_sha256(t, m.name) == disk):
                    return True
        except Exception:
            continue
    return False


def pinned_commit():
    out = git("ls-tree", "HEAD", "ModelBlaster").stdout.split()
    return out[2] if len(out) > 2 and out[0] == "160000" else None


def have_object(sha):
    """True when the submodule's own object store can serve the pinned commit."""
    if not os.path.exists(os.path.join(SUB, ".git")):
        return False
    return git("cat-file", "-e", f"{sha}^{{commit}}", cwd=SUB).returncode == 0


def manifest_commits(bundles):
    """{bundle filename: commit id} out of the TRACKED artifact/history/MANIFEST.sha256.

    The manifest is the record of which tip each bundle was cut at; the bundles themselves are
    large and untracked, the same convention archive_v3/ uses for its tars.
    """
    path = os.path.join(bundles, "MANIFEST.sha256")
    if not os.path.exists(path):
        return {}, {}
    text = open(path).read()
    tips = {}
    cur = None
    sizes = {}
    for line in text.splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+(\d+)\s+(\S+\.bundle)\s*$", line.strip())
        if m:
            sizes[m.group(3)] = (m.group(1), int(m.group(2)))
            continue
        m = re.match(r"^#\s*(\S+\.bundle)\b", line)
        if m:
            cur = m.group(1)
        m = re.search(r"bundled at\s+([0-9a-f]{40})", line)
        if m and cur:
            tips[cur] = m.group(1)
    return tips, sizes


def doc_named_paths():
    """Every path inside the submodule that a doc or an artifact page names, and who names it.

    Derived from the prose rather than listed here: a doc that starts naming a new file is checked
    from the moment it does, and one that names a path that never existed is caught the same way
    (that is how `measurements_and_ablations.md`'s top-level citation of `run_xpurt_k1.sh`, which
    lives under `scripts/`, was found).

    Two spellings reach the same file and both count. A page may write the path from the repository
    root (`ModelBlaster/kernels/...`), or it may `cd ModelBlaster` in a fenced block and write it
    from there (`scripts/ime_fused_conv_bench.py`). The second is the form the reader actually runs,
    and it is `verify_doc_commands.py` that already knows which directory a fenced command runs in --
    so that pass is reused rather than reimplemented, and the two checks cannot drift.
    """
    out = {}

    def add(rel, who):
        rel = rel.rstrip(".,)")
        if rel.startswith("build/"):
            return            # generated by a board build; gitignored on purpose
        out.setdefault(rel, set()).add(who)

    pat = re.compile(r"ModelBlaster/([A-Za-z0-9_./-]+\.(?:c|h|py|csv|json|md|sh|cpp))")
    # recursive, like verify_doc_commands: docs/ is grouped into per-target subdirectories
    # (K1/, Feature/, Demo/, ...) and a page that moved into one still names the same files.
    srcs = glob.glob(os.path.join(REPO, "docs/**/*.md"), recursive=True) \
        + glob.glob(os.path.join(REPO, "artifact/**/*.md"), recursive=True)
    for s in srcs:
        for m in pat.findall(open(s, errors="ignore").read()):
            add(m, os.path.relpath(s, REPO))

    sys.path.insert(0, os.path.join(REPO, "scripts"))
    try:
        from verify_doc_commands import cited_paths
    except Exception:
        return out
    for (path, cwd), docs in cited_paths(["docs", "artifact"]).items():
        if cwd == "ModelBlaster":
            for d in docs:
                add(path, d)
    return out


def conv_tables(pin):
    """The measured tables the only-if-better rule reads, as ime_cost.py itself names them.

    Read out of the pinned source so the check follows the rule rather than a copy of it: adding a
    third conv op-kind to CONV_TABLES puts its table under this check automatically.
    """
    src = git("show", f"{pin}:pipeline/ime_cost.py", cwd=SUB).stdout
    body = re.search(r"CONV_TABLES\s*=\s*\{(.*?)\}", src, re.S)
    if not body:
        return {}
    names = dict(re.findall(r'"([^"]+)"\s*:\s*(\w+)', body.group(1)))
    consts = {}
    for m in re.finditer(r'^(_\w*CSV)\s*=\s*os\.path\.join\(_ARTIFACTS,\s*"([^"]+)",\s*"([^"]+)"\)', src, re.M):
        consts[m.group(1)] = f"artifacts/{m.group(2)}/{m.group(3)}"
    return {op: consts[c] for op, c in names.items() if c in consts}


def table_rows(pin, rel):
    text = git("show", f"{pin}:{rel}", cwd=SUB).stdout
    return list(csv.DictReader(text.splitlines()))


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--bundles", default=os.path.join(REPO, "artifact/history"),
                    help="where the offline git bundles live (default artifact/history)")
    ap.add_argument("--archives", default=os.environ.get("ARCHIVES") or
                    os.path.join(REPO, "results/codesign_feedback/archive_v3"),
                    help="the archive directory the untracked board traces are in "
                         "(default $ARCHIVES, else results/codesign_feedback/archive_v3)")
    a = ap.parse_args()
    VERBOSE = a.verbose

    # ---------------------------------------------------------------- the pin
    print("=== the submodule pointer")
    pin = pinned_commit()
    if not check(pin is not None, f"ModelBlaster is pinned as a submodule commit ({pin or 'absent'})"):
        print("\n1 failed")
        return 1
    print(f"      pinned at {pin}")

    tips, sizes = manifest_commits(a.bundles)
    mb_tip = tips.get("modelblaster.bundle")
    check(mb_tip is not None,
          "artifact/history/MANIFEST.sha256 records modelblaster.bundle's tip"
          + ("" if mb_tip else " -- it does not, so nothing says the bundle carries the pin"))
    if mb_tip:
        check(mb_tip == pin,
              "the bundle's tip IS the commit the superproject pins"
              + ("" if mb_tip == pin else f" -- manifest says {mb_tip[:12]}, pin is {pin[:12]};"
                                          " re-cut the bundle or move the pointer"))
    check(tracked("artifact/history/MANIFEST.sha256"),
          "the manifest is tracked (the bundles themselves are large and are not)")

    # The pin is reachable one of two ways. Neither is assumed: a pin on origin needs no bundle, and
    # a pin off origin needs one that is actually on disk with the right bytes.
    on_origin = False
    if os.path.exists(os.path.join(SUB, ".git")):
        on_origin = bool(git("branch", "-r", "--contains", pin, cwd=SUB).stdout.strip())
    bundle = os.path.join(a.bundles, "modelblaster.bundle")
    if on_origin:
        check(True, "the pin is reachable from a remote branch -- a plain clone can fetch it")
    elif os.path.exists(bundle):
        heads = subprocess.run(["git", "bundle", "list-heads", bundle],
                               capture_output=True, text=True).stdout
        check(pin in heads,
              f"the pin is not on any remote; {os.path.relpath(bundle, REPO)} carries it"
              + ("" if pin in heads else " -- but the bundle does NOT contain it"))
    else:
        skip(f"the pin is not on any remote and {os.path.relpath(bundle, REPO)} is not on disk"
             " -- the manifest above is what says it is recoverable; cut or fetch the bundle to check the bytes")

    # ------------------------------------------------------- what the docs name
    print("\n=== what the docs name, at the pinned commit")
    named = doc_named_paths()
    if not have_object(pin):
        skip(f"the submodule is not checked out at {pin[:12]}, so its {len(named)} doc-named paths"
             " cannot be checked -- clone artifact/history/modelblaster.bundle (artifact/history/README.md)")
        skip("the measured IME tables cannot be read for the same reason")
        skip("the only-if-better rule's tests cannot be located for the same reason")
        tables = {}
    else:
        miss = {}
        for rel, who in sorted(named.items()):
            if git("cat-file", "-e", f"{pin}:{rel}", cwd=SUB).returncode != 0:
                miss[rel] = who
            elif VERBOSE:
                print(f"      {rel}")
        check(not miss, f"every ModelBlaster path the docs name exists at the pinned commit"
                        f" ({len(named)} named across docs/ and artifact/)")
        for rel, who in sorted(miss.items()):
            print(f"      MISSING {rel}   named by {', '.join(sorted(who))}")

        # ------------------------------------------- the tables the picker reads
        print("\n=== the measured IME-vs-RVV tables the only-if-better rule reads")
        tables = conv_tables(pin)
        check(bool(tables), f"pipeline/ime_cost.py names a table per conv op-kind ({len(tables)} found)")
        for op, rel in sorted(tables.items()):
            if git("cat-file", "-e", f"{pin}:{rel}", cwd=SUB).returncode != 0:
                check(False, f"{op} -> {rel} is named by the rule and is NOT at the pinned commit")
                continue
            rows = table_rows(pin, rel)
            bad = [r for r in rows if (r.get("verify") or "OK") != "OK"]
            wins = [r for r in rows if float(r.get("speedup") or 0) > 1.0]
            check(rows and not bad,
                  f"{op} -> {rel}: {len(rows)} measured shapes, {len(wins)} won by the IME, "
                  f"every row bit-exact against RVV"
                  + (f" -- {len(bad)} row(s) are NOT" if bad else ""))

        # ---------------------------------------------------- the rule is testable
        print("\n=== the only-if-better rule is testable")
        for t in ("pipeline/tests/test_ime_cost.py", "pipeline/tests/test_ime_force.py"):
            check(git("cat-file", "-e", f"{pin}:{t}", cwd=SUB).returncode == 0,
                  f"{t} is at the pinned commit")

    # --------------------------------------------- the claims, re-derived here
    print("\n=== the IME claims the docs state, re-derived from the tracked files behind them")
    summary = os.path.join(REPO, "results/codesign_feedback/ime_mixed_schedule_36hz.json")
    if not check(os.path.exists(summary) and tracked(os.path.relpath(summary, REPO)),
                 "results/codesign_feedback/ime_mixed_schedule_36hz.json is tracked"):
        print(f"\n{FAILED} failed, {SKIPPED} skipped")
        return 1
    S = json.load(open(summary))

    for arm, spec in S["arms"].items():
        rel = spec["schedule"]
        if not os.path.exists(os.path.join(REPO, rel)):
            check(False, f"{arm}: {rel} is named by the summary and is not on disk")
            continue
        sched = json.load(open(os.path.join(REPO, rel)))
        # `dispatches` is keyed by dispatch name, not a list: iterating it directly counts strings.
        entries = sched.get("dispatches", {})
        entries = list(entries.values()) if isinstance(entries, dict) else list(entries)
        impls = [d.get("impl") for d in entries if isinstance(d, dict) and "impl" in d]
        n_ime = sum(1 for i in impls if i == "ime")
        # `impl` is written by postprocessing only when the spec sets scheduler.enable_impls, so the
        # all-RVV arm carries none at all. Either every dispatch records the engine that ran it, or
        # none does and the arm claims no IME dispatches -- a schedule that claims some and records
        # them on only part of its dispatches is the case worth catching.
        if spec["ime_dispatches"]:
            check(len(impls) == len(entries),
                  f"{arm}: every one of the {len(entries)} dispatches records which engine ran it"
                  + (f" -- {len(entries) - len(impls)} do not" if len(impls) != len(entries) else ""))
        else:
            check(not impls,
                  f"{arm}: solved without enable_impls, so none of its {len(entries)} dispatches "
                  f"records an engine" + (f" -- {len(impls)} do" if impls else ""))
        check(tracked(rel), f"{arm}: {rel} is tracked")
        check(n_ime == spec["ime_dispatches"],
              f"{arm}: {n_ime} of {len(impls)} dispatches carry impl=ime, and the summary records "
              f"{spec['ime_dispatches']}")
        miss = [t for t in spec["traces"]
                if not tracked(f"results/codesign_feedback/xpurt_long/{t}")
                and not archived(f"results/codesign_feedback/xpurt_long/{t}", a.archives)]
        check(not miss, f"{arm}: all {len(spec['traces'])} board traces behind its median are tracked"
                        " or archived"
                        + (f" -- {', '.join(miss)} are not" if miss else ""))

    # the win count the prose states, against the table it was counted from
    doc = os.path.join(REPO, "docs/K1/ime_kernel_reproduction.md")
    fused = tables.get("conv2d_batchnorm2d_silu_s8") if tables else None
    m = re.search(r"IME wins\s+\d+\s*(?:->|→)\s*(\d+)\s+of\s+(\d+)", open(doc).read())
    if not m:
        check(False, "docs/K1/ime_kernel_reproduction.md states a per-shape win count this can re-derive")
    elif not fused:
        skip(f"the doc states {m.group(1)} of {m.group(2)} shapes won; the table is not readable here")
    else:
        rows = table_rows(pin, fused)
        wins = sum(1 for r in rows if float(r.get("speedup") or 0) > 1.0)
        check((wins, len(rows)) == (int(m.group(1)), int(m.group(2))),
              f"the doc's per-shape win count re-derives from {fused}: {wins} of {len(rows)}"
              f" (doc says {m.group(1)} of {m.group(2)})")

    print(f"\n{FAILED} failed, {SKIPPED} skipped")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
