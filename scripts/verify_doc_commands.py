#!/usr/bin/env python3
"""Every script a document tells a reader to run is actually at the path it gives.

A reproduction page is only a recipe if its commands resolve. Two ways they stop resolving, both
found in this tree: a script moves (the Isaac drivers live under `sims/scripts/`, the kernel and
board tools under `ModelBlaster/scripts/` and `runtime/scripts/`, but pages cited them as
`scripts/…`), and a script is retired while the page that names it is not. Neither shows up in any
other check -- `verify_showdown_figure.py` reads sidecars, not prose.

Two kinds of citation are not this repository's to resolve, and both are skipped by name rather
than by silently passing:

  * a page documenting another component (`docs/K1/`, `docs/Firesim/`, `docs/Demo/`,
    `docs/Qualcomm/`, `docs/Feature/`), whose commands run in the runtime, the board tree or
    `zephyr-chipyard-sw/` -- trees this checkout does not contain, so a citation there cannot be
    checked here and its absence says nothing about this artifact;
  * `attic/session_log.md`, a verbatim archive of past commit messages rather than a recipe. It
    quotes the filenames a change removed, and reading it as a page of commands would mean keeping
    retired names alive forever. `verify_flight_inputs.py` holds it out for the same reason.

`--strict` checks them anyway, which is how the skip list is audited.

    scripts/verify_doc_commands.py [--roots docs artifact] [--strict] [-v]

Exit 0 when every cited path this repository owns exists. A path that does not is printed with the documents citing it,
so the fix is either the right path or a page that no longer claims something the repo cannot do.
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# a path that looks like a runnable script in one of the repo's script directories
CITED = re.compile(r"(?<![\w/.])((?:scripts|sims/scripts|ModelBlaster/scripts|runtime/scripts|"
                   r"board/scripts|xpu-rt/scripts)/[A-Za-z0-9_.-]+\.(?:sh|py))")


FENCE = re.compile(r"^\s*```")
# `cd ModelBlaster`, `cd ModelBlaster && ...`, `cd xpu-rt/build` -- a literal directory, nothing
# built out of a variable, which we cannot resolve and do not guess at.
CD = re.compile(r"(?:^|\s|&&\s*|;\s*)cd\s+(\.\.(?:/[A-Za-z0-9_./-]+)?|[A-Za-z0-9_][A-Za-z0-9_./-]*)")


def cited_paths(roots):
    """(path, cwd) -> the documents that name it, where cwd is the directory the command runs in.

    Resolving every citation against the repository root is how a command that CANNOT run reads as
    fine: `cd ModelBlaster` followed by `ModelBlaster/scripts/ime_fused_conv_bench.py` names a path
    that exists from the root and resolves to `ModelBlaster/ModelBlaster/...` where the reader
    actually is. So a fenced block carries its own working directory -- set by a literal `cd`, reset
    at the fence -- and its citations resolve against that. Prose outside a fence is not a command
    and keeps resolving against the root.
    """
    out = collections.defaultdict(set)
    for root in roots:
        for f in glob.glob(os.path.join(REPO, root, "**", "*.md"), recursive=True):
            try:
                text = open(f).read()
            except Exception:
                continue
            rel = os.path.relpath(f, REPO)
            in_fence, cwd = False, ""
            for line in text.splitlines():
                if FENCE.match(line):
                    in_fence, cwd = not in_fence, ""
                    continue
                if in_fence:
                    for d in CD.findall(line):
                        if os.path.isdir(os.path.join(REPO, cwd, d)):
                            nxt = os.path.normpath(os.path.join(cwd, d))
                            cwd = "" if nxt in (".", os.curdir) or nxt.startswith("..") else nxt
                for m in CITED.findall(line):
                    out[(m, cwd if in_fence else "")].add(rel)
    return out


# pages that document a component whose tree this checkout does not contain
OTHER_COMPONENTS = ("docs/K1/", "docs/Firesim/", "docs/Demo/", "docs/Qualcomm/", "docs/Feature/")
# a verbatim archive of past commit messages, not a recipe (see verify_flight_inputs.RECORDS)
RECORDS = ("docs/session_log.md", "attic/session_log.md")


def owned(docs):
    """The citing pages whose commands this repository is supposed to be able to run."""
    return {d for d in docs
            if not d.startswith(OTHER_COMPONENTS) and d not in RECORDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=["docs", "artifact"])
    ap.add_argument("--strict", action="store_true",
                    help="also check the pages that document other components and the commit archive")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    cited = cited_paths(a.roots)
    absent = {k: docs for k, docs in cited.items()
              if not os.path.exists(os.path.join(REPO, k[1], k[0]))}
    # A path the repository ships an `.example` for is one the reader writes, not one the
    # repository owes them -- `scripts/env.local.sh` beside `scripts/env.local.sh.example` is the
    # case. It exists here and not in a clone, so this verifier passed locally and failed where it
    # matters. The rule is the template, not a list of names, so a second such file needs no edit.
    templated = {k for k in absent
                 if os.path.exists(os.path.join(REPO, k[1], k[0] + ".example"))}
    absent = {k: v for k, v in absent.items() if k not in templated}
    if a.strict:
        missing, skipped = absent, {}
    else:
        missing = {k: owned(d) for k, d in absent.items() if owned(d)}
        skipped = {k: d for k, d in absent.items() if not owned(d)}
    for p, cwd in sorted(missing):
        where = ", ".join(sorted(missing[(p, cwd)])[:3])
        frm = f" from {cwd}/" if cwd else ""
        print(f"FAIL  {p} is named by {len(missing[(p, cwd)])} document(s){frm} and is not in the "
              f"repo  ({where})")
    for p, cwd in sorted(skipped):
        where = ", ".join(sorted(skipped[(p, cwd)])[:2])
        print(f"SKIP  {p} is named only by a page this repository does not own  ({where})")
    for p, cwd in sorted(templated):
        print(f"SKIP  {p} is a file the reader writes; {p}.example is the tracked template")
    if a.verbose:
        for k in sorted(cited):
            if k not in absent:
                print(f"PASS  {os.path.join(k[1], k[0])} ({len(cited[k])} document(s))")
    print(f"\n{len(cited) - len(absent)} cited scripts resolve, {len(missing)} do not"
          + (f", {len(skipped)} belong to another component or the commit archive" if skipped else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
