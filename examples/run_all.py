#!/usr/bin/env python3
"""Run the examples that need neither a board nor a licence.

    .venv/bin/python examples/run_all.py

The test suite runs this, which is the point: an example that has rotted into
a description of code that no longer exists is worse than no example, because
it is confidently wrong. Anything here has to keep working or the suite goes
red.

`compare_solvers.py` is deliberately NOT in the default set -- it shells out
to the scheduler eight times and takes about ten seconds, which is fine to run
by hand and too slow to put in front of every commit. Pass `--all` for it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CHEAP = [
    "verbs/all_five_verbs.py",
    "workloads/anatomy_of_a_spec.py",
    "k1_board/board_flow.py",
    "feedback_loop/one_revolution.py",
]
SLOW = [
    "solvers/compare_solvers.py",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true",
                    help="also run the slow ones (compare_solvers)")
    ap.add_argument("--quiet", action="store_true",
                    help="only report pass/fail, not each example's output")
    a = ap.parse_args()

    scripts = CHEAP + (SLOW if a.all else [])
    failures = []
    for rel in scripts:
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, str(HERE / rel)],
                           cwd=REPO, capture_output=True, text=True)
        secs = time.perf_counter() - t0
        ok = p.returncode == 0
        print(f"{'ok  ' if ok else 'FAIL'}  {rel:<38} {secs:6.1f}s")
        if not ok:
            failures.append(rel)
            print((p.stderr or p.stdout).strip()[-1500:])
        elif not a.quiet:
            # One line of evidence it did something, not just exited 0.
            body = [ln for ln in p.stdout.splitlines() if ln.startswith("[")]
            for ln in body[:1]:
                print(f"        {ln}")

    print()
    if failures:
        print(f"{len(failures)} example(s) failed: {', '.join(failures)}")
        return 1
    print(f"all {len(scripts)} example(s) ran")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
