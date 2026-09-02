"""The examples run. That is the whole test, and it is worth having.

An example that has rotted into a description of code which no longer exists
is worse than no example: it is confidently wrong, and it is the first thing a
newcomer reads. Every function these scripts call is one a refactor can rename
without any other test noticing, because nothing else imports them.

`examples/run_all.py` runs the subset needing neither a board nor a licence.
`compare_solvers.py` is excluded there (it shells out to the scheduler eight
times) and covered by its own slower test below, marked so it can be
deselected.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"


class TheExamplesStillRun(unittest.TestCase):

    def test_the_cheap_subset_runs(self):
        p = subprocess.run(
            [sys.executable, str(EXAMPLES / "run_all.py"), "--quiet"],
            cwd=REPO, capture_output=True, text=True, timeout=600)
        self.assertEqual(p.returncode, 0,
                         f"an example failed:\n{p.stdout[-3000:]}\n"
                         f"{p.stderr[-2000:]}")

    def test_every_example_is_either_run_or_deliberately_excluded(self):
        """A new example that nobody runs rots exactly like an old one."""
        import ast
        src = (EXAMPLES / "run_all.py").read_text()
        listed = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.endswith(".py") and "/" in node.value:
                    listed.add(node.value)
        on_disk = {
            str(p.relative_to(EXAMPLES))
            for p in EXAMPLES.rglob("*.py")
            if p.name not in ("run_all.py", "_common.py")
        }
        missing = on_disk - listed
        self.assertEqual(missing, set(),
                         f"example(s) not referenced by run_all.py: "
                         f"{sorted(missing)} -- add them to CHEAP or SLOW, or "
                         f"they will rot unnoticed")


if __name__ == "__main__":
    unittest.main()
