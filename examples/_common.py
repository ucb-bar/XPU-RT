"""Shared plumbing for the examples. Import paths, printing, and a Gantt.

Kept deliberately small. An example that spends thirty lines on setup before it
shows anything is not an example.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
XPU_RT = REPO / "xpu-rt"
SCRIPTS = REPO / "scripts"
MB = REPO / "ModelBlaster"

for p in (str(XPU_RT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)


def head(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def step(n, text: str) -> None:
    print(f"\n[{n}] {text}")


def note(text: str) -> None:
    for line in text.strip().split("\n"):
        print(f"    {line}")


def need_board() -> bool:
    """True when a board is reachable. Examples that need one say so and stop.

    An example that silently substitutes synthetic numbers for measured ones is
    worse than an example that refuses: the output looks the same.
    """
    return bool(os.environ.get("K1_HOST"))


def need_file(path: Path, what: str) -> bool:
    if path.exists():
        return True
    print(f"    SKIP: no {what} at {path}")
    print(f"          this example needs it and will not invent one.")
    return False


def out_dir(name: str) -> Path:
    """Examples write under out/examples/, which is gitignored."""
    d = REPO / "out" / "examples" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def gantt(schedule_path, png_path, title=None) -> bool:
    """Render a predicted schedule. Returns False (loudly) if it cannot.

    Uses `xpu-rt/plot_gantt.py`, the renderer the rest of the repo uses. There
    is deliberately no second renderer here: a figure in an example that does
    not come out of the same code as a figure in the paper is a figure that can
    disagree with it.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import plot_gantt
    except ImportError as e:
        print(f"    (no Gantt: {e})")
        return False
    try:
        plot_gantt.render_fixture_gantt(str(schedule_path), str(png_path),
                                        title=title)
    except Exception as e:                              # pragma: no cover
        print(f"    (no Gantt: {type(e).__name__}: {e})")
        return False
    print(f"    Gantt -> {png_path}")
    return True
