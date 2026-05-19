#!/usr/bin/env python3
"""Reproduce the paper figure via the XPU-RT QNN agentic CLI.

Thin shim around::

    uv run xpu-rt qnn demo paper-figure

Lets users (and CI) drive the demo with a single script invocation
without remembering the click sub-sub-command path. Forwards every
argument unchanged.

Examples::

    python3 scripts/qnn_paper_figure_demo.py --dry-run
    python3 scripts/qnn_paper_figure_demo.py --max-rounds 2 --no-interactive
"""

from __future__ import annotations

import os
import sys

# The CLI module owns the implementation. Import lazily so this shim
# doesn't pay for the click-group registration cost twice.
def _main() -> int:
    try:
        from xpu_rt.cli import main as cli_main
    except ImportError as exc:
        print(
            f"error: cannot import xpu_rt.cli: {exc}.\n"
            "Run `uv sync` first, then try again "
            "(or invoke `uv run scripts/qnn_paper_figure_demo.py ...`).",
            file=sys.stderr,
        )
        return 2
    argv = ["qnn", "demo", "paper-figure", *sys.argv[1:]]
    try:
        cli_main(argv, standalone_mode=False)
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
