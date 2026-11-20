#!/usr/bin/env python3
"""Run scripts/run_xpurt_schedule.py without drawing the schedule picture.

The driver always draws the whole schedule (one bar per layer instance) before it writes the plan file.
For a 3-model mix with a 170 Hz model that drawing takes about 40 of the 47 minutes; the plan itself is
found in about 4.5 minutes. The picture is not an input to anything, so this wrapper replaces the drawing
function with a no-op and runs the driver unchanged. The plan file is byte-identical.

usage: run_schedule_noplot.py <xpurt checkout> <driver arguments ...>
"""
from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    xpurt = os.path.abspath(sys.argv[1])
    driver = os.path.join(xpurt, "scripts", "run_xpurt_schedule.py")
    sys.path.insert(0, os.path.join(xpurt, "xpu-rt"))
    import plot  # the driver's own `import plot` gets this same module object

    def skip_plot(*_args, **kwargs) -> None:
        sys.stdout.write(f"[no-plot wrapper] schedule picture skipped ({kwargs.get('save_path', '')})\n")

    plot.plot_optimization_schedule = skip_plot
    sys.argv = [driver] + sys.argv[2:]
    runpy.run_path(driver, run_name="__main__")


if __name__ == "__main__":
    main()
