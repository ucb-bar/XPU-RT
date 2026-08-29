"""Stage 6 — the two Gantt charts, rendered by modelblaster's plotter.

`scripts/run_xpurt_schedule.py` already draws the predicted timeline when
it solves.  For predicted-vs-actual we reuse
`modelblaster/scripts/plot_xpurt_trace.py` exactly as the spike flow does:
it stacks the scheduler's plan over what the hardware did, red-bordering
any entry that ran past its predicted finish.  The Flow C runtime emits
its trace in that script's own column schema with microsecond ticks, so
`--clock-mhz 1` is the whole adaptation.

A second, zoomed render covers the first window of the run — with a 33 ms
yolov8n tile in the same axes, a 0.03 ms control dispatch is a hairline,
and the periodic cadence is the interesting part.
"""

from __future__ import annotations

import importlib.util
import os
import sys

from . import mb


def _load_plotter():
    path = os.path.join(mb.modelblaster_root(), "scripts", "plot_xpurt_trace.py")
    spec = importlib.util.spec_from_file_location("mb_plot_xpurt_trace", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def render(log_path: str, out_full: str, out_zoom: str | None = None,
           csv_path: str | None = None, source: str = "QRB5165 (QNN)",
           zoom_ms: float | None = None) -> dict:
    plotter = _load_plotter()
    with open(log_path) as f:
        text = f.read()
    rows = plotter.parse_trace(text, clock_mhz=1.0)   # trace ticks are microseconds
    os.makedirs(os.path.dirname(os.path.abspath(out_full)), exist_ok=True)
    plotter.render_plot(rows, out_full, source=source)
    if csv_path:
        plotter.write_csv(rows, csv_path)

    out = {"entries": len(rows), "full": out_full,
           "summary": plotter._summary(rows)}
    if out_zoom:
        if zoom_ms is None:
            # Default window: two periods past the slowest periodic network,
            # or the first quarter of the run, whichever is larger.
            makespan = max(r.actual_end_ms for r in rows)
            zoom_ms = max(makespan / 4.0, 20.0)
        sub = [r for r in rows if r.predicted_start_ms <= zoom_ms]
        if sub:
            plotter.render_plot(sub, out_zoom, source=f"{source} — first {zoom_ms:.0f} ms")
            out["zoom"] = out_zoom
            out["zoom_ms"] = zoom_ms
    return out
