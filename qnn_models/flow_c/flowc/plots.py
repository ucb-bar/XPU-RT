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


def _render_by_hardware(rows, out_path: str, source: str) -> None:
    """Both panels lane-by-hardware.

    modelblaster's renderer lanes the actual panel by worker index, which
    reads as `worker[0]`/`worker[1]` and doesn't line up with the predicted
    panel's `kind#hart` rows — and in kind-network mode the worker indices
    are an implementation detail (two HTA lanes are still one HTA). Rows
    carry `core_kind` and `hart`, so the actual panel can use the same key
    the predicted panel does, and the two stack read directly against each
    other.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nets = sorted({r.network for r in rows})
    palette = plt.get_cmap("tab10")
    color_for = {n: palette(i % 10) for i, n in enumerate(nets)}

    lane_keys = sorted({(r.core_kind, r.hart) for r in rows})
    lane_idx = {k: i for i, k in enumerate(lane_keys)}
    labels = [f"{kind}#hart{hart}" for kind, hart in lane_keys]

    pred_makespan = max(r.predicted_start_ms + r.predicted_duration_ms for r in rows)
    actual_makespan = max(r.actual_end_ms for r in rows)
    xmax = max(pred_makespan, actual_makespan) * 1.02

    fig, (ax_pred, ax_actual) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"hspace": 0.35})
    bar_h = 0.6
    for r in rows:
        key = lane_idx[(r.core_kind, r.hart)]
        c = color_for[r.network]
        ax_pred.barh(key, r.predicted_duration_ms, left=r.predicted_start_ms,
                     height=bar_h, color=c, edgecolor="black", linewidth=0.3)
        pred_end = r.predicted_start_ms + r.predicted_duration_ms
        overrun = r.actual_end_ms > pred_end + 0.001
        ax_actual.barh(key, r.actual_duration_ms, left=r.actual_start_ms,
                       height=bar_h, color=c,
                       edgecolor="red" if overrun else "black",
                       linewidth=1.0 if overrun else 0.3)

    for ax, title in ((ax_pred, "XPU-RT predicted schedule"),
                      (ax_actual, f"Actual execution on {source} "
                                  f"(red border = ran past predicted finish)")):
        ax.set_yticks(list(lane_idx.values()))
        ax.set_yticklabels(labels)
        ax.set_title(title)
        ax.set_xlim(0, xmax)
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.3)
        ax.axvline(pred_makespan, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(actual_makespan, color="black", linestyle="-", linewidth=1.0, alpha=0.6)
    ax_actual.set_xlabel("time (ms)")
    ax_pred.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=color_for[n], label=n)
                            for n in nets],
                   loc="upper right", framealpha=0.9)
    fig.suptitle(f"xpurt timeline — predicted {pred_makespan:.2f} ms vs actual "
                 f"{actual_makespan:.2f} ms ({actual_makespan / pred_makespan:.2f}x)",
                 fontsize=11)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def render(log_path: str, out_full: str, out_zoom: str | None = None,
           csv_path: str | None = None, source: str = "QRB5165 (QNN)",
           zoom_ms: float | None = None, style: str = "hw") -> dict:
    plotter = _load_plotter()
    with open(log_path) as f:
        text = f.read()
    rows = plotter.parse_trace(text, clock_mhz=1.0)   # trace ticks are microseconds
    os.makedirs(os.path.dirname(os.path.abspath(out_full)), exist_ok=True)
    draw = (_render_by_hardware if style == "hw"
            else lambda rr, path, source: plotter.render_plot(rr, path, source=source))
    draw(rows, out_full, source)
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
            draw(sub, out_zoom, f"{source} — first {zoom_ms:.0f} ms")
            out["zoom"] = out_zoom
            out["zoom_ms"] = zoom_ms
    return out
