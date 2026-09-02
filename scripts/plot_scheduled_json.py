#!/usr/bin/env python3
"""Render a solved schedule on physical machine lanes without re-solving it.

Multi-hart targets are drawn as one bar spanning every held core. Both PNG and
PDF are written so this command shares the visual semantics used by the
feedback-loop and scheduler-sweep figures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))
sys.path.insert(0, _HERE)

import plot_k1_evolution as gantt  # noqa: E402
import schedule_trace  # noqa: E402


def load_and_plot(json_path: str, save_path: str | None = None,
                  window_ms: float | None = None,
                  deadline_model: str | None = None):
    with open(json_path) as f:
        schedule = json.load(f)

    dispatches = schedule.get("dispatches") or {}
    if not dispatches:
        raise ValueError(f"no dispatches in {json_path}")
    rows = schedule_trace.trace_rows_from_schedule(schedule)
    periods = schedule_trace.periods_ms(schedule)
    if window_ms is None:
        window_ms = max(float(r["end_us"]) for r in rows) / 1000.0

    if save_path is None:
        stem = os.path.splitext(os.path.basename(json_path))[0]
        save_path = os.path.join(_REPO, "plots", stem)
    else:
        save_path = os.path.splitext(save_path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    title = os.path.splitext(os.path.basename(json_path))[0]
    png, pdf = gantt.render_gantt_panels(
        [{"title": title, "rows": rows, "sched": dispatches}], save_path,
        periods=periods, cores=gantt.cores_from_schedule(dispatches),
        window_ms=window_ms, deadline_model=deadline_model,
        xlabel="Predicted time from K1 profiles (ms)", panel_labels=False,
        panel_height_mm=42.0)
    print(f"Done. Plot saved to {png} and {pdf}")
    return png, pdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", help="solved schedule JSON")
    parser.add_argument("--save", default=None,
                        help="output stem or .png path")
    parser.add_argument("--window-ms", type=float, default=None)
    parser.add_argument("--deadline-model", default=None)
    args = parser.parse_args()
    load_and_plot(args.json_path, args.save, args.window_ms,
                  args.deadline_model)
