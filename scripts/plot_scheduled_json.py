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


# The x-axis used to say "Predicted time from K1 profiles (ms)" for every
# schedule this renders, K1 or not -- so a PYNQ-Z1 Rocket schedule came out
# labelled with a SpacemiT X60's profile basis.  `--xlabel` (and `--title`)
# make the two strings that name what is being shown settable; both default to
# the previous behaviour exactly, so every existing caller is unchanged.
DEFAULT_XLABEL = "Predicted time from K1 profiles (ms)"


def load_and_plot(json_path: str, save_path: str | None = None,
                  window_ms: float | None = None,
                  deadline_model: str | None = None,
                  xlabel: str | None = None,
                  title: str | None = None):
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

    if title is None:
        title = os.path.splitext(os.path.basename(json_path))[0]
    png, pdf = gantt.render_gantt_panels(
        [{"title": title, "rows": rows, "sched": dispatches}], save_path,
        periods=periods, cores=gantt.cores_from_schedule(dispatches),
        window_ms=window_ms, deadline_model=deadline_model,
        xlabel=xlabel or DEFAULT_XLABEL, panel_labels=False,
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
    parser.add_argument("--xlabel", default=None,
                        help="x-axis label; default names the K1 profile "
                             "basis, which is wrong for any other target")
    parser.add_argument("--title", default=None,
                        help="panel title; default is the schedule's filename "
                             "stem")
    args = parser.parse_args()
    load_and_plot(args.json_path, args.save, args.window_ms,
                  args.deadline_model, args.xlabel, args.title)
