#!/usr/bin/env python3
"""Extract the shortest qualified steady-state repeat frame from a schedule."""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))

import repeat_window  # noqa: E402
import workload_spec  # noqa: E402


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write(path: str, value: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--anchor-model", required=True,
                        help="large model whose complete first instance must fit")
    parser.add_argument("--quantum-ms", type=float,
                        help="allowed frame-length grid (default: shortest period)")
    parser.add_argument("--max-window-ms", type=float,
                        help="largest prefix to consider")
    parser.add_argument("--out", required=True,
                        help="materialized repeatable schedule JSON")
    parser.add_argument("--report",
                        help="qualification report JSON (default: OUT.report.json)")
    args = parser.parse_args()

    schedule = _load(args.schedule)
    workload = _load(args.workload)
    periods = workload_spec.periods_ms(workload)
    _, known = workload_spec.windows_and_names(workload)
    report = repeat_window.find(
        schedule, periods, args.anchor_model, known,
        quantum_ms=args.quantum_ms, max_window_ms=args.max_window_ms)
    frame = repeat_window.extract_frame(schedule, report)
    report_path = args.report or args.out + ".report.json"
    _write(args.out, frame)
    _write(report_path, report)
    print(
        f"qualified {report['window_ms']:g} ms repeat frame: "
        f"{report['dispatches_shown']} dispatches included, "
        f"{report['dispatches_excluded']} trailing dispatches excluded"
    )
    print(f"wrote {args.out}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
