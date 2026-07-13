#!/usr/bin/env python3
"""
Feedback-driven compilation: retroactively analyze a previously-scheduled
JSON file for dispatch-granularity mismatches, without re-running the
scheduler or profiling pipeline.

Motivating scenario: a prior run partitioned each model with the compiler's
default dispatch granularity, profiled it, and let xpu-rt's scheduler
compute a minimum-makespan schedule -- and the result still can't meet its
deadline. This script re-reads that schedule's JSON and flags which
non-periodic job's dispatches are too coarse (or needlessly too fine)
relative to the periodic jobs sharing the schedule. It's advisory only --
xpu-rt can't split an already-coarse dispatch itself; that has to happen
upstream, in whatever compiler produced the dispatch graph.

Usage:
    python scripts/analyze_granularity.py <scheduled_json_path> [--json]

Works on any schedule JSON this repo produces, old or new: if the file
already carries metadata["granularity_advice"] (written by a scheduler run
after this feature landed), that's printed directly. Otherwise falls back to
analyze_granularity.from_schedule_json(), which infers periodicity from
dispatch-key naming (see granularity_advisor.py's module docstring for the
precision trade-off that implies).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'xpu-rt'))

from granularity_advisor import analyze_granularity, from_schedule_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a schedule JSON for dispatch-granularity advice."
    )
    parser.add_argument("schedule_json", help="Path to a scheduled_*.json file")
    parser.add_argument(
        "--json", action="store_true",
        help="Print machine-readable JSON instead of human-readable lines",
    )
    args = parser.parse_args()

    with open(args.schedule_json) as f:
        schedule = json.load(f)

    cached = schedule.get("metadata", {}).get("granularity_advice")
    if cached is not None:
        advice_dicts = cached
        source = "metadata (computed at scheduling time)"
    else:
        records = from_schedule_json(schedule)
        advice_dicts = [a.as_dict() for a in analyze_granularity(records)]
        source = "inferred from dispatch-key naming (no periodicity metadata in this file)"

    if args.json:
        print(json.dumps({"source": source, "advice": advice_dicts}, indent=2))
        return 0

    print(f"{args.schedule_json}")
    print(f"  granularity advice source: {source}")
    if not advice_dicts:
        print("  no periodic job found in this schedule -- nothing to compare against")
        return 0

    for a in advice_dicts:
        marker = "  " if a["recommended"] == "unchanged" else "! "
        print(f"{marker}[{a['recommended']}] {a['reason']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
