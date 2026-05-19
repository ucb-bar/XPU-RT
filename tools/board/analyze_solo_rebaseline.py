"""Compare a board solo-E2E measurement against expected baselines.

Reads ``measurement.json`` (schema ``measurement_record_board_v1``) and the
companion ``plan.json`` (which carries ``expected_us_from_measurements_json``
per partition) and reports per-partition drift + a PASS/FAIL verdict.

Usage:
    uv run python scripts/board/analyze_solo_rebaseline.py \
        build/board_run/solo_rebaseline/measurement.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PASS_THRESHOLD_PCT = 15.0


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <measurement.json>", file=sys.stderr)
        return 2

    meas_path = Path(sys.argv[1])
    plan_path = meas_path.parent / "plan.json"
    if not meas_path.exists():
        print(f"missing: {meas_path}", file=sys.stderr)
        return 2
    if not plan_path.exists():
        print(f"missing companion plan.json next to measurement: {plan_path}", file=sys.stderr)
        return 2

    meas = json.loads(meas_path.read_text())
    plan = json.loads(plan_path.read_text())

    expected_by_pid = {p["partition_id"]: p["expected_us_from_measurements_json"] for p in plan["partitions"]}

    rows = []
    for entry in meas.get("raw_per_partition_us", []):
        pid = entry["partition_id"]
        measured = entry.get("mean_us")
        expected = expected_by_pid.get(pid)
        if measured is None or expected is None:
            drift = None
        else:
            drift = (measured - expected) / expected * 100.0
        rows.append((pid, entry.get("backend"), expected, measured, drift, entry.get("ok", True)))

    print(f"\n{'partition':22s} {'backend':7s} {'expected (ms)':>14s} {'measured (ms)':>14s} {'drift %':>9s}  ok")
    print("-" * 80)
    max_abs_drift = 0.0
    failures: list[str] = []
    for pid, backend, exp, meas_us, drift, ok in rows:
        exp_ms = f"{exp/1000:.1f}" if exp is not None else "?"
        meas_ms = f"{meas_us/1000:.1f}" if meas_us is not None else "?"
        drift_s = f"{drift:+.1f}" if drift is not None else "?"
        ok_s = "Y" if ok else "N"
        print(f"{pid:22s} {backend:7s} {exp_ms:>14s} {meas_ms:>14s} {drift_s:>9s}  {ok_s}")
        if drift is not None:
            max_abs_drift = max(max_abs_drift, abs(drift))
            if abs(drift) > PASS_THRESHOLD_PCT:
                failures.append(f"{pid} drift {drift:+.1f}%")
        if not ok:
            failures.append(f"{pid} ok=false")

    print()
    print(f"max |drift|: {max_abs_drift:.1f}%   threshold: {PASS_THRESHOLD_PCT:.1f}%")
    if failures:
        print(f"FAIL — {len(failures)} partition(s) outside the threshold:")
        for f in failures:
            print(f"  - {f}")
        print()
        print("Action: regenerate xpu-rt/data/profiled/qnn_e2e/measurements.json from this run")
        print("        and re-bootstrap calibration via bootstrap_from_solo_measurements().")
        return 1
    print("PASS — calibration anchors are valid; downstream predictions stand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
