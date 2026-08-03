"""Shared infrastructure for the four scheduling policies.

The `run_underlying_solver` helper invokes scripts/run_xpurt_schedule.py
as a subprocess so the policy gets the SAME fixture format the rest of
the pipeline consumes (with automerge + compaction post-passes wired in
via the registry wrapper). Subprocess isolation also means a solver
that crashes doesn't take the policy driver with it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Re-invoke whatever interpreter is running us, so the subprocess inherits this
# environment's MOSEK/cvxpy install. Override with XPURT_PYTHON when the policy
# must run under a different env than the driver.
_PY = os.environ.get("XPURT_PYTHON") or sys.executable


def run_underlying_solver(
    workload_path: str,
    *,
    solver: str = "milp",
    scheduler: str = "cpsat",
    time_limit: float = 60.0,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], float, str]:
    """Run `scripts/run_xpurt_schedule.py` and return
    (fixture_path, wall_s, status). solver/scheduler match the script's
    flag semantics.
    """
    if solver == "milp":
        flags = ["--solver", "milp", "--scheduler", scheduler]
    else:
        flags = ["--solver", solver]
    cmd = [
        _PY,
        str(_REPO_ROOT / "scripts" / "run_xpurt_schedule.py"),
        "--networks-json", workload_path,
        *flags,
        "--use-profiled",
        "--time-limit", str(time_limit),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT) + ":" + str(_REPO_ROOT / "xpu-rt") + ":" + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)

    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=str(_REPO_ROOT),
                                capture_output=True, text=True,
                                env=env, timeout=600)
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - t0, "timeout"
    wall = time.perf_counter() - t0

    if result.returncode != 0:
        tail = (result.stderr or "")[-300:].replace("\n", " ")
        return None, wall, f"rc={result.returncode}: {tail}"

    stem = Path(workload_path).stem
    solver_tag = solver_tag_for(solver, scheduler)
    fixture_path = _REPO_ROOT / "schedules" / f"scheduled_{stem}{solver_tag}_profiled.json"
    if not fixture_path.exists():
        return None, wall, f"fixture missing: {fixture_path}"
    return str(fixture_path), wall, "ok"


def solver_tag_for(solver: str, scheduler: str) -> str:
    """Match the tag-suffix convention in scripts/run_xpurt_schedule.py
    lines 550-564. Used to reconstruct the output fixture path."""
    if solver == "greedy":
        return "_greedy"
    if solver == "greedy_periodic":
        return "_greedy_periodic"
    if solver == "decomposed":
        return "_decomposed"
    # milp/registry: blank for mosek, _<scheduler> otherwise.
    if solver == "milp" and scheduler != "mosek":
        return f"_{scheduler}"
    return ""


def summarize_fixture(fixture_path: str,
                       workload_data: Dict[str, Any],
                       solver_label: str = "") -> Dict[str, Any]:
    """Compute summary metrics + band report for a fixture."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT / "xpu-rt"))
    from diagnostics import check_band_invariant

    fixture = json.loads(Path(fixture_path).read_text())
    report = check_band_invariant(fixture, workload_data, solver=solver_label)
    makespan = float(fixture.get("metadata", {}).get("makespan", 0.0))
    n_misses = report.n_deadline_violations
    n_dispatches = len(fixture.get("dispatches", {}))
    return {
        "fixture_path": fixture_path,
        "makespan": makespan,
        "n_deadline_miss": n_misses,
        "n_release_viol": report.n_release_violations,
        "n_dispatches": n_dispatches,
        "band_report": {
            "worst_release_overrun": report.worst_release_overrun_us,
            "worst_deadline_overrun": report.worst_deadline_overrun_us,
            "per_network": dict(report.per_network),
        },
    }
