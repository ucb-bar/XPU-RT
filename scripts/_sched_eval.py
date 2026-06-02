"""Shared helpers for the iterative-loop drivers (compare_backends, iterate_firesim).

Runs run_xpurt_schedule.py for a given (profile_hw, solver, scheduler) by writing
a temp networks JSON (so outputs don't collide), then loads the emitted
SchedulerReport and runs the advisor. Predicted-only — no FireSim/spike.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))

GREEDY_FAMILY = {"decomposed", "greedy", "greedy_periodic"}


def available_backends(target: str = "firesim_rocket_saturn",
                       needed=("dronet", "mlp_control", "yolov8_nano")) -> List[str]:
    """profile_hw dirs under gen/profile/<hw>/<target> that have all needed models."""
    root = os.path.join(REPO, "gen", "profile")
    out: List[str] = []
    if not os.path.isdir(root):
        return out
    for hw in sorted(os.listdir(root)):
        base = os.path.join(root, hw, target)
        if os.path.isdir(base) and all(os.path.isdir(os.path.join(base, m)) for m in needed):
            out.append(hw)
    return out


def _solver_tag(solver: str, scheduler: Optional[str]) -> str:
    if solver in ("greedy", "greedy_periodic", "decomposed"):
        return f"_{solver}"
    if scheduler and scheduler != "mosek":
        return f"_{scheduler}"
    return ""


def report_path(stem: str, solver: str, scheduler: Optional[str], profiled: bool = True) -> str:
    tag = _solver_tag(solver, scheduler)
    ptag = "_profiled" if profiled else ""
    return os.path.join(REPO, "schedules", f"scheduled_{stem}{tag}{ptag}_report.json")


def run_candidate(base_spec_path: str, *, profile_hw: Optional[Dict[str, str]] = None,
                  solver: str = "greedy", scheduler: Optional[str] = None,
                  stem: Optional[str] = None, timeout: int = 180,
                  profiled: bool = True) -> Dict[str, Any]:
    """Run one config; return {status, report_path?, report?, stem, wall_s?}."""
    with open(base_spec_path) as f:
        spec = json.load(f)
    if profile_hw:
        spec.setdefault("hardware", {})["profile_hw"] = dict(profile_hw)
    if stem is None:
        label = (solver if solver in GREEDY_FAMILY else (scheduler or solver))
        hw = "-".join(sorted(set((profile_hw or {}).values()))) or "base"
        stem = f"_iter_{label}_{hw}"
    tmpdir = tempfile.mkdtemp(prefix="iter_spec_")
    spec_path = os.path.join(tmpdir, f"{stem}.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f)

    cmd = [sys.executable, os.path.join(REPO, "scripts", "run_xpurt_schedule.py"),
           "--networks-json", spec_path, "--solver", solver]
    if solver not in GREEDY_FAMILY and scheduler:
        cmd += ["--scheduler", scheduler]
    cmd += (["--profiled"] if profiled else ["--no-profiled"])
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "stem": stem}
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-300:].replace("\n", " ")
        return {"status": f"error: {tail}", "stem": stem}
    rp = report_path(stem, solver, scheduler, profiled)
    if not os.path.exists(rp):
        return {"status": f"error: no report at {rp}", "stem": stem}
    with open(rp) as f:
        report = json.load(f)
    return {"status": "ok", "report_path": rp, "report": report, "stem": stem}


def advise(report: Dict[str, Any], deadline_us: Optional[float]):
    import advisor
    return advisor.advise_schedule(report, deadline_us=deadline_us)
