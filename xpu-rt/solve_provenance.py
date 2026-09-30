"""What a solve was actually solved *against*, recorded with the schedule.

WHY THIS EXISTS. `pdb_hash` fingerprints the profile CSVs the solver read, and nothing
else. So an additive solve and a `--board-calibration` re-solve of the same spec read the same
CSVs, carry the SAME `pdb_hash`, and `compare_candidates.py` refuses to adjudicate them:

    REFUSING: both schedules carry the same pdb_hash, so they were solved against the
    SAME measured costs.

That is exactly the comparison the board-feedback story is built on -- panel 3 (board re-cost)
against panel 4 (re-solve on board costs) -- so `pdb_hash` alone leaves the flagship claim
outside the guard rail that protects every other claim. Nor does it record the environment:
`XPURT_CPSAT_WORKERS` changes CP-SAT's search (and its determinism), and compaction and
automerge rewrite the emitted schedule.

`solve_hash` folds all of that in: profile content, calibration table content, solver,
scheduler, and the env switches that change the result. Two schedules that differ in any of
them get different hashes and can be compared; two that differ in none of them cannot, which
is the honest answer.

Usage: `record(...)` once per process, as early as the facts are known;
`as_metadata()` at emit time.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

# Env switches that change a solve's result and must therefore be part of its identity.
TRACKED_ENV = ("XPURT_CPSAT_WORKERS", "XPURT_COMPACT", "XPURT_NO_COMPACT",
               "XPURT_AUTOMERGE", "XPURT_NO_AUTOMERGE", "XPURT_MAX_HARTS",
               "XPURT_MAX_POOL_WIDTH", "XPURT_MAX_POOLS", "XPURT_TICKS_PER_SEC")

_state: Dict[str, Any] = {}


def record(*, solver: Optional[str] = None, scheduler: Optional[str] = None,
           time_limit: Optional[float] = None, random_seed: Optional[int] = None,
           board_calibration: Optional[Dict[str, Any]] = None,
           calibration_path: Optional[str] = None,
           max_periodic_iters: Optional[int] = None) -> None:
    """Record the solve context. Safe to call more than once; later calls update."""
    _state.update({k: v for k, v in dict(
        solver=solver, scheduler=scheduler, time_limit=time_limit,
        random_seed=random_seed, max_periodic_iters=max_periodic_iters).items()
        if v is not None})
    _state["env"] = {k: os.environ[k] for k in TRACKED_ENV if k in os.environ}
    if calibration_path:
        _state["board_calibration"] = calibration_summary(board_calibration,
                                                          calibration_path)


def calibration_summary(calib: Optional[Dict[str, Any]],
                        path: str) -> Dict[str, Any]:
    """Which calibration tiers this table can actually serve.

    `_board_calibration_mult` falls through exact per-dispatch key -> per-op kind
    (EXTRAPOLATED) -> aggregate scalar (last resort) and records nothing about which tier
    fired. Counting the keys at least tells a reader how much of the table is exact.
    """
    if calib is None:
        return {"path": path, "loaded": False}
    blob = json.dumps(calib, sort_keys=True).encode()
    return {
        "path": path,
        "loaded": True,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "schema": calib.get("schema"),
        "source": calib.get("source"),
        "workload": calib.get("workload"),
        "aggregate_multiplier": calib.get("aggregate_multiplier"),
        "n_exact_dispatch_keys": len(calib.get("per_dispatch_multiplier") or {}),
        "n_op_kind_keys": len(calib.get("per_op_multiplier") or {}),
    }


def solve_hash(pdb_hash: Optional[str] = None) -> str:
    """A fingerprint of everything this solve read and every switch that changed it."""
    h = hashlib.sha256()
    h.update((pdb_hash or "").encode())
    payload = {k: v for k, v in _state.items()}
    cal = payload.get("board_calibration") or {}
    h.update(json.dumps({
        "calibration_sha256": cal.get("sha256"),
        "calibration_loaded": cal.get("loaded"),
        "solver": payload.get("solver"),
        "scheduler": payload.get("scheduler"),
        "time_limit": payload.get("time_limit"),
        "random_seed": payload.get("random_seed"),
        "max_periodic_iters": payload.get("max_periodic_iters"),
        "env": payload.get("env") or {},
    }, sort_keys=True).encode())
    return h.hexdigest()


def as_metadata(pdb_hash: Optional[str] = None) -> Dict[str, Any]:
    """The block to merge into a schedule's `metadata`. Empty when nothing was recorded."""
    if not _state:
        return {}
    out = {"solve_hash": solve_hash(pdb_hash), "solve_env": dict(_state)}
    if "board_calibration" not in _state:
        # Absence is a fact worth recording: it says this schedule is the additive view.
        out["solve_env"]["board_calibration"] = {"loaded": False, "path": None}
    return out


def reset() -> None:
    """Drop recorded state (tests)."""
    _state.clear()
