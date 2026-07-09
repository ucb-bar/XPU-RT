# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for replaying SW-scheduler dispatch JSON files at sim time.

Schedule JSON format (produced by the XPURT scheduler / profiler):

.. code-block:: text

    {
      "metadata": {"makespan": <float>},
      "dispatches": {
        "<model>_dispatch_<idx>": {"start_time": <float>, "duration": <float>},
        ...
      }
    }

``start_time`` and ``duration`` are in milliseconds by default (configurable
via ``time_unit``). The model name is derived from the dispatch key prefix,
with trailing digits stripped so ``mlp0_dispatch_0`` and ``mlp1_dispatch_0``
both normalize to ``"mlp"``.

Functions here are pure (no torch / no isaac dependencies) so they can be
imported eagerly in scripts that haven't launched the simulation yet.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


def _model_from_dispatch_key(key: str) -> str:
    """Extract the (normalized) model name from a dispatch key.

    Example: ``"dronet_dispatch_0"`` -> ``"dronet"``. Trailing digits are
    stripped so ``"mlp0_dispatch_0"`` -> ``"mlp"``.
    """
    if "_dispatch" not in key:
        raise ValueError(f"Dispatch key has no '_dispatch' segment: {key!r}")
    model_name = key.split("_dispatch", 1)[0]
    normalized = re.sub(r"\d+$", "", model_name)
    return normalized or model_name


def _merge_sorted_times(times: list[float], eps: float) -> list[float]:
    """Merge breakpoints closer than ``eps`` (helps numerically-noisy schedules)."""
    out: list[float] = []
    for t in sorted(times):
        if not out or abs(t - out[-1]) > eps:
            out.append(t)
    return out


def load_schedule(
    schedule_path: Path,
    merge_eps: float = 1e-9,
    time_unit: str = "ms",
) -> tuple[float, dict[str, list[tuple[float, float]]]]:
    """Load a schedule JSON.

    Args:
        schedule_path: Path to the JSON file.
        merge_eps: Reserved for future schedule simplification (kept for API
            compatibility with the original inlined version).
        time_unit: Either ``"ms"`` or ``"s"``; converts to seconds internally.

    Returns:
        ``(period_seconds, by_model)`` where ``by_model`` maps the normalized
        model name to a list of ``(start_s, end_s)`` intervals within one period.
    """
    if not Path(schedule_path).is_file():
        raise FileNotFoundError(f"Schedule file not found: {schedule_path}")

    if time_unit not in ("ms", "s"):
        raise ValueError(f"time_unit must be 'ms' or 's', got {time_unit!r}")
    to_seconds = 1e-3 if time_unit == "ms" else 1.0

    data = json.loads(Path(schedule_path).read_text())
    dispatches = data.get("dispatches")
    if not isinstance(dispatches, dict):
        raise ValueError("Schedule JSON must contain a 'dispatches' object.")

    by_model: dict[str, list[tuple[float, float]]] = defaultdict(list)
    max_end = 0.0
    for key, d in dispatches.items():
        m = _model_from_dispatch_key(str(key))
        start = float(d["start_time"]) * to_seconds
        end = start + float(d["duration"]) * to_seconds
        by_model[m].append((start, end))
        max_end = max(max_end, end)

    meta = data.get("metadata") or {}
    makespan = float(meta["makespan"]) * to_seconds if "makespan" in meta else max_end
    period = max(makespan, max_end)

    # `merge_eps` is currently a no-op kept for API parity with the original
    # inlined helper. If you ever need to merge near-coincident breakpoints,
    # apply `_merge_sorted_times(...)` here.
    _ = merge_eps

    return period, dict(by_model)


def load_schedule_by_hw(
    schedule_path: Path,
    time_unit: str = "ms",
) -> tuple[float, dict[str, list[tuple[float, float, str]]]]:
    """Load a schedule JSON, grouping dispatches by hardware target.

    Each interval is annotated with its (normalized) model name so the
    visualization can colour-code by model while laying out one row per HW
    resource — matching how real-time HW utilization is usually visualized.

    Args:
        schedule_path: Path to the JSON file.
        time_unit: Either ``"ms"`` or ``"s"``; converts to seconds internally.

    Returns:
        ``(period_seconds, by_hw)`` where ``by_hw`` maps a hardware-target
        string (e.g. ``"CPU_P#0"``) to a list of ``(start_s, end_s, model_name)``
        intervals within one period.  Returns an empty dict if no dispatch
        carries a ``hardware_target`` field (older schedules).
    """
    if not Path(schedule_path).is_file():
        raise FileNotFoundError(f"Schedule file not found: {schedule_path}")
    if time_unit not in ("ms", "s"):
        raise ValueError(f"time_unit must be 'ms' or 's', got {time_unit!r}")
    to_seconds = 1e-3 if time_unit == "ms" else 1.0

    data = json.loads(Path(schedule_path).read_text())
    dispatches = data.get("dispatches")
    if not isinstance(dispatches, dict):
        raise ValueError("Schedule JSON must contain a 'dispatches' object.")

    by_hw: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    max_end = 0.0
    for key, d in dispatches.items():
        hw = d.get("hardware_target")
        if hw is None:
            continue
        model = _model_from_dispatch_key(str(key))
        start = float(d["start_time"]) * to_seconds
        end = start + float(d["duration"]) * to_seconds
        by_hw[str(hw)].append((start, end, model))
        max_end = max(max_end, end)

    meta = data.get("metadata") or {}
    makespan = float(meta["makespan"]) * to_seconds if "makespan" in meta else max_end
    period = max(makespan, max_end)
    return period, dict(by_hw)


def load_schedule_jobs(
    schedule_path: Path,
    time_unit: str = "ms",
) -> tuple[float, dict[str, list[tuple[float, float]]]]:
    """Load a schedule JSON, returning per-job (start, end) intervals.

    A "job" is all dispatches sharing the same ``job_name`` field (e.g.
    ``dronet0``).  The interval is ``(min start_time, max end_time)`` across
    all dispatches in that job — representing when the full model invocation
    begins and completes.

    The returned dict maps the *normalized* model name (trailing digits
    stripped, e.g. ``"dronet"``) to a sorted list of ``(start_s, end_s)``
    job intervals.  This is the correct granularity for modeling "sample
    input at job start, apply output at job end" behaviour.
    """
    if not Path(schedule_path).is_file():
        raise FileNotFoundError(f"Schedule file not found: {schedule_path}")
    if time_unit not in ("ms", "s"):
        raise ValueError(f"time_unit must be 'ms' or 's', got {time_unit!r}")
    to_seconds = 1e-3 if time_unit == "ms" else 1.0

    data = json.loads(Path(schedule_path).read_text())
    dispatches = data.get("dispatches")
    if not isinstance(dispatches, dict):
        raise ValueError("Schedule JSON must contain a 'dispatches' object.")

    # Group by job_name (e.g. "dronet0", "mlp_control3", "yolov8_nano")
    job_bounds: dict[str, tuple[float, float]] = {}
    for _key, d in dispatches.items():
        job = d.get("job_name")
        if job is None:
            continue
        start = float(d["start_time"]) * to_seconds
        end = start + float(d["duration"]) * to_seconds
        if job in job_bounds:
            s, e = job_bounds[job]
            job_bounds[job] = (min(s, start), max(e, end))
        else:
            job_bounds[job] = (start, end)

    # Normalize job names → model names (strip trailing digits)
    by_model: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for job, (s, e) in job_bounds.items():
        normalized = re.sub(r"\d+$", "", job) or job
        by_model[normalized].append((s, e))

    # Sort intervals within each model
    for m in by_model:
        by_model[m].sort()

    meta = data.get("metadata") or {}
    max_end = max((e for intervals in by_model.values() for _, e in intervals), default=0.0)
    makespan = float(meta["makespan"]) * to_seconds if "makespan" in meta else max_end
    period = max(makespan, max_end)
    return period, dict(by_model)


def is_model_active(
    model_name: str,
    time_in_period: float,
    model_dispatches: dict[str, list[tuple[float, float]]],
    tol: float = 1e-9,
) -> bool:
    """Whether ``model_name`` is mid-dispatch at ``time_in_period`` (seconds).

    A small tolerance ``tol`` (default 1 ns) is applied to ``start`` so that
    schedule entries with floating-point-noisy starts like ``8.88e-19 s``
    correctly cover ``time_in_period == 0``. Without this, the very first
    dispatch in a period can be silently skipped on the very first probe.
    """
    if model_name not in model_dispatches:
        return False
    for start, end in model_dispatches[model_name]:
        if (start - tol) <= time_in_period < end:
            return True
    return False


def did_model_complete(
    model_name: str,
    time_in_period: float,
    control_dt: float,
    model_dispatches: dict[str, list[tuple[float, float]]],
) -> bool:
    """Whether any dispatch of ``model_name`` completed in ``(t - control_dt, t]``.

    When simulating a schedule at a coarser control rate than the dispatch
    granularity, individual dispatches (often sub-ms) will never be "active"
    at the exact probe instant.  This function checks whether a full dispatch
    *ended* during the last control step — meaning the model's output became
    available to consume at this control tick.
    """
    if model_name not in model_dispatches:
        return False
    t_lo = time_in_period - control_dt
    for _start, end in model_dispatches[model_name]:
        if t_lo < end <= time_in_period:
            return True
    return False
