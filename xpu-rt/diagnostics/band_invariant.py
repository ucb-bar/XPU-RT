"""Frequency-as-band invariant checker.

A network N with period P_N and instance k defines an execution band

    [R_k = start_time + k*P_N,  D_k = R_k + window_duration_N]

Every dispatch belonging to instance k must satisfy

    R_k  <=  dispatch.start_time         (release)
    dispatch.start_time + dispatch.duration  <=  D_k         (deadline)

This module is read-only: it inspects a scheduled fixture against a
workload JSON and returns the violation set. No mutation, no patching.
The aggregate counts feed `scripts/audit_band_compliance.py`.

The fixture format is what `postprocessing.output_scheduled_json`
produces — each entry in `dispatches` has `start_time`, `duration`,
`job_name`, `hardware_target`. The dispatch *name* carries the
instance index: periodic instances are emitted as `<base><i>_<...>`
(see `workload_factory.py` line ~564 where instance_identifier =
f"{network_identifier}{i}"). So we parse the instance from the name.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Tolerance for float comparison. Fixture units = workload-period units
# (typically ms). 1e-4 ms = 100ns of float noise, tight enough to surface
# real overruns and loose enough not to flag fp roundoff.
EPS = 1e-4


@dataclass
class BandViolation:
    dispatch: str
    network: str
    instance_idx: int
    start_us: float
    finish_us: float
    R_k_us: float
    D_k_us: float
    release_overrun_us: float   # max(0, R_k - start). 0 if no release overrun.
    deadline_overrun_us: float  # max(0, finish - D_k). 0 if no deadline overrun.

    @property
    def is_release_violation(self) -> bool:
        return self.release_overrun_us > EPS

    @property
    def is_deadline_violation(self) -> bool:
        return self.deadline_overrun_us > EPS

    @property
    def is_any(self) -> bool:
        return self.is_release_violation or self.is_deadline_violation


@dataclass
class BandReport:
    """Aggregate band-compliance report for a scheduled fixture."""

    solver: str
    workload_label: str
    n_ops: int
    n_release_violations: int
    n_deadline_violations: int
    worst_release_overrun_us: float
    worst_deadline_overrun_us: float
    violations: List[BandViolation] = field(default_factory=list)
    # Per-network breakdown for quick read-off in the audit CSV.
    per_network: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        """One-row CSV summary."""
        return {
            "solver": self.solver,
            "workload": self.workload_label,
            "n_ops": self.n_ops,
            "n_release_violations": self.n_release_violations,
            "n_deadline_violations": self.n_deadline_violations,
            "worst_release_overrun_us": round(self.worst_release_overrun_us, 3),
            "worst_deadline_overrun_us": round(self.worst_deadline_overrun_us, 3),
            "n_networks_with_misses": sum(
                1 for v in self.per_network.values() if v.get("n_violations", 0) > 0
            ),
        }


def _parse_instance(dispatch_name: str, job_name: str,
                    periodic_bases: Dict[str, Tuple[int, float, float, float]],
                    nonperiodic_bases: set) -> Tuple[str, int]:
    """Map a dispatch name to (network_base, instance_idx).

    Periodic instances are named `<base><i>_<rest>` where <i> is a run
    of digits. We try each known periodic base in descending length
    order so `dronet` is matched before `dr` (paranoia for short
    names).
    """
    # Sort by descending length so the longest base prefix wins.
    for base in sorted(periodic_bases.keys(), key=lambda s: -len(s)):
        if dispatch_name.startswith(base):
            tail = dispatch_name[len(base):]
            m = re.match(r"^(\d+)_", tail)
            if m:
                return base, int(m.group(1))
    # Non-periodic: instance index 0.
    if job_name in nonperiodic_bases:
        return job_name, 0
    # Fallback: split on first underscore.
    if "_" in dispatch_name:
        head = dispatch_name.split("_", 1)[0]
        # Try stripping trailing digits for an instance.
        m = re.match(r"^(.+?)(\d+)$", head)
        if m:
            return m.group(1), int(m.group(2))
        return head, 0
    return job_name or "unknown", 0


def _periodic_metadata(workload_data: Dict[str, Any]) -> Tuple[Dict[str, Tuple[int, float, float, float]], set]:
    """Extract per-network periodic metadata from the workload JSON.

    Returns:
        periodic_bases — {network_base: (num_instances, period_ms,
                                         window_duration_ms, start_time_ms)}
        nonperiodic_bases — set of non-periodic network names
    """
    networks = workload_data.get("networks", {})
    periodic_bases: Dict[str, Tuple[int, float, float, float]] = {}
    nonperiodic_bases: set = set()
    for name, info in networks.items():
        period = info.get("period")
        if period is None:
            nonperiodic_bases.add(name)
            continue
        num_instances = int(info.get("num_instances", 1))
        window_duration = float(info.get("window_duration", period))
        start_time = float(info.get("start_time", 0.0))
        periodic_bases[name] = (num_instances, float(period), window_duration, start_time)
    return periodic_bases, nonperiodic_bases


def check_band_invariant(fixture: Dict[str, Any], workload_data: Dict[str, Any],
                          solver: str = "unknown",
                          workload_label: str = "") -> BandReport:
    """Per-op band check: each dispatch must lie within its instance's
    [R_k, D_k] band. Returns a BandReport with aggregate counts and a
    per-violation breakdown.

    fixture: the JSON produced by `output_scheduled_json` (top-level
        contains "dispatches" dict).
    workload_data: the top-level networks JSON (period / window_duration /
        num_instances per network).
    """
    periodic_bases, nonperiodic_bases = _periodic_metadata(workload_data)

    dispatches = fixture.get("dispatches", {})
    violations: List[BandViolation] = []
    per_network: Dict[str, Dict[str, Any]] = {}

    # Units: fixtures from postprocessing.output_scheduled_json store
    # start_time / duration in the same units as the workload's profile
    # data (milliseconds in the run_xpurt pipeline). The workload JSON's
    # `period` / `window_duration` are also in ms. We compare directly
    # without unit conversion. Field names below say "_us" for legacy
    # reasons but are unit-agnostic — they hold whatever the fixture
    # holds.
    UNIT = 1.0

    for dispatch_name, entry in dispatches.items():
        start_us = float(entry.get("start_time", 0.0))
        duration_us = float(entry.get("duration", 0.0))
        finish_us = start_us + duration_us
        job_name = entry.get("job_name") or ""

        # Strip trailing instance digits from job_name to match periodic_bases.
        base_job = re.sub(r"\d+$", "", job_name) if job_name else ""
        if base_job in periodic_bases:
            base, inst = base_job, 0
            # Re-parse from dispatch name to get accurate instance.
            base_parsed, inst_parsed = _parse_instance(
                dispatch_name, job_name, periodic_bases, nonperiodic_bases
            )
            # If parsing recovered a known base, trust it.
            if base_parsed in periodic_bases:
                base, inst = base_parsed, inst_parsed
        else:
            base, inst = _parse_instance(
                dispatch_name, job_name, periodic_bases, nonperiodic_bases
            )

        # Compute R_k, D_k.
        if base in periodic_bases:
            num_inst, period_ms, window_ms, start_ms = periodic_bases[base]
            R_k_us = (start_ms + inst * period_ms) * UNIT
            D_k_us = (start_ms + inst * period_ms + window_ms) * UNIT
        else:
            # Non-periodic op: no band constraint. Track it for n_ops
            # but not for violations.
            net_stat = per_network.setdefault(
                base, {"n_ops": 0, "n_violations": 0, "is_periodic": False}
            )
            net_stat["n_ops"] += 1
            continue

        net_stat = per_network.setdefault(
            base, {"n_ops": 0, "n_violations": 0, "is_periodic": True}
        )
        net_stat["n_ops"] += 1

        release_overrun = max(0.0, R_k_us - start_us)
        deadline_overrun = max(0.0, finish_us - D_k_us)

        if release_overrun > EPS or deadline_overrun > EPS:
            v = BandViolation(
                dispatch=dispatch_name,
                network=base,
                instance_idx=inst,
                start_us=start_us,
                finish_us=finish_us,
                R_k_us=R_k_us,
                D_k_us=D_k_us,
                release_overrun_us=release_overrun,
                deadline_overrun_us=deadline_overrun,
            )
            violations.append(v)
            net_stat["n_violations"] += 1

    n_release = sum(1 for v in violations if v.is_release_violation)
    n_deadline = sum(1 for v in violations if v.is_deadline_violation)
    worst_release = max((v.release_overrun_us for v in violations), default=0.0)
    worst_deadline = max((v.deadline_overrun_us for v in violations), default=0.0)

    return BandReport(
        solver=solver,
        workload_label=workload_label,
        n_ops=len(dispatches),
        n_release_violations=n_release,
        n_deadline_violations=n_deadline,
        worst_release_overrun_us=worst_release,
        worst_deadline_overrun_us=worst_deadline,
        violations=violations,
        per_network=per_network,
    )
