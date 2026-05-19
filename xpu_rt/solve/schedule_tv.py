"""Translation validation for scheduler outputs.

Given a concrete scheduler solution (start times, end times, device
assignments, makespan), this module checks five obligations:

1. Assignment validity — every partition is assigned to exactly one
   device, and the device index is in ``[0, num_devices)``.
2. Duration consistency — ``end - start`` equals the declared duration
   on the chosen device (within float tolerance).
3. Dependency edges — for every ``succ -> preds`` arc,
   ``start[succ] >= end[pred] + transfer[d_pred][d_succ]``.
4. Per-device no-overlap — on any device, intervals are pairwise
   disjoint in time.
5. Makespan equality — the reported makespan matches
   ``max(end_times.values())``.

For concrete numeric output these checks are trivial — plain Python
suffices. The Z3 path exists for (a) a uniform structured
counterexample shape across check kinds, (b) forward-compatibility with
parametric durations / conditional schedules, and (c) drop-in
integration with the obligation harness in :mod:`z3_obligations`.

Z3 obligations are encoded in integer microseconds (input floats are
rounded by ``_SCALE``). For each obligation type we assert the
negation; an ``unsat`` answer proves it, ``sat`` returns a structured
counterexample, ``unknown`` (timeout) is reported as ``proved=False``
with a ``"timeout"`` violation kind so the caller never gets a silent
pass.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "ScheduleTVResult",
    "ScheduleTVViolation",
    "translation_validate_schedule",
]


_SCALE = 1000  # microseconds -> integer solver units (matches schedule_joint_cpsat)
_FLOAT_TOL_US = 1e-3  # tolerance for end-start vs declared duration


@dataclass(frozen=True)
class ScheduleTVViolation:
    """A single TV failure.

    Attributes:
        kind: One of ``"dep_violated"``, ``"device_overlap"``,
            ``"unassigned"``, ``"multi_assigned"``,
            ``"duration_mismatch"``, ``"makespan_mismatch"``, or
            ``"timeout"``.
        detail: Structured payload identifying the offending objects
            (partition IDs, device indices, observed vs expected
            values).
    """

    kind: str
    detail: dict[str, Any]


@dataclass(frozen=True)
class ScheduleTVResult:
    """Output of :func:`translation_validate_schedule`.

    Attributes:
        proved: ``True`` iff every obligation discharged successfully.
            False on any violation or solver timeout.
        violations: Per-check failures (empty when ``proved`` is True).
        z3_time_ms: Wall-clock spent in Z3 (zero if ``use_z3=False``).
        python_time_ms: Wall-clock spent in the pure-Python check path.
        n_deps_checked: Number of dependency arcs checked.
        n_overlap_pairs_checked: Number of same-device interval pairs
            checked for overlap.
    """

    proved: bool
    violations: list[ScheduleTVViolation] = field(default_factory=list)
    z3_time_ms: float = 0.0
    python_time_ms: float = 0.0
    n_deps_checked: int = 0
    n_overlap_pairs_checked: int = 0


def _to_int_us(value: float) -> int:
    return int(round(float(value) * _SCALE))


def _python_checks(
    partition_ids: list[str],
    durations_us_by_device: dict[str, list[float]],
    dependencies: dict[str, list[str]],
    num_devices: int,
    start_times: dict[str, float],
    end_times: dict[str, float],
    device_assignments: dict[str, int],
    makespan_us: float,
    transfer_us: list[list[float]],
) -> tuple[list[ScheduleTVViolation], int, int]:
    """Pure-Python obligation evaluator.

    Returns:
        ``(violations, n_deps_checked, n_overlap_pairs_checked)``.
    """
    violations: list[ScheduleTVViolation] = []

    # (1) Assignment validity + (2) Duration consistency.
    for pid in partition_ids:
        d = device_assignments.get(pid)
        if d is None:
            violations.append(ScheduleTVViolation("unassigned", {"partition": pid}))
            continue
        if d < 0 or d >= num_devices:
            violations.append(
                ScheduleTVViolation(
                    "multi_assigned",
                    {"partition": pid, "device": d, "num_devices": num_devices},
                )
            )
            continue
        per_dev = durations_us_by_device.get(pid)
        if per_dev is None or d >= len(per_dev):
            violations.append(
                ScheduleTVViolation(
                    "duration_mismatch",
                    {"partition": pid, "device": d, "reason": "missing_duration_row"},
                )
            )
            continue
        declared = float(per_dev[d])
        observed = float(end_times[pid]) - float(start_times[pid])
        if not math.isfinite(declared) or abs(observed - declared) > _FLOAT_TOL_US:
            violations.append(
                ScheduleTVViolation(
                    "duration_mismatch",
                    {
                        "partition": pid,
                        "device": d,
                        "declared_us": declared,
                        "observed_us": observed,
                    },
                )
            )

    # (3) Dependency edges.
    n_deps = 0
    for succ in partition_ids:
        for pred in dependencies.get(succ, []):
            if pred not in start_times or succ not in start_times:
                continue
            n_deps += 1
            d_pred = device_assignments.get(pred, -1)
            d_succ = device_assignments.get(succ, -1)
            if d_pred < 0 or d_succ < 0:
                # Already flagged by (1); skip arc check to avoid noise.
                continue
            transfer = float(transfer_us[d_pred][d_succ])
            required_start = float(end_times[pred]) + transfer
            if float(start_times[succ]) + _FLOAT_TOL_US < required_start:
                violations.append(
                    ScheduleTVViolation(
                        "dep_violated",
                        {
                            "pred": pred,
                            "succ": succ,
                            "end_pred_us": end_times[pred],
                            "start_succ_us": start_times[succ],
                            "transfer_us": transfer,
                            "slack_us": float(start_times[succ]) - required_start,
                        },
                    )
                )

    # (4) Per-device no-overlap.
    by_device: dict[int, list[str]] = {d: [] for d in range(num_devices)}
    for pid in partition_ids:
        d = device_assignments.get(pid, -1)
        if 0 <= d < num_devices:
            by_device[d].append(pid)
    n_overlap_pairs = 0
    for d, pids in by_device.items():
        # Sort by start for cheaper sequential overlap detection; we
        # still report all overlapping pairs (not just the first) so
        # callers see the full picture.
        pids_sorted = sorted(pids, key=lambda p: start_times[p])
        for i in range(len(pids_sorted)):
            for j in range(i + 1, len(pids_sorted)):
                n_overlap_pairs += 1
                p, q = pids_sorted[i], pids_sorted[j]
                # Disjoint iff end[p] <= start[q] (or symmetric); the
                # sort guarantees start[p] <= start[q].
                if float(end_times[p]) > float(start_times[q]) + _FLOAT_TOL_US:
                    violations.append(
                        ScheduleTVViolation(
                            "device_overlap",
                            {
                                "device": d,
                                "a": p,
                                "b": q,
                                "end_a_us": end_times[p],
                                "start_b_us": start_times[q],
                            },
                        )
                    )

    # (5) Makespan equality.
    if end_times:
        observed_makespan = max(end_times.values())
        if abs(float(makespan_us) - observed_makespan) > _FLOAT_TOL_US:
            violations.append(
                ScheduleTVViolation(
                    "makespan_mismatch",
                    {
                        "reported_us": float(makespan_us),
                        "max_end_us": observed_makespan,
                    },
                )
            )

    return violations, n_deps, n_overlap_pairs


def _z3_check_dependency(
    pred: str,
    succ: str,
    end_pred: int,
    start_succ: int,
    transfer: int,
    timeout_ms: int,
) -> ScheduleTVViolation | None:
    """Z3 query: prove ``start_succ >= end_pred + transfer``.

    All values are integer microseconds. We assert the negation and
    expect ``unsat``.
    """
    import z3

    solver = z3.Solver()
    solver.set(timeout=int(timeout_ms))
    s = z3.Int("start_succ")
    e = z3.Int("end_pred")
    t = z3.Int("transfer")
    solver.add(s == start_succ, e == end_pred, t == transfer)
    solver.add(s < e + t)
    res = solver.check()
    if res == z3.unsat:
        return None
    if res == z3.sat:
        return ScheduleTVViolation(
            "dep_violated",
            {
                "pred": pred,
                "succ": succ,
                "end_pred_us": end_pred / _SCALE,
                "start_succ_us": start_succ / _SCALE,
                "transfer_us": transfer / _SCALE,
            },
        )
    return ScheduleTVViolation("timeout", {"check": "dep_violated", "pred": pred, "succ": succ})


def _z3_check_overlap(
    p: str,
    q: str,
    device: int,
    start_p: int,
    end_p: int,
    start_q: int,
    end_q: int,
    timeout_ms: int,
) -> ScheduleTVViolation | None:
    """Z3 query: prove ``end_p <= start_q ∨ end_q <= start_p``.

    The disjunction is the standard no-overlap form. Negation is
    ``end_p > start_q ∧ end_q > start_p`` (both intervals strictly
    overlap).
    """
    import z3

    solver = z3.Solver()
    solver.set(timeout=int(timeout_ms))
    sp = z3.Int("start_p")
    ep = z3.Int("end_p")
    sq = z3.Int("start_q")
    eq = z3.Int("end_q")
    solver.add(sp == start_p, ep == end_p, sq == start_q, eq == end_q)
    solver.add(ep > sq, eq > sp)
    res = solver.check()
    if res == z3.unsat:
        return None
    if res == z3.sat:
        return ScheduleTVViolation(
            "device_overlap",
            {
                "device": device,
                "a": p,
                "b": q,
                "end_a_us": end_p / _SCALE,
                "start_b_us": start_q / _SCALE,
            },
        )
    return ScheduleTVViolation("timeout", {"check": "device_overlap", "a": p, "b": q})


def _z3_check_makespan(reported: int, max_end: int, timeout_ms: int) -> ScheduleTVViolation | None:
    import z3

    solver = z3.Solver()
    solver.set(timeout=int(timeout_ms))
    r = z3.Int("reported")
    m = z3.Int("max_end")
    solver.add(r == reported, m == max_end)
    solver.add(r != m)
    res = solver.check()
    if res == z3.unsat:
        return None
    if res == z3.sat:
        return ScheduleTVViolation(
            "makespan_mismatch",
            {"reported_us": reported / _SCALE, "max_end_us": max_end / _SCALE},
        )
    return ScheduleTVViolation("timeout", {"check": "makespan_mismatch"})


def _z3_checks(
    partition_ids: list[str],
    dependencies: dict[str, list[str]],
    num_devices: int,
    start_times: dict[str, float],
    end_times: dict[str, float],
    device_assignments: dict[str, int],
    makespan_us: float,
    transfer_us: list[list[float]],
    timeout_ms: int,
) -> tuple[list[ScheduleTVViolation], int, int]:
    """Run Z3 obligations corresponding to checks (3), (4), (5).

    Assignment (1) and duration consistency (2) are pure data shape
    checks for which Z3 adds no value — those still run in the Python
    path. The Z3 path here is the part that benefits from a uniform
    counterexample format and is forward-compatible with parametric
    schedules.
    """
    violations: list[ScheduleTVViolation] = []

    # Dependencies.
    n_deps = 0
    for succ in partition_ids:
        for pred in dependencies.get(succ, []):
            if pred not in start_times or succ not in start_times:
                continue
            n_deps += 1
            d_pred = device_assignments.get(pred, -1)
            d_succ = device_assignments.get(succ, -1)
            if d_pred < 0 or d_succ < 0 or d_pred >= num_devices or d_succ >= num_devices:
                continue
            v = _z3_check_dependency(
                pred=pred,
                succ=succ,
                end_pred=_to_int_us(end_times[pred]),
                start_succ=_to_int_us(start_times[succ]),
                transfer=_to_int_us(transfer_us[d_pred][d_succ]),
                timeout_ms=timeout_ms,
            )
            if v is not None:
                violations.append(v)

    # Per-device overlap pairs.
    by_device: dict[int, list[str]] = {d: [] for d in range(num_devices)}
    for pid in partition_ids:
        d = device_assignments.get(pid, -1)
        if 0 <= d < num_devices:
            by_device[d].append(pid)
    n_overlap_pairs = 0
    for d, pids in by_device.items():
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                n_overlap_pairs += 1
                p, q = pids[i], pids[j]
                v = _z3_check_overlap(
                    p=p,
                    q=q,
                    device=d,
                    start_p=_to_int_us(start_times[p]),
                    end_p=_to_int_us(end_times[p]),
                    start_q=_to_int_us(start_times[q]),
                    end_q=_to_int_us(end_times[q]),
                    timeout_ms=timeout_ms,
                )
                if v is not None:
                    violations.append(v)

    # Makespan equality.
    if end_times:
        max_end = max(_to_int_us(v) for v in end_times.values())
        v = _z3_check_makespan(_to_int_us(makespan_us), max_end, timeout_ms)
        if v is not None:
            violations.append(v)

    return violations, n_deps, n_overlap_pairs


def translation_validate_schedule(
    partition_ids: list[str],
    durations_us_by_device: dict[str, list[float]],
    dependencies: dict[str, list[str]],
    num_devices: int,
    start_times: dict[str, float],
    end_times: dict[str, float],
    device_assignments: dict[str, int],
    makespan_us: float,
    *,
    transfer_us: list[list[float]] | None = None,
    use_z3: bool = True,
    timeout_ms: int = 5000,
) -> ScheduleTVResult:
    """Translation-validate a scheduler solution.

    Args:
        partition_ids: Stable partition order.
        durations_us_by_device: ``pid -> [duration on device 0, ...]``
            (microseconds).
        dependencies: ``succ_id -> [pred_id, ...]``.
        num_devices: Topology size.
        start_times: ``pid -> start time (µs)`` from the scheduler.
        end_times: ``pid -> end time (µs)`` from the scheduler.
        device_assignments: ``pid -> device index`` from the scheduler.
        makespan_us: The makespan the scheduler reported.
        transfer_us: ``num_devices × num_devices`` transfer matrix in
            microseconds. Defaults to an all-zero matrix.
        use_z3: When True, run the Z3 obligation path on dependency,
            overlap, and makespan checks alongside the Python checks.
            Assignment / duration consistency always uses Python (Z3
            adds no value there).
        timeout_ms: Per-query Z3 timeout in milliseconds.

    Returns:
        A :class:`ScheduleTVResult`. ``proved=True`` iff no violation
        was raised by either path.
    """
    if transfer_us is None:
        transfer_us = [[0.0] * num_devices for _ in range(num_devices)]

    t0 = time.perf_counter()
    py_violations, n_deps, n_overlap = _python_checks(
        partition_ids=partition_ids,
        durations_us_by_device=durations_us_by_device,
        dependencies=dependencies,
        num_devices=num_devices,
        start_times=start_times,
        end_times=end_times,
        device_assignments=device_assignments,
        makespan_us=makespan_us,
        transfer_us=transfer_us,
    )
    python_time_ms = (time.perf_counter() - t0) * 1000.0

    z3_violations: list[ScheduleTVViolation] = []
    z3_time_ms = 0.0
    if use_z3:
        t1 = time.perf_counter()
        z3_violations, _z3_n_deps, _z3_n_overlap = _z3_checks(
            partition_ids=partition_ids,
            dependencies=dependencies,
            num_devices=num_devices,
            start_times=start_times,
            end_times=end_times,
            device_assignments=device_assignments,
            makespan_us=makespan_us,
            transfer_us=transfer_us,
            timeout_ms=timeout_ms,
        )
        z3_time_ms = (time.perf_counter() - t1) * 1000.0

    # Merge violations, deduplicating by (kind, sorted detail items).
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    merged: list[ScheduleTVViolation] = []
    for v in py_violations + z3_violations:
        key = (v.kind, tuple(sorted((k, str(val)) for k, val in v.detail.items())))
        if key in seen:
            continue
        seen.add(key)
        merged.append(v)

    proved = len(merged) == 0
    return ScheduleTVResult(
        proved=proved,
        violations=merged,
        z3_time_ms=z3_time_ms,
        python_time_ms=python_time_ms,
        n_deps_checked=n_deps,
        n_overlap_pairs_checked=n_overlap,
    )
