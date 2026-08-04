"""Adapters from schedule artifacts to the freshness evaluator's trace format.

The evaluator consumes `freshness.Invocation` intervals and knows nothing about
XPU-RT. This module is the only place that understands the emitted fixture
schema, so a simulator or ModelBlaster trace can be plugged in later by adding
one function here rather than by touching the semantics.

Instance intervals
------------------
A fixture holds per-DISPATCH rows; freshness is a per-INSTANCE property. An
instance's interval is [min(start), max(start+duration)] over its dispatches.
That is the right envelope for the question being asked: the perception result
is not usable until the last dispatch of that instance has written its output,
and the control command is not emitted until the last dispatch of the
controller has run.

Release and deadline come from the workload spec, not the fixture — the
schedule says when work ran, the spec says when it was allowed to run:

    release  = start_time_phase + instance * period
    deadline = release + window_duration

Aperiodic networks get release 0 and no deadline.

Units are whatever the fixture uses. output_scheduled_json writes `start_time`
and `duration` in milliseconds, and periods/window_durations in the toplevel
JSON are milliseconds too, so the whole trace is millisecond-denominated and
`time_unit="ms"` is declared to the evaluator. (Note the metrics sidecar
mislabels the same quantity `makespan_us`; that key is not used here.)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))

from freshness import Invocation, split_instance_name  # noqa: E402


def periodic_spec(networks_data: Dict) -> Dict[str, Dict[str, Optional[float]]]:
    """Map network identifier -> {period, window_duration, start_time}.

    Values are None for an aperiodic network.
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for name, info in (networks_data.get("networks", {}) or {}).items():
        if not isinstance(info, dict):
            continue
        ident = str(info.get("identifier", name))
        period = info.get("period")
        window = info.get("window_duration")
        out[ident] = {
            "period": float(period) if period is not None else None,
            "window_duration": float(window) if window is not None else None,
            "start_time": float(info.get("start_time", 0) or 0),
        }
    return out


def invocations_from_fixture(
    fixture: Dict,
    networks_data: Dict,
) -> List[Invocation]:
    """Collapse a schedule fixture's dispatch rows into per-instance intervals."""
    dispatches = fixture.get("dispatches") or {}
    if not dispatches:
        raise ValueError("fixture has no dispatches")

    spec = periodic_spec(networks_data)
    known_tasks = sorted(spec)
    if not known_tasks:
        raise ValueError("workload spec declares no networks")

    # job_name -> (min start, max end)
    spans: Dict[str, Tuple[float, float]] = {}
    for key, row in dispatches.items():
        job = row.get("job_name")
        if not job:
            raise ValueError(f"dispatch {key!r} has no job_name")
        start = float(row.get("start_time", 0.0))
        end = start + float(row.get("duration", 0.0))
        if job in spans:
            lo, hi = spans[job]
            spans[job] = (min(lo, start), max(hi, end))
        else:
            spans[job] = (start, end)

    invocations: List[Invocation] = []
    for job, (start, end) in sorted(spans.items()):
        task, instance = split_instance_name(job, known_tasks)
        s = spec[task]
        period = s["period"]
        if period is not None:
            release = float(s["start_time"] or 0.0) + instance * period
            window = s["window_duration"]
            deadline = release + window if window is not None else None
        else:
            release = 0.0
            deadline = None

        # A schedule that starts an instance before its release would violate
        # the solver's own min_start_t constraint; catching it here stops a
        # negative-age record from being reported as a freshness result.
        if start + 1e-6 < release:
            raise ValueError(
                f"{job}: scheduled start {start} precedes its release {release}; "
                f"the fixture violates the workload's release constraint"
            )

        invocations.append(
            Invocation(
                task=task,
                instance=instance,
                release_time=release,
                start_time=start,
                end_time=end,
                deadline=deadline,
            )
        )
    return invocations


def load_fixture(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def deadline_compliance(
    invocations: List[Invocation],
    task: str,
    reference_window: Optional[float] = None,
) -> Dict[str, object]:
    """Deadline statistics for one task, whichever role it plays.

    The freshness evaluator only emits records for CONSUMER invocations, so the
    producer's own deadline behaviour is otherwise invisible — and at high
    contention the producer is exactly what is late. Reporting only the
    consumer's deadline success would let "the controller is fine but its input
    is old" be confused with "everything is on time but the data is old", which
    are different failures with different fixes.

    `reference_window` guards a comparison hazard. One protection mechanism is
    to TIGHTEN the producer's own window, which moves `Invocation.deadline` —
    so the candidate's `_deadline_success_rate` is measured against a different
    bar than the baseline's and the two are not comparable. Passing the
    baseline window adds a `_vs_ref` series measured against a FIXED bar, which
    is the one to compare across candidates. The self-relative series is still
    reported, because "did it honour the deadline it was given" is the question
    the solver was actually asked.
    """
    sel = [i for i in invocations if i.task == task and i.deadline is not None]
    if not sel:
        out: Dict[str, object] = {
            f"{task}_deadline_success_rate": None,
            f"{task}_max_lateness_ms": None,
            f"{task}_n_invocations": 0,
        }
        if reference_window is not None:
            out[f"{task}_deadline_success_rate_vs_ref"] = None
            out[f"{task}_max_lateness_ms_vs_ref"] = None
            out[f"{task}_reference_window_ms"] = reference_window
        return out
    late = [i.end_time - i.deadline for i in sel]
    out = {
        f"{task}_deadline_success_rate": sum(1 for d in late if d <= 0) / len(sel),
        f"{task}_max_lateness_ms": max(late),
        f"{task}_n_invocations": len(sel),
    }
    if reference_window is not None:
        ref_late = [i.end_time - (i.release_time + reference_window) for i in sel]
        out[f"{task}_deadline_success_rate_vs_ref"] = (
            sum(1 for d in ref_late if d <= 0) / len(sel)
        )
        out[f"{task}_max_lateness_ms_vs_ref"] = max(ref_late)
        out[f"{task}_reference_window_ms"] = reference_window
    return out


def soft_utility(
    invocations: List[Invocation],
    soft_tasks: List[str],
    epoch_length: Optional[float],
) -> Dict[str, object]:
    """Count completed soft-work instances.

    An instance counts as completed only if it finished inside the epoch. Work
    still running when the epoch ends produced no usable output, so counting it
    would credit a policy for work it did not deliver — which is exactly the
    quantity the adaptive-vs-conservative comparison turns on.
    """
    per_task: Dict[str, int] = {t: 0 for t in soft_tasks}
    started: Dict[str, int] = {t: 0 for t in soft_tasks}
    for inv in invocations:
        if inv.task not in per_task:
            continue
        started[inv.task] += 1
        if epoch_length is None or inv.end_time <= epoch_length:
            per_task[inv.task] += 1
    return {
        "soft_instances_completed": sum(per_task.values()),
        "soft_instances_released": sum(started.values()),
        "soft_completed_by_task": per_task,
        "soft_released_by_task": started,
    }
