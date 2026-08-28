"""Splitting a job name into `(network, instance)`, once.

A job name is `<network><instance>` with NO separator -- `dronet3`,
`mlp_control12`, `yolov8_nano_64x960`. Recovering the two halves is trivial
only while no network name ends in a digit, and `yolov8_nano_64x96` (the
deployed detector, at 64x96) is a real one. Under a trailing-digit split it
reads as `yolov8_nano_64x` + instance 96.

WHY THIS IS A MODULE. That split had been written independently five times --
`figstyle.model_of`, `generate_xpurt_main._split_job_name`,
`plot_k1_trace_gantt.model_of`, `granularity_advisor._strip_trailing_digits`,
`trace_metrics.model_of` -- and each new copy was written after a previous one
broke on exactly this. Four of them were cosmetic (a legend said
`yolov8_nano_64x`, a colour fell back to grey). The fifth was not:

    trace_metrics.instance_index("yolov8_nano_64x960") -> 960

and the deadline for instance k is `k*T + D`, so the detector's deadline became
960 * 50 ms = 48 SECONDS. Measured on the featured schedule, it reported
`instance_deadline_misses: 0` with `response_p50 = -47954.45 ms`. The zero was
not a measurement, it was a structural impossibility -- and `trace_metrics` is
the one place in this repo allowed to say what a periodic run achieved, feeding
`candidate_objective` terms 1 (hard deadline misses) and 4 (p99 response). A
scorer that cannot report a miss silently accepts every candidate.

THE RULE. Given the set of real network names, longest match wins, and what
follows must be all digits. Without that set, fall back to stripping trailing
digits -- which is what every caller did before, and is correct for every
network whose name does not end in one.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple


def strip_trailing_digits(s: str) -> str:
    """The name-free fallback. Correct unless a network name ends in a digit."""
    i = len(s)
    while i > 0 and s[i - 1].isdigit():
        i -= 1
    return s[:i]


def split_job_name(job: str, known: Optional[Iterable[str]] = None,
                   ) -> Tuple[str, int]:
    """`('yolov8_nano_64x96', 0)` from `'yolov8_nano_64x960'`, given the names.

    Longest match first, so `yolov8_nano_64x96` is preferred over any shorter
    prefix that also matches. An exact hit means instance 0.
    """
    if job and known:
        for base in sorted(known, key=len, reverse=True):
            if job == base:
                return base, 0
            if job.startswith(base):
                rest = job[len(base):]
                if rest.isdigit():
                    return base, int(rest)
    base = strip_trailing_digits(job or "")
    if not base:
        return job or "", 0
    rest = (job or "")[len(base):]
    return base, int(rest) if rest.isdigit() else 0


def model_of(job: str, known: Optional[Iterable[str]] = None) -> str:
    return split_job_name(job, known)[0]


def instance_index(job: str, known: Optional[Iterable[str]] = None) -> int:
    return split_job_name(job, known)[1]
