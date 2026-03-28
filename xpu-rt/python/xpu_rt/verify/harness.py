"""Verification harness for comparing reference and candidate callables.

Runs both callables, measures latency, compares outputs tensor-by-tensor,
and writes a ``verification.json`` report to disk.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import structlog
import torch

from compgen.verify.compare import NumericComparison, compare_tensors

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class VerificationRun:
    """Outcome of a single verification run.

    Attributes:
        name: Human-readable label for the run.
        passed: Whether **all** output tensors matched within tolerance.
        latency_ref_ms: Wall-clock time for the reference callable (ms).
        latency_got_ms: Wall-clock time for the candidate callable (ms).
        comparisons: Per-output-tensor comparison results.
    """

    name: str
    passed: bool
    latency_ref_ms: float
    latency_got_ms: float
    comparisons: tuple[NumericComparison, ...]


def _time_callable(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Call *fn* and return ``(result, elapsed_ms)``."""
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0


def _to_tensor_list(value: Any) -> list[torch.Tensor]:
    """Normalise an output value into a flat list of tensors."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        tensors: list[torch.Tensor] = []
        for item in value:
            if isinstance(item, torch.Tensor):
                tensors.append(item)
        if not tensors:
            raise TypeError(f"Sequence contains no tensors: {type(value)!r}")
        return tensors
    raise TypeError(f"Unsupported output type: {type(value)!r}")


def verify_callable_against_reference(
    *,
    name: str,
    ref_fn: Callable[[], Any],
    got_fn: Callable[[], Any],
    out_dir: str | Path,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> VerificationRun:
    """Run *ref_fn* and *got_fn*, compare outputs, and persist a report.

    Both callables are invoked with **no arguments** -- they must capture
    their inputs via closure or ``functools.partial``.

    If *got_fn* raises an exception, the run is marked as failed and the
    exception message is recorded.

    Args:
        name: Label for the verification run.
        ref_fn: Reference callable (e.g. eager model forward).
        got_fn: Candidate callable (e.g. transformed/compiled forward).
        out_dir: Directory where ``verification.json`` will be written.
        atol: Absolute tolerance forwarded to :func:`compare_tensors`.
        rtol: Relative tolerance forwarded to :func:`compare_tensors`.

    Returns:
        A :class:`VerificationRun` summarising the outcome.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_out, ref_ms = _time_callable(ref_fn)

    try:
        got_out, got_ms = _time_callable(got_fn)
    except Exception as exc:
        logger.warning("candidate callable raised", name=name, error=str(exc))
        failed_cmp = NumericComparison(
            passed=False,
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            atol=atol,
            rtol=rtol,
            num_mismatched=-1,
        )
        result = VerificationRun(
            name=name,
            passed=False,
            latency_ref_ms=ref_ms,
            latency_got_ms=0.0,
            comparisons=(failed_cmp,),
        )
        _write_report(result, out_dir)
        return result

    ref_tensors = _to_tensor_list(ref_out)
    got_tensors = _to_tensor_list(got_out)

    cmps: list[NumericComparison] = []
    for i, (r, g) in enumerate(zip(ref_tensors, got_tensors)):
        cmp = compare_tensors(r, g, atol=atol, rtol=rtol)
        cmps.append(cmp)
        if not cmp.passed:
            logger.info("mismatch", name=name, output_index=i, max_abs=cmp.max_abs_error)

    all_passed = all(c.passed for c in cmps)

    result = VerificationRun(
        name=name,
        passed=all_passed,
        latency_ref_ms=ref_ms,
        latency_got_ms=got_ms,
        comparisons=tuple(cmps),
    )
    _write_report(result, out_dir)
    return result


def _write_report(result: VerificationRun, out_dir: Path) -> None:
    """Persist a verification report as JSON."""
    payload = {
        "name": result.name,
        "passed": result.passed,
        "latency_ref_ms": result.latency_ref_ms,
        "latency_got_ms": result.latency_got_ms,
        "comparisons": [asdict(c) for c in result.comparisons],
    }
    report_path = out_dir / "verification.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.debug("verification report written", path=str(report_path))


__all__ = [
    "VerificationRun",
    "verify_callable_against_reference",
]
