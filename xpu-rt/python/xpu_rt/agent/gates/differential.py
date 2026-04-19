"""Differential gate — wraps compgen.semantic.verify.verify_callable_against_reference.

Caller provides two callables in the context::

    ctx = {
        "ref_fn": callable,          # zero-arg reference
        "got_fn": callable,          # zero-arg candidate
        "atol": 1e-5,                # optional
        "rtol": 1e-5,                # optional
    }

Gate returns a GateResult dict with per-output NumericComparison
details on rejection.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any


def differential_gate(proposal: dict[str, Any], **ctx: Any) -> dict[str, Any]:
    ref_fn = ctx.get("ref_fn")
    got_fn = ctx.get("got_fn")
    if ref_fn is None or got_fn is None:
        return {
            "status": "deferred",
            "details": {
                "reason": "differential gate requires ctx.ref_fn + ctx.got_fn"
            },
        }

    atol = float(ctx.get("atol", 1e-5))
    rtol = float(ctx.get("rtol", 1e-5))

    try:
        import torch
        from compgen.semantic.verify.compare import compare_tensors
    except ImportError as e:   # pragma: no cover
        return {
            "status": "deferred",
            "details": {"reason": f"missing dependency: {e}"},
        }

    def _as_tensor_list(x: Any) -> list[Any]:
        if isinstance(x, torch.Tensor):
            return [x]
        if isinstance(x, (list, tuple)):
            return [v for v in x if isinstance(v, torch.Tensor)]
        return []

    t0 = time.perf_counter()
    try:
        ref_out = ref_fn()
    except Exception as e:   # noqa: BLE001
        return {
            "status": "rejected",
            "details": {
                "reason": "ref_fn raised",
                "error": f"{type(e).__name__}: {e}",
            },
        }
    ref_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    try:
        got_out = got_fn()
    except Exception as e:   # noqa: BLE001
        return {
            "status": "rejected",
            "details": {
                "reason": "got_fn raised",
                "error": f"{type(e).__name__}: {e}",
            },
        }
    got_ms = (time.perf_counter() - t0) * 1000.0

    ref_list = _as_tensor_list(ref_out)
    got_list = _as_tensor_list(got_out)
    if len(ref_list) != len(got_list):
        return {
            "status": "rejected",
            "details": {
                "reason": "tensor count mismatch",
                "ref_count": len(ref_list),
                "got_count": len(got_list),
            },
        }
    if not ref_list:
        return {
            "status": "rejected",
            "details": {"reason": "no tensor outputs to compare"},
        }

    comparisons: list[dict[str, Any]] = []
    all_passed = True
    for i, (r, g) in enumerate(zip(ref_list, got_list)):
        cmp = compare_tensors(r, g, atol=atol, rtol=rtol)
        comparisons.append({"index": i, **asdict(cmp)})
        if not cmp.passed:
            all_passed = False

    return {
        "status": "accepted" if all_passed else "rejected",
        "details": {
            "comparisons": comparisons,
            "latency_ref_ms": round(ref_ms, 3),
            "latency_got_ms": round(got_ms, 3),
            "atol": atol,
            "rtol": rtol,
        },
    }


__all__ = ["differential_gate"]
