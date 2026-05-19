"""Cost-table-backed per-dispatch profile lookup.

The agentic loop's ``--dry-run`` path needs to produce a
``profiled_manifest.json`` *without* an on-board run. It does so by
looking each dispatch up in the cached ``qrb5165_costs.json`` cost
table. The same machinery is useful even in the real flow as a
fallback for dispatches the profiler didn't manage to measure.

This module owns the lookup. The original
``scripts/profile_qnn_per_dispatch.py`` re-exports
``synthesise_profiled_manifest`` for back-compat — anything outside
``scripts/`` should import from here instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from xpu_rt.targets.backends.qnn.cost_table import CostTable

_TARGET_BACKEND = {"cpu": "CPU", "qnn_gpu": "GPU", "qnn_hta": "HTA"}


def profile_one_dispatch(
    dispatch_id: str,
    target: str,
    *,
    cost_table_path: Path,
    op_key_hint: str | None = None,
) -> dict:
    """Return a profiled_manifest cell for ``(dispatch_id, target)``."""
    backend = _TARGET_BACKEND.get(target, target.upper())
    if not cost_table_path.is_file():
        return {"mean_us": None, "source": "no_cost_table",
                "dispatch_name": dispatch_id, "target": target}
    try:
        table = CostTable.load(cost_table_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"mean_us": None, "source": f"cost_table_error: {exc}",
                "dispatch_name": dispatch_id, "target": target}

    candidates: list[str] = []
    if op_key_hint:
        candidates.append(f"{op_key_hint}::{backend}::0")
        candidates.append(f"{op_key_hint}::{backend}::1")
    for key, row in table.execute.items():
        if not isinstance(row, dict):
            continue
        if (row.get("dispatch_name") == dispatch_id
                and f"::{backend}::" in key):
            candidates.append(key)

    for key in candidates:
        row = table.execute.get(key)
        if isinstance(row, dict) and row.get("mean_us") is not None:
            return {
                "mean_us": float(row["mean_us"]),
                "min_us": row.get("min_us"),
                "max_us": row.get("max_us"),
                "iters": row.get("iters"),
                "source": "cost_table",
                "cost_table_key": key,
                "dispatch_name": dispatch_id,
                "target": target,
            }
    return {"mean_us": None, "source": "not_in_cost_table",
            "dispatch_name": dispatch_id, "target": target}


def synthesise_profiled_manifest(
    *,
    schedule_dispatches: dict[str, dict],
    targets: list[str],
    cost_table_path: Path,
    matrix_path: Path | None = None,
) -> dict:
    """Build a profiled_manifest.json shape from the cost table."""
    out: dict[str, dict[str, dict]] = {}
    for name, row in schedule_dispatches.items():
        hint = (
            row.get("op_key")
            or row.get("canonical")
            or row.get("op_kind")
        ) if isinstance(row, dict) else None
        per_target: dict[str, dict] = {}
        for t in targets:
            per_target[t] = profile_one_dispatch(
                name, t, cost_table_path=cost_table_path,
                op_key_hint=hint,
            )
        out[name] = per_target
    return {"dispatches": out,
            "source": "synthesised_from_cost_table",
            "cost_table_path": str(cost_table_path),
            "matrix_path": str(matrix_path) if matrix_path else None}
