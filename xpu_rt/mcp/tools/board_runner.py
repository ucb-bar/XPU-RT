"""MCP tools for emitting board run plans and ingesting measurements.

Three host-side tools wire the QRB5165 measurement round-trip into the
Stage-4 feedback loop:

* :func:`xpu_rt_emit_board_plan` — Reads a loop-state dict and returns the
  ``loop_plan_board_v1`` JSON the user copies to the board. The host
  never ssh's anywhere; copy/scp is the user's manual action.
* :func:`xpu_rt_ingest_board_measurement` — Reads a
  ``measurement_record_board_v1`` JSON the user copied back from the
  board, validates target_id/workload_id alignment, adapts it into per-
  ``(workload, backend)`` :class:`MeasurementRecord` updates, and calls
  :func:`xpu_rt_apply_measurement` once per backend so the calibration
  absorbs the new data.
* :func:`xpu_rt_run_board_loop_step` — Convenience: emit the plan, return
  ``status='awaiting_measurement'`` (default), or poll for a measurement
  file at ``build/loops/measurements/<workload>__<target>__<run>.json``
  and ingest it automatically when ``wait_for_measurement=True``.

The actual ``ssh root@<board> bash run_loop_plan_on_qrb5165.sh`` step is
never invoked by these tools.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from xpu_rt.mcp.session import SessionManager
from xpu_rt.mcp.tools.feedback_loop_tools import xpu_rt_apply_measurement
from xpu_rt.scheduler.qnn_real_workload import load_cost_matrix
from xpu_rt.scheduling.feedback_loop import state_from_dict

log = structlog.get_logger(__name__)

PLAN_SCHEMA_VERSION = "loop_plan_board_v1"
MEASUREMENT_SCHEMA_VERSION = "measurement_record_board_v1"

DEFAULT_MEASUREMENTS_DIR = Path("build") / "loops" / "measurements"

# Backend → DLC variant suffix. Mirrors
# scripts/board/emit_loop_plan.py::BACKEND_DLC_VARIANT and
# xpu_rt.targets.backends.qnn.on_board_runner.BACKEND_DLC_VARIANT.
_BACKEND_DLC_VARIANT: dict[str, str] = {
    "CPU": "",
    "GPU": "_fp16",
    "DSP": "_quantized",
    "HTP": "_quantized",
}

_DEFAULT_BOARD_MODEL_ROOT = "/root/models"


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _resolve_dlc_path(model_dir: str, model_name: str, backend: str) -> str:
    """Return the conventional DLC path for ``(model_name, backend)``."""

    variant = _BACKEND_DLC_VARIANT.get(backend, "")
    return f"{model_dir.rstrip('/')}/{model_name}{variant}.dlc"


def _resolve_backend(chunk: dict[str, Any]) -> str:
    """Pick the backend to run a chunk on.

    Honours the chunk's ``preferred_backend`` unless it is ``UNKNOWN``,
    in which case falls back to the cheapest entry in
    ``durations_us_by_backend`` (or DSP if no costs are present).
    """

    backend = str(chunk.get("preferred_backend", "DSP") or "DSP")
    if backend == "UNKNOWN":
        durations = chunk.get("durations_us_by_backend", {}) or {}
        ranked = [
            (b, v) for b, v in durations.items()
            if v is not None and float(v) > 0.0
        ]
        if ranked:
            backend = min(ranked, key=lambda kv: float(kv[1]))[0]
        else:
            backend = "DSP"
    return backend


def _per_op_sum_for(
    cost_matrix: dict[str, Any] | None,
    workload_id: str,
    backend: str,
) -> float:
    """Sum per-op durations for ``(workload, backend)`` from the cost matrix.

    Returns 0.0 when the cost matrix is unavailable or the backend has
    no costs for this workload; the loop tolerates a zero per-op sum
    (it shows up as overhead = measured).
    """

    if cost_matrix is None:
        return 0.0
    total = 0.0
    for _op, lanes in cost_matrix.get(workload_id, {}).items():
        if isinstance(lanes, dict) and backend in lanes:
            value = lanes[backend]
            if value is None:
                continue
            try:
                total += float(value)
            except (TypeError, ValueError):
                continue
    return total


# --------------------------------------------------------------------------- #
# Public MCP tool handlers
# --------------------------------------------------------------------------- #


def xpu_rt_emit_board_plan(
    sm: SessionManager | None = None,  # noqa: ARG001
    *,
    loop_state_dict: dict[str, Any],
    model_dir: str | None = None,
    model_name: str | None = None,
    input_list: str | None = None,
    iters: int = 10,
    dlc_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Emit a ``loop_plan_board_v1`` plan from a loop-state dict.

    Args:
        sm: MCP session manager (unused).
        loop_state_dict: JSON-typed loop state, as produced by
            :func:`xpu_rt.scheduling.feedback_loop.state_to_dict`.
        model_dir: Remote directory holding the DLCs on the board.
            Defaults to ``/root/models/<workload_id>``.
        model_name: Base model name (without variant suffix). Defaults
            to ``workload_id``.
        input_list: Default ``input_list.txt`` path on the board.
            Optional; partitions may override.
        iters: Iterations per partition (default 10).
        dlc_overrides: ``chunk_id → dlc_path`` overrides for special-case
            partitions.

    Returns:
        A JSON-typed plan dict the caller writes to disk and copies to
        the board (e.g. ``scp plan.json qrb5165:/data/local/tmp/``).

    Raises:
        ValueError: ``loop_state_dict`` is missing ``workload_id`` or
            has an empty ``current_chunks`` list.
    """

    workload_id = str(loop_state_dict.get("workload_id", "") or "")
    target_id = str(loop_state_dict.get("target_id", "qrb5165") or "qrb5165")
    if not workload_id:
        raise ValueError("loop_state_dict is missing workload_id")

    chunks = loop_state_dict.get("current_chunks", []) or []
    if not chunks:
        raise ValueError(
            "loop_state_dict.current_chunks is empty — has the loop been "
            "stepped at least once?"
        )

    effective_model_name = model_name or workload_id
    effective_model_dir = model_dir or f"{_DEFAULT_BOARD_MODEL_ROOT}/{workload_id}"
    overrides = dlc_overrides or {}

    partitions: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        backend = _resolve_backend(chunk)
        dlc_path = overrides.get(chunk_id) or _resolve_dlc_path(
            effective_model_dir, effective_model_name, backend
        )
        partitions.append(
            {
                "partition_id": chunk_id,
                "backend": backend,
                "dlc_path": dlc_path,
                "iters": int(iters),
                "n_ops": len(chunk.get("op_ids", []) or []),
                "input_list": input_list or "",
            }
        )

    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "workload_id": workload_id,
        "target_id": target_id,
        "iters": int(iters),
        "loop_iteration": int(loop_state_dict.get("iteration", 0)),
        "loop_status": str(loop_state_dict.get("status", "init")),
        "loop_predicted_makespan_us": loop_state_dict.get(
            "current_predicted_makespan_us"
        ),
        "partitions": partitions,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    log.info(
        "board_plan_emitted",
        workload_id=workload_id,
        target_id=target_id,
        n_partitions=len(partitions),
        iters=iters,
    )
    return plan


def xpu_rt_ingest_board_measurement(
    sm: SessionManager | None = None,
    *,
    measurement_json_path: str,
    loop_state_dict: dict[str, Any],
    cost_matrix_path: str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Read a board ``measurement_record_board_v1`` and apply it.

    Builds one :class:`MeasurementRecord` per ``(workload, backend)``
    pair in ``per_backend_mean_us`` and routes them sequentially through
    :func:`xpu_rt_apply_measurement`. The calibration's EMA absorbs each
    backend's mean once; the chunks / solver choice are untouched (this
    is the same contract as ``xpu_rt_apply_measurement`` proper).

    Args:
        sm: MCP session manager (forwarded to
            :func:`xpu_rt_apply_measurement`).
        measurement_json_path: Path to the JSON file copied back from
            the board.
        loop_state_dict: The loop state to update.
        cost_matrix_path: Optional cost-matrix path used to recompute
            per-op sums per backend. Without it the per-op sum defaults
            to 0 — the calibration treats the entire measured wall time
            as overhead, which is the safe (conservative) fallback.
        persist: Forwarded to :func:`xpu_rt_apply_measurement`.

    Returns:
        ``{"ok", "state", "n_applied", "per_backend_mean_us",
        "workload_id", "target_id"}``.

    Raises:
        FileNotFoundError: ``measurement_json_path`` does not exist.
        ValueError: schema_version / target_id / workload_id mismatch.
    """

    path = Path(measurement_json_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"measurement json not found: {measurement_json_path}"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))

    schema = str(payload.get("schema_version", ""))
    if schema != MEASUREMENT_SCHEMA_VERSION:
        raise ValueError(
            f"unexpected measurement schema_version={schema!r}; "
            f"expected {MEASUREMENT_SCHEMA_VERSION!r}"
        )

    expected_target = str(loop_state_dict.get("target_id", "") or "")
    expected_workload = str(loop_state_dict.get("workload_id", "") or "")
    measured_target = str(payload.get("target_id", "") or "")
    measured_workload = str(payload.get("workload_id", "") or "")

    if expected_target and measured_target and measured_target != expected_target:
        raise ValueError(
            f"target_id mismatch: loop state has {expected_target!r} "
            f"but measurement has {measured_target!r}"
        )
    if (
        expected_workload
        and measured_workload
        and measured_workload != expected_workload
    ):
        raise ValueError(
            f"workload_id mismatch: loop state has {expected_workload!r} "
            f"but measurement has {measured_workload!r}"
        )

    per_backend = payload.get("per_backend_mean_us") or {}
    if not isinstance(per_backend, dict) or not per_backend:
        raise ValueError(
            "measurement is missing a non-empty per_backend_mean_us mapping"
        )

    cost_matrix: dict[str, Any] | None = None
    if cost_matrix_path:
        cost_matrix = load_cost_matrix(cost_matrix_path)

    workload_id = measured_workload or expected_workload
    predicted = (
        loop_state_dict.get("current_predicted_makespan_us") or 0.0
    )

    current_state = loop_state_dict
    n_applied = 0
    for backend, mean_us in per_backend.items():
        try:
            mean_us_f = float(mean_us)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"non-numeric mean_us for backend {backend!r}: {mean_us!r}"
            ) from exc
        if mean_us_f <= 0.0:
            log.warning(
                "board_measurement_skipped",
                workload_id=workload_id,
                backend=backend,
                reason="non_positive_mean",
            )
            continue
        measurement_dict = {
            "workload_id": workload_id,
            "backend": str(backend),
            "measured_us": mean_us_f,
            "per_op_sum_us": _per_op_sum_for(
                cost_matrix, workload_id, str(backend)
            ),
            "predicted_us": float(predicted),
        }
        result = xpu_rt_apply_measurement(
            sm,
            loop_state_dict=current_state,
            measurement_dict=measurement_dict,
            persist=persist,
        )
        if not result.get("ok"):
            raise ValueError(
                f"xpu_rt_apply_measurement failed for backend={backend!r}: "
                f"{result!r}"
            )
        current_state = result["state"]
        n_applied += 1

    log.info(
        "board_measurement_ingested",
        workload_id=workload_id,
        target_id=measured_target or expected_target,
        n_applied=n_applied,
        path=str(path),
    )
    return {
        "ok": True,
        "state": current_state,
        "n_applied": n_applied,
        "per_backend_mean_us": {str(k): float(v) for k, v in per_backend.items()},
        "workload_id": workload_id,
        "target_id": measured_target or expected_target,
    }


def xpu_rt_run_board_loop_step(
    sm: SessionManager | None = None,
    *,
    loop_state_dict: dict[str, Any],
    cost_matrix_path: str,
    model_dir: str | None = None,
    model_name: str | None = None,
    input_list: str | None = None,
    iters: int = 10,
    wait_for_measurement: bool = False,
    measurement_path: str | None = None,
    measurement_poll_interval_s: float = 30.0,
    measurement_max_wait_s: float = 600.0,
) -> dict[str, Any]:
    """Emit a board plan; optionally wait for the user's measurement.

    Two modes:

    * ``wait_for_measurement=False`` (default) — Returns
      ``status='awaiting_measurement'`` with the embedded plan. The user
      runs the board script, then calls
      :func:`xpu_rt_ingest_board_measurement` themselves.
    * ``wait_for_measurement=True`` — Polls ``measurement_path`` (or a
      conventional drop location) every
      ``measurement_poll_interval_s`` seconds until the file appears or
      ``measurement_max_wait_s`` elapses. On success, ingests the
      measurement and returns ``status='measurement_applied'``. On
      timeout, returns ``status='measurement_timeout'``.

    Args:
        sm: MCP session manager (forwarded to ingest).
        loop_state_dict: Current loop state.
        cost_matrix_path: Cost matrix path (forwarded to ingest for
            per-op sums).
        model_dir / model_name / input_list / iters: Forwarded to
            :func:`xpu_rt_emit_board_plan`.
        wait_for_measurement: Polling mode flag.
        measurement_path: Override for the polled file location. When
            unset and ``wait_for_measurement=True``, defaults to
            ``build/loops/measurements/<workload>__<target>__iter<N>.json``.
        measurement_poll_interval_s: Polling interval in seconds.
        measurement_max_wait_s: Total wait budget before timeout.

    Returns:
        ``{"ok", "status", "plan", "state", "measurement_path",
        "waited_s"}``. ``state`` is unchanged when no measurement was
        applied. ``status`` is ``awaiting_measurement`` /
        ``measurement_applied`` / ``measurement_timeout``.
    """

    plan = xpu_rt_emit_board_plan(
        sm,
        loop_state_dict=loop_state_dict,
        model_dir=model_dir,
        model_name=model_name,
        input_list=input_list,
        iters=iters,
    )

    workload_id = plan["workload_id"]
    target_id = plan["target_id"]
    iteration = int(loop_state_dict.get("iteration", 0))

    resolved_measurement_path = (
        Path(measurement_path)
        if measurement_path
        else DEFAULT_MEASUREMENTS_DIR
        / f"{workload_id}__{target_id}__iter{iteration:03d}.json"
    )

    if not wait_for_measurement:
        return {
            "ok": True,
            "status": "awaiting_measurement",
            "plan": plan,
            "state": loop_state_dict,
            "measurement_path": str(resolved_measurement_path),
            "waited_s": 0.0,
        }

    if measurement_poll_interval_s <= 0:
        raise ValueError(
            "measurement_poll_interval_s must be positive when "
            "wait_for_measurement=True"
        )
    if measurement_max_wait_s < 0:
        raise ValueError("measurement_max_wait_s must be non-negative")

    started = time.monotonic()
    while True:
        if resolved_measurement_path.is_file():
            ingested = xpu_rt_ingest_board_measurement(
                sm,
                measurement_json_path=str(resolved_measurement_path),
                loop_state_dict=loop_state_dict,
                cost_matrix_path=cost_matrix_path,
            )
            return {
                "ok": True,
                "status": "measurement_applied",
                "plan": plan,
                "state": ingested["state"],
                "measurement_path": str(resolved_measurement_path),
                "waited_s": time.monotonic() - started,
                "n_applied": ingested["n_applied"],
            }
        elapsed = time.monotonic() - started
        if elapsed >= measurement_max_wait_s:
            log.warning(
                "board_measurement_timeout",
                workload_id=workload_id,
                target_id=target_id,
                waited_s=elapsed,
                measurement_path=str(resolved_measurement_path),
            )
            return {
                "ok": False,
                "status": "measurement_timeout",
                "plan": plan,
                "state": loop_state_dict,
                "measurement_path": str(resolved_measurement_path),
                "waited_s": elapsed,
            }
        time.sleep(measurement_poll_interval_s)


# --------------------------------------------------------------------------- #
# Validation helper — keeps the round-trip honest in unit tests.
# --------------------------------------------------------------------------- #


def _validate_loop_state(loop_state_dict: dict[str, Any]) -> None:
    """Light validation of the loop-state shape.

    Currently only used as a smoke-check entry point for tests; the real
    parsing happens in :func:`state_from_dict` when callers need a
    :class:`LoopState`.
    """

    state_from_dict(loop_state_dict)


# --------------------------------------------------------------------------- #
# Registration — joined into ALL_TOOLS via xpu_rt.mcp.tools.__init__
# --------------------------------------------------------------------------- #


BOARD_RUNNER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "xpu_rt_emit_board_plan",
        "description": (
            "Emit a board-runnable plan.json (loop_plan_board_v1) from a "
            "loop-state dict. The caller writes the result to disk and "
            "copies it to the QRB5165; this tool never ssh's anywhere."
        ),
        "phase": "transform",
        "handler": xpu_rt_emit_board_plan,
        "input_schema": {
            "type": "object",
            "properties": {
                "loop_state_dict": {"type": "object"},
                "model_dir": {"type": ["string", "null"]},
                "model_name": {"type": ["string", "null"]},
                "input_list": {"type": ["string", "null"]},
                "iters": {"type": "integer"},
                "dlc_overrides": {"type": ["object", "null"]},
            },
            "required": ["loop_state_dict"],
        },
    },
    {
        "name": "xpu_rt_ingest_board_measurement",
        "description": (
            "Read a measurement_record_board_v1 JSON copied back from the "
            "board, validate target_id/workload_id alignment, and apply "
            "one MeasurementRecord per (workload, backend) to the loop "
            "state's calibration."
        ),
        "phase": "transform",
        "handler": xpu_rt_ingest_board_measurement,
        "input_schema": {
            "type": "object",
            "properties": {
                "measurement_json_path": {"type": "string"},
                "loop_state_dict": {"type": "object"},
                "cost_matrix_path": {"type": ["string", "null"]},
                "persist": {"type": "boolean"},
            },
            "required": ["measurement_json_path", "loop_state_dict"],
        },
    },
    {
        "name": "xpu_rt_run_board_loop_step",
        "description": (
            "Convenience wrapper: emit the board plan and either return "
            "status=awaiting_measurement (default) or poll for a "
            "measurement JSON at build/loops/measurements/... and ingest "
            "it once it appears."
        ),
        "phase": "transform",
        "handler": xpu_rt_run_board_loop_step,
        "input_schema": {
            "type": "object",
            "properties": {
                "loop_state_dict": {"type": "object"},
                "cost_matrix_path": {"type": "string"},
                "model_dir": {"type": ["string", "null"]},
                "model_name": {"type": ["string", "null"]},
                "input_list": {"type": ["string", "null"]},
                "iters": {"type": "integer"},
                "wait_for_measurement": {"type": "boolean"},
                "measurement_path": {"type": ["string", "null"]},
                "measurement_poll_interval_s": {"type": "number"},
                "measurement_max_wait_s": {"type": "number"},
            },
            "required": ["loop_state_dict", "cost_matrix_path"],
        },
    },
]


__all__ = [
    "BOARD_RUNNER_TOOLS",
    "MEASUREMENT_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "xpu_rt_emit_board_plan",
    "xpu_rt_ingest_board_measurement",
    "xpu_rt_run_board_loop_step",
]
