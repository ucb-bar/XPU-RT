"""Closed-loop MCP tools for the QNN flow.

Four tools that together let the agent (this Claude Code session)
drive the closed loop end-to-end via MCP only:

* ``xpu_rt_qnn_profile_detailed_on_board`` — real per-op profiling
  via ``qnn-net-run --profiling_level=detailed`` +
  ``qnn-profile-viewer``. Output is a list of typed ``OpTiming``
  rows tagged ``provenance="measured"``.
* ``xpu_rt_qnn_execute_schedule_on_board`` — run a MILP schedule on
  the board (parallel SSH lanes), return per-island measured
  (start, finish) and per-lane wall times.
* ``xpu_rt_qnn_compute_contention_feedback`` — given predicted +
  measured, update the multiplicative per-backend contention
  factors; return the new latency matrix to feed back into the
  scheduler.
* ``xpu_rt_qnn_propose_granularity`` — enumerate ``Island``s for a
  named granularity flavour (per-op / fused / sharded / whole-net)
  and return the legal-action set with per-cell provenance.

All four append to ``QNN_FLOW_TOOLS`` via ``qnn_flow.py``.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from xpu_rt.mcp.session import SessionManager


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _qnn_event(run_dir: Path, *, event: str, **fields: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = {"schema_version": "qnn_event_v1",
           "event": event, "timestamp_utc": _now(), **fields}
    with (run_dir / "qnn_events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- #
# Tool: detailed profiling
# --------------------------------------------------------------------------- #


def xpu_rt_qnn_profile_detailed_on_board(
    sm: SessionManager,  # noqa: ARG001
    *,
    out_dir: str,
    round_index: int,
    dlc_path: str,
    backend: str,
    input_list: str,
    iters: int = 10,
    workload_id: str | None = None,
) -> dict[str, Any]:
    """Run qnn-net-run --profiling_level=detailed; parse per-op timings."""
    from xpu_rt.targets.backends.qnn.board import load_board_config
    from xpu_rt.targets.backends.qnn.profile_detailed import profile_whole_net

    cfg = load_board_config()
    timings, stderr_tail = profile_whole_net(
        cfg, dlc_path=dlc_path, backend=backend,
        input_list=input_list, iters=iters, timeout_s=600,
    )
    run = Path(out_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    rd = run / f"round_{round_index}"
    rd.mkdir(parents=True, exist_ok=True)
    wid = workload_id or Path(dlc_path).stem
    target = rd / f"profile_detailed_{wid}_{backend}.json"
    target.write_text(json.dumps({
        "schema_version": "qnn_profile_detailed_v1",
        "workload_id": wid, "backend": backend,
        "iters": iters, "dlc_path": dlc_path,
        "stderr_tail": stderr_tail,
        "n_ops": len(timings),
        "timings": [t.to_dict() for t in timings],
    }, indent=2))
    _qnn_event(run, event="qnn_profile_detailed",
               round=round_index, workload=wid, backend=backend,
               n_ops=len(timings), ok=len(timings) > 0)
    return {
        "ok": len(timings) > 0,
        "round_index": round_index,
        "workload_id": wid,
        "backend": backend,
        "n_ops": len(timings),
        "timings_path": str(target),
        "stderr_tail": stderr_tail,
    }


# --------------------------------------------------------------------------- #
# Tool: execute schedule
# --------------------------------------------------------------------------- #


def xpu_rt_qnn_execute_schedule_on_board(
    sm: SessionManager,  # noqa: ARG001
    *,
    out_dir: str,
    round_index: int,
    schedule_path: str,
    workload_specs: dict[str, dict[str, Any]],
    timeout_s: int = 600,
) -> dict[str, Any]:
    """Run a MILP schedule on board via parallel SSH lanes.

    ``workload_specs[workload_id]`` carries per-backend
    ``executor_artifact`` info (either ``dlc_path`` or
    ``context_paths[backend]``) plus ``input_list`` and ``iters``.
    """
    from xpu_rt.targets.backends.qnn.board import load_board_config
    from xpu_rt.targets.backends.qnn.execute_schedule import execute_schedule

    run = Path(out_dir).resolve()
    rd = run / f"round_{round_index}"
    rd.mkdir(parents=True, exist_ok=True)
    try:
        sched = json.loads(Path(schedule_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"schedule unreadable: {exc}"}
    cfg = load_board_config()
    try:
        result = execute_schedule(
            cfg, sched, workload_specs=workload_specs, timeout_s=timeout_s,
        )
    except KeyError as exc:
        return {"ok": False, "error": str(exc),
                "round_index": round_index}
    target = rd / "measured.json"
    target.write_text(json.dumps(result.to_dict(), indent=2))
    _qnn_event(run, event="qnn_execute_schedule",
               round=round_index,
               predicted_makespan_us=result.schedule_makespan_us,
               measured_makespan_us=result.measured_makespan_us,
               ok=result.ok)
    # Pretty markdown for the agent to paste.
    rows: list[str] = []
    rows.append(f"### Schedule execution — round {round_index}")
    rows.append("")
    rows.append(f"- Predicted makespan: **{result.schedule_makespan_us/1000:.1f} ms**")
    rows.append(f"- Measured makespan:  **{result.measured_makespan_us/1000:.1f} ms**")
    if result.schedule_makespan_us > 0:
        slip = (result.measured_makespan_us - result.schedule_makespan_us) / result.schedule_makespan_us * 100
        rows.append(f"- Slip: {slip:+.1f}%")
    rows.append("")
    rows.append("| backend | predicted lane (ms) | measured lane (ms) | factor |")
    rows.append("|---|---:|---:|---:|")
    for b, m_us in sorted(result.lane_finish_us.items()):
        pred_lane = sum(
            float(o.get("predicted_us") or 0.0)
            for o in sched.get("ops", []) if o.get("machine") == b
        )
        f = m_us / pred_lane if pred_lane > 0 else float("nan")
        rows.append(f"| {b} | {pred_lane/1000:.1f} | {m_us/1000:.1f} | {f:.2f}× |")
    return {
        "ok": result.ok,
        "round_index": round_index,
        "schedule_makespan_us": result.schedule_makespan_us,
        "measured_makespan_us": result.measured_makespan_us,
        "lane_finish_us": dict(result.lane_finish_us),
        "measured_path": str(target),
        "pretty_markdown": "\n".join(rows),
    }


# --------------------------------------------------------------------------- #
# Tool: contention feedback
# --------------------------------------------------------------------------- #


def xpu_rt_qnn_compute_contention_feedback(
    sm: SessionManager,  # noqa: ARG001
    *,
    out_dir: str,
    round_index: int,
    schedule_path: str,
    measured_path: str,
    solo_latency_matrix: dict[str, dict[str, float | None]],
    prior_state_path: str | None = None,
) -> dict[str, Any]:
    """Update the multiplicative per-backend contention factor.

    Returns the updated latency matrix to feed back into the MOSEK
    bridge for the next round, plus a convergence flag.
    """
    from xpu_rt.targets.backends.qnn.contention import (
        ContentionState,
        per_backend_measured_from_execution,
        per_backend_predicted_from_schedule,
        write_contention_log,
    )

    run = Path(out_dir).resolve()
    try:
        sched = json.loads(Path(schedule_path).read_text())
        execution = json.loads(Path(measured_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"input unreadable: {exc}"}

    state = ContentionState()
    if prior_state_path and Path(prior_state_path).is_file():
        try:
            prior = json.loads(Path(prior_state_path).read_text())
            state = ContentionState(
                factors=dict(prior.get("factors") or {}),
                history=list(prior.get("history") or []),
                last_delta=dict(prior.get("last_delta") or {}),
                max_factor=float(prior.get("max_factor", 2.5)),
                ema_weight=float(prior.get("ema_weight", 0.5)),
            )
        except (OSError, json.JSONDecodeError, TypeError):
            state = ContentionState()

    lane_finish = execution.get("lane_finish_us")
    if isinstance(lane_finish, dict) and lane_finish:
        backends = list(lane_finish.keys())
    else:
        backends = list({op.get("machine") for op in sched.get("ops") or []
                          if op.get("machine")})
    state.ensure(backends)

    predicted = per_backend_predicted_from_schedule(sched)
    measured = per_backend_measured_from_execution(execution)
    new_factors = state.update(
        per_backend_predicted_us=predicted,
        per_backend_measured_us=measured,
    )
    converged = state.is_converged()
    write_contention_log(run, round_index=round_index, state=state)
    # Apply factors to solo latency matrix to get the next-round matrix.
    new_matrix = state.apply(solo_latency_matrix)
    state_path = run / "contention_state.json"
    state_path.write_text(json.dumps(state.to_dict(), indent=2))
    _qnn_event(run, event="qnn_contention_update",
               round=round_index, factors=new_factors,
               last_delta=state.last_delta, converged=converged)
    # Markdown summary.
    rows = ["### Contention feedback — round " + str(round_index), ""]
    rows.append("| backend | predicted lane (ms) | measured lane (ms) | factor | Δ vs prev |")
    rows.append("|---|---:|---:|---:|---:|")
    for b in sorted(new_factors):
        rows.append(
            f"| {b} | {predicted.get(b, 0.0)/1000:.1f} "
            f"| {measured.get(b, 0.0)/1000:.1f} "
            f"| {new_factors[b]:.3f}× "
            f"| {state.last_delta.get(b, 0.0)*100:.1f}% |"
        )
    rows.append("")
    rows.append(f"_Converged: {'✅ yes' if converged else '… not yet (need Δ<5% × 2 rounds)'}_")
    return {
        "ok": True,
        "round_index": round_index,
        "factors": dict(new_factors),
        "last_delta": dict(state.last_delta),
        "converged": converged,
        "new_latency_matrix": new_matrix,
        "state_path": str(state_path),
        "pretty_markdown": "\n".join(rows),
    }


# --------------------------------------------------------------------------- #
# Tool: propose granularity (enumerate candidate Islands)
# --------------------------------------------------------------------------- #


def xpu_rt_qnn_propose_granularity(
    sm: SessionManager,  # noqa: ARG001
    *,
    out_dir: str,
    round_index: int,
    workload_id: str,
    op_ids: list[str],
    kind: str = "whole_net",
    fusion_groups: list[list[str]] | None = None,
    n_shards: int = 0,
    target_op_id: str | None = None,
    backend_candidates: list[str] | None = None,
) -> dict[str, Any]:
    """Build a GranularityProposal and return its legal-action set."""
    from xpu_rt.targets.backends.qnn.granularity_proposal import (
        propose_fusions, propose_per_op, propose_shards, propose_whole_net,
    )

    backends = tuple(backend_candidates or ())
    if kind == "whole_net":
        proposal = propose_whole_net(workload_id, op_ids,
                                      backend_candidates=backends)
    elif kind == "per_op":
        proposal = propose_per_op(workload_id, op_ids,
                                   backend_candidates=backends)
    elif kind == "fused":
        proposal = propose_fusions(workload_id, op_ids,
                                    fusion_groups or [],
                                    backend_candidates=backends)
    elif kind == "sharded":
        if not target_op_id or n_shards < 2:
            return {"ok": False,
                    "error": "sharded kind needs target_op_id + n_shards>=2"}
        proposal = propose_shards(workload_id, target_op_id, n_shards,
                                   backend_candidates=backends)
    else:
        return {"ok": False, "error": f"unknown kind {kind!r}"}

    run = Path(out_dir).resolve() / f"round_{round_index}"
    run.mkdir(parents=True, exist_ok=True)
    target = run / f"granularity_{workload_id}_{kind}.json"
    target.write_text(json.dumps(proposal.to_dict(), indent=2))
    _qnn_event(Path(out_dir).resolve(),
               event="qnn_granularity_proposed",
               round=round_index, workload=workload_id,
               kind=kind, n_islands=len(proposal.islands))
    return {
        "ok": True,
        "round_index": round_index,
        "label": proposal.label,
        "n_islands": len(proposal.islands),
        "proposal_path": str(target),
        "proposal": proposal.to_dict(),
    }


# --------------------------------------------------------------------------- #
# Tool registration (appended to QNN_FLOW_TOOLS in qnn_flow.py)
# --------------------------------------------------------------------------- #


QNN_CLOSED_LOOP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "xpu_rt_qnn_profile_detailed_on_board",
        "description": (
            "Run qnn-net-run with --profiling_level=detailed on the "
            "QRB5165 board, parse qnn-profile-viewer's CSV, return "
            "per-op real measurements tagged provenance=measured. "
            "This is the real-only source for per-op cost cells."
        ),
        "phase": "job",
        "handler": xpu_rt_qnn_profile_detailed_on_board,
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string"},
                "round_index": {"type": "integer"},
                "dlc_path": {"type": "string"},
                "backend": {"type": "string"},
                "input_list": {"type": "string"},
                "iters": {"type": "integer", "default": 10},
                "workload_id": {"type": "string"},
            },
            "required": ["out_dir", "round_index", "dlc_path",
                         "backend", "input_list"],
        },
    },
    {
        "name": "xpu_rt_qnn_execute_schedule_on_board",
        "description": (
            "Execute a MOSEK MILP schedule on the board via parallel "
            "SSH lane scripts. Returns per-island measured "
            "(start_ns, end_ns) plus per-lane wall times — the "
            "ground truth we compare against the MILP's prediction."
        ),
        "phase": "job",
        "handler": xpu_rt_qnn_execute_schedule_on_board,
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string"},
                "round_index": {"type": "integer"},
                "schedule_path": {"type": "string"},
                "workload_specs": {"type": "object"},
                "timeout_s": {"type": "integer", "default": 600},
            },
            "required": ["out_dir", "round_index", "schedule_path",
                         "workload_specs"],
        },
    },
    {
        "name": "xpu_rt_qnn_compute_contention_feedback",
        "description": (
            "Update the multiplicative per-backend contention factor "
            "from a (predicted, measured) pair. Returns the new "
            "latency matrix to feed back into the next MOSEK round, "
            "plus a convergence flag (Δ<5% across last 2 rounds)."
        ),
        "phase": "transform",
        "handler": xpu_rt_qnn_compute_contention_feedback,
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string"},
                "round_index": {"type": "integer"},
                "schedule_path": {"type": "string"},
                "measured_path": {"type": "string"},
                "solo_latency_matrix": {"type": "object"},
                "prior_state_path": {"type": "string"},
            },
            "required": ["out_dir", "round_index", "schedule_path",
                         "measured_path", "solo_latency_matrix"],
        },
    },
    {
        "name": "xpu_rt_qnn_propose_granularity",
        "description": (
            "Enumerate Island candidates for a named granularity "
            "flavour (whole_net | per_op | fused | sharded) and "
            "return the legal-action set with per-cell provenance "
            "info. The agent uses this to express arbitrary "
            "partitions (per-op, conv+relu fused, matmul sharded, "
            "etc.) before sending to the MILP."
        ),
        "phase": "inspect",
        "handler": xpu_rt_qnn_propose_granularity,
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string"},
                "round_index": {"type": "integer"},
                "workload_id": {"type": "string"},
                "op_ids": {"type": "array", "items": {"type": "string"}},
                "kind": {"type": "string",
                         "enum": ["whole_net", "per_op", "fused", "sharded"]},
                "fusion_groups": {"type": "array"},
                "n_shards": {"type": "integer"},
                "target_op_id": {"type": "string"},
                "backend_candidates": {"type": "array",
                                       "items": {"type": "string"}},
            },
            "required": ["out_dir", "round_index", "workload_id", "op_ids"],
        },
    },
]
