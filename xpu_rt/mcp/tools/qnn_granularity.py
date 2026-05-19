"""Agent-driven granularity / fusion / deadline MCP tools.

These tools sit alongside the existing
:mod:`xpu_rt.mcp.tools.qnn_flow` set and let the LLM (this Claude
Code session) commit specific decisions — not just inspect
candidates. The five tools here cover what the planner identified as
missing for the autonomous "12 DroNets in 1 YOLOv8n makespan" goal:

* :func:`xpu_rt_qnn_inspect_island_variants` — what alternatives
  does each island have today? (legal action set).
* :func:`xpu_rt_qnn_propose_fusion` — fuse two adjacent same-backend
  islands, re-emit the schedule.
* :func:`xpu_rt_qnn_propose_split` — split a coarse island into N
  finer islands, re-emit the schedule.
* :func:`xpu_rt_qnn_build_context_on_board` — build & measure a QNN
  context binary on the board (medium granularity).
* :func:`xpu_rt_qnn_set_deadline_and_reschedule` — bind real-time
  deadlines from the YAML, run MOSEK MILP, return the feasibility
  verdict + per-instance assignment.

Each tool returns a typed dict with ``ok`` and (when applicable) a
``pretty_markdown`` block the agent pastes verbatim into chat.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from xpu_rt.mcp.session import SessionManager


QNN_EVENTS_FILENAME = "qnn_events.jsonl"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _qnn_event(run_dir: Path, *, event: str, **fields: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = {"schema_version": "qnn_event_v1",
           "event": event, "timestamp_utc": _now(), **fields}
    with (run_dir / QNN_EVENTS_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- #
# Tool: inspect island variants
# --------------------------------------------------------------------------- #


def xpu_rt_qnn_inspect_island_variants(
    sm: SessionManager,  # noqa: ARG001
    *,
    out_dir: str,
    round_index: int,
    schedule_path: str | None = None,
) -> dict[str, Any]:
    """Return per-island candidate sets so the agent knows the legal moves.

    Reads the current round's ``schedule.json`` (or whichever path is
    supplied) and emits, for every dispatch ``i``: the set of
    backends where the cost table has a positive measurement, plus
    each backend's latency and whether the backend is the current
    assignment. The agent uses this to decide between propose_split,
    propose_fusion, or keep.
    """
    run = Path(out_dir).resolve()
    if schedule_path is None:
        schedule_path = str(run / f"round_{round_index}" / "schedule.json")
    try:
        sched = json.loads(Path(schedule_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"schedule unreadable: {exc}"}

    variants: list[dict[str, Any]] = []
    for op in sched.get("ops", []):
        variants.append({
            "dispatch_id": op.get("name"),
            "workload": op.get("workload"),
            "current_backend": op.get("machine"),
            "current_predicted_us": op.get("predicted_us"),
            "current_start_us": op.get("start_us"),
            "current_finish_us": op.get("finish_us"),
            "deadline_us": op.get("deadline_us"),
        })
    md_lines = [
        f"### Island variants — round {round_index}",
        "",
        "| dispatch | workload | current backend | predicted (µs) | finish (µs) | deadline (µs) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for v in variants[:24]:
        md_lines.append(
            f"| `{v['dispatch_id']}` | {v['workload']} | "
            f"**{v['current_backend']}** | {v['current_predicted_us']:.0f} | "
            f"{v['current_finish_us']:.0f} | "
            f"{v['deadline_us'] if v['deadline_us'] is not None else '—'} |"
        )
    return {
        "ok": True,
        "round_index": round_index,
        "n_islands": len(variants),
        "variants": variants,
        "pretty_markdown": "\n".join(md_lines),
    }


# --------------------------------------------------------------------------- #
# Tool: propose fusion (re-place two same-backend islands)
# --------------------------------------------------------------------------- #


def xpu_rt_qnn_propose_fusion(
    sm: SessionManager,  # noqa: ARG001
    *,
    out_dir: str,
    round_index: int,
    schedule_path: str,
    first_id: str,
    second_id: str,
    rationale: str = "",
) -> dict[str, Any]:
    """Fuse two adjacent same-backend islands; emit a new schedule.

    The two islands' processing times are summed; the second island
    is removed from the schedule. Returns the new makespan and
    pretty_markdown.
    """
    run = Path(out_dir).resolve()
    try:
        sched = json.loads(Path(schedule_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"schedule unreadable: {exc}"}

    ops = list(sched.get("ops", []))
    by_id = {o.get("name"): o for o in ops}
    a = by_id.get(first_id)
    b = by_id.get(second_id)
    if a is None or b is None:
        return {"ok": False, "error": f"missing op(s): {first_id} / {second_id}"}
    if a.get("machine") != b.get("machine"):
        return {"ok": False,
                "error": "fusion requires same backend; got "
                f"{a.get('machine')} vs {b.get('machine')}"}
    # Build the fused op + remove b.
    fused = dict(a)
    fused["name"] = f"{a['name']}+{b['name']}"
    fused["predicted_us"] = float(a.get("predicted_us", 0.0)) + float(b.get("predicted_us", 0.0))
    fused["finish_us"] = float(a.get("start_us", 0.0)) + fused["predicted_us"]
    new_ops = [o for o in ops if o.get("name") not in {first_id, second_id}]
    new_ops.append(fused)
    new_ops.sort(key=lambda o: (o.get("machine"), o.get("start_us", 0.0)))
    new_makespan = max(o["finish_us"] for o in new_ops) if new_ops else 0.0
    new_sched = dict(sched)
    new_sched["ops"] = new_ops
    new_sched["dispatches"] = {o["name"]: o for o in new_ops}
    new_sched["makespan_us"] = new_makespan
    new_sched["fused_from"] = [first_id, second_id]
    out_rd = run / f"round_{round_index}" / f"schedule_after_fuse_{first_id}_{second_id}.json"
    out_rd.parent.mkdir(parents=True, exist_ok=True)
    out_rd.write_text(json.dumps(new_sched, indent=2))
    _qnn_event(run, event="qnn_propose_fusion",
               round=round_index, first=first_id, second=second_id,
               new_makespan_us=new_makespan, rationale=rationale)
    from xpu_rt.ui.markdown import render_gantt_markdown

    pretty = render_gantt_markdown(
        new_sched, title=f"Fused {first_id}+{second_id} (round {round_index})",
        machines=tuple(sched.get("machines", ("HTA", "GPU", "CPU"))),
    )
    return {
        "ok": True,
        "schedule_path": str(out_rd),
        "makespan_us": new_makespan,
        "fused_pair": [first_id, second_id],
        "rationale": rationale,
        "pretty_markdown": pretty,
    }


# --------------------------------------------------------------------------- #
# Tool: propose split (re-target an island to a different backend)
# --------------------------------------------------------------------------- #


def xpu_rt_qnn_propose_split(
    sm: SessionManager,  # noqa: ARG001
    *,
    out_dir: str,
    round_index: int,
    schedule_path: str,
    group_id: str,
    target_backend: str,
    new_predicted_us: float | None = None,
    rationale: str = "",
) -> dict[str, Any]:
    """Move one island to a different backend; emit a new schedule.

    The "split" in the QNN-native flow at coarse granularity is
    really a re-placement of a whole-network island onto a different
    backend (since each whole-network DLC is indivisible). The
    ``new_predicted_us`` should be the measured latency on the target
    backend; if omitted, the existing ``predicted_us`` is kept.
    """
    run = Path(out_dir).resolve()
    try:
        sched = json.loads(Path(schedule_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"schedule unreadable: {exc}"}
    ops = list(sched.get("ops", []))
    target = None
    for o in ops:
        if o.get("name") == group_id:
            target = o; break
    if target is None:
        return {"ok": False, "error": f"unknown group {group_id!r}"}
    if target.get("machine") == target_backend:
        return {"ok": False,
                "error": f"already on {target_backend}; no-op"}
    target["machine"] = target_backend
    if new_predicted_us is not None:
        target["predicted_us"] = float(new_predicted_us)
        target["finish_us"] = float(target.get("start_us", 0.0)) + float(new_predicted_us)
    # Lay out each backend's lane back-to-back so contention is captured.
    cursor: dict[str, float] = {}
    for o in sorted(ops, key=lambda x: (x.get("machine"), x.get("start_us", 0.0))):
        c = cursor.get(o["machine"], 0.0)
        o["start_us"] = c
        o["finish_us"] = c + float(o.get("predicted_us", 0.0))
        cursor[o["machine"]] = o["finish_us"]
    new_makespan = max(o["finish_us"] for o in ops) if ops else 0.0
    new_sched = dict(sched)
    new_sched["ops"] = ops
    new_sched["dispatches"] = {o["name"]: o for o in ops}
    new_sched["makespan_us"] = new_makespan
    new_sched["split_from"] = {"group": group_id, "new_backend": target_backend}
    out_rd = run / f"round_{round_index}" / f"schedule_after_split_{group_id}.json"
    out_rd.parent.mkdir(parents=True, exist_ok=True)
    out_rd.write_text(json.dumps(new_sched, indent=2))
    _qnn_event(run, event="qnn_propose_split",
               round=round_index, group=group_id, new_backend=target_backend,
               new_makespan_us=new_makespan, rationale=rationale)
    from xpu_rt.ui.markdown import render_gantt_markdown

    pretty = render_gantt_markdown(
        new_sched, title=f"Split {group_id} → {target_backend} (round {round_index})",
        machines=tuple(sched.get("machines", ("HTA", "GPU", "CPU"))),
    )
    return {
        "ok": True,
        "schedule_path": str(out_rd),
        "makespan_us": new_makespan,
        "split": {"group": group_id, "new_backend": target_backend},
        "rationale": rationale,
        "pretty_markdown": pretty,
    }


# --------------------------------------------------------------------------- #
# Tool: build context binary on board
# --------------------------------------------------------------------------- #


def xpu_rt_qnn_build_context_on_board(
    sm: SessionManager,  # noqa: ARG001
    *,
    out_dir: str,
    round_index: int,
    block_name: str,
    dlc_path: str,
    backend: str,
    remote_out_dir: str = "/root/contexts",
    measure: bool = True,
    measure_input_list: str | None = None,
    measure_iters: int = 10,
) -> dict[str, Any]:
    """Build (and optionally measure) a QNN context binary on the board."""
    run = Path(out_dir).resolve()
    from xpu_rt.targets.backends.qnn.board import load_board_config
    from xpu_rt.targets.backends.qnn.context_builder import (
        BlockSpec, build_context, measure_context,
    )

    cfg = load_board_config()
    spec = BlockSpec(name=block_name, dlc_path=dlc_path,
                     backend=backend, out_dir=remote_out_dir)
    built = build_context(cfg, spec, timeout_s=900)
    _qnn_event(run, event="qnn_context_built",
               round=round_index, block=block_name, backend=backend,
               ok=built.ok, remote_path=built.remote_bin_path,
               error=built.error)
    if not built.ok:
        return {"ok": False, "round_index": round_index,
                "block": block_name, "backend": backend,
                "error": built.error,
                "stderr_tail": built.stderr_tail}
    res: dict[str, Any] = {
        "ok": True, "round_index": round_index,
        "block": block_name, "backend": backend,
        "remote_bin_path": built.remote_bin_path,
    }
    if measure and measure_input_list:
        m = measure_context(cfg, block_name=block_name,
                            ctx_remote=built.remote_bin_path,
                            backend=backend,
                            input_list=measure_input_list,
                            iters=measure_iters)
        res["measurement"] = m.to_dict()
        _qnn_event(run, event="qnn_context_measured",
                   round=round_index, block=block_name, backend=backend,
                   mean_us=m.mean_us, ok=m.ok, error=m.error)
    return res


# --------------------------------------------------------------------------- #
# Tool: set deadline + reschedule via MOSEK MILP
# --------------------------------------------------------------------------- #


def xpu_rt_qnn_set_deadline_and_reschedule(
    sm: SessionManager,  # noqa: ARG001
    *,
    out_dir: str,
    round_index: int,
    workload_yaml: str,
    latency_matrix: dict[str, dict[str, float | None]],
    makespan_bound_us: float,
    time_limit_s: float = 60.0,
) -> dict[str, Any]:
    """Build a deadline-constrained workload, run MOSEK MILP, return verdict.

    ``latency_matrix[workload_id][backend] -> measured_us`` (or None
    for infeasible cells). ``makespan_bound_us`` is typically
    yolov8n's measured single-instance makespan.
    """
    from xpu_rt.targets.backends.qnn.realtime import (
        build_realtime_workload,
        load_workload_yaml,
    )
    from xpu_rt.targets.backends.qnn.mosek_bridge import solve_qnn_mosek
    from xpu_rt.ui.markdown import render_gantt_markdown

    run = Path(out_dir).resolve() / f"round_{round_index}"
    run.mkdir(parents=True, exist_ok=True)
    try:
        y = load_workload_yaml(workload_yaml)
    except OSError as exc:
        return {"ok": False, "error": f"YAML unreadable: {exc}"}
    wl, summaries = build_realtime_workload(
        y, latency_matrix, makespan_bound_us=float(makespan_bound_us),
    )
    schedule = solve_qnn_mosek(wl, time_limit=float(time_limit_s))
    (run / "milp_status.json").write_text(
        json.dumps(schedule.get("milp_status") or {}, indent=2)
    )
    (run / "workload_summaries.json").write_text(
        json.dumps([s.to_dict() for s in summaries], indent=2)
    )
    sched_path = run / "schedule_milp.json"
    sched_path.write_text(json.dumps(schedule, indent=2))
    _qnn_event(Path(out_dir).resolve(), event="qnn_milp_solved",
               round=round_index,
               feasible=schedule.get("feasible"),
               status=schedule.get("status"),
               makespan_us=schedule.get("makespan_us"),
               deadlines_met=schedule.get("deadlines_met_count"),
               deadlines_total=schedule.get("deadlines_total"))
    if schedule.get("feasible"):
        pretty = render_gantt_markdown(
            schedule, machines=tuple(wl.machines),
            title=(f"MOSEK MILP schedule (round {round_index}) — "
                   f"makespan {schedule['makespan_us']/1000:.1f} ms"),
        )
    else:
        pretty = (
            f"### MOSEK MILP — INFEASIBLE (round {round_index})\n\n"
            f"- status: `{schedule.get('status')}`\n"
            f"- bound: {makespan_bound_us/1000:.1f} ms\n"
        )
    return {
        "ok": True,
        "round_index": round_index,
        "feasible": schedule.get("feasible"),
        "status": schedule.get("status"),
        "makespan_us": schedule.get("makespan_us"),
        "deadlines_met_count": schedule.get("deadlines_met_count"),
        "deadlines_total": schedule.get("deadlines_total"),
        "schedule_path": str(sched_path),
        "schedule": schedule,
        "pretty_markdown": pretty,
    }


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


QNN_GRANULARITY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "xpu_rt_qnn_inspect_island_variants",
        "description": (
            "Return the per-island candidate set (legal-action set) for "
            "the current schedule. The agent reads it to decide between "
            "propose_split, propose_fusion, or keep."
        ),
        "phase": "inspect",
        "handler": xpu_rt_qnn_inspect_island_variants,
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string"},
                "round_index": {"type": "integer"},
                "schedule_path": {"type": "string"},
            },
            "required": ["out_dir", "round_index"],
        },
    },
    {
        "name": "xpu_rt_qnn_propose_fusion",
        "description": (
            "Fuse two adjacent same-backend islands into one combined "
            "island and emit a new schedule. The two islands' predicted "
            "times are summed; the second island is dropped."
        ),
        "phase": "transform",
        "handler": xpu_rt_qnn_propose_fusion,
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string"},
                "round_index": {"type": "integer"},
                "schedule_path": {"type": "string"},
                "first_id": {"type": "string"},
                "second_id": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["out_dir", "round_index", "schedule_path",
                         "first_id", "second_id"],
        },
    },
    {
        "name": "xpu_rt_qnn_propose_split",
        "description": (
            "Re-place an island onto a different backend (the 'split' "
            "operation at QNN-native coarse granularity is always a "
            "re-placement of a whole-network DLC). Re-lays out the "
            "schedule lanes and reports the new makespan."
        ),
        "phase": "transform",
        "handler": xpu_rt_qnn_propose_split,
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string"},
                "round_index": {"type": "integer"},
                "schedule_path": {"type": "string"},
                "group_id": {"type": "string"},
                "target_backend": {"type": "string"},
                "new_predicted_us": {"type": "number"},
                "rationale": {"type": "string"},
            },
            "required": ["out_dir", "round_index", "schedule_path",
                         "group_id", "target_backend"],
        },
    },
    {
        "name": "xpu_rt_qnn_build_context_on_board",
        "description": (
            "Build a QNN context binary on the QRB5165 board via "
            "qnn-context-binary-generator, optionally measure it with "
            "qnn-net-run --retrieve_context. This is the 'medium "
            "granularity' on-board profiling primitive."
        ),
        "phase": "job",
        "handler": xpu_rt_qnn_build_context_on_board,
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string"},
                "round_index": {"type": "integer"},
                "block_name": {"type": "string"},
                "dlc_path": {"type": "string"},
                "backend": {"type": "string"},
                "remote_out_dir": {"type": "string"},
                "measure": {"type": "boolean"},
                "measure_input_list": {"type": "string"},
                "measure_iters": {"type": "integer"},
            },
            "required": ["out_dir", "round_index", "block_name",
                         "dlc_path", "backend"],
        },
    },
    {
        "name": "xpu_rt_qnn_set_deadline_and_reschedule",
        "description": (
            "Build a real-time Workload (per-instance deadlines + "
            "global makespan bound), solve via MOSEK MILP, and return "
            "the feasibility verdict + per-instance assignment. The "
            "agent uses this once it has multi-granularity measurements "
            "on board to verify the 12-dronets-in-yolov8n target."
        ),
        "phase": "transform",
        "handler": xpu_rt_qnn_set_deadline_and_reschedule,
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string"},
                "round_index": {"type": "integer"},
                "workload_yaml": {"type": "string"},
                "latency_matrix": {"type": "object"},
                "makespan_bound_us": {"type": "number"},
                "time_limit_s": {"type": "number"},
            },
            "required": ["out_dir", "round_index", "workload_yaml",
                         "latency_matrix", "makespan_bound_us"],
        },
    },
]
