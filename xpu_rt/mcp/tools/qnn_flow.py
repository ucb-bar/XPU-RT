"""MCP tools for the agent-driven QNN flow.

Three handlers, all keyed to a single QNN run directory:

* ``xpu_rt_qnn_schedule_round`` — invokes the heterogeneous loop's
  ``run_single_round`` in-process, emits ``qnn_events.jsonl`` entries
  for the dashboard, and returns the round's schedule digest.
* ``xpu_rt_qnn_profile_on_board`` — wraps the on-board push/run/pull;
  in ``dry_run`` mode synthesises a ``profiled_manifest.json`` from
  the cached ``qrb5165_costs.json`` so the loop runs end-to-end
  without the QRB5165 powered on.
* ``xpu_rt_qnn_decide_granularity`` — packages the split/coarsen
  candidates + predicted-vs-measured table into a request that the
  agent (this Claude Code session, or a Codex client) reads and
  commits via the existing
  ``xpu_rt_emit_agent_decision_request`` /
  ``xpu_rt_commit_agent_decision_response`` pair.

These tools register through the standard ``QNN_FLOW_TOOLS`` list and
join ``ALL_TOOLS`` exactly like every other in-tree group.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from xpu_rt.mcp.session import SessionManager


# --------------------------------------------------------------------------- #
# QNN events jsonl — separate from stage_ledger.jsonl on purpose:
# stage_ledger.jsonl is bound to the StageEvent v1 schema (allowed
# events: start / finish / artifact_written / validation_pass /
# validation_fail). The QNN flow needs its own ``qnn_*`` events for
# the dashboard, so we keep them in their own file. The dashboard
# reads both.
# --------------------------------------------------------------------------- #


QNN_EVENTS_FILENAME = "qnn_events.jsonl"
QNN_DECISIONS_FILENAME = "granularity_decisions.jsonl"


def _import_heterogeneous_loop():
    """Import the heterogeneous_loop module from scripts/ regardless of cwd.

    The script lives at ``<repo>/scripts/heterogeneous_loop.py``, which
    isn't on the package import path. We resolve it via importlib so
    callers don't have to manipulate ``sys.path`` themselves.
    """
    import importlib.util
    import sys as _sys
    # mcp/tools/qnn_flow.py → 5 parents = repo root
    # (.../xpu-rt-integration/xpu-rt/python/xpu_rt/mcp/tools/qnn_flow.py)
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "scripts" / "heterogeneous_loop.py",
        Path.cwd() / "scripts" / "heterogeneous_loop.py",
    ]
    src = next((c for c in candidates if c.is_file()), None)
    if src is None:
        raise FileNotFoundError(
            "could not locate scripts/heterogeneous_loop.py (looked at "
            f"{[str(c) for c in candidates]})"
        )
    spec = importlib.util.spec_from_file_location("xpu_rt._het_loop", src)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    _sys.modules.setdefault("xpu_rt._het_loop", mod)
    spec.loader.exec_module(mod)
    return mod


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _qnn_event(run_dir: Path, *, event: str, **fields: Any) -> dict[str, Any]:
    rec = {"schema_version": "qnn_event_v1",
           "event": event,
           "timestamp_utc": _now(),
           **fields}
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / QNN_EVENTS_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def _resolve_loop_config(
    *,
    out_dir: Path,
    workload_id: str,
    source_mlir: Path | None,
    source_onnx: Path | None,
    targets: list[str],
    diversity_weight: float,
    max_rounds: int,
    cost_table: Path,
    ssh_host: str,
    ssh_identity: Path | None,
    merlin_root: Path,
) -> Any:
    """Build a ``heterogeneous_loop.LoopConfig`` without going through argparse."""
    het = _import_heterogeneous_loop()
    target_to_machine = {"cpu": "CPU", "qnn_gpu": "GPU", "qnn_hta": "HTA"}
    machines = [target_to_machine[t] for t in targets]
    machine_to_target = {m: t for t, m in target_to_machine.items()}
    if source_mlir is None and source_onnx is not None:
        bridge_out = out_dir / "payload_from_onnx.mlir"
        bridge_out.parent.mkdir(parents=True, exist_ok=True)
        source_mlir = het.materialise_source_from_onnx(
            source_onnx, bridge_out, workload_id=workload_id,
        )
    if source_mlir is None:
        raise ValueError("one of source_mlir / source_onnx is required")
    iree_compile = (merlin_root / "build/host-merlin-release-qrb/tools/iree-compile")
    return het.LoopConfig(
        source=Path(source_mlir).resolve(),
        out_dir=out_dir.resolve(),
        targets=targets,
        machines=machines,
        target_to_machine=target_to_machine,
        machine_to_target=machine_to_target,
        diversity_weight=diversity_weight,
        max_rounds=max_rounds,
        transfer_us=200.0,
        iterations=10,
        warmup=2,
        iree_compile=iree_compile,
        skip_profile=False,
        merlin_root=merlin_root.resolve(),
        cost_table=cost_table.resolve(),
        ssh_host=ssh_host,
        ssh_identity=ssh_identity.resolve() if ssh_identity else None,
        profile_input_mode="zero",
        capture_dir=None,
        dispatch_graph_json=None,
    )


# --------------------------------------------------------------------------- #
# Tool: schedule one round
# --------------------------------------------------------------------------- #


def xpu_rt_qnn_schedule_round(
    sm: SessionManager,  # noqa: ARG001
    *,
    out_dir: str,
    round_index: int,
    workload_id: str,
    source_mlir: str | None = None,
    source_onnx: str | None = None,
    targets: list[str] | None = None,
    diversity_weight: float = 100.0,
    max_rounds: int = 4,
    cost_table: str,
    ssh_host: str = "root@10.44.120.201",
    ssh_identity: str | None = None,
    merlin_root: str = "/scratch2/agustin/merlin",
    prev_schedule: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute one heterogeneous-loop round in-process.

    In ``dry_run`` mode skips the merlin compile-matrix + on-board
    profile steps and synthesises both from the cached cost table.
    Always emits ``qnn_island_dag_built`` and ``qnn_schedule_emitted``
    events into the QNN-events log.
    """
    rd_root = Path(out_dir).resolve()
    rd_root.mkdir(parents=True, exist_ok=True)
    targets = targets or ["cpu", "qnn_gpu", "qnn_hta"]
    if dry_run:
        return _dry_run_schedule_round(
            out_dir=rd_root,
            round_index=round_index,
            workload_id=workload_id,
            source_onnx=Path(source_onnx) if source_onnx else None,
            targets=targets,
            cost_table=Path(cost_table),
        )

    try:
        cfg = _resolve_loop_config(
            out_dir=rd_root,
            workload_id=workload_id,
            source_mlir=Path(source_mlir) if source_mlir else None,
            source_onnx=Path(source_onnx) if source_onnx else None,
            targets=targets,
            diversity_weight=diversity_weight,
            max_rounds=max_rounds,
            cost_table=Path(cost_table),
            ssh_host=ssh_host,
            ssh_identity=Path(ssh_identity) if ssh_identity else None,
            merlin_root=Path(merlin_root),
        )
        het = _import_heterogeneous_loop()
        outcome = het.run_single_round(
            cfg,
            k=round_index,
            cur_source=cfg.source,
            prev_schedule=Path(prev_schedule) if prev_schedule else None,
        )
    except Exception as exc:  # noqa: BLE001 - report typed
        _qnn_event(rd_root, event="qnn_schedule_failed",
                   round=round_index, error=str(exc),
                   exc_type=type(exc).__name__)
        return {"ok": False, "round": round_index,
                "error": str(exc), "exception_type": type(exc).__name__}

    _qnn_event(rd_root, event="qnn_schedule_emitted",
               round=round_index,
               makespan_us=outcome.makespan_us,
               schedule_path=str(outcome.schedule),
               placement_stable=outcome.placement_stable)
    from xpu_rt.ui.markdown import render_gantt_markdown

    pretty = render_gantt_markdown(
        outcome.schedule, title=f"Schedule (round {round_index})",
    )
    return {
        "ok": True,
        "round": round_index,
        "makespan_us": outcome.makespan_us,
        "schedule_path": str(outcome.schedule),
        "profiled_manifest_path": str(outcome.profiled_manifest),
        "matrix_path": str(outcome.matrix),
        "round_dir": str(outcome.round_dir),
        "placement_stable": outcome.placement_stable,
        "next_source": str(outcome.next_source),
        "pretty_markdown": pretty,
    }


def _dry_run_schedule_round(
    *,
    out_dir: Path,
    round_index: int,
    workload_id: str,
    source_onnx: Path | None,
    targets: list[str],
    cost_table: Path,
) -> dict[str, Any]:
    """Stand-alone dry-run path that uses only the cost table + scheduler.

    Builds one IslandVariantGroup per workload (coarse granularity)
    from arbitrary CostTable rows so the QNN scheduler has something
    to schedule. Subsequent rounds (split/coarsen choices applied)
    grow / shrink that island list.
    """
    rd = out_dir / f"round_{round_index}"
    rd.mkdir(parents=True, exist_ok=True)
    from xpu_rt.targets.backends.qnn.cost_table import CostTable

    schedule = {
        "schema_version": "qnn_dry_schedule_v1",
        "workload_id": workload_id,
        "round": round_index,
        "machines": ["HTA", "GPU", "CPU"],
        "makespan_us": 0.0,
        "ops": [],
        "dispatches": {},
    }
    table = (CostTable.load(cost_table) if cost_table.is_file() else CostTable())
    # Pick a few measured rows to populate the schedule, biased by
    # round_index so successive rounds shrink the makespan slightly
    # (simulates the agent's split/coarsen decisions converging).
    keys = sorted(table.execute.keys())[: max(4, 12 - round_index)]
    cursor_us = {"HTA": 0.0, "GPU": 0.0, "CPU": 0.0}
    for key in keys:
        row = table.execute.get(key) or {}
        mean = float(row.get("mean_us") or 1000.0)
        if "::HTA::" in key:
            machine = "HTA"
        elif "::GPU::" in key:
            machine = "GPU"
        else:
            machine = "CPU"
        name = f"{workload_id}.{key.replace('::', '_')[:48]}"
        start = cursor_us[machine]
        finish = start + mean
        cursor_us[machine] = finish
        op = {
            "name": name,
            "machine": machine,
            "workload": workload_id,
            "start_us": start,
            "finish_us": finish,
            "predicted_us": mean,
            "cost_table_key": key,
        }
        schedule["ops"].append(op)
        schedule["dispatches"][name] = dict(op)
    schedule["makespan_us"] = max(cursor_us.values()) if cursor_us else 0.0
    schedule_path = rd / "schedule.json"
    schedule_path.write_text(json.dumps(schedule, indent=2))

    # Synthesise a profiled_manifest mirroring the schedule via the
    # measured cost table.
    from xpu_rt.targets.backends.qnn.profile_lookup import synthesise_profiled_manifest

    manifest = synthesise_profiled_manifest(
        schedule_dispatches=schedule["dispatches"],
        targets=targets,
        cost_table_path=cost_table,
    )
    manifest_path = rd / "profiled_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    _qnn_event(out_dir, event="qnn_island_dag_built",
               round=round_index, num_groups=len(schedule["ops"]),
               num_candidates=len(schedule["ops"]), dry_run=True)
    _qnn_event(out_dir, event="qnn_schedule_emitted",
               round=round_index, makespan_us=schedule["makespan_us"],
               schedule_path=str(schedule_path), dry_run=True)
    _qnn_event(out_dir, event="qnn_trace_ingested",
               round=round_index, trace_path=str(manifest_path),
               dry_run=True, source="cost_table")
    from xpu_rt.ui.markdown import render_gantt_markdown

    pretty = render_gantt_markdown(
        schedule, title=f"Schedule (round {round_index}, dry-run)",
    )
    return {
        "ok": True,
        "round": round_index,
        "makespan_us": schedule["makespan_us"],
        "schedule_path": str(schedule_path),
        "profiled_manifest_path": str(manifest_path),
        "round_dir": str(rd),
        "placement_stable": False,
        "dry_run": True,
        "pretty_markdown": pretty,
    }


# --------------------------------------------------------------------------- #
# Tool: profile on board (or synthesise from cost table in dry-run)
# --------------------------------------------------------------------------- #


def xpu_rt_qnn_profile_on_board(
    sm: SessionManager,  # noqa: ARG001
    *,
    out_dir: str,
    round_index: int,
    schedule_path: str,
    cost_table: str,
    ssh_host: str = "root@10.44.120.201",
    ssh_identity: str | None = None,
    targets: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Profile every dispatch in the round's schedule on the board.

    In ``dry_run`` mode the call only reads the cached cost table —
    no SSH, no scp, no board interaction. The output manifest shape
    is identical to the real path so downstream consumers don't care
    which mode produced it.
    """
    rd_root = Path(out_dir).resolve()
    rd = rd_root / f"round_{round_index}"
    rd.mkdir(parents=True, exist_ok=True)
    targets = targets or ["cpu", "qnn_gpu", "qnn_hta"]
    try:
        schedule = json.loads(Path(schedule_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"could not read schedule: {exc}"}
    dispatches = schedule.get("dispatches") or {
        op.get("name", f"op_{i}"): op for i, op in enumerate(schedule.get("ops", []))
    }
    if dry_run:
        from xpu_rt.targets.backends.qnn.profile_lookup import synthesise_profiled_manifest

        manifest = synthesise_profiled_manifest(
            schedule_dispatches=dispatches,
            targets=targets,
            cost_table_path=Path(cost_table),
        )
        manifest_path = rd / "profiled_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        _qnn_event(rd_root, event="qnn_trace_ingested",
                   round=round_index, trace_path=str(manifest_path),
                   dry_run=True, source="cost_table")
        from xpu_rt.targets.backends.qnn.granularity import (
            predicted_vs_measured_table,
        )
        from xpu_rt.ui.markdown import render_deltas_markdown

        try:
            schedule_doc = json.loads(Path(schedule_path).read_text())
        except (OSError, json.JSONDecodeError):
            schedule_doc = {"ops": []}
        deltas = predicted_vs_measured_table(
            profile=manifest, schedule=schedule_doc,
        )
        pretty = render_deltas_markdown(
            deltas,
            title=f"Predicted vs measured (round {round_index}, dry-run)",
        )
        return {"ok": True, "round": round_index, "dry_run": True,
                "profiled_manifest_path": str(manifest_path),
                "pretty_markdown": pretty}

    # Real path: shell out to run_on_board_flow.py.
    import shlex
    import subprocess

    onboard_dir = rd / "onboard"
    onboard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = rd / "profiled_manifest.json"
    if not manifest_path.is_file():
        # The heterogeneous loop's step_profile is the canonical
        # producer; this tool's contract is "make sure the manifest is
        # on disk", not "redo the profiling". If the caller did not
        # already invoke step_profile, fail loudly.
        return {"ok": False, "round": round_index,
                "error": f"no profiled_manifest.json at {manifest_path}; "
                         "run xpu_rt_qnn_schedule_round first or use dry_run=True"}

    repo_root = Path(__file__).resolve().parents[3]
    cmd = [
        "python3", str(repo_root / "scripts" / "run_on_board_flow.py"),
        "--schedule", str(schedule_path),
        "--manifest", str(manifest_path),
        "--out-dir", str(onboard_dir),
        "--ssh-host", ssh_host,
    ]
    if ssh_identity:
        cmd.extend(["--ssh-identity", str(ssh_identity)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    _qnn_event(rd_root, event="qnn_board_pushed",
               round=round_index, returncode=proc.returncode,
               board_host=ssh_host,
               cmd=" ".join(shlex.quote(c) for c in cmd))
    if proc.returncode != 0:
        return {"ok": False, "round": round_index,
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-1000:]}
    _qnn_event(rd_root, event="qnn_trace_ingested",
               round=round_index,
               trace_path=str(onboard_dir / "trace.csv"))
    from xpu_rt.targets.backends.qnn.granularity import (
        predicted_vs_measured_table,
    )
    from xpu_rt.ui.markdown import render_deltas_markdown

    try:
        schedule_doc = json.loads(Path(schedule_path).read_text())
        manifest_doc = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        schedule_doc, manifest_doc = {"ops": []}, {"dispatches": {}}
    deltas = predicted_vs_measured_table(
        profile=manifest_doc, schedule=schedule_doc,
    )
    pretty = render_deltas_markdown(
        deltas, title=f"Predicted vs measured (round {round_index})",
    )
    return {"ok": True, "round": round_index, "dry_run": False,
            "onboard_dir": str(onboard_dir),
            "trace_path": str(onboard_dir / "trace.csv"),
            "profiled_manifest_path": str(manifest_path),
            "pretty_markdown": pretty}


# --------------------------------------------------------------------------- #
# Tool: decide granularity (split / coarsen / keep)
# --------------------------------------------------------------------------- #


def xpu_rt_qnn_decide_granularity(
    sm: SessionManager,  # noqa: ARG001
    *,
    out_dir: str,
    round_index: int,
    schedule_path: str,
    profile_path: str,
    dossier_path: str | None = None,
    write_request: bool = True,
) -> dict[str, Any]:
    """Build the agent-facing decision request for this round.

    Returns a typed dict matching the standard
    ``agent_decision_request`` envelope, with a ``kind:
    "qnn_granularity"`` discriminator so the skill driver can route
    it to the right rubric. When ``write_request`` is True (default)
    the request is also written to
    ``<out_dir>/round_<k>/qnn_granularity_request.json`` so the
    dashboard can render it without re-calling the tool.
    """
    rd_root = Path(out_dir).resolve()
    rd = rd_root / f"round_{round_index}"
    rd.mkdir(parents=True, exist_ok=True)

    from xpu_rt.targets.backends.qnn.granularity import (
        compute_coarsen_candidates,
        compute_split_candidates,
        predicted_vs_measured_table,
    )

    try:
        schedule = json.loads(Path(schedule_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"schedule unreadable: {exc}"}
    try:
        profile = json.loads(Path(profile_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"profile unreadable: {exc}"}
    dossier = None
    if dossier_path:
        try:
            dossier = json.loads(Path(dossier_path).read_text())
        except (OSError, json.JSONDecodeError):
            dossier = None

    splits = compute_split_candidates(
        dossier=dossier, profile=profile, schedule=schedule,
    )
    coarsens = compute_coarsen_candidates(schedule=schedule)
    deltas = predicted_vs_measured_table(profile=profile, schedule=schedule)

    request = {
        "schema_version": "agent_decision_request_v1",
        "kind": "qnn_granularity",
        "round_index": round_index,
        "makespan_us": float(schedule.get("makespan_us", 0.0)),
        "candidate_ids_allowed": (
            [f"split:{c.dispatch_id}" for c in splits]
            + [f"coarsen:{c.first_dispatch_id}+{c.second_dispatch_id}"
               for c in coarsens]
            + ["keep:all"]
        ),
        "split_candidates": [c.to_dict() for c in splits],
        "coarsen_candidates": [c.to_dict() for c in coarsens],
        "predicted_vs_measured": deltas[:64],  # cap for prompt size
        "greedy_pick": (
            f"split:{splits[0].dispatch_id}" if splits
            else (f"coarsen:{coarsens[0].first_dispatch_id}+"
                  f"{coarsens[0].second_dispatch_id}" if coarsens
                  else "keep:all")
        ),
    }
    if write_request:
        (rd / "qnn_granularity_request.json").write_text(
            json.dumps(request, indent=2)
        )
    _qnn_event(rd_root, event="qnn_granularity_decision",
               round=round_index,
               n_split=len(splits), n_coarsen=len(coarsens),
               greedy_pick=request["greedy_pick"])
    from xpu_rt.ui.markdown import (
        render_decision_markdown,
        render_round_summary_markdown,
    )

    decision_md = render_decision_markdown(
        round_index=round_index,
        makespan_us=request["makespan_us"],
        greedy_pick=request["greedy_pick"],
        split_candidates=request["split_candidates"],
        coarsen_candidates=request["coarsen_candidates"],
        legal_candidate_ids=request["candidate_ids_allowed"],
    )
    summary_md = render_round_summary_markdown(
        round_index=round_index,
        makespan_us=request["makespan_us"],
        schedule=schedule,
        deltas=deltas,
        greedy_pick=request["greedy_pick"],
        split_candidates=request["split_candidates"],
        coarsen_candidates=request["coarsen_candidates"],
        legal_candidate_ids=request["candidate_ids_allowed"],
    )
    return {"ok": True, **request,
            "pretty_markdown": decision_md,
            "round_summary_markdown": summary_md}


# --------------------------------------------------------------------------- #
# Tool list (consumed by mcp/tools/__init__.py)
# --------------------------------------------------------------------------- #


QNN_FLOW_TOOLS: list[dict[str, Any]] = [
    {
        "name": "xpu_rt_qnn_schedule_round",
        "description": (
            "Run one heterogeneous-loop round in-process: compile the "
            "dispatch matrix (or, in dry_run, synthesise from the "
            "cached cost table), profile on board, run the MOSEK "
            "scheduler, optionally re-quantise. Emits qnn_* events "
            "into <out_dir>/qnn_events.jsonl for the dashboard."
        ),
        "phase": "transform",
        "handler": xpu_rt_qnn_schedule_round,
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string"},
                "round_index": {"type": "integer"},
                "workload_id": {"type": "string"},
                "source_mlir": {"type": "string"},
                "source_onnx": {"type": "string"},
                "targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["cpu", "qnn_gpu", "qnn_hta"],
                },
                "diversity_weight": {"type": "number", "default": 100.0},
                "max_rounds": {"type": "integer", "default": 4},
                "cost_table": {"type": "string"},
                "ssh_host": {"type": "string"},
                "ssh_identity": {"type": "string"},
                "merlin_root": {"type": "string"},
                "prev_schedule": {"type": "string"},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["out_dir", "round_index", "workload_id", "cost_table"],
        },
    },
    {
        "name": "xpu_rt_qnn_profile_on_board",
        "description": (
            "Push the round's VMFBs to the QRB5165 board, run the "
            "flow runner, pull back trace.csv + profiled_manifest.json. "
            "In dry_run mode synthesises a profiled_manifest from the "
            "cached cost table so the loop runs without the board."
        ),
        "phase": "job",
        "handler": xpu_rt_qnn_profile_on_board,
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string"},
                "round_index": {"type": "integer"},
                "schedule_path": {"type": "string"},
                "cost_table": {"type": "string"},
                "ssh_host": {"type": "string"},
                "ssh_identity": {"type": "string"},
                "targets": {"type": "array", "items": {"type": "string"}},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["out_dir", "round_index", "schedule_path", "cost_table"],
        },
    },
    {
        "name": "xpu_rt_qnn_decide_granularity",
        "description": (
            "Build the per-round granularity decision request: "
            "split candidates (predicted-vs-measured ratio > 1.3 AND "
            "region ≥10% of makespan), coarsen candidates (same-backend "
            "adjacent islands with transfer ≥20% of compute), and the "
            "predicted-vs-measured deltas table. Returns an "
            "agent_decision_request_v1 envelope with kind=qnn_granularity."
        ),
        "phase": "inspect",
        "handler": xpu_rt_qnn_decide_granularity,
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string"},
                "round_index": {"type": "integer"},
                "schedule_path": {"type": "string"},
                "profile_path": {"type": "string"},
                "dossier_path": {"type": "string"},
                "write_request": {"type": "boolean", "default": True},
            },
            "required": ["out_dir", "round_index", "schedule_path",
                         "profile_path"],
        },
    },
]

# Append the granularity / fusion / deadline tools (kept in their own
# module to keep this file focused on the round-driver primitives).
from xpu_rt.mcp.tools.qnn_granularity import QNN_GRANULARITY_TOOLS  # noqa: E402
from xpu_rt.mcp.tools.qnn_closed_loop import QNN_CLOSED_LOOP_TOOLS  # noqa: E402

QNN_FLOW_TOOLS.extend(QNN_GRANULARITY_TOOLS)
QNN_FLOW_TOOLS.extend(QNN_CLOSED_LOOP_TOOLS)
