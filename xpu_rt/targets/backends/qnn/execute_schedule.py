"""On-board execution of a MOSEK MILP schedule.

The MILP returns ``ops`` with ``(name, machine, start_us,
finish_us, predicted_us)``. To validate that the prediction holds
under real concurrent execution, we translate the schedule into one
shell script per backend lane (containing the lane's ordered
qnn-net-run invocations) and run all lanes simultaneously via
parallel SSH sessions. Each invocation logs ns-precision
``START_NS`` / ``END_NS`` markers; we pull the trace back and
compare to the predicted timings.

Each island's executor is one of:

* ``--dlc_path``: whole-network DLC (used when the island covers
  the full workload).
* ``--retrieve_context``: pre-built QNN context binary (used when
  the island is a sub-network with an ``executor_artifact`` of
  kind ``context_binary``).

When an island has no ``executor_artifact`` for its assigned
backend, the executor raises a typed error rather than fabricating
output — never silently "schedule but skip".
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import re
import shlex
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from xpu_rt.targets.backends.qnn.board import BoardConfig
from xpu_rt.targets.backends.qnn.on_board_runner import BACKEND_LIB, _ssh_cmd


@dataclasses.dataclass(frozen=True)
class LaneInvocation:
    """One scheduled run-line on a backend lane."""

    op_id: str                     # the island id
    workload_id: str
    dlc_path: str | None           # for --dlc_path mode
    context_path: str | None       # for --retrieve_context mode
    backend: str                   # CPU / GPU / DSP / HTA / HTP
    input_list: str
    iters: int                     # 1 per island for honest timing
    predicted_us: float            # what the MILP said this should take


@dataclasses.dataclass(frozen=True)
class IslandMeasurement:
    op_id: str
    workload_id: str
    machine: str
    predicted_us: float
    start_ns: int
    end_ns: int
    iters: int
    ok: bool
    error: str = ""

    @property
    def duration_us(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000.0

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "duration_us": self.duration_us}


@dataclasses.dataclass(frozen=True)
class ExecutionResult:
    schedule_makespan_us: float    # MILP's predicted
    measured_makespan_us: float    # max lane finish
    lane_finish_us: dict[str, float]
    islands: list[IslandMeasurement]
    stderr_tails: dict[str, str]   # per-backend stderr tail
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "qnn_execution_result_v1",
            "schedule_makespan_us": self.schedule_makespan_us,
            "measured_makespan_us": self.measured_makespan_us,
            "lane_finish_us": dict(self.lane_finish_us),
            "islands": [i.to_dict() for i in self.islands],
            "stderr_tails": dict(self.stderr_tails),
            "ok": self.ok,
        }


_LANE_START_RE = re.compile(r"^LANE_START\s+(\S+)\s+(\d+)\s*$", re.MULTILINE)
_LANE_END_RE = re.compile(r"^LANE_END\s+(\S+)\s+(\d+)\s*$", re.MULTILINE)
_START_RE = re.compile(r"^START_NS\s+(\S+)\s+(\d+)\s*$", re.MULTILINE)
_END_RE = re.compile(r"^END_NS\s+(\S+)\s+(\d+)\s*$", re.MULTILINE)


def _lane_script(
    lane_id: str,
    invocations: Iterable[LaneInvocation],
    *,
    qnn_sdk_root: str = "/root/qairt",
    out_dir: str = "/tmp/qnn_lane_out",
) -> str:
    """Build the bash script for one backend lane.

    Emits one ``START_NS <op_id> <ns>`` and ``END_NS <op_id> <ns>``
    per invocation so we can pull a single trace and parse all
    timings without per-invocation scp.
    """
    lines: list[str] = [
        f'export LD_LIBRARY_PATH={qnn_sdk_root}/lib/target:${{LD_LIBRARY_PATH:-}}',
        f'export ADSP_LIBRARY_PATH="{qnn_sdk_root}/lib/hexagon-v66;/dsp/cdsp;/dsp"',
        f'mkdir -p {shlex.quote(out_dir)} && cd {shlex.quote(out_dir)}',
        f'echo "LANE_START {lane_id} $(date +%s%N)"',
    ]
    for inv in invocations:
        lib = BACKEND_LIB[inv.backend]
        # Build the per-invocation input list (line repeated iters times).
        lines.append('MULTI=$(mktemp)')
        lines.append(f'LINE=$(head -n1 {shlex.quote(inv.input_list)})')
        lines.append(
            'i=0 && while [ $i -lt '
            f'{inv.iters} ]; do echo "$LINE" >> "$MULTI"; i=$((i+1)); done'
        )
        # Per-invocation output dir to keep qnn-net-run happy.
        lines.append(f'OUT="{out_dir}/{lane_id}_{inv.op_id.replace("/", "_")}"')
        lines.append('rm -rf "$OUT" && mkdir -p "$OUT"')
        lines.append(f'echo "START_NS {inv.op_id} $(date +%s%N)"')
        if inv.context_path:
            cmd = (
                f'{qnn_sdk_root}/bin/target/qnn-net-run '
                f'--retrieve_context {shlex.quote(inv.context_path)} '
                f'--backend {qnn_sdk_root}/lib/target/{lib} '
                f'--input_list "$MULTI" --output_dir "$OUT" 1>/dev/null 2>&1'
            )
        else:
            assert inv.dlc_path, "either dlc_path or context_path must be set"
            cmd = (
                f'{qnn_sdk_root}/bin/target/qnn-net-run '
                f'--dlc_path {shlex.quote(inv.dlc_path)} '
                f'--backend {qnn_sdk_root}/lib/target/{lib} '
                f'--input_list "$MULTI" --output_dir "$OUT" 1>/dev/null 2>&1'
            )
        lines.append(cmd)
        lines.append(f'echo "END_NS {inv.op_id} $(date +%s%N)"')
        lines.append('rm -f "$MULTI"')
    lines.append(f'echo "LANE_END {lane_id} $(date +%s%N)"')
    return ' && '.join(lines)


def _run_lane(
    cfg: BoardConfig,
    lane_id: str,
    invocations: list[LaneInvocation],
    *,
    timeout_s: int = 600,
) -> tuple[str, str, int]:
    """Run one lane via SSH. Returns (stdout, stderr, returncode)."""
    if not invocations:
        return ("", "", 0)
    script = _lane_script(lane_id, invocations)
    cmd = _ssh_cmd(cfg, script, timeout=timeout_s)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
        return (proc.stdout, proc.stderr, proc.returncode)
    except subprocess.TimeoutExpired:
        return ("", f"ssh timeout after {timeout_s}s", 124)


def _parse_lane_output(stdout: str) -> tuple[dict[str, int], dict[str, int]]:
    """Pull (op_id → start_ns) and (op_id → end_ns) from a lane stdout."""
    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    for m in _START_RE.finditer(stdout):
        starts[m.group(1)] = int(m.group(2))
    for m in _END_RE.finditer(stdout):
        ends[m.group(1)] = int(m.group(2))
    return starts, ends


def _collapse_invocations(
    lane_invocations: list[LaneInvocation],
) -> list[LaneInvocation]:
    """Merge adjacent invocations of the same (workload, dlc, ctx, backend).

    When the MILP places N copies of the same workload on the same
    lane, naive execution runs N separate qnn-net-run processes,
    each paying its own init cost. We collapse runs that share an
    artifact into ONE invocation with ``iters = sum(iters)``. The
    collapsed island ID is the concatenation of constituent op ids
    so per-op trace markers stay unambiguous.
    """
    if not lane_invocations:
        return []
    out: list[LaneInvocation] = []
    current = lane_invocations[0]
    accumulated_ids: list[str] = [current.op_id]
    accumulated_iters: int = current.iters
    accumulated_pred: float = current.predicted_us
    for nxt in lane_invocations[1:]:
        same = (nxt.workload_id == current.workload_id
                and nxt.dlc_path == current.dlc_path
                and nxt.context_path == current.context_path
                and nxt.backend == current.backend
                and nxt.input_list == current.input_list)
        if same:
            accumulated_ids.append(nxt.op_id)
            accumulated_iters += nxt.iters
            accumulated_pred += nxt.predicted_us
        else:
            out.append(LaneInvocation(
                op_id="+".join(accumulated_ids),
                workload_id=current.workload_id,
                dlc_path=current.dlc_path, context_path=current.context_path,
                backend=current.backend, input_list=current.input_list,
                iters=accumulated_iters, predicted_us=accumulated_pred,
            ))
            current = nxt
            accumulated_ids = [nxt.op_id]
            accumulated_iters = nxt.iters
            accumulated_pred = nxt.predicted_us
    out.append(LaneInvocation(
        op_id="+".join(accumulated_ids),
        workload_id=current.workload_id,
        dlc_path=current.dlc_path, context_path=current.context_path,
        backend=current.backend, input_list=current.input_list,
        iters=accumulated_iters, predicted_us=accumulated_pred,
    ))
    return out


def _resolve_invocations(
    schedule: Mapping[str, Any],
    *,
    workload_specs: Mapping[str, Mapping[str, Any]],
    collapse: bool = True,
) -> dict[str, list[LaneInvocation]]:
    """Group schedule ops by backend lane, materialise LaneInvocations.

    ``workload_specs[workload_id]`` carries:
      - ``dlc_path``: whole-network DLC remote path, OR
      - ``context_paths[backend]``: per-backend context binary, OR
      both (per-island ``executor_artifact``).
      - ``input_list``: remote path with first line = input bytes.
    """
    lanes: dict[str, list[LaneInvocation]] = {}
    ops = schedule.get("ops") or []
    # Sort by start_us so the script's order matches the MILP plan.
    ops_sorted = sorted(ops, key=lambda o: float(o.get("start_us", 0.0)))
    for op in ops_sorted:
        wid = str(op.get("workload") or op.get("name", ""))
        b = str(op.get("machine"))
        spec = workload_specs.get(wid)
        if not spec:
            raise KeyError(
                f"workload spec missing for {wid!r}; provide "
                f"workload_specs[{wid!r}] with dlc_path / context_paths."
            )
        ctx = (spec.get("context_paths") or {}).get(b)
        # Per-backend DLC takes precedence over a single dlc_path.
        dlc = None
        if ctx is None:
            dlc = (spec.get("dlc_paths") or {}).get(b) or spec.get("dlc_path")
        if not ctx and not dlc:
            raise KeyError(
                f"no executor_artifact for ({wid}, {b}); refuse to execute."
            )
        lanes.setdefault(b, []).append(LaneInvocation(
            op_id=str(op.get("name", wid)),
            workload_id=wid,
            dlc_path=dlc,
            context_path=ctx,
            backend=b,
            input_list=spec["input_list"],
            iters=int(spec.get("iters", 1)),
            predicted_us=float(op.get("predicted_us", 0.0)),
        ))
    if collapse:
        for b in list(lanes):
            lanes[b] = _collapse_invocations(lanes[b])
    return lanes


def execute_schedule(
    cfg: BoardConfig,
    schedule: Mapping[str, Any],
    *,
    workload_specs: Mapping[str, Mapping[str, Any]],
    timeout_s: int = 600,
) -> ExecutionResult:
    """Run a MILP schedule on the board, return per-island timings."""
    lanes = _resolve_invocations(schedule, workload_specs=workload_specs)
    pred_makespan = float(schedule.get("makespan_us") or 0.0)
    if not lanes:
        return ExecutionResult(
            schedule_makespan_us=pred_makespan,
            measured_makespan_us=0.0,
            lane_finish_us={},
            islands=[],
            stderr_tails={},
            ok=True,
        )

    # Run all lanes simultaneously.
    futures: dict[str, concurrent.futures.Future] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(lanes)),
    ) as pool:
        for b, invocations in lanes.items():
            futures[b] = pool.submit(
                _run_lane, cfg, b, invocations, timeout_s=timeout_s,
            )
        outcomes = {b: f.result() for b, f in futures.items()}

    islands: list[IslandMeasurement] = []
    lane_finish_us: dict[str, float] = {}
    stderr_tails: dict[str, str] = {}
    ok_all = True
    # Find the earliest LANE_START across all lanes to normalise epoch
    # timestamps into relative durations from t=0.
    lane_starts_ns: dict[str, int] = {}
    for b, (stdout, _stderr, _rc) in outcomes.items():
        m = _LANE_START_RE.search(stdout)
        if m:
            lane_starts_ns[b] = int(m.group(2))
    global_start_ns = min(lane_starts_ns.values()) if lane_starts_ns else 0
    for b, (stdout, stderr, rc) in outcomes.items():
        stderr_tails[b] = stderr[-500:] if stderr else ""
        starts, ends = _parse_lane_output(stdout)
        # Convert per-island timestamps to relative-since-global-start.
        rel_starts = {k: v - global_start_ns for k, v in starts.items()}
        rel_ends = {k: v - global_start_ns for k, v in ends.items()}
        # Lane finish = max END_NS in this lane, normalised.
        lane_end_m = _LANE_END_RE.search(stdout)
        if lane_end_m:
            lane_finish_us[b] = (int(lane_end_m.group(2)) - global_start_ns) / 1_000.0
        elif rel_ends:
            lane_finish_us[b] = max(rel_ends.values()) / 1_000.0
        for inv in lanes[b]:
            s = rel_starts.get(inv.op_id)
            e = rel_ends.get(inv.op_id)
            if s is None or e is None or rc != 0:
                ok_all = False
                islands.append(IslandMeasurement(
                    op_id=inv.op_id, workload_id=inv.workload_id,
                    machine=b, predicted_us=inv.predicted_us,
                    start_ns=0, end_ns=0, iters=inv.iters,
                    ok=False,
                    error=f"rc={rc}; missing start/end markers",
                ))
                continue
            islands.append(IslandMeasurement(
                op_id=inv.op_id, workload_id=inv.workload_id,
                machine=b, predicted_us=inv.predicted_us,
                start_ns=s, end_ns=e, iters=inv.iters, ok=True,
            ))

    measured_makespan_us = max(lane_finish_us.values()) if lane_finish_us else 0.0
    return ExecutionResult(
        schedule_makespan_us=pred_makespan,
        measured_makespan_us=measured_makespan_us,
        lane_finish_us=lane_finish_us,
        islands=islands,
        stderr_tails=stderr_tails,
        ok=ok_all,
    )
