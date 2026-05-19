"""Autonomous-run proof recorder for the QNN agentic loop.

Two artifacts are written for every run:

* ``agent_trace.jsonl`` — one line per MCP tool invocation the agent
  makes (or per agent-level decision). Each entry carries the tool
  name, the round index, the input args (compactly), the returned
  outcome summary, and the rationale string the agent supplied.
* ``final_report.md`` — a human-readable, Claude-Code-friendly
  summary written after the loop terminates. Includes the
  optimization arc (round-by-round makespans), the agent's decision
  log, and the final assignment table (yolov8n + 12 dronets), with
  per-instance deadline-met flags.

Together they form the "proof" that the loop ran autonomously and
hit the target ("12 DroNets within YOLOv8n's makespan, all deadlines
met").
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from pathlib import Path
from typing import Any


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


@dataclasses.dataclass
class RoundSummary:
    """One row in the final report's optimization-arc table."""

    round_index: int
    granularity: str          # "coarse" | "medium" | "fine"
    action: str               # "calibrate" | "place" | "split" | "fuse" | "reschedule" | "validate" | "stop"
    predicted_makespan_us: float | None
    measured_makespan_us: float | None
    feasibility: str          # "pass" | "infeasible" | "n/a"
    deadlines_met: int | None = None
    deadlines_total: int | None = None
    rationale: str = ""
    assignment: dict[str, str] | None = None
    # Closed-loop additions: per-backend contention factor at the
    # END of this round (after applying the round's feedback), and a
    # bool flag for whether the factor converged to within tolerance.
    contention_factors: dict[str, float] | None = None
    contention_converged: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ProofWriter:
    """Append-only sink for agent trace + final report builder."""

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "agent_trace.jsonl"
        self.report_path = self.run_dir / "final_report.md"

    def record(
        self,
        *,
        tool_name: str,
        round_index: int,
        args: dict[str, Any],
        result: dict[str, Any] | None,
        rationale: str,
    ) -> None:
        """Append one trace entry. Never raises."""
        line = {
            "schema_version": "agent_trace_v1",
            "timestamp_utc": _now(),
            "tool": tool_name,
            "round": round_index,
            "args": _redact(args),
            "result_summary": _summarise(result or {}),
            "rationale": rationale,
        }
        try:
            with self.trace_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
        except OSError:
            pass

    def write_final_report(
        self,
        rounds: list[RoundSummary],
        *,
        target_makespan_us: float | None = None,
        n_dronet_copies: int = 12,
    ) -> Path:
        """Render the final markdown report and return its path."""
        lines: list[str] = []
        lines.append("# Autonomous QNN scheduling proof — final report")
        lines.append("")
        lines.append(f"_Generated {_now()}_")
        lines.append("")
        # Target line.
        if target_makespan_us is not None:
            lines.append(
                f"**Target:** schedule **{n_dronet_copies}× DroNet** within "
                f"YOLOv8n's measured makespan ({target_makespan_us / 1000:.1f} ms) "
                "while every DroNet instance meets its 40 ms latency budget."
            )
        else:
            lines.append("**Target:** see round table below.")
        lines.append("")

        # Final outcome.
        if rounds:
            last = rounds[-1]
            ok = (last.feasibility == "pass"
                  and (last.deadlines_met or 0) == (last.deadlines_total or 0))
            verdict = "✅ **PASS**" if ok else "❌ **NEEDS WORK**"
            lines.append(f"**Final verdict:** {verdict}")
            if last.assignment:
                lines.append("")
                lines.append("Final assignment:")
                for k, v in last.assignment.items():
                    lines.append(f"- `{k}` → **{v}**")
            lines.append("")

        # Optimization arc table.
        lines.append("## Optimization arc")
        lines.append("")
        lines.append("| round | granularity | action | predicted (ms) | measured (ms) | feasibility | deadlines |")
        lines.append("|---:|---|---|---:|---:|---|---|")
        for r in rounds:
            pred = (
                f"{r.predicted_makespan_us/1000:.1f}"
                if r.predicted_makespan_us is not None else "—"
            )
            meas = (
                f"{r.measured_makespan_us/1000:.1f}"
                if r.measured_makespan_us is not None else "—"
            )
            dl = (
                f"{r.deadlines_met}/{r.deadlines_total}"
                if r.deadlines_total else "—"
            )
            lines.append(
                f"| {r.round_index} | {r.granularity} | {r.action} "
                f"| {pred} | {meas} | {r.feasibility} | {dl} |"
            )
        lines.append("")

        # Contention convergence — only render when at least one
        # round recorded contention factors.
        contention_rounds = [r for r in rounds
                             if r.contention_factors is not None]
        if contention_rounds:
            lines.append("## Contention convergence")
            lines.append("")
            backends = sorted({b for r in contention_rounds
                                 for b in r.contention_factors})
            header = "| round | " + " | ".join(f"{b}" for b in backends) + " | converged? |"
            sep = "|---:|" + "|".join("---:" for _ in backends) + "|:--:|"
            lines.append(header)
            lines.append(sep)
            for r in contention_rounds:
                row = [str(r.round_index)]
                for b in backends:
                    v = r.contention_factors.get(b)
                    row.append(f"{v:.3f}" if v is not None else "—")
                row.append("✅" if r.contention_converged else "…")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        # Decision log section sources from the trace itself.
        lines.append("## Agent decision log")
        lines.append("")
        for r in rounds:
            if r.rationale:
                lines.append(f"### Round {r.round_index} · {r.action} ({r.granularity})")
                lines.append("")
                lines.append(r.rationale)
                lines.append("")

        # MCP-tool call summary table from the trace.
        try:
            tools_seen: dict[str, int] = {}
            with self.trace_path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    tools_seen[rec["tool"]] = tools_seen.get(rec["tool"], 0) + 1
            if tools_seen:
                lines.append("## MCP tool invocations")
                lines.append("")
                lines.append("| tool | calls |")
                lines.append("|---|---:|")
                for tool, n in sorted(tools_seen.items(), key=lambda x: -x[1]):
                    lines.append(f"| `{tool}` | {n} |")
                lines.append("")
        except OSError:
            pass

        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        return self.report_path


def _redact(args: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky fields from arg dicts (e.g., raw_stderr_tail)."""
    out: dict[str, Any] = {}
    for k, v in args.items():
        if k in {"raw_stderr_tail", "pretty_markdown", "round_summary_markdown"}:
            continue
        if isinstance(v, str) and len(v) > 200:
            out[k] = v[:200] + "…"
        else:
            out[k] = v
    return out


def _summarise(result: dict[str, Any]) -> dict[str, Any]:
    """Compact a tool result for the trace line.

    Keeps only fields useful for offline auditing — no Gantt blobs,
    no raw schedule.json — so the trace stays small.
    """
    if not isinstance(result, dict):
        return {"raw": str(result)[:200]}
    keep = {
        "ok", "feasible", "status", "round", "makespan_us",
        "deadlines_met_count", "deadlines_total",
        "greedy_pick", "n_split", "n_coarsen",
        "placement_stable", "schedule_path",
        "profiled_manifest_path",
    }
    return {k: result[k] for k in keep if k in result}
