"""Rich Live dashboard for the agentic QNN flow.

Four named panels, rendered in a single ``rich.live.Live`` block:

* **stages** — per-stage spinner / status / elapsed, derived from
  ``stage_ledger.jsonl`` + ``qnn_events.jsonl`` events.
* **gantt** — ASCII Gantt of the most recent round's schedule, via
  :func:`xpu_rt.ui.gantt.render_ascii_gantt`.
* **deltas** — predicted-vs-measured table for the most recent
  round, with rows above the split-threshold highlighted.
* **decisions** — bottom panel streaming
  ``qnn_granularity_decision`` events: the agent's chosen action
  and the rationale.

The dashboard tails the events JSONLs in a background thread so the
panels update in real time. ``prompt_override()`` stops the Live
display, defers to ``ask_user_question``, then restarts Live — both
prompt_toolkit and Rich want exclusive control of the terminal.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xpu_rt.ui.gantt import render_ascii_gantt
from xpu_rt.ui.question import QuestionAnswer, QuestionOption, ask_user_question

QNN_EVENTS_FILENAME = "qnn_events.jsonl"
LEDGER_FILENAME = "stage_ledger.jsonl"
DECISIONS_FILENAME = "granularity_decisions.jsonl"


@dataclass
class _StageRow:
    stage: str
    status: str = "pending"  # pending | running | done | failed
    started_at: float | None = None
    finished_at: float | None = None
    note: str = ""

    @property
    def elapsed_s(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or time.time()
        return end - self.started_at


@dataclass
class _DashboardState:
    stages: dict[str, _StageRow] = field(default_factory=dict)
    schedule_path: Path | None = None
    profile_path: Path | None = None
    decisions: list[dict] = field(default_factory=list)
    deltas: list[dict] = field(default_factory=list)
    last_event_at: float = 0.0
    round_index: int = -1


class QnnDashboard:
    """Live four-panel dashboard for a QNN run directory."""

    def __init__(
        self,
        run_dir: Path,
        *,
        stage_order: Sequence[str] | None = None,
        refresh_per_second: float = 4.0,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._stage_order = list(stage_order or _default_stage_order())
        for s in self._stage_order:
            self._state_init_stage(s)
        self._state = _DashboardState(stages={
            s: _StageRow(stage=s) for s in self._stage_order
        })
        self._refresh = refresh_per_second
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Lazy-imported Rich pieces.
        self._live = None
        self._layout = None
        self._console = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self.stop()
        return False

    def start(self) -> None:
        from rich.console import Console
        from rich.layout import Layout
        from rich.live import Live

        self._console = Console()
        self._layout = Layout()
        self._layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="decisions", size=8),
        )
        self._layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )
        self._layout["left"].split_column(
            Layout(name="stages"),
            Layout(name="gantt"),
        )
        self._layout["right"].update(self._render_deltas())
        self._layout["header"].update(self._render_header())
        self._layout["stages"].update(self._render_stages())
        self._layout["gantt"].update(self._render_gantt())
        self._layout["decisions"].update(self._render_decisions())

        self._live = Live(
            self._layout,
            console=self._console,
            refresh_per_second=self._refresh,
            screen=False,
            transient=False,
        )
        self._live.start()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._live is not None:
            try:
                self._live.stop()
            finally:
                self._live = None

    # ------------------------------------------------------------------ #
    # Background poll
    # ------------------------------------------------------------------ #

    def _poll_loop(self) -> None:
        last_qnn_pos = 0
        last_ledger_pos = 0
        while not self._stop.is_set():
            try:
                last_qnn_pos = self._consume_jsonl(
                    self.run_dir / QNN_EVENTS_FILENAME, last_qnn_pos,
                    self._handle_qnn_event,
                )
                last_ledger_pos = self._consume_jsonl(
                    self.run_dir / LEDGER_FILENAME, last_ledger_pos,
                    self._handle_ledger_event,
                )
                self._refresh_panels()
            except Exception:  # noqa: BLE001 - never let the UI crash the run
                pass
            self._stop.wait(1.0 / max(1.0, self._refresh))

    def _consume_jsonl(
        self, path: Path, start: int, callback,
    ) -> int:
        if not path.is_file():
            return start
        try:
            with path.open("r", encoding="utf-8") as fh:
                fh.seek(start)
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        callback(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                return fh.tell()
        except OSError:
            return start

    def _handle_qnn_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("event", "")
        self._state.last_event_at = time.time()
        if kind == "qnn_schedule_emitted":
            self._state.round_index = int(ev.get("round", -1))
            sp = ev.get("schedule_path")
            if sp:
                self._state.schedule_path = Path(sp)
        elif kind == "qnn_trace_ingested":
            tp = ev.get("trace_path")
            if tp:
                self._state.profile_path = Path(tp)
        elif kind == "qnn_granularity_decision":
            self._state.decisions.append(dict(ev))

    def _handle_ledger_event(self, ev: dict[str, Any]) -> None:
        sid = ev.get("stage_id", "")
        evt = ev.get("event", "")
        if not sid:
            return
        if sid not in self._state.stages:
            self._state.stages[sid] = _StageRow(stage=sid)
            if sid not in self._stage_order:
                self._stage_order.append(sid)
        row = self._state.stages[sid]
        if evt == "start":
            row.status = "running"
            row.started_at = time.time()
        elif evt == "finish":
            row.status = "done"
            row.finished_at = time.time()
        elif evt == "validation_fail":
            row.status = "failed"
            row.finished_at = time.time()
            row.note = ev.get("note") or row.note

    # ------------------------------------------------------------------ #
    # Renderers
    # ------------------------------------------------------------------ #

    def _state_init_stage(self, s: str) -> None:
        # placeholder hook; reserved for plugin stages
        pass

    def _refresh_panels(self) -> None:
        if self._layout is None:
            return
        self._layout["header"].update(self._render_header())
        self._layout["stages"].update(self._render_stages())
        self._layout["gantt"].update(self._render_gantt())
        self._layout["right"].update(self._render_deltas())
        self._layout["decisions"].update(self._render_decisions())

    def _render_header(self):
        from rich.panel import Panel
        from rich.text import Text

        text = Text()
        text.append("XPU-RT  ", style="bold cyan")
        text.append("QNN agentic flow  ", style="bold")
        text.append(f"run_dir={self.run_dir}\n")
        text.append(f"round={self._state.round_index}", style="dim")
        return Panel(text, border_style="cyan")

    def _render_stages(self):
        from rich import box
        from rich.panel import Panel
        from rich.table import Table

        tbl = Table(box=box.MINIMAL, expand=True, show_header=True)
        tbl.add_column("stage", overflow="fold")
        tbl.add_column("status", width=8)
        tbl.add_column("elapsed", justify="right", width=8)
        tbl.add_column("note", overflow="fold")
        for s in self._stage_order:
            row = self._state.stages.get(s) or _StageRow(stage=s)
            color = {
                "pending": "dim", "running": "yellow",
                "done": "green", "failed": "red",
            }.get(row.status, "white")
            elapsed = row.elapsed_s
            elapsed_s = f"{elapsed:.1f}s" if elapsed is not None else "-"
            tbl.add_row(
                f"[{color}]{row.stage}[/{color}]",
                f"[{color}]{row.status}[/{color}]",
                elapsed_s,
                row.note,
            )
        return Panel(tbl, title="Stages", border_style="cyan")

    def _render_gantt(self):
        from rich.panel import Panel
        from rich.text import Text

        sp = self._state.schedule_path
        if sp is None or not sp.is_file():
            return Panel(Text("(no schedule yet)", style="dim"),
                         title="Gantt", border_style="cyan")
        try:
            schedule = json.loads(sp.read_text())
        except (OSError, json.JSONDecodeError):
            return Panel(Text("(schedule unreadable)", style="red"),
                         title="Gantt", border_style="red")
        width = (self._console.width if self._console else 100) // 2 - 6
        return render_ascii_gantt(schedule, width=max(20, width))

    def _render_deltas(self):
        from rich import box
        from rich.panel import Panel
        from rich.table import Table

        sp = self._state.schedule_path
        pp = self._state.profile_path
        tbl = Table(box=box.MINIMAL, expand=True, show_header=True)
        tbl.add_column("dispatch", overflow="fold")
        tbl.add_column("dev", width=4)
        tbl.add_column("pred (µs)", justify="right")
        tbl.add_column("meas (µs)", justify="right")
        tbl.add_column("Δ", justify="right")
        tbl.add_column("ratio", justify="right")
        if sp is None or pp is None:
            return Panel(tbl, title="Δ (predicted vs measured)",
                         border_style="cyan")
        try:
            from xpu_rt.targets.backends.qnn.granularity import (
                predicted_vs_measured_table,
            )
            schedule = json.loads(sp.read_text())
            profile = json.loads(pp.read_text())
            rows = predicted_vs_measured_table(profile=profile, schedule=schedule)
        except Exception:  # noqa: BLE001
            return Panel(tbl, title="Δ (read failed)", border_style="red")
        for r in rows[:18]:
            ratio = r.get("ratio")
            style = ""
            if ratio is not None and ratio > 1.3:
                style = "red"
            elif ratio is not None and ratio > 1.1:
                style = "yellow"
            delta = r.get("delta_us")
            tbl.add_row(
                f"[{style}]{r['dispatch'][:32]}[/{style}]" if style
                else r["dispatch"][:32],
                str(r["machine"]),
                f"{r['predicted_us']:.0f}",
                f"{r['measured_us']:.0f}" if r["measured_us"] is not None else "-",
                f"{delta:+.0f}" if delta is not None else "-",
                f"{ratio:.2f}" if ratio is not None else "-",
            )
        return Panel(tbl, title="Δ (predicted vs measured)",
                     border_style="cyan")

    def _render_decisions(self):
        from rich import box
        from rich.panel import Panel
        from rich.table import Table

        tbl = Table(box=box.MINIMAL, expand=True, show_header=True)
        tbl.add_column("round", width=5)
        tbl.add_column("pick")
        tbl.add_column("n_split", justify="right")
        tbl.add_column("n_coarsen", justify="right")
        tbl.add_column("timestamp", overflow="fold")
        for d in self._state.decisions[-6:]:
            tbl.add_row(
                str(d.get("round", "-")),
                str(d.get("greedy_pick", "-")),
                str(d.get("n_split", "-")),
                str(d.get("n_coarsen", "-")),
                str(d.get("timestamp_utc", ""))[:19],
            )
        return Panel(tbl, title="Decisions", border_style="cyan")

    # ------------------------------------------------------------------ #
    # Inline prompt: pause Live, ask, resume Live.
    # ------------------------------------------------------------------ #

    def prompt_override(
        self,
        title: str,
        options: Iterable[QuestionOption],
        *,
        kind: str = "radio",
        allow_skip: bool = True,
        allow_freeform: bool = True,
        default_value: str | None = None,
    ) -> QuestionAnswer:
        """Ask the user a follow-up question, with Live paused."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:  # noqa: BLE001
                pass
        try:
            return ask_user_question(
                title, list(options),
                kind=kind,  # type: ignore[arg-type]
                allow_skip=allow_skip,
                allow_freeform=allow_freeform,
                default_value=default_value,
            )
        finally:
            if self._live is not None:
                try:
                    self._live.start()
                except Exception:  # noqa: BLE001
                    pass


def _default_stage_order() -> tuple[str, ...]:
    return (
        "qnn_island_dag_built",
        "qnn_schedule_emitted",
        "qnn_board_pushed",
        "qnn_trace_ingested",
        "qnn_granularity_decision",
    )
