"""Per-stage / per-pass IR dumping, IREE-style.

Opt-in via :class:`compgen.options.CompGenOptions` ``dump_ir`` or the
``COMPGEN_DUMP_IR=1`` environment variable. When enabled, every pass
invocation (through ``pipeline/driver.py::_run_with_report``) and every
stage invocation (through ``stages/base.py::CompilationStage.run``)
writes ``<output_dir>/ir_dumps/NNN_<name>_<before|after>.mlir`` plus a
per-entry row in ``index.json``.

The final glued module is written as ``final.mlir`` by
:meth:`IRDumpWriter.write_final`.

Hashing: the module is serialized once via xDSL ``Printer``; the SHA-256
of the serialized text is cached on the writer by ``(id(module),
counter_snapshot)`` so repeated ``after`` dumps do not rehash.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer

from compgen.trace.bus import get_active_bus
from compgen.trace.events import EventKind, Phase

log = structlog.get_logger()

ENV_DUMP = "COMPGEN_DUMP_IR"


def dump_enabled_from_env() -> bool:
    value = os.environ.get(ENV_DUMP, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _print_module(module: ModuleOp) -> str:
    buf = io.StringIO()
    Printer(stream=buf).print_op(module)
    return buf.getvalue()


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class IRDumpEntry:
    index: int
    name: str
    phase: str
    path: str
    ir_hash: str
    duration_ms: float = 0.0
    trace_event_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class IRDumpWriter:
    """Single-file IR dump writer.

    Not thread-safe-free — the lock serializes the counter and the
    ``index.json`` rewrite so two concurrent passes can't collide on a
    sequence number. For single-threaded compiles (the common case) the
    lock is uncontested.
    """

    def __init__(self, output_dir: Path, *, enabled: bool) -> None:
        self.output_dir = Path(output_dir) / "ir_dumps"
        self.enabled = enabled
        self._counter = 0
        self._lock = threading.Lock()
        self._entries: list[IRDumpEntry] = []
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def dump(
        self,
        name: str,
        phase: str,
        module: ModuleOp,
        *,
        duration_ms: float = 0.0,
        trace_event_id: str = "",
        meta: dict[str, Any] | None = None,
    ) -> tuple[Path | None, str]:
        """Write one IR file. Returns ``(path, ir_hash)``.

        Returns ``(None, "")`` when dumping is disabled or the module is
        None. ``phase`` is one of ``"before"`` / ``"after"`` / ``"entry"``
        / ``"exit"``.
        """
        if not self.enabled or module is None:
            return None, ""
        safe_name = name.replace("/", "_").replace(" ", "_")
        try:
            text = _print_module(module)
        except Exception as exc:  # noqa: BLE001
            log.debug("ir_dump.print_failed", name=safe_name, error=str(exc))
            return None, ""
        ir_hash = _hash_text(text)
        with self._lock:
            self._counter += 1
            idx = self._counter
        filename = f"{idx:04d}_{safe_name}_{phase}.mlir"
        out_path = self.output_dir / filename
        try:
            out_path.write_text(text)
        except OSError as exc:
            log.warning("ir_dump.write_failed", path=str(out_path), error=str(exc))
            return None, ir_hash
        entry = IRDumpEntry(
            index=idx,
            name=safe_name,
            phase=phase,
            path=str(out_path.relative_to(self.output_dir.parent)),
            ir_hash=ir_hash,
            duration_ms=float(duration_ms),
            trace_event_id=trace_event_id,
            meta=meta or {},
        )
        with self._lock:
            self._entries.append(entry)
            self._flush_index_locked()
        bus = get_active_bus()
        if bus is not None:
            bus.publish(
                kind=EventKind.IR_DUMP.value,
                phase=Phase.POINT.value,
                payload={
                    "index": idx,
                    "name": safe_name,
                    "phase_tag": phase,
                    "path": entry.path,
                    "ir_hash": ir_hash,
                    "duration_ms": float(duration_ms),
                    "span_id": trace_event_id,
                },
            )
        return out_path, ir_hash

    def write_final(self, module: ModuleOp) -> Path | None:
        if not self.enabled or module is None:
            return None
        try:
            text = _print_module(module)
        except Exception as exc:  # noqa: BLE001
            log.debug("ir_dump.final_print_failed", error=str(exc))
            return None
        final_path = self.output_dir / "final.mlir"
        try:
            final_path.write_text(text)
        except OSError as exc:
            log.warning("ir_dump.final_write_failed", error=str(exc))
            return None
        bus = get_active_bus()
        if bus is not None:
            bus.publish(
                kind=EventKind.IR_DUMP.value,
                phase=Phase.POINT.value,
                payload={
                    "name": "final",
                    "phase_tag": "final",
                    "path": str(final_path.relative_to(self.output_dir.parent)),
                    "ir_hash": _hash_text(text),
                },
            )
        return final_path

    def _flush_index_locked(self) -> None:
        index_path = self.output_dir / "index.json"
        payload = {
            "count": len(self._entries),
            "entries": [
                {
                    "index": e.index,
                    "name": e.name,
                    "phase": e.phase,
                    "path": e.path,
                    "ir_hash": e.ir_hash,
                    "duration_ms": e.duration_ms,
                    "trace_event_id": e.trace_event_id,
                    "meta": e.meta,
                }
                for e in self._entries
            ],
        }
        tmp = index_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, default=str))
            tmp.replace(index_path)
        except OSError as exc:
            log.warning("ir_dump.index_write_failed", error=str(exc))

    @property
    def entries(self) -> list[IRDumpEntry]:
        return list(self._entries)


# ---------------------------------------------------------------------------
# Process-wide active writer (installed by api.compile_model)
# ---------------------------------------------------------------------------

_ACTIVE: IRDumpWriter | None = None
_ACTIVE_LOCK = threading.Lock()


def install_ir_dump_writer(writer: IRDumpWriter | None) -> None:
    global _ACTIVE
    with _ACTIVE_LOCK:
        _ACTIVE = writer


def get_ir_dump_writer() -> IRDumpWriter | None:
    with _ACTIVE_LOCK:
        return _ACTIVE


__all__ = [
    "IRDumpEntry",
    "IRDumpWriter",
    "dump_enabled_from_env",
    "get_ir_dump_writer",
    "install_ir_dump_writer",
]
