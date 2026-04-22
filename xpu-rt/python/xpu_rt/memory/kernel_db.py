"""Kernel performance + fusion-decision history (SQLite-backed).

Companion to ``compgen.memory.store.CompilerMemory`` — that one tracks
tasks/candidates/evaluations/knowledge for the broader agent loop;
this one tracks two narrow questions:

  1. **kernel_perf**  — for each (target, op_family, fingerprint),
     what's the best-known measured perf? The kernel_optimizer reads
     this to decide whether to skip codegen entirely (cache hit) or
     escalate to autocomp (cached-best is too slow).

  2. **fusion_decisions** — for each (target, producer_role,
     consumer_role) pair, the running history of fusion decisions and
     their *observed* speedups. The fusion-oracle's cost model
     calibrates against this — if the oracle predicted 1.5× and we
     measured 0.9×, the next prediction adjusts.

Lives at ``~/.compgen/kernel_db.sqlite`` (overridable via
``COMPGEN_KERNEL_DB``). Single-file, single-writer assumed; SQLite's
WAL mode makes concurrent read safe.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


def default_db_path() -> Path:
    override = os.environ.get("COMPGEN_KERNEL_DB")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".compgen" / "kernel_db.sqlite"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelPerfRecord:
    target: str
    op_family: str
    fingerprint: str
    perf_us: float
    correctness_passed: bool
    source_path: str = ""
    measured_at: float = 0.0  # unix ts
    notes: str = ""


@dataclass(frozen=True)
class FusionDecisionRecord:
    target: str
    producer_role: str
    consumer_role: str
    decision: str  # "fuse" | "dont_fuse" | "ineligible"
    predicted_speedup: float
    observed_speedup: float | None  # None = not yet measured
    measured_at: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS kernel_perf (
    target              TEXT NOT NULL,
    op_family           TEXT NOT NULL,
    fingerprint         TEXT NOT NULL,
    perf_us             REAL NOT NULL,
    correctness_passed  INTEGER NOT NULL,
    source_path         TEXT NOT NULL DEFAULT '',
    measured_at         REAL NOT NULL,
    notes               TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (target, op_family, fingerprint, measured_at)
);

CREATE INDEX IF NOT EXISTS kernel_perf_lookup
    ON kernel_perf(target, op_family, fingerprint);

CREATE TABLE IF NOT EXISTS fusion_decisions (
    target              TEXT NOT NULL,
    producer_role       TEXT NOT NULL,
    consumer_role       TEXT NOT NULL,
    decision            TEXT NOT NULL,
    predicted_speedup   REAL NOT NULL,
    observed_speedup    REAL,
    measured_at         REAL NOT NULL,
    notes               TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (target, producer_role, consumer_role, measured_at)
);

CREATE INDEX IF NOT EXISTS fusion_decisions_lookup
    ON fusion_decisions(target, producer_role, consumer_role);
"""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class KernelDB:
    """Thin SQLite wrapper for the two tables."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ----- kernel_perf -----

    def record_kernel_perf(self, rec: KernelPerfRecord) -> None:
        self._conn.execute(
            """INSERT INTO kernel_perf
               (target, op_family, fingerprint, perf_us, correctness_passed,
                source_path, measured_at, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.target,
                rec.op_family,
                rec.fingerprint,
                rec.perf_us,
                1 if rec.correctness_passed else 0,
                rec.source_path,
                rec.measured_at or time.time(),
                rec.notes,
            ),
        )
        self._conn.commit()

    def best_kernel_perf(
        self,
        target: str,
        op_family: str,
        fingerprint: str,
    ) -> KernelPerfRecord | None:
        """Lowest-perf_us record that passed correctness, or ``None``."""
        cur = self._conn.execute(
            """SELECT target, op_family, fingerprint, perf_us, correctness_passed,
                      source_path, measured_at, notes
               FROM kernel_perf
               WHERE target=? AND op_family=? AND fingerprint=? AND correctness_passed=1
               ORDER BY perf_us ASC
               LIMIT 1""",
            (target, op_family, fingerprint),
        )
        row = cur.fetchone()
        return _row_to_kernel_perf(row) if row else None

    def list_kernel_perf(
        self,
        target: str | None = None,
        op_family: str | None = None,
    ) -> list[KernelPerfRecord]:
        sql = "SELECT target, op_family, fingerprint, perf_us, correctness_passed, source_path, measured_at, notes FROM kernel_perf"
        params: list = []
        wheres: list[str] = []
        if target is not None:
            wheres.append("target=?")
            params.append(target)
        if op_family is not None:
            wheres.append("op_family=?")
            params.append(op_family)
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)
        sql += " ORDER BY measured_at DESC"
        return [_row_to_kernel_perf(r) for r in self._conn.execute(sql, params)]

    # ----- fusion_decisions -----

    def record_fusion_decision(self, rec: FusionDecisionRecord) -> None:
        self._conn.execute(
            """INSERT INTO fusion_decisions
               (target, producer_role, consumer_role, decision,
                predicted_speedup, observed_speedup, measured_at, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.target,
                rec.producer_role,
                rec.consumer_role,
                rec.decision,
                rec.predicted_speedup,
                rec.observed_speedup,
                rec.measured_at or time.time(),
                rec.notes,
            ),
        )
        self._conn.commit()

    def fusion_history(
        self,
        target: str,
        producer_role: str,
        consumer_role: str,
    ) -> list[FusionDecisionRecord]:
        cur = self._conn.execute(
            """SELECT target, producer_role, consumer_role, decision,
                      predicted_speedup, observed_speedup, measured_at, notes
               FROM fusion_decisions
               WHERE target=? AND producer_role=? AND consumer_role=?
               ORDER BY measured_at DESC""",
            (target, producer_role, consumer_role),
        )
        return [_row_to_fusion_decision(r) for r in cur]

    def average_observed_speedup(
        self,
        target: str,
        producer_role: str,
        consumer_role: str,
    ) -> float | None:
        """Mean of ``observed_speedup`` over the history, ignoring NULL.

        Used by the fusion oracle to calibrate predictions: if past
        observations cluster at 0.9× when we predicted 1.5×, the next
        prediction should weight that down.
        """
        cur = self._conn.execute(
            """SELECT AVG(observed_speedup) FROM fusion_decisions
               WHERE target=? AND producer_role=? AND consumer_role=?
                     AND observed_speedup IS NOT NULL""",
            (target, producer_role, consumer_role),
        )
        row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else None


def _row_to_kernel_perf(row) -> KernelPerfRecord:
    return KernelPerfRecord(
        target=row[0],
        op_family=row[1],
        fingerprint=row[2],
        perf_us=row[3],
        correctness_passed=bool(row[4]),
        source_path=row[5],
        measured_at=row[6],
        notes=row[7],
    )


def _row_to_fusion_decision(row) -> FusionDecisionRecord:
    return FusionDecisionRecord(
        target=row[0],
        producer_role=row[1],
        consumer_role=row[2],
        decision=row[3],
        predicted_speedup=row[4],
        observed_speedup=row[5],
        measured_at=row[6],
        notes=row[7],
    )


# ---------------------------------------------------------------------------
# Singleton (mirrors KernelStore / KnowledgeStore)
# ---------------------------------------------------------------------------


_singleton: KernelDB | None = None


def shared_db() -> KernelDB:
    global _singleton
    if _singleton is None:
        _singleton = KernelDB()
    return _singleton


def set_shared_db(db: KernelDB | None) -> None:
    global _singleton
    if _singleton is not None and db is not _singleton:
        _singleton.close()
    _singleton = db


__all__ = [
    "FusionDecisionRecord",
    "KernelDB",
    "KernelPerfRecord",
    "default_db_path",
    "set_shared_db",
    "shared_db",
]
