"""SQLite backend for the Compiler Memory System (Layer B).

Stores relational metadata: tasks, candidates, evaluations,
knowledge items, promotions, state signatures, episode steps.

Designed for local-first operation (zero config, just a file).
Upgradeable to Postgres by swapping this module.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import structlog

log = structlog.get_logger()

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    task_kind TEXT NOT NULL,
    workload_key TEXT DEFAULT '',
    region_key TEXT DEFAULT '',
    target_key TEXT DEFAULT '',
    hardware_key TEXT DEFAULT '',
    objective TEXT DEFAULT 'latency',
    input_artifact_hash TEXT DEFAULT '',
    created_at TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS state_signatures (
    state_id TEXT PRIMARY KEY,
    task_id TEXT REFERENCES tasks(task_id),
    op_family TEXT DEFAULT '',
    shape_signature TEXT DEFAULT '',
    dtype_signature TEXT DEFAULT '',
    layout_signature TEXT DEFAULT '',
    hardware_signature TEXT DEFAULT '',
    memory_signature TEXT DEFAULT '',
    config_signature TEXT DEFAULT '',
    bottleneck_signature TEXT DEFAULT '',
    profile_signature TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    task_id TEXT REFERENCES tasks(task_id),
    artifact_hash TEXT DEFAULT '',
    parent_candidate_id TEXT DEFAULT '',
    generator_kind TEXT DEFAULT 'llm',
    generator_model TEXT DEFAULT '',
    generation_round INTEGER DEFAULT 0,
    state_signature_id TEXT DEFAULT '',
    status TEXT DEFAULT 'new',
    created_at TEXT DEFAULT '',
    metadata_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS evaluations (
    eval_id TEXT PRIMARY KEY,
    candidate_id TEXT REFERENCES candidates(candidate_id),
    backend TEXT DEFAULT '',
    compile_ok INTEGER DEFAULT 0,
    correctness_ok INTEGER DEFAULT 0,
    perf_ok INTEGER DEFAULT 0,
    score REAL DEFAULT 0.0,
    latency_us REAL DEFAULT 0.0,
    throughput REAL DEFAULT 0.0,
    energy REAL DEFAULT 0.0,
    verifier_summary TEXT DEFAULT '',
    profile_summary TEXT DEFAULT '',
    profile_artifact_hash TEXT DEFAULT '',
    created_at TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS knowledge_items (
    knowledge_id TEXT PRIMARY KEY,
    knowledge_kind TEXT DEFAULT 'optimization_tactic',
    scope_kind TEXT DEFAULT 'global',
    scope_key TEXT DEFAULT '',
    summary TEXT DEFAULT '',
    artifact_hash TEXT DEFAULT '',
    quality_score REAL DEFAULT 0.0,
    uses INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    failures INTEGER DEFAULT 0,
    last_used_at TEXT DEFAULT '',
    source TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS promotions (
    promotion_id TEXT PRIMARY KEY,
    candidate_id TEXT REFERENCES candidates(candidate_id),
    promotion_key TEXT DEFAULT '',
    version INTEGER DEFAULT 1,
    reason TEXT DEFAULT '',
    measured_gain REAL DEFAULT 0.0,
    verified_by TEXT DEFAULT '',
    created_at TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS episode_steps (
    step_id TEXT PRIMARY KEY,
    task_id TEXT REFERENCES tasks(task_id),
    candidate_id TEXT DEFAULT '',
    action TEXT DEFAULT '',
    reward REAL DEFAULT 0.0,
    step_number INTEGER DEFAULT 0,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    source_kind TEXT DEFAULT '',
    repo_url TEXT DEFAULT '',
    commit_hash TEXT DEFAULT '',
    path TEXT DEFAULT '',
    ingested_at TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_candidates_task ON candidates(task_id);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);
CREATE INDEX IF NOT EXISTS idx_evaluations_candidate ON evaluations(candidate_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_kind ON knowledge_items(knowledge_kind);
CREATE INDEX IF NOT EXISTS idx_knowledge_scope ON knowledge_items(scope_kind, scope_key);
CREATE INDEX IF NOT EXISTS idx_episode_task ON episode_steps(task_id);
CREATE INDEX IF NOT EXISTS idx_state_op_family ON state_signatures(op_family);
CREATE INDEX IF NOT EXISTS idx_state_hardware ON state_signatures(hardware_signature);
"""


class SQLiteBackend:
    """SQLite-backed metadata store.

    Attributes:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path = Path(".compgen_cache/memory.db")) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL statement."""
        return self._get_conn().execute(sql, params)

    def executemany(self, sql: str, params_list: list[tuple]) -> sqlite3.Cursor:
        """Execute a SQL statement for each param tuple."""
        return self._get_conn().executemany(sql, params_list)

    def commit(self) -> None:
        """Commit the current transaction."""
        if self._conn is not None:
            self._conn.commit()

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        """Execute and fetch one row."""
        return self.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute and fetch all rows."""
        return self.execute(sql, params).fetchall()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def table_count(self, table: str) -> int:
        """Count rows in a table."""
        row = self.fetchone(f"SELECT COUNT(*) as c FROM {table}")  # noqa: S608
        return row["c"] if row else 0


__all__ = ["SQLiteBackend"]
