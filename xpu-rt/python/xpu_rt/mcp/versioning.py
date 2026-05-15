"""H5 — schema versioning + tool_schema_log (Section 11 Dream 5).

Every MCP tool dict carries a ``schema_version`` (default ``"v1"``)
and a derived ``schema_hash`` (canonical-JSON SHA-256 of the input
schema). A client can pin a version at session open; if the registry
later serves a different hash for the same (tool_id, schema_version)
pair, the audit fires a typed ``schema_hash_mismatch_on_pin``
violation.

The schema log is append-only JSONL; one row per
(tool_id, schema_hash, first_seen_commit) tuple.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def canonical_schema_hash(schema: dict[str, Any]) -> str:
    """SHA-256 of canonical-JSON schema (first 16 hex chars).

    Two schemas with identical content hash to the same value
    regardless of key order or whitespace.
    """

    try:
        blob = json.dumps(schema, sort_keys=True, default=str).encode("utf-8")
    except Exception:  # noqa: BLE001
        blob = repr(sorted(schema.items())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class ToolSchemaPin:
    """One session-level schema pin.

    A pin says: "this session wants ``tool_id`` at exactly
    ``schema_version`` + ``schema_hash``". Dispatch refuses if the
    served schema's hash doesn't match the pinned hash.
    """

    tool_id: str
    schema_version: str
    schema_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
        }


@dataclass
class ToolSchemaLog:
    """Append-only ledger of every (tool_id, schema_version, hash) seen.

    Persisted to ``results/tool_evidence_pack/tool_schema_log.json``
    by callers; in-memory by default.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)

    def append(
        self,
        *,
        tool_id: str,
        schema_version: str,
        schema_hash: str,
        first_seen_commit: str = "",
    ) -> bool:
        """Record a new entry; idempotent on (tool_id, hash) pairs.

        Returns True if a new row was added; False if a row with the
        same (tool_id, schema_hash) already exists.
        """

        for row in self.rows:
            if row["tool_id"] == tool_id and row["schema_hash"] == schema_hash:
                return False
        self.rows.append(
            {
                "tool_id": tool_id,
                "schema_version": schema_version,
                "schema_hash": schema_hash,
                "first_seen_commit": first_seen_commit,
            }
        )
        return True

    def lookup(self, *, tool_id: str, schema_version: str) -> list[dict[str, Any]]:
        """Return all rows for ``(tool_id, schema_version)``."""

        return [
            r
            for r in self.rows
            if r["tool_id"] == tool_id and r["schema_version"] == schema_version
        ]

    def write(self, path: Path) -> None:
        """Persist the log to ``path`` as pretty JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": self.rows}, indent=2, sort_keys=True))

    @classmethod
    def read(cls, path: Path) -> "ToolSchemaLog":
        if not path.exists():
            return cls()
        blob = json.loads(path.read_text())
        return cls(rows=list(blob.get("rows", [])))


def annotate_tool_with_schema_version(tool: dict[str, Any]) -> dict[str, Any]:
    """Return ``tool`` with ``schema_version`` + ``schema_hash`` ensured.

    Defaults: ``schema_version="v1"``, ``schema_hash`` derived from
    the tool's ``input_schema`` (empty dict if absent). Mutates in
    place AND returns for chaining.
    """

    tool.setdefault("schema_version", "v1")
    if "schema_hash" not in tool:
        tool["schema_hash"] = canonical_schema_hash(tool.get("input_schema", {}))
    return tool


def detect_pin_mismatch(
    *,
    pin: ToolSchemaPin,
    served_hash: str,
) -> str | None:
    """Return ``"schema_hash_mismatch_on_pin"`` if the served schema's
    hash differs from the pin's hash; ``None`` when they match.
    """

    if pin.schema_hash != served_hash:
        return "schema_hash_mismatch_on_pin"
    return None


__all__ = [
    "ToolSchemaLog",
    "ToolSchemaPin",
    "annotate_tool_with_schema_version",
    "canonical_schema_hash",
    "detect_pin_mismatch",
]
