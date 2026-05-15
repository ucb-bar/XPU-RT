"""H5 — schema versioning + ``tool_schema_log``.

Coverage:

1. ``canonical_schema_hash`` is byte-stable and key-order-insensitive.
2. Differing schemas hash to different values.
3. ``annotate_tool_with_schema_version`` defaults version to "v1" and
   computes the schema hash from ``input_schema``.
4. The ledger is idempotent: a duplicate ``append`` is a no-op.
5. ``lookup`` returns matching rows; non-matching returns empty.
6. ``detect_pin_mismatch`` returns the typed violation when hashes
   differ and ``None`` when they match.
7. The ledger round-trips through ``write`` / ``read``.
"""

from __future__ import annotations

from pathlib import Path

from compgen.mcp.versioning import (
    ToolSchemaLog,
    ToolSchemaPin,
    annotate_tool_with_schema_version,
    canonical_schema_hash,
    detect_pin_mismatch,
)


def test_canonical_schema_hash_stable() -> None:
    s1 = {"type": "object", "properties": {"x": {"type": "string"}}}
    s2 = {"properties": {"x": {"type": "string"}}, "type": "object"}
    assert canonical_schema_hash(s1) == canonical_schema_hash(s2)
    assert len(canonical_schema_hash(s1)) == 16


def test_canonical_schema_hash_distinguishes() -> None:
    s1 = {"type": "object"}
    s2 = {"type": "array"}
    assert canonical_schema_hash(s1) != canonical_schema_hash(s2)


def test_annotate_defaults_version_and_hash() -> None:
    tool = {"name": "echo", "input_schema": {"type": "object"}}
    annotated = annotate_tool_with_schema_version(tool)
    assert annotated["schema_version"] == "v1"
    assert len(annotated["schema_hash"]) == 16


def test_annotate_idempotent() -> None:
    tool = {"name": "echo", "schema_version": "v2", "schema_hash": "abc123"}
    annotated = annotate_tool_with_schema_version(tool)
    assert annotated["schema_version"] == "v2"
    assert annotated["schema_hash"] == "abc123"


def test_schema_log_append_idempotent() -> None:
    log = ToolSchemaLog()
    added1 = log.append(
        tool_id="echo", schema_version="v1", schema_hash="deadbeef00000000"
    )
    added2 = log.append(
        tool_id="echo", schema_version="v1", schema_hash="deadbeef00000000"
    )
    assert added1 is True
    assert added2 is False
    assert len(log.rows) == 1


def test_schema_log_lookup() -> None:
    log = ToolSchemaLog()
    log.append(tool_id="echo", schema_version="v1", schema_hash="h1")
    log.append(tool_id="echo", schema_version="v1", schema_hash="h2")
    log.append(tool_id="echo", schema_version="v2", schema_hash="h3")
    rows = log.lookup(tool_id="echo", schema_version="v1")
    assert len(rows) == 2
    assert log.lookup(tool_id="echo", schema_version="v9") == []


def test_detect_pin_mismatch() -> None:
    pin = ToolSchemaPin(tool_id="echo", schema_version="v1", schema_hash="h1")
    assert detect_pin_mismatch(pin=pin, served_hash="h1") is None
    assert (
        detect_pin_mismatch(pin=pin, served_hash="h2")
        == "schema_hash_mismatch_on_pin"
    )


def test_schema_log_round_trip(tmp_path: Path) -> None:
    log = ToolSchemaLog()
    log.append(tool_id="echo", schema_version="v1", schema_hash="h1")
    log.append(tool_id="apply_recipe", schema_version="v1", schema_hash="h2")
    out = tmp_path / "schema_log.json"
    log.write(out)
    assert out.exists()

    restored = ToolSchemaLog.read(out)
    assert len(restored.rows) == 2
    assert restored.lookup(tool_id="echo", schema_version="v1") == [
        r for r in log.rows if r["tool_id"] == "echo"
    ]


def test_schema_log_read_missing_path(tmp_path: Path) -> None:
    """Reading a non-existent log returns an empty ledger (graceful)."""

    restored = ToolSchemaLog.read(tmp_path / "missing.json")
    assert restored.rows == []
