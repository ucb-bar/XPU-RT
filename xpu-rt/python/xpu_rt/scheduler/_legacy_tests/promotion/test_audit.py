"""Tests for promotion/audit.py -- audit trail."""

from __future__ import annotations

from pathlib import Path

from compgen.promotion.audit import AuditEvent, AuditLog, create_event


def test_audit_event_construction() -> None:
    event = AuditEvent(
        event_id="evt-001",
        event_type="promotion",
        timestamp="2025-01-15T12:00:00Z",
        data={"recipe_key": "abc123"},
        actor="system",
    )
    assert event.event_id == "evt-001"
    assert event.event_type == "promotion"


def test_audit_event_defaults() -> None:
    event = AuditEvent(event_id="evt-002", event_type="generation", timestamp="2025-01-15T13:00:00Z")
    assert event.data == {}
    assert event.actor == "system"


def test_create_event() -> None:
    event = create_event("promotion", data={"key": "abc"}, actor="test")
    assert event.event_type == "promotion"
    assert event.data == {"key": "abc"}
    assert event.actor == "test"
    assert event.event_id  # non-empty
    assert event.timestamp  # non-empty


def test_audit_log_record_and_query(tmp_path: Path) -> None:
    log = AuditLog(log_path=tmp_path / "audit.jsonl")
    e1 = create_event("promotion", data={"v": 1})
    e2 = create_event("generation", data={"v": 2})
    log.record(e1)
    log.record(e2)

    all_events = log.query()
    assert len(all_events) == 2

    promotions = log.query(event_type="promotion")
    assert len(promotions) == 1
    assert promotions[0].event_type == "promotion"


def test_audit_log_query_empty(tmp_path: Path) -> None:
    log = AuditLog(log_path=tmp_path / "nonexistent.jsonl")
    assert log.query() == []


def test_audit_log_replay(tmp_path: Path) -> None:
    log = AuditLog(log_path=tmp_path / "audit.jsonl")
    event = create_event("test", data={"foo": "bar"})
    log.record(event)

    result = log.replay(event.event_id)
    assert "event" in result
    assert result["event"]["data"]["foo"] == "bar"


def test_audit_log_replay_missing(tmp_path: Path) -> None:
    log = AuditLog(log_path=tmp_path / "audit.jsonl")
    result = log.replay("nonexistent")
    assert "error" in result
