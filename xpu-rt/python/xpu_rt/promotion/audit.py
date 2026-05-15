"""Audit trail and replay.

Every significant event in CompGen is recorded:
- LLM generation calls (via llm.recorder)
- Verification results
- Promotion events
- Cache operations
- Bundle creation

The audit log enables:
- Reproducibility (replay exact pipeline runs)
- Debugging (trace what happened and why)
- Compliance (who generated what, when, with what model)

Invariants:
    - Audit entries are append-only (never modified or deleted).
    - Each entry has a unique ID and timestamp.
    - Entries are serializable to JSON.
    - The audit log can be queried by event type, time range, or key.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    """A single audit log entry.

    Attributes:
        event_id: Unique event identifier.
        event_type: Type of event (e.g., "generation", "verification", "promotion").
        timestamp: ISO 8601 timestamp.
        data: Event-specific data.
        actor: Who/what triggered the event (e.g., model ID, user, system).
    """

    event_id: str
    event_type: str
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)
    actor: str = "system"


def create_event(event_type: str, data: dict[str, Any] | None = None, actor: str = "system") -> AuditEvent:
    """Create a new AuditEvent with auto-generated ID and timestamp."""
    return AuditEvent(
        event_id=uuid.uuid4().hex[:12],
        event_type=event_type,
        timestamp=datetime.now(UTC).isoformat(),
        data=data or {},
        actor=actor,
    )


@dataclass
class AuditLog:
    """Append-only audit log backed by JSON-lines file.

    Attributes:
        log_path: Path to the audit log file.
    """

    log_path: Path

    def record(self, event: AuditEvent) -> None:
        """Append an event to the audit log."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(event), default=str) + "\n")

    def query(
        self,
        event_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[AuditEvent]:
        """Query the audit log with optional filters."""
        if not self.log_path.exists():
            return []

        events: list[AuditEvent] = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                event = AuditEvent(**data)

                if event_type and event.event_type != event_type:
                    continue
                if since:
                    event_time = datetime.fromisoformat(event.timestamp)
                    if event_time < since:
                        continue
                if until:
                    event_time = datetime.fromisoformat(event.timestamp)
                    if event_time > until:
                        continue

                events.append(event)

        return events

    def replay(self, event_id: str) -> dict[str, Any]:
        """Look up a recorded event by ID and return its data."""
        events = self.query()
        for event in events:
            if event.event_id == event_id:
                return {"event": asdict(event)}
        return {"error": f"Event {event_id} not found"}


__all__ = ["AuditEvent", "AuditLog", "create_event"]
