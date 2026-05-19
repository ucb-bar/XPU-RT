"""Strategy DB — running confidence/usage stats per :class:`StateVector`.

Lives on disk at ``<card>/strategies.json``, one document per target.
Each strategy row records what action worked for a state vector, how
many times it has worked, the running mean speedup, and the last time
it was applied. The agent loop bumps these stats every time a
candidate is accepted; the prompt builder reads the top-K entries for
a state and folds them into the next propose request.

The DB is a thin JSON-on-disk store. No SQLite, no embeddings, no
locks beyond an atomic write through ``.tmp``. The volume per target
is bounded (a few hundred rows max in practice).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from xpu_rt.kernels.kernelblaster_v2.contract_state import StateVector
from xpu_rt.memory.target_knowledge import TargetKnowledgeCard


@dataclass(frozen=True)
class StrategyEntry:
    """One (state, action) → (confidence, usage_count, gain) row."""

    state_key: str
    action: str
    sample_count: int = 0
    accepted_count: int = 0
    mean_speedup: float = 1.0
    last_applied_at: str = ""
    notes: str = ""

    @property
    def confidence(self) -> float:
        """Beta-style confidence: accepted / (sample + 1)."""
        return self.accepted_count / max(self.sample_count + 1, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            **dataclasses.asdict(self),
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> StrategyEntry:
        return cls(
            state_key=str(body["state_key"]),
            action=str(body["action"]),
            sample_count=int(body.get("sample_count", 0)),
            accepted_count=int(body.get("accepted_count", 0)),
            mean_speedup=float(body.get("mean_speedup", 1.0)),
            last_applied_at=str(body.get("last_applied_at", "")),
            notes=str(body.get("notes", "")),
        )


@dataclass
class StrategyDB:
    """Per-card strategy database.

    Construct via :meth:`for_card`. Reads from ``card.strategies_path``
    if it exists, otherwise starts empty. Persists on every
    :meth:`record` so a crash mid-loop still preserves what was seen.
    """

    path: Path
    rows: dict[tuple[str, str], StrategyEntry] = field(default_factory=dict)

    @classmethod
    def for_card(cls, card: TargetKnowledgeCard) -> StrategyDB:
        db = cls(path=card.strategies_path)
        db.reload()
        return db

    def reload(self) -> None:
        self.rows.clear()
        if not self.path.exists():
            return
        try:
            body = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for raw in body.get("rows", []):
            entry = StrategyEntry.from_dict(raw)
            self.rows[(entry.state_key, entry.action)] = entry

    def save(self) -> None:
        payload = {
            "version": "xpu_rt_strategy_db_v1",
            "rows": [r.to_dict() for r in sorted(self.rows.values(), key=lambda r: r.state_key)],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)

    # ------------------------------------------------------------------ ops

    def record(
        self,
        *,
        state: StateVector,
        action: str,
        accepted: bool,
        speedup: float,
        notes: str = "",
    ) -> StrategyEntry:
        """Bump stats for ``(state, action)`` and return the updated row.

        ``speedup`` is folded into a running mean over accepted samples
        only — rejected samples bump ``sample_count`` but not
        ``mean_speedup``.
        """
        key = (state.key(), action)
        prior = self.rows.get(key)
        if prior is None:
            sample_count = 1
            accepted_count = 1 if accepted else 0
            mean = speedup if accepted else 1.0
        else:
            sample_count = prior.sample_count + 1
            accepted_count = prior.accepted_count + (1 if accepted else 0)
            if accepted:
                n = accepted_count
                mean = ((prior.mean_speedup * max(n - 1, 1)) + speedup) / max(n, 1)
            else:
                mean = prior.mean_speedup
        entry = StrategyEntry(
            state_key=state.key(),
            action=action,
            sample_count=sample_count,
            accepted_count=accepted_count,
            mean_speedup=mean,
            last_applied_at=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )
        self.rows[key] = entry
        self.save()
        return entry

    def top_for_state(
        self,
        state: StateVector,
        *,
        limit: int = 5,
    ) -> list[StrategyEntry]:
        """Return the top-``limit`` strategies for ``state`` by mean speedup.

        Ties broken by accepted_count, then by recency. Only entries
        with at least one acceptance are returned — pure-failure rows
        are filtered out so the prompt builder doesn't waste tokens
        on them.
        """
        matches = [
            r
            for r in self.rows.values()
            if r.state_key == state.key() and r.accepted_count > 0
        ]
        matches.sort(
            key=lambda r: (-r.mean_speedup, -r.accepted_count, r.last_applied_at),
            reverse=False,
        )
        return matches[:limit]

    def all_states(self) -> Iterable[str]:
        return {r.state_key for r in self.rows.values()}
