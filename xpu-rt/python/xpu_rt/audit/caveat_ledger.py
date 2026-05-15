"""Machine-readable caveat ledger.

Every caveat that affects a paper-claimable row must live here, not in
prose. Free-text caveats are rejected by the schema validator.

Schema (JSON):

.. code-block:: json

    {
      "caveats": [
        {
          "id": "portable_gate_single_target",
          "claim_affected": "portable_recipe_promotion",
          "status": "blocked_by_hardware",
          "is_bug": false,
          "blocks_paper_claim": false,
          "required_to_close":
            "run same region_signature on >=2 target_class values (need cuda)",
          "evidence_paths": [
            "docs/architecture/promotion-and-memory.md#m-30-honest-residual"
          ],
          "created_at_utc": "2026-05-05T00:00:00Z",
          "last_verified_at_utc": "2026-05-05T00:00:00Z"
        }
      ]
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from compgen.audit.errors import CaveatLedgerError, StaleCaveatError

CAVEAT_STATUSES: tuple[str, ...] = (
    "open",
    "blocked_by_hardware",
    "blocked_by_external",
    "resolved",
    "rejected",
)

# A caveat that has not been verified within this window is stale unless
# its status is ``resolved``.
DEFAULT_STALE_DAYS: int = 30

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime(_TS_FMT)


def _parse_ts(ts: str, *, field_name: str) -> datetime:
    try:
        return datetime.strptime(ts, _TS_FMT).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise CaveatLedgerError(
            f"caveat field {field_name!r}={ts!r} must be {_TS_FMT}"
        ) from exc


@dataclass(frozen=True)
class Caveat:
    """A single caveat row.

    A caveat documents a known limitation that a paper claim depends on.
    It is *not* a TODO. Every row must point at concrete evidence, declare
    whether it blocks a paper claim, and include the action required to
    close it.
    """

    id: str
    claim_affected: str
    status: str
    is_bug: bool
    blocks_paper_claim: bool
    required_to_close: str
    evidence_paths: tuple[str, ...]
    created_at_utc: str
    last_verified_at_utc: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim_affected": self.claim_affected,
            "status": self.status,
            "is_bug": self.is_bug,
            "blocks_paper_claim": self.blocks_paper_claim,
            "required_to_close": self.required_to_close,
            "evidence_paths": list(self.evidence_paths),
            "created_at_utc": self.created_at_utc,
            "last_verified_at_utc": self.last_verified_at_utc,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Caveat:
        for required in (
            "id",
            "claim_affected",
            "status",
            "is_bug",
            "blocks_paper_claim",
            "required_to_close",
            "evidence_paths",
            "created_at_utc",
            "last_verified_at_utc",
        ):
            if required not in data:
                raise CaveatLedgerError(
                    f"caveat missing required field {required!r}"
                )
        evidence = data.get("evidence_paths") or []
        if not isinstance(evidence, list) or not all(
            isinstance(e, str) and e for e in evidence
        ):
            raise CaveatLedgerError(
                f"caveat {data.get('id')!r}: evidence_paths must be a non-empty list of strings"
            )
        if not evidence:
            raise CaveatLedgerError(
                f"caveat {data.get('id')!r}: evidence_paths cannot be empty"
            )
        if not isinstance(data.get("is_bug"), bool):
            raise CaveatLedgerError(
                f"caveat {data.get('id')!r}: is_bug must be bool"
            )
        if not isinstance(data.get("blocks_paper_claim"), bool):
            raise CaveatLedgerError(
                f"caveat {data.get('id')!r}: blocks_paper_claim must be bool"
            )
        return cls(
            id=str(data["id"]),
            claim_affected=str(data["claim_affected"]),
            status=str(data["status"]),
            is_bug=bool(data["is_bug"]),
            blocks_paper_claim=bool(data["blocks_paper_claim"]),
            required_to_close=str(data["required_to_close"]),
            evidence_paths=tuple(evidence),
            created_at_utc=str(data["created_at_utc"]),
            last_verified_at_utc=str(data["last_verified_at_utc"]),
            notes=str(data.get("notes", "")),
        )

    def validate(self) -> None:
        if not _ID_RE.match(self.id):
            raise CaveatLedgerError(
                f"caveat id {self.id!r} must match {_ID_RE.pattern}"
            )
        if self.status not in CAVEAT_STATUSES:
            raise CaveatLedgerError(
                f"caveat {self.id}: status {self.status!r} must be one of {CAVEAT_STATUSES}"
            )
        if not self.required_to_close.strip():
            raise CaveatLedgerError(
                f"caveat {self.id}: required_to_close must be non-empty"
            )
        if not self.claim_affected.strip():
            raise CaveatLedgerError(
                f"caveat {self.id}: claim_affected must be non-empty"
            )
        if not self.evidence_paths or not all(
            isinstance(e, str) and e.strip() for e in self.evidence_paths
        ):
            raise CaveatLedgerError(
                f"caveat {self.id}: evidence_paths must be a non-empty list of strings"
            )
        _parse_ts(self.created_at_utc, field_name="created_at_utc")
        _parse_ts(self.last_verified_at_utc, field_name="last_verified_at_utc")
        if self.status == "resolved" and self.blocks_paper_claim:
            raise CaveatLedgerError(
                f"caveat {self.id}: status=resolved is incompatible with blocks_paper_claim=true"
            )

    def is_stale(self, *, now: datetime | None = None, stale_days: int = DEFAULT_STALE_DAYS) -> bool:
        if self.status == "resolved":
            return False
        ref = now or datetime.now(tz=timezone.utc)
        last = _parse_ts(self.last_verified_at_utc, field_name="last_verified_at_utc")
        return (ref - last) > timedelta(days=stale_days)


@dataclass
class CaveatLedger:
    """A loaded caveat ledger. Mutable; persist via :meth:`dump`."""

    caveats: list[Caveat] = field(default_factory=list)
    source_path: Path | None = None

    def __iter__(self) -> Iterator[Caveat]:
        return iter(self.caveats)

    def __len__(self) -> int:
        return len(self.caveats)

    def get(self, caveat_id: str) -> Caveat | None:
        for c in self.caveats:
            if c.id == caveat_id:
                return c
        return None

    def add(self, caveat: Caveat) -> None:
        caveat.validate()
        if self.get(caveat.id) is not None:
            raise CaveatLedgerError(f"caveat id {caveat.id!r} already present")
        self.caveats.append(caveat)

    def upsert(self, caveat: Caveat) -> None:
        caveat.validate()
        for i, existing in enumerate(self.caveats):
            if existing.id == caveat.id:
                self.caveats[i] = caveat
                return
        self.caveats.append(caveat)

    def update_status(
        self,
        caveat_id: str,
        *,
        status: str,
        evidence_path: str | None = None,
        now: str | None = None,
    ) -> Caveat:
        existing = self.get(caveat_id)
        if existing is None:
            raise CaveatLedgerError(f"caveat {caveat_id!r} not found")
        if status not in CAVEAT_STATUSES:
            raise CaveatLedgerError(f"status {status!r} must be in {CAVEAT_STATUSES}")
        new_evidence = list(existing.evidence_paths)
        if evidence_path and evidence_path not in new_evidence:
            new_evidence.append(evidence_path)
        updated = Caveat(
            id=existing.id,
            claim_affected=existing.claim_affected,
            status=status,
            is_bug=existing.is_bug,
            blocks_paper_claim=False if status == "resolved" else existing.blocks_paper_claim,
            required_to_close=existing.required_to_close,
            evidence_paths=tuple(new_evidence),
            created_at_utc=existing.created_at_utc,
            last_verified_at_utc=now or _utc_now(),
            notes=existing.notes,
        )
        updated.validate()
        self.upsert(updated)
        return updated

    def validate(self, *, allow_stale: bool = False, stale_days: int = DEFAULT_STALE_DAYS) -> None:
        seen: set[str] = set()
        for c in self.caveats:
            c.validate()
            if c.id in seen:
                raise CaveatLedgerError(f"duplicate caveat id {c.id!r}")
            seen.add(c.id)
            if not allow_stale and c.is_stale(stale_days=stale_days):
                raise StaleCaveatError(
                    f"caveat {c.id} is stale: last verified "
                    f"{c.last_verified_at_utc} (>{stale_days} days)"
                )

    def stale(self, *, stale_days: int = DEFAULT_STALE_DAYS) -> list[Caveat]:
        return [c for c in self.caveats if c.is_stale(stale_days=stale_days)]

    def to_dict(self) -> dict[str, Any]:
        return {"caveats": [c.to_dict() for c in self.caveats]}

    def dump(self, path: Path) -> None:
        self.validate(allow_stale=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        self.source_path = path

    @classmethod
    def load(cls, path: Path) -> CaveatLedger:
        if not path.exists():
            raise CaveatLedgerError(f"caveat ledger not found: {path}")
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or "caveats" not in raw:
            raise CaveatLedgerError(f"caveat ledger {path}: must be a mapping with 'caveats'")
        caveats_raw = raw["caveats"]
        if not isinstance(caveats_raw, list):
            raise CaveatLedgerError(f"caveat ledger {path}: 'caveats' must be a list")
        caveats = [Caveat.from_dict(c) for c in caveats_raw]
        ledger = cls(caveats=caveats, source_path=path)
        return ledger


def make_caveat(
    *,
    id: str,
    claim_affected: str,
    status: str,
    is_bug: bool,
    blocks_paper_claim: bool,
    required_to_close: str,
    evidence_paths: Iterable[str],
    created_at_utc: str | None = None,
    last_verified_at_utc: str | None = None,
    notes: str = "",
) -> Caveat:
    now = _utc_now()
    caveat = Caveat(
        id=id,
        claim_affected=claim_affected,
        status=status,
        is_bug=is_bug,
        blocks_paper_claim=blocks_paper_claim,
        required_to_close=required_to_close,
        evidence_paths=tuple(evidence_paths),
        created_at_utc=created_at_utc or now,
        last_verified_at_utc=last_verified_at_utc or now,
        notes=notes,
    )
    caveat.validate()
    return caveat
