"""Realness contracts — machine-readable per-feature claim records.

A realness contract names a single feature, the claim it makes, and the
evidence it requires. The contract is the *canonical* claim; the audit
proves or rejects it.

Realness levels (ascending strength):

- ``schema_only``        format exists, not consumed (paper-claimable: no)
- ``write_only``         artifact emitted, not used downstream            (no)
- ``read_only``          artifact consumed but does not affect behavior   (limited)
- ``decision_affecting`` artifact changes candidate/pass choice           (yes)
- ``production_path``    affects real end-to-end run                      (yes)
- ``hardware_backed``    exercised with real kernel/profile/runtime evidence (strongest)

A feature is not "done" until its declared level is reached by an
end-to-end test recorded in the trust report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from compgen.audit.errors import RealnessContractError

REALNESS_LEVELS: tuple[str, ...] = (
    "schema_only",
    "write_only",
    "read_only",
    "decision_affecting",
    "production_path",
    "hardware_backed",
)

PAPER_CLAIMABLE_LEVELS: frozenset[str] = frozenset(
    {"decision_affecting", "production_path", "hardware_backed"}
)

_FEATURE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class RealnessContract:
    """One feature's realness contract.

    Fields mirror the YAML schema. Validation happens in
    :func:`validate_contract` so that loading a contract never silently
    accepts a malformed entry.
    """

    feature_id: str
    claim: str
    realness_level: str
    forbidden: tuple[str, ...]
    required_evidence: tuple[str, ...]
    commit: str
    created_at_utc: str
    source_path: Path | None = None
    notes: str = ""

    @property
    def is_paper_claimable(self) -> bool:
        return self.realness_level in PAPER_CLAIMABLE_LEVELS

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "claim": self.claim,
            "realness_level": self.realness_level,
            "forbidden": list(self.forbidden),
            "required_evidence": list(self.required_evidence),
            "commit": self.commit,
            "created_at_utc": self.created_at_utc,
            "notes": self.notes,
        }


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_ts(value: Any) -> str:
    """YAML safe_load parses ISO-8601 timestamps as datetime objects.

    Coerce back to the canonical 'YYYY-MM-DDTHH:MM:SSZ' string form so
    validation and round-trip are stable.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(value)


def _require_field(data: dict[str, Any], key: str, source: Path | None) -> Any:
    if key not in data:
        raise RealnessContractError(
            f"realness contract missing required field '{key}'"
            + (f" in {source}" if source else "")
        )
    value = data[key]
    if value is None or value == "":
        raise RealnessContractError(
            f"realness contract field '{key}' is empty"
            + (f" in {source}" if source else "")
        )
    return value


def _require_list(data: dict[str, Any], key: str, source: Path | None) -> list[str]:
    value = data.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise RealnessContractError(
            f"realness contract field '{key}' must be a list"
            + (f" in {source}" if source else "")
        )
    out: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry:
            raise RealnessContractError(
                f"realness contract field '{key}' must contain non-empty strings"
                + (f" in {source}" if source else "")
            )
        out.append(entry)
    return out


def validate_contract(contract: RealnessContract) -> None:
    """Raise :class:`RealnessContractError` if the contract is malformed."""
    if not _FEATURE_ID_RE.match(contract.feature_id):
        raise RealnessContractError(
            f"feature_id {contract.feature_id!r} must match {_FEATURE_ID_RE.pattern}"
        )
    if contract.realness_level not in REALNESS_LEVELS:
        raise RealnessContractError(
            f"realness_level {contract.realness_level!r} must be one of {REALNESS_LEVELS}"
        )
    if not contract.claim or len(contract.claim.strip()) < 8:
        raise RealnessContractError(
            f"claim must be a non-trivial sentence; got {contract.claim!r}"
        )
    if not contract.required_evidence:
        raise RealnessContractError(
            f"contract {contract.feature_id} declares no required_evidence"
        )
    # Timestamp must parse as ISO-8601 UTC.
    try:
        datetime.strptime(contract.created_at_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise RealnessContractError(
            f"created_at_utc {contract.created_at_utc!r} must be 'YYYY-MM-DDTHH:MM:SSZ'"
        ) from exc


def load_contract(path: Path) -> RealnessContract:
    """Load and validate a single realness contract YAML file."""
    if not path.exists():
        raise RealnessContractError(f"realness contract not found: {path}")
    text = path.read_text()
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise RealnessContractError(f"realness contract must be a YAML mapping: {path}")
    contract = RealnessContract(
        feature_id=str(_require_field(raw, "feature_id", path)),
        claim=str(_require_field(raw, "claim", path)).strip(),
        realness_level=str(_require_field(raw, "realness_level", path)),
        forbidden=tuple(_require_list(raw, "forbidden", path)),
        required_evidence=tuple(_require_list(raw, "required_evidence", path)),
        commit=str(_require_field(raw, "commit", path)),
        created_at_utc=_coerce_ts(_require_field(raw, "created_at_utc", path)),
        source_path=path,
        notes=str(raw.get("notes", "")),
    )
    validate_contract(contract)
    return contract


def iter_contracts(root: Path) -> Iterator[RealnessContract]:
    """Yield every realness contract under ``root`` (sorted by feature_id)."""
    if not root.exists():
        return
    contracts: list[RealnessContract] = []
    for path in sorted(root.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        contracts.append(load_contract(path))
    contracts.sort(key=lambda c: c.feature_id)
    yield from contracts


def write_contract(contract: RealnessContract, path: Path) -> None:
    """Write a contract to YAML and re-validate by round-trip."""
    validate_contract(contract)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(contract.to_dict(), sort_keys=True))
    # Round-trip validation.
    reloaded = load_contract(path)
    if reloaded.to_dict() != contract.to_dict():
        raise RealnessContractError(
            f"round-trip mismatch writing contract {contract.feature_id} to {path}"
        )


def make_contract(
    *,
    feature_id: str,
    claim: str,
    realness_level: str,
    forbidden: list[str] | tuple[str, ...] = (),
    required_evidence: list[str] | tuple[str, ...] = (),
    commit: str,
    created_at_utc: str | None = None,
    notes: str = "",
) -> RealnessContract:
    """Construct + validate a contract programmatically."""
    contract = RealnessContract(
        feature_id=feature_id,
        claim=claim.strip(),
        realness_level=realness_level,
        forbidden=tuple(forbidden),
        required_evidence=tuple(required_evidence),
        commit=commit,
        created_at_utc=created_at_utc or _utc_now(),
        notes=notes,
    )
    validate_contract(contract)
    return contract
