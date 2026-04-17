"""Promotion and artifact persistence for synthesized guards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from compgen.semantic.synthesis.guard_lang import Expr, expr_from_json, expr_to_json


@dataclass(frozen=True)
class GuardArtifact:
    """Persisted synthesized guard artifact."""

    guard_key: str
    transform_family: str
    guard_kind: str
    target_class: str = ""
    fragments: tuple[Expr, ...] = field(default_factory=tuple)
    sound_fragments: int = 0
    precise_unsound_fragments: int = 0
    repaired_fragments: int = 0
    proved_sound: bool = False
    proof_status: str = "unknown"
    verification_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "guard_key": self.guard_key,
            "transform_family": self.transform_family,
            "guard_kind": self.guard_kind,
            "target_class": self.target_class,
            "fragments": [expr_to_json(fragment) for fragment in self.fragments],
            "sound_fragments": self.sound_fragments,
            "precise_unsound_fragments": self.precise_unsound_fragments,
            "repaired_fragments": self.repaired_fragments,
            "proved_sound": self.proved_sound,
            "proof_status": self.proof_status,
            "verification_time_ms": self.verification_time_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> GuardArtifact:
        return cls(
            guard_key=str(data["guard_key"]),
            transform_family=str(data["transform_family"]),
            guard_kind=str(data.get("guard_kind", "legality")),
            target_class=str(data.get("target_class", "")),
            fragments=tuple(expr_from_json(item) for item in data.get("fragments", [])),
            sound_fragments=int(data.get("sound_fragments", 0)),
            precise_unsound_fragments=int(data.get("precise_unsound_fragments", 0)),
            repaired_fragments=int(data.get("repaired_fragments", 0)),
            proved_sound=bool(data.get("proved_sound", False)),
            proof_status=str(data.get("proof_status", "unknown")),
            verification_time_ms=float(data.get("verification_time_ms", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


def _default_guard_key(
    transform_family: str,
    guard_kind: str,
    target_class: str,
    fragments: Sequence[Expr],
) -> str:
    payload = json.dumps(
        {
            "family": transform_family,
            "kind": guard_kind,
            "target": target_class,
            "fragments": [expr_to_json(fragment) for fragment in fragments],
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    target_suffix = target_class or "generic"
    return f"guard.{transform_family}.{guard_kind}.{target_suffix}.{digest}"


def promote_guard(
    out_dir: Path,
    *,
    transform_family: str,
    guard_kind: str,
    fragments: Sequence[Expr],
    target_class: str = "",
    guard_key: str | None = None,
    sound_fragments: int = 0,
    precise_unsound_fragments: int = 0,
    repaired_fragments: int = 0,
    proved_sound: bool = False,
    proof_status: str = "unknown",
    verification_time_ms: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> GuardArtifact:
    """Persist a synthesized guard artifact on disk."""

    resolved_key = guard_key or _default_guard_key(transform_family, guard_kind, target_class, fragments)
    artifact = GuardArtifact(
        guard_key=resolved_key,
        transform_family=transform_family,
        guard_kind=guard_kind,
        target_class=target_class,
        fragments=tuple(fragments),
        sound_fragments=sound_fragments,
        precise_unsound_fragments=precise_unsound_fragments,
        repaired_fragments=repaired_fragments,
        proved_sound=proved_sound,
        proof_status=proof_status,
        verification_time_ms=verification_time_ms,
        metadata=dict(metadata or {}),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{resolved_key}.json"
    path.write_text(json.dumps(artifact.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def load_guard_artifact(path: str | Path) -> GuardArtifact:
    """Load a guard artifact from disk."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GuardArtifact.from_json_dict(data)


__all__ = ["GuardArtifact", "load_guard_artifact", "promote_guard"]
