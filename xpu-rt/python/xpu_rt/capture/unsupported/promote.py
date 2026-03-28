"""Promotion metadata for unsupported-op recoveries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from compgen.capture.unsupported.classify import UnsupportedClassification
from compgen.capture.unsupported.introspect import UnsupportedOpDossier


@dataclass(frozen=True)
class PromotionRecord:
    """Version-stamped identity for a recovery artifact."""

    cache_key: str
    runtime_versions: tuple[tuple[str, str], ...]
    policy: str = "cache-first"


def build_promotion_record(
    dossier: UnsupportedOpDossier,
    classification: UnsupportedClassification,
    runtime_versions: dict[str, str],
) -> PromotionRecord:
    """Create a stable cache key for a recovery under the current runtime."""

    digest = hashlib.sha256()
    digest.update(dossier.target.encode())
    digest.update(dossier.schema.encode())
    digest.update(classification.bucket.encode())
    for key, value in sorted(runtime_versions.items()):
        digest.update(key.encode())
        digest.update(value.encode())
    cache_key = digest.hexdigest()[:16]
    return PromotionRecord(
        cache_key=cache_key,
        runtime_versions=tuple(sorted(runtime_versions.items())),
    )


__all__ = ["PromotionRecord", "build_promotion_record"]
