"""ArtifactIndex actor — indexes generated compilation artifacts.

Stores metadata only — actual artifacts live on shared storage.
Provides lookup by (target, model, objective) triple.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from infra.ray._require import require_ray

ray = require_ray()


@dataclass
class ArtifactEntry:
    """Metadata for a stored artifact."""

    artifact_id: str
    artifact_type: str  # "bundle", "execution_plan", "kernel", "recipe"
    target_name: str
    model_hash: str = ""
    objective: str = ""
    storage_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "target_name": self.target_name,
            "model_hash": self.model_hash,
            "objective": self.objective,
            "storage_path": self.storage_path,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@ray.remote
class ArtifactIndexActor:
    """Index of all generated artifacts.

    Provides lookup by target, model hash, and objective.
    Stores metadata only — artifacts are on shared storage.
    """

    def __init__(self, storage_root: str = "/tmp/compgen_artifacts") -> None:
        self._index: dict[str, ArtifactEntry] = {}
        self._storage_root = storage_root

    def register_artifact(
        self,
        artifact_type: str,
        target_name: str,
        storage_path: str,
        *,
        model_hash: str = "",
        objective: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a new artifact.

        Returns:
            The generated artifact_id.
        """
        artifact_id = str(uuid.uuid4())
        entry = ArtifactEntry(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            target_name=target_name,
            model_hash=model_hash,
            objective=objective,
            storage_path=storage_path,
            metadata=metadata or {},
            created_at=datetime.now(UTC).isoformat(),
        )
        self._index[artifact_id] = entry
        return artifact_id

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Get artifact metadata by ID."""
        entry = self._index.get(artifact_id)
        return entry.to_dict() if entry else None

    def find_artifacts(
        self,
        *,
        target_name: str | None = None,
        model_hash: str | None = None,
        objective: str | None = None,
        artifact_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find artifacts matching the given criteria."""
        results: list[dict[str, Any]] = []
        for entry in self._index.values():
            if target_name and entry.target_name != target_name:
                continue
            if model_hash and entry.model_hash != model_hash:
                continue
            if objective and entry.objective != objective:
                continue
            if artifact_type and entry.artifact_type != artifact_type:
                continue
            results.append(entry.to_dict())
        return results

    def list_artifacts(self) -> list[dict[str, Any]]:
        """List all indexed artifacts."""
        return [e.to_dict() for e in self._index.values()]

    def delete_artifact(self, artifact_id: str) -> bool:
        """Remove an artifact from the index."""
        return self._index.pop(artifact_id, None) is not None

    def count(self) -> int:
        """Return the number of indexed artifacts."""
        return len(self._index)


__all__ = ["ArtifactEntry", "ArtifactIndexActor"]
