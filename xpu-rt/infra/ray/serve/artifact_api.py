"""ArtifactAPI — Ray Serve deployment for artifact management.

Exposes REST endpoints for listing, fetching, and registering artifacts.
"""

from __future__ import annotations

from typing import Any

from infra.ray._require import require_ray, require_serve

ray = require_ray()
serve = require_serve()


@serve.deployment(route_prefix="/api/v1/artifacts")
class ArtifactAPI:
    """REST API for artifact management.

    Endpoints:
        GET  /api/v1/artifacts        — list all artifacts
        GET  /api/v1/artifacts/{id}   — get artifact metadata
        POST /api/v1/artifacts        — register new artifact
    """

    def __init__(self, artifact_actor: Any) -> None:
        self._artifacts = artifact_actor

    async def __call__(self, request: Any) -> dict[str, Any]:
        """Handle HTTP requests."""
        return {"status": "ok", "service": "compgen-artifact-api"}

    async def list_artifacts(
        self,
        target_name: str | None = None,
        artifact_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """List artifacts with optional filters."""
        if target_name or artifact_type:
            return ray.get(
                self._artifacts.find_artifacts.remote(
                    target_name=target_name,
                    artifact_type=artifact_type,
                )
            )
        return ray.get(self._artifacts.list_artifacts.remote())

    async def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Get artifact by ID."""
        return ray.get(self._artifacts.get_artifact.remote(artifact_id))

    async def register_artifact(
        self,
        artifact_type: str,
        target_name: str,
        storage_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a new artifact."""
        artifact_id = ray.get(
            self._artifacts.register_artifact.remote(
                artifact_type=artifact_type,
                target_name=target_name,
                storage_path=storage_path,
                metadata=metadata or {},
            )
        )
        return {"artifact_id": artifact_id, "status": "registered"}


__all__ = ["ArtifactAPI"]
