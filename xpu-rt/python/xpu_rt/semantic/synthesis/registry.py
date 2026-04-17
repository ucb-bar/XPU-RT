"""Registry for promoted synthesized guard artifacts."""

from __future__ import annotations

from pathlib import Path

from compgen.semantic.synthesis.promote import GuardArtifact, load_guard_artifact


class GuardRegistry:
    """In-memory index of promoted guard artifacts."""

    def __init__(self) -> None:
        self._artifacts: dict[str, GuardArtifact] = {}

    def register(self, artifact: GuardArtifact) -> None:
        self._artifacts[artifact.guard_key] = artifact

    def load_dir(self, path: str | Path) -> None:
        root = Path(path)
        if not root.exists():
            return
        for artifact_path in sorted(root.glob("guard.*.json")):
            artifact = load_guard_artifact(artifact_path)
            self.register(artifact)

    def get(self, guard_key: str) -> GuardArtifact:
        return self._artifacts[guard_key]

    def try_get(self, guard_key: str) -> GuardArtifact | None:
        return self._artifacts.get(guard_key)

    def keys(self) -> list[str]:
        return sorted(self._artifacts.keys())


__all__ = ["GuardRegistry"]
