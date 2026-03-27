"""Recipe cache management.

Provides lookup, storage, and invalidation for promoted recipes.
The cache is keyed by promotion key (target + model + objective hash).

Invariants:
    - Cache lookups are O(1) (hash-based via filesystem).
    - Cache entries point to the recipe library (no duplication).
    - Cache invalidation logs the reason.
    - Cache supports listing all recipes for a target profile.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from compgen.promotion.promote import RecipeKey
from compgen.runtime.bundle import Bundle


@dataclass
class RecipeCache:
    """Cache for promoted recipes backed by the filesystem.

    Each recipe lives at ``library_path / key.key / manifest.json``.

    Attributes:
        library_path: Path to the recipe library.
    """

    library_path: Path

    def get(self, key: RecipeKey) -> Bundle | None:
        """Look up a recipe by key. Returns None if not found."""
        recipe_dir = self.library_path / key.key
        manifest_path = recipe_dir / "manifest.json"
        if not manifest_path.exists():
            return None

        with open(manifest_path) as f:
            data = json.load(f)

        return Bundle(
            version=data.get("version", "1.0"),
            target_profile=data.get("target_profile", ""),
            model_hash=data.get("model_hash", ""),
            objective=data.get("objective", ""),
            artifacts=data.get("artifacts", {}),
            creation_timestamp=data.get("creation_timestamp", ""),
        )

    def put(self, key: RecipeKey, bundle: Bundle) -> None:
        """Store a recipe in the cache."""
        recipe_dir = self.library_path / key.key
        recipe_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = recipe_dir / "manifest.json"
        manifest_path.write_text(json.dumps(bundle.to_dict(), indent=2))

    def invalidate(self, key: RecipeKey, reason: str = "") -> bool:
        """Mark a cached recipe as invalid (renames dir, doesn't delete).

        Returns True if the recipe was found and invalidated.
        """
        recipe_dir = self.library_path / key.key
        if not recipe_dir.exists():
            return False

        invalid_dir = self.library_path / f"{key.key}.invalid"
        if invalid_dir.exists():
            shutil.rmtree(invalid_dir)
        recipe_dir.rename(invalid_dir)

        # Write invalidation reason
        reason_path = invalid_dir / "invalidation_reason.txt"
        reason_path.write_text(reason or "no reason given")

        return True

    def list_recipes(self, target_hash: str | None = None) -> list[RecipeKey]:
        """List all cached recipes, optionally filtered by target."""
        if not self.library_path.exists():
            return []

        keys: list[RecipeKey] = []
        for entry in sorted(self.library_path.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.endswith(".invalid"):
                continue
            manifest = entry / "manifest.json"
            if not manifest.exists():
                continue

            # Parse key from directory name: target_model_obj_vN
            parts = entry.name.rsplit("_v", 1)
            if len(parts) != 2:
                continue
            key_parts = parts[0].split("_", 2)
            if len(key_parts) != 3:
                continue

            try:
                version = int(parts[1])
            except ValueError:
                continue

            rk = RecipeKey(
                target_hash=key_parts[0],
                model_hash=key_parts[1],
                objective_hash=key_parts[2],
                version=version,
            )

            if target_hash and rk.target_hash != target_hash:
                continue

            keys.append(rk)

        return keys


__all__ = ["RecipeCache"]
