"""Perturbation utilities for catching hardcoded behavior.

The audit ladder demands that the pipeline still completes honestly
under perturbations of inputs, paths, and library state. Each function
in this module mutates one specific axis; the audit suite then re-runs
a known-good fixture and asserts the result is still verified-or-typed-
blocked, never silent partial.

Perturbations:

- ``rename_regions``         — rewrite region_ids in a saved run
- ``change_output_dir``      — move a run dir to a new path
- ``vary_tile_divisibility`` — patch a model YAML to use non-clean shapes
- ``corrupt_promotion_library`` — flip a byte in a sidecar
- ``empty_promotion_library``   — wipe the cache before a run
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PerturbationResult:
    """Outcome of one perturbation."""

    name: str
    target: Path
    before: str  # short description of pre-state
    after: str  # short description of post-state


def rename_regions(run_dir: Path, mapping: dict[str, str]) -> PerturbationResult:
    """Rewrite ``region_id`` occurrences in run-dir JSONs.

    Walks every ``*.json`` under ``run_dir`` and replaces literal
    occurrences of ``mapping`` keys with their values. This is a
    deliberately blunt instrument: it surfaces hardcoded references to
    specific region_ids in either source code or audit logic.
    """
    affected: list[Path] = []
    for path in run_dir.rglob("*.json"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        replaced = text
        for old, new in mapping.items():
            replaced = replaced.replace(old, new)
        if replaced != text:
            path.write_text(replaced, encoding="utf-8")
            affected.append(path)
    return PerturbationResult(
        name="rename_regions",
        target=run_dir,
        before=f"{len(mapping)} pattern(s) requested",
        after=f"{len(affected)} files affected",
    )


def change_output_dir(run_dir: Path, new_path: Path) -> PerturbationResult:
    """Move ``run_dir`` to ``new_path`` (renames the directory)."""
    new_path = Path(new_path).resolve()
    if new_path.exists():
        shutil.rmtree(new_path)
    shutil.move(str(run_dir), str(new_path))
    return PerturbationResult(
        name="change_output_dir",
        target=new_path,
        before=str(run_dir),
        after=str(new_path),
    )


def vary_tile_divisibility(model_yaml_path: Path) -> PerturbationResult:
    """Mark the model YAML as using non-clean shapes.

    Adds a ``perturbation: vary_tile_divisibility`` annotation to the
    YAML so the capture stage knows to use a perturbed shape variant.
    Models that read this flag in their adapter switch to non-clean
    shapes; other models ignore it.
    """
    text = model_yaml_path.read_text(encoding="utf-8")
    if "perturbation:" not in text:
        text += "\nperturbation: vary_tile_divisibility\n"
    model_yaml_path.write_text(text)
    return PerturbationResult(
        name="vary_tile_divisibility",
        target=model_yaml_path,
        before="clean shapes",
        after="non-clean shapes (annotation written)",
    )


def corrupt_promotion_library(library_path: Path) -> PerturbationResult:
    """Flip the contract_hash on every promoted-recipe sidecar.

    A corrupted ``contract_hash`` should cause the retrieval to
    miss exact-contract matches; the audit verifies that the failure
    mode is "no exact match" rather than "silently use stale recipe".
    """
    affected: list[Path] = []
    if not library_path.exists():
        return PerturbationResult(
            name="corrupt_promotion_library",
            target=library_path,
            before="library missing",
            after="nothing to corrupt",
        )
    for sidecar in library_path.rglob("promoted_recipe.json"):
        try:
            data: dict[str, Any] = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "key" in data and isinstance(data["key"], dict) and "contract_hash" in data["key"]:
            data["key"]["contract_hash"] = "0" * 16  # zero-out
            sidecar.write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            affected.append(sidecar)
    return PerturbationResult(
        name="corrupt_promotion_library",
        target=library_path,
        before=f"{len(list(library_path.rglob('promoted_recipe.json')))} sidecar(s)",
        after=f"{len(affected)} sidecar(s) corrupted (contract_hash → 0...)",
    )


def empty_promotion_library(library_path: Path) -> PerturbationResult:
    """Wipe the recipe library so the next run is forced cold.

    Equivalent to ``COMPGEN_DISABLE_RECIPE_MEMORY=1`` for the file
    layer (the env var short-circuits at the retrieval layer; this
    function nukes the on-disk state).
    """
    if library_path.exists():
        shutil.rmtree(library_path)
    return PerturbationResult(
        name="empty_promotion_library",
        target=library_path,
        before="library possibly populated",
        after="library wiped (does not exist)",
    )
