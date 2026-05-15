"""Synthesize and attach promoted guards for a Recipe IR module."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compgen.ir.recipe.serialize import mlir_to_recipe, recipe_to_mlir
from compgen.semantic.synthesis.integration import synthesize_and_attach_guards


@click.command()
@click.argument("recipe_mlir", type=click.Path(exists=True, path_type=Path))
@click.option("--guard-dir", type=click.Path(path_type=Path), default=Path("artifacts/guards"))
@click.option("--target-class", type=str, default="")
@click.option("--emit-updated-recipe", type=click.Path(path_type=Path), default=None)
def main(recipe_mlir: Path, guard_dir: Path, target_class: str, emit_updated_recipe: Path | None) -> None:
    """Synthesize promoted guards for RECIPE_MLIR and persist them."""

    module = mlir_to_recipe(recipe_mlir.read_text(encoding="utf-8"))
    registry, _, summary = synthesize_and_attach_guards(
        module,
        out_dir=guard_dir,
        target_class=target_class,
    )
    click.echo(json.dumps({"registered_guards": registry.keys(), "summary": summary}, indent=2))
    if emit_updated_recipe is not None:
        emit_updated_recipe.write_text(recipe_to_mlir(module), encoding="utf-8")


if __name__ == "__main__":
    main()
