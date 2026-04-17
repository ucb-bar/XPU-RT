"""Generate standalone MLIR C++ compiler from xDSL prototypes.

This is the "compiler generator" — it introspects CompGen's xDSL dialect
definitions and emits a complete C++ MLIR project (TableGen, headers,
sources, CMake, driver) that can be built against ``third_party/llvm-project/``.

Usage (Python API)::

    from compgen.extensions.mlir_cppgen import generate_compiler
    generate_compiler(
        dialects=["layout", "tile", "accel"],
        output_dir=Path("artifacts/compiler"),
    )

Usage (CLI)::

    python -m compgen.extensions.mlir_cppgen \\
        --dialects layout,tile,accel \\
        --output artifacts/compiler/ \\
        --docker
"""

from __future__ import annotations

from pathlib import Path

import structlog

from compgen.extensions.mlir_cppgen.cmake_emitter import write_cmake_files
from compgen.extensions.mlir_cppgen.cpp_emitter import write_header_files, write_source_files
from compgen.extensions.mlir_cppgen.docker_emitter import write_dockerfile
from compgen.extensions.mlir_cppgen.driver_emitter import write_opt_driver
from compgen.extensions.mlir_cppgen.introspect import (
    DialectInfo,
    introspect_accel_dialect,
    introspect_layout_dialect,
    introspect_recipe_base,
    introspect_tile_dialect,
)
from compgen.extensions.mlir_cppgen.pass_emitter import get_layout_passes, write_pass_files
from compgen.extensions.mlir_cppgen.tablegen_emitter import write_tablegen_files
from compgen.extensions.mlir_cppgen.test_emitter import write_test_files

logger = structlog.get_logger()

# Registry of well-known dialects
_DIALECT_REGISTRY: dict[str, callable] = {
    "layout": introspect_layout_dialect,
    "tile": introspect_tile_dialect,
    "accel": introspect_accel_dialect,
}

# Dependency graph: dialect prefix → list of dependency prefixes
_DEP_GRAPH: dict[str, list[str]] = {
    "Layout": ["RecipeBase"],
    "Tile": ["RecipeBase"],
    "Accel": [],
    "RecipeBase": [],
}

# Cross-dialect TableGen includes
_TD_DEP_INCLUDES: dict[str, list[str]] = {
    "Layout": ["RecipeBase/RecipeBaseAttrs.td"],
    "Tile": ["RecipeBase/RecipeBaseAttrs.td"],
    "Accel": [],
    "RecipeBase": [],
}

# Cross-dialect C++ header includes
_H_DEP_INCLUDES: dict[str, list[str]] = {
    "Layout": ["RecipeBase/RecipeBaseAttrs.h"],
    "Tile": ["RecipeBase/RecipeBaseAttrs.h"],
    "Accel": [],
    "RecipeBase": [],
}


def generate_compiler(
    dialects: list[str] | None = None,
    output_dir: Path | str = Path("artifacts/compiler"),
    *,
    include_passes: list[str] | None = None,
    include_docker: bool = False,
    llvm_dir: str = "third_party/llvm-project",
) -> Path:
    """Generate a complete MLIR C++ compiler project.

    Introspects the specified xDSL dialects and emits TableGen, C++,
    CMake, and driver files to ``output_dir``.

    Args:
        dialects: List of dialect names (e.g., ["layout", "tile", "accel"]).
                  Defaults to all registered dialects.
        output_dir: Root directory for the generated project.
        include_passes: List of pass groups to generate (e.g., ["layout"]).
                        None means generate all available passes.
        include_docker: Whether to generate a Dockerfile.
        llvm_dir: Relative path to LLVM source for Docker.

    Returns:
        Path to the generated project root.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dialects is None:
        dialects = list(_DIALECT_REGISTRY.keys())

    logger.info("mlir_cppgen.generate", dialects=dialects, output=str(output_dir))

    # Always include recipe_base for shared attrs
    all_infos: list[DialectInfo] = [introspect_recipe_base()]
    logger.info("mlir_cppgen.introspect", dialect="recipe_base", attrs=len(all_infos[0].attrs))

    for name in dialects:
        if name not in _DIALECT_REGISTRY:
            raise ValueError(f"Unknown dialect: {name}. Available: {sorted(_DIALECT_REGISTRY)}")
        info = _DIALECT_REGISTRY[name]()
        all_infos.append(info)
        logger.info(
            "mlir_cppgen.introspect",
            dialect=name,
            ops=len(info.ops),
            attrs=len(info.attrs),
        )

    # Generate per-dialect files
    all_written: list[Path] = []
    for info in all_infos:
        include_dir = output_dir / "include" / info.prefix
        lib_dir = output_dir / "lib" / info.prefix

        td_deps = _TD_DEP_INCLUDES.get(info.prefix, [])
        h_deps = _H_DEP_INCLUDES.get(info.prefix, [])

        # TableGen
        written = write_tablegen_files(info, include_dir, dep_includes=td_deps)
        all_written.extend(written)
        logger.debug("mlir_cppgen.tablegen", dialect=info.name, files=len(written))

        # Headers
        written = write_header_files(info, include_dir, dep_headers=h_deps)
        all_written.extend(written)
        logger.debug("mlir_cppgen.headers", dialect=info.name, files=len(written))

        # Sources
        written = write_source_files(info, lib_dir)
        all_written.extend(written)
        logger.debug("mlir_cppgen.sources", dialect=info.name, files=len(written))

    # Generate passes
    pass_groups = include_passes if include_passes is not None else ["layout"]
    dialects_with_passes: set[str] = set()
    for group in pass_groups:
        if group == "layout":
            layout_info = next((i for i in all_infos if i.name == "layout"), None)
            if layout_info:
                passes = get_layout_passes()
                written = write_pass_files(layout_info, passes, output_dir)
                all_written.extend(written)
                dialects_with_passes.add("Layout")
                logger.info(
                    "mlir_cppgen.passes",
                    dialect="layout",
                    passes=len(passes),
                    files=len(written),
                )

    # CMake files
    cmake_written = write_cmake_files(
        all_infos, output_dir,
        dep_graph=_DEP_GRAPH,
        dialects_with_passes=dialects_with_passes,
    )
    all_written.extend(cmake_written)
    logger.debug("mlir_cppgen.cmake", files=len(cmake_written))

    # Driver
    opt_dir = output_dir / "compgen-opt"
    driver_path = write_opt_driver(all_infos, opt_dir, dialects_with_passes=dialects_with_passes)
    all_written.append(driver_path)
    logger.debug("mlir_cppgen.driver", path=str(driver_path))

    # Test files
    all_passes = []
    if "layout" in (include_passes or ["layout"]):
        all_passes.extend(get_layout_passes())
    test_written = write_test_files(all_infos, all_passes, output_dir)
    all_written.extend(test_written)
    logger.debug("mlir_cppgen.tests", files=len(test_written))

    # Dockerfile (optional)
    if include_docker:
        docker_path = write_dockerfile(output_dir, llvm_dir)
        all_written.append(docker_path)
        logger.debug("mlir_cppgen.docker", path=str(docker_path))

    logger.info("mlir_cppgen.complete", total_files=len(all_written), output=str(output_dir))
    return output_dir


__all__ = [
    "generate_compiler",
]
