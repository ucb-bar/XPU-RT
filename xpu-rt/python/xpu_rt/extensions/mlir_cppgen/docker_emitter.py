"""Emit Dockerfile for building the generated compiler.

Generates a multi-stage Dockerfile:
  Stage 1: Build LLVM/MLIR
  Stage 2: Build CompGen compiler
  Stage 3: Runtime image with compgen-opt
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _make_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def emit_dockerfile(llvm_dir: str = "third_party/llvm-project") -> str:
    """Generate Dockerfile content.

    Args:
        llvm_dir: Relative path to LLVM source (for COPY directive).

    Returns:
        Dockerfile content string.
    """
    env = _make_env()
    tmpl = env.get_template("dockerfile.j2")
    return tmpl.render(llvm_dir=llvm_dir)


def write_dockerfile(output_dir: Path, llvm_dir: str = "third_party/llvm-project") -> Path:
    """Write Dockerfile to the output directory.

    Args:
        output_dir: Root of the generated compiler project.
        llvm_dir: Relative path to LLVM source.

    Returns:
        Path to the written Dockerfile.
    """
    path = output_dir / "Dockerfile"
    path.write_text(emit_dockerfile(llvm_dir))
    return path


__all__ = [
    "emit_dockerfile",
    "write_dockerfile",
]
