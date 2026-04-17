"""CLI entry point for the CompGen MLIR C++ compiler generator.

Usage::

    python -m compgen.extensions.mlir_cppgen \\
        --dialects layout,tile,accel \\
        --output artifacts/compiler/ \\
        --docker
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="compgen-mlir-cppgen",
        description="Generate standalone MLIR C++ compiler from xDSL prototypes.",
    )
    parser.add_argument(
        "--dialects",
        type=str,
        default="layout,tile,accel",
        help="Comma-separated dialect names to generate (default: layout,tile,accel)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/compiler"),
        help="Output directory for the generated project (default: artifacts/compiler/)",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Also generate a Dockerfile",
    )
    parser.add_argument(
        "--passes",
        type=str,
        default="layout",
        help="Comma-separated pass groups to generate (default: layout)",
    )
    parser.add_argument(
        "--llvm-dir",
        type=str,
        default="third_party/llvm-project",
        help="Relative path to LLVM source for Dockerfile (default: third_party/llvm-project)",
    )

    args = parser.parse_args()
    dialect_names = [d.strip() for d in args.dialects.split(",") if d.strip()]
    pass_groups = [p.strip() for p in args.passes.split(",") if p.strip()] if args.passes else None

    from compgen.extensions.mlir_cppgen import generate_compiler

    output = generate_compiler(
        dialects=dialect_names,
        output_dir=args.output,
        include_passes=pass_groups,
        include_docker=args.docker,
        llvm_dir=args.llvm_dir,
    )
    print(f"Generated compiler project at: {output}")
    sys.exit(0)


if __name__ == "__main__":
    main()
