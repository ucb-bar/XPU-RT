"""Subprocess wrapper for running xpu_rt-opt on MLIR text.

Bridges the Python xDSL pipeline to the generated C++ compiler:
  1. Python exports MLIR text via ``xdsl.printer.Printer``
  2. This module shells out to ``xpu_rt-opt`` with the desired passes
  3. Python parses the result back via ``xdsl.parser.Parser``

Usage::

    from xpu_rt.extensions.mlir_cppgen.runner import run_xpu_rt_opt

    # Run a single pass
    output = run_xpu_rt_opt(
        mlir_text, ["--layout-propagate-layouts"],
        opt_binary=Path("build/bin/xpu_rt-opt"),
    )

    # Run the full layout pipeline
    output = run_xpu_rt_opt(
        mlir_text, ["--layout-pipeline"],
        opt_binary=Path("build/bin/xpu_rt-opt"),
    )
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger()


class CompgenOptError(RuntimeError):
    """Raised when xpu_rt-opt exits with non-zero status."""

    def __init__(self, returncode: int, stderr: str) -> None:
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"xpu_rt-opt failed (rc={returncode}): {stderr[:500]}")


def find_xpu_rt_opt(
    search_paths: list[Path | str] | None = None,
) -> Path | None:
    """Locate the xpu_rt-opt binary.

    Searches in order:
    1. Provided search paths
    2. ``artifacts/compiler/build/bin/xpu_rt-opt``
    3. System PATH

    Returns:
        Path to the binary, or None if not found.
    """
    candidates: list[Path] = []
    if search_paths:
        candidates.extend(Path(p) for p in search_paths)
    candidates.append(Path("artifacts/compiler/build/bin/xpu_rt-opt"))
    candidates.append(Path("build/bin/xpu_rt-opt"))

    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate

    # Fall back to system PATH
    found = shutil.which("xpu_rt-opt")
    return Path(found) if found else None


def run_xpu_rt_opt(
    mlir_text: str,
    passes: list[str],
    *,
    opt_binary: Path | str | None = None,
    timeout_seconds: int = 120,
) -> str:
    """Run xpu_rt-opt on MLIR text and return the transformed output.

    Args:
        mlir_text: Input MLIR text (from xDSL Printer).
        passes: Pass flags (e.g., ["--layout-propagate-layouts"]).
        opt_binary: Path to the xpu_rt-opt binary. Auto-detected if None.
        timeout_seconds: Maximum execution time.

    Returns:
        Transformed MLIR text string.

    Raises:
        CompgenOptError: If xpu_rt-opt exits with non-zero status.
        FileNotFoundError: If the binary is not found.
        TimeoutError: If execution exceeds timeout.
    """
    if opt_binary is None:
        opt_binary = find_xpu_rt_opt()
    if opt_binary is None:
        raise FileNotFoundError(
            "xpu_rt-opt binary not found. Build with: cmake -G Ninja -S artifacts/compiler -B build && ninja -C build"
        )

    opt_binary = Path(opt_binary)
    cmd = [str(opt_binary)] + passes

    logger.debug("xpu_rt_opt.run", cmd=cmd, input_size=len(mlir_text))

    try:
        proc = subprocess.run(
            cmd,
            input=mlir_text,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"xpu_rt-opt timed out after {timeout_seconds}s") from e

    if proc.returncode != 0:
        raise CompgenOptError(proc.returncode, proc.stderr)

    logger.debug(
        "xpu_rt_opt.done",
        output_size=len(proc.stdout),
        stderr=proc.stderr[:200] if proc.stderr else "",
    )
    return proc.stdout


# Convenience: layout pipeline
LAYOUT_PIPELINE_PASSES = [
    "--layout-canonicalize-transposes",
    "--layout-attach-layout-hints",
    "--layout-set-virtual-encodings",
    "--layout-propagate-layouts",
    "--layout-hoist-layout-ops",
    "--layout-fuse-layout-into-producers",
    "--layout-introduce-prepacking",
    "--layout-specialize-layouts",
    "--layout-materialize-boundaries",
    "--layout-cleanup-artifacts",
]


def run_layout_pipeline(
    mlir_text: str,
    *,
    opt_binary: Path | str | None = None,
) -> str:
    """Run the full 10-pass layout pipeline through xpu_rt-opt.

    Args:
        mlir_text: Input MLIR text.
        opt_binary: Path to the xpu_rt-opt binary.

    Returns:
        Transformed MLIR text after all 10 layout passes.
    """
    return run_xpu_rt_opt(
        mlir_text,
        LAYOUT_PIPELINE_PASSES,
        opt_binary=opt_binary,
    )


__all__ = [
    "CompgenOptError",
    "LAYOUT_PIPELINE_PASSES",
    "find_xpu_rt_opt",
    "run_xpu_rt_opt",
    "run_layout_pipeline",
]
