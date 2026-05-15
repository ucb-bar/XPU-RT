"""Drive ``cuda-tile-translate`` (or gracefully degrade) to emit an artifact.

When ``cuda-tile-translate`` is on PATH:
  * Invoke ``cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module
    -bytecode-version=13.1 <mlir> -o <out>`` against the lowered text.
  * Read the ``.tileirbc`` bytecode, base64-encode it for the
    :class:`CompiledArtifact.code` field, and record the on-disk path
    in ``metadata["artifact_path"]``.
  * Verify the bytecode magic (``\\x7fTileIR\\x00``) before claiming
    success.
  * ``format`` is ``"cuda-tile-bitcode"``.

When the binary is absent:
  * Write the MLIR text under ``output_dir`` and return a
    :class:`CompiledArtifact` with ``format="mlir-cuda-tile"`` —
    text only, no bytecode. Metadata records the toolchain miss so
    downstream gates know not to require bytecode-only verification.

This is the canonical degrade-rather-than-fail pattern: a fresh
``pip install compgen`` on a CUDA-less laptop still produces a usable
artifact for documentation / pre-flight inspection.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path
from typing import Any

import structlog

from compgen.extensions.vendor_dialect.adapter import LoweringResult
from compgen.extensions.vendor_dialect.descriptor import VendorDialectDescriptor
from compgen.targets.backend import CompiledArtifact

log = structlog.get_logger()

_BYTECODE_MAGIC = b"\x7fTileIR\x00"
_TOOLCHAIN_BIN = "cuda-tile-translate"
_BYTECODE_VERSION = "13.1"


def _toolchain_path() -> str | None:
    """Return the absolute path to ``cuda-tile-translate``, or None if absent."""
    return shutil.which(_TOOLCHAIN_BIN)


def emit_cuda_tile_artifact(
    lowering: LoweringResult,
    *,
    descriptor: VendorDialectDescriptor,
    output_dir: Path,
    options: dict[str, Any] | None = None,
) -> CompiledArtifact:
    """Emit a :class:`CompiledArtifact` from a lowered cuda_tile module.

    Behavior is conditional on whether ``cuda-tile-translate`` is on
    PATH; see module docstring for details.
    """
    del options  # reserved for future toolchain flags
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlir_text = lowering.vendor_mlir
    if not mlir_text:
        raise ValueError("emit_cuda_tile_artifact: lowering.vendor_mlir is empty")

    mlir_path = Path(lowering.metadata.get("vendor_mlir_path") or output_dir / "ffn_kernel.mlir")
    if not mlir_path.exists():
        mlir_path.write_text(mlir_text)

    toolchain = _toolchain_path()
    base_metadata = {
        "vendor": "cuda_tile",
        "vendor_mlir_path": str(mlir_path),
        "bundle_steps": list(descriptor.bundle.steps),
        "in_tree_reference": True,
        **lowering.metadata,
    }

    if toolchain is None:
        log.info(
            "cuda_tile.bundle.toolchain_absent",
            mlir_path=str(mlir_path),
            note="emitting MLIR text only",
        )
        return CompiledArtifact(
            code=mlir_text,
            format="mlir-cuda-tile",
            target_name=descriptor.target,
            metadata={
                **base_metadata,
                "toolchain_present": False,
                "toolchain_required": _TOOLCHAIN_BIN,
                "artifact_path": str(mlir_path),
                "artifact_size_bytes": mlir_path.stat().st_size,
            },
        )

    bytecode_path = output_dir / "ffn_kernel.tileirbc"
    cmd = [
        toolchain,
        "-mlir-to-cudatilebc",
        "-no-implicit-module",
        f"-bytecode-version={_BYTECODE_VERSION}",
        str(mlir_path),
        "-o",
        str(bytecode_path),
    ]
    log.info("cuda_tile.bundle.translate.start", cmd=" ".join(cmd))
    result = subprocess.run(  # noqa: S603 - args are constructed from controlled inputs
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cuda-tile-translate failed (rc={result.returncode}):\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )

    if not bytecode_path.exists():
        raise RuntimeError(f"cuda-tile-translate succeeded but {bytecode_path} was not produced")

    blob = bytecode_path.read_bytes()
    if not blob.startswith(_BYTECODE_MAGIC):
        raise RuntimeError(
            f"cuda-tile-translate output is missing the TileIR bytecode magic; first 16 bytes (hex): {blob[:16].hex()}"
        )

    log.info(
        "cuda_tile.bundle.translate.done",
        size=len(blob),
        magic_ok=True,
    )

    return CompiledArtifact(
        code=base64.b64encode(blob).decode("ascii"),
        format="cuda-tile-bitcode",
        target_name=descriptor.target,
        metadata={
            **base_metadata,
            "toolchain_present": True,
            "toolchain_path": toolchain,
            "bytecode_version": _BYTECODE_VERSION,
            "artifact_path": str(bytecode_path),
            "artifact_size_bytes": len(blob),
            "bytecode_magic_hex": _BYTECODE_MAGIC.hex(),
        },
    )


__all__ = ["emit_cuda_tile_artifact"]
