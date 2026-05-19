"""ONNX → payload MLIR bridge — canonical home under merlin/.

Originally lived under ``backends/qnn/onnx_bridge.py``. Lifted here
because the importer is a merlin feature (``$MERLIN_ROOT/tools/
onnx_to_mlir.py``) used by more than just the QNN flow: saturn_opu,
gemmini, spacemit also start from ONNX inputs in the paper-figure
demos. The QNN module now re-exports from this location so existing
callers keep working.

Resolution order, in priority:

1. **Merlin torch-mlir importer** when ``$MERLIN_ROOT`` exposes
   ``tools/onnx_to_mlir.py`` — the path the heterogeneous loop and
   the paper-figure scripts already shell out to.
2. **`torch-mlir-import-onnx` binary** on ``PATH`` — upstream CLI
   fallback when merlin isn't around.
3. **Stub fallback** (``allow_stub=True``) — writes a small
   ``builtin.module`` placeholder MLIR with the ONNX path / sha
   embedded so ``--dry-run`` flows still have a file to walk.

The result is cached on ``(sha256(onnx), opset, importer)`` so the
dashboard's iterative loop doesn't re-import on every round.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OnnxImportResult:
    mlir_path: Path
    importer: str  # "merlin" | "torch-mlir" | "stub"
    cache_hit: bool
    sha256: str


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _merlin_importer() -> Path | None:
    merlin_root = os.environ.get("MERLIN_ROOT", "/scratch2/agustin/merlin")
    cand = Path(merlin_root) / "tools" / "onnx_to_mlir.py"
    if cand.is_file():
        return cand
    return None


def _torch_mlir_binary() -> str | None:
    return shutil.which("torch-mlir-import-onnx") or None


def _write_stub_mlir(
    out_path: Path,
    *,
    onnx_path: Path,
    onnx_sha: str,
    opset: int,
    workload_id: str,
) -> None:
    """Emit a minimal MLIR module pointing at the ONNX we couldn't import.

    Downstream consumers (e.g. the heterogeneous loop's matrix-compile
    step) skip real lowering when they see
    ``xpu_rt.onnx.stub = true`` and fall back to a coarse single-island
    manifest seeded from measured CostTable entries. Keeps
    ``--dry-run`` honest about the fact that no real compilation
    happened, without crashing the loop.
    """
    content = (
        f'module attributes {{xpu_rt.onnx.stub = unit, '
        f'xpu_rt.onnx.path = "{onnx_path}", '
        f'xpu_rt.onnx.sha256 = "{onnx_sha}", '
        f'xpu_rt.onnx.opset = {opset} : i32, '
        f'xpu_rt.workload_id = "{workload_id}"}} {{\n'
        f'  // Placeholder produced by xpu_rt.targets.backends.merlin.onnx_bridge\n'
        f'  // when no torch-mlir or merlin importer was available.\n'
        f'}}\n'
    )
    out_path.write_text(content)


def onnx_to_payload_mlir(
    onnx_path: Path,
    out_mlir: Path,
    *,
    opset_check: int = 18,
    workload_id: str | None = None,
    allow_stub: bool = True,
    cache_dir: Path | None = None,
) -> OnnxImportResult:
    """Convert ONNX → payload MLIR, returning the path + which importer ran.

    Cache key: ``sha256(onnx) + opset + importer-name``. When
    ``allow_stub`` is True (the default in ``--dry-run``), the stub
    path is used instead of raising when no real importer is on the
    system.
    """
    onnx_path = Path(onnx_path).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    out_mlir = Path(out_mlir)
    out_mlir.parent.mkdir(parents=True, exist_ok=True)
    sha = _sha256_of(onnx_path)
    wid = workload_id or onnx_path.stem

    cache_root = cache_dir or out_mlir.parent / ".onnx_mlir_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    merlin = _merlin_importer()
    torch_mlir = _torch_mlir_binary()
    importer = "merlin" if merlin else ("torch-mlir" if torch_mlir else "stub")

    cache_key = f"{sha[:16]}_op{opset_check}_{importer}.mlir"
    cached = cache_root / cache_key
    if cached.is_file():
        shutil.copyfile(cached, out_mlir)
        return OnnxImportResult(out_mlir, importer, True, sha)

    if importer == "merlin" and merlin is not None:
        cmd = ["python3", str(merlin), str(onnx_path), "-o", str(out_mlir),
               "--opset", str(opset_check)]
        rc = subprocess.run(cmd, capture_output=True, text=True)
        if rc.returncode == 0 and out_mlir.is_file():
            shutil.copyfile(out_mlir, cached)
            return OnnxImportResult(out_mlir, "merlin", False, sha)
        # Fall through to next importer.
    if importer in {"torch-mlir", "merlin"} and torch_mlir is not None:
        cmd = [torch_mlir, str(onnx_path), "-o", str(out_mlir)]
        rc = subprocess.run(cmd, capture_output=True, text=True)
        if rc.returncode == 0 and out_mlir.is_file():
            shutil.copyfile(out_mlir, cached)
            return OnnxImportResult(out_mlir, "torch-mlir", False, sha)
    if not allow_stub:
        raise RuntimeError(
            f"no ONNX→MLIR importer available for {onnx_path}. "
            "Install merlin or torch-mlir-import-onnx, or pass "
            "allow_stub=True for the dry-run path."
        )
    _write_stub_mlir(out_mlir, onnx_path=onnx_path, onnx_sha=sha,
                     opset=opset_check, workload_id=wid)
    shutil.copyfile(out_mlir, cached)
    return OnnxImportResult(out_mlir, "stub", False, sha)
