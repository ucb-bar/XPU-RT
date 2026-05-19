"""Wrapper around ``tools/compile.py`` — one (target, source) → artifact.

Merlin's ``./merlin compile`` writes outputs under
``build/compiled_models/<model>/<target>/``. We can override the
output directory via ``--build-dir`` so XPU-RT keeps artifacts in its
own run scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bridge import MerlinBridge, MerlinCallResult


@dataclass(frozen=True)
class MerlinCompileResult:
    """Outcome of one ``compile_program`` call.

    ``vmfb_path`` is best-effort: when merlin's tool writes to a
    well-known path under ``out_dir`` we surface it; otherwise the
    caller must inspect ``call_result.stdout`` / ``out_dir``.
    """

    target: str
    source: Path
    out_dir: Path
    returncode: int
    vmfb_path: Path | None
    call_result: MerlinCallResult


def _guess_vmfb(out_dir: Path, target: str) -> Path | None:
    """Find the first ``.vmfb`` under merlin's compile output layout."""
    candidates = list(out_dir.rglob(f"*{target}*.vmfb"))
    if candidates:
        return candidates[0]
    fallback = list(out_dir.rglob("*.vmfb"))
    return fallback[0] if fallback else None


def compile_program(
    bridge: MerlinBridge,
    *,
    target: str,
    source: Path,
    out_dir: Path,
    hw: str | None = None,
    quantized: bool = False,
    extra_args: list[str] | None = None,
) -> MerlinCompileResult:
    """Compile ``source`` (``.mlir`` or ``.onnx``) for ``target``.

    Args:
        bridge: Resolved :class:`MerlinBridge`.
        target: Merlin target YAML stem (e.g. ``"spacemit_x60"``,
            ``"saturn_opu_v128"``).
        source: Input file. ``.mlir`` and ``.onnx`` are both accepted
            by merlin's own importer.
        out_dir: Build root. Maps to merlin's ``--build-dir`` so we
            can keep artifacts off ``$MERLIN_ROOT/build``.
        hw: Optional hardware sub-target (e.g. ``"OPU"``, ``"RVV"``).
        quantized: Force merlin's ``--quantized`` flag.
        extra_args: Additional argv tokens forwarded verbatim.
    """
    source = Path(source).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cli: list[str] = [
        str(source),
        "--target", target,
        "--build-dir", str(out_dir),
    ]
    if hw:
        cli.extend(["--hw", hw])
    if quantized:
        cli.append("--quantized")
    if extra_args:
        cli.extend(extra_args)
    result = bridge.call("compile", cli_args=cli)
    return MerlinCompileResult(
        target=target,
        source=source,
        out_dir=out_dir,
        returncode=result.returncode,
        vmfb_path=_guess_vmfb(out_dir, target) if result.returncode == 0 else None,
        call_result=result,
    )
