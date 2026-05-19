"""Wrapper around ``tools/chipyard.py`` — chipyard image build flow.

Merlin's chipyard driver reads hardware recipes from
``build_tools/hardware/*.yaml`` and automates simulator / firesim
image builds. We expose a single typed entrypoint that returns the
final image path (best-effort) plus the captured logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bridge import MerlinBridge, MerlinCallResult


@dataclass(frozen=True)
class MerlinChipyardImage:
    """Outcome of one chipyard build."""

    hardware: str
    out_dir: Path
    returncode: int
    image_path: Path | None
    call_result: MerlinCallResult


def _guess_image(out_dir: Path) -> Path | None:
    """Pick the most likely chipyard image artefact under ``out_dir``."""
    for pattern in ("*.elf", "*.bin", "*.img", "simulator-*"):
        hits = list(out_dir.rglob(pattern))
        if hits:
            return hits[0]
    return None


def build_chipyard_image(
    bridge: MerlinBridge,
    *,
    hardware: str,
    out_dir: Path,
    subcommand: str = "build",
    extra_args: list[str] | None = None,
) -> MerlinChipyardImage:
    """Drive ``tools/chipyard.py <subcommand> --hardware <hardware>``.

    ``subcommand`` is one of merlin's chipyard subcommands (``build``,
    ``firesim``, ``validate``, ...). Defaults to ``build`` which
    produces a simulator image.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cli: list[str] = [subcommand, "--hardware", hardware, "--out-dir", str(out_dir)]
    if extra_args:
        cli.extend(extra_args)
    result = bridge.call("chipyard", cli_args=cli)
    return MerlinChipyardImage(
        hardware=hardware,
        out_dir=out_dir,
        returncode=result.returncode,
        image_path=_guess_image(out_dir) if result.returncode == 0 else None,
        call_result=result,
    )
