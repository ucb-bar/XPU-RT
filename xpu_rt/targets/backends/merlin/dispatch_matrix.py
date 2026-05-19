"""Wrapper around ``tools/compile_dispatch_matrix.py``.

Compiles a single source MLIR for N targets and produces a
``matrix.json`` describing per-(dispatch, target) feasibility +
artifact paths. Used by the heterogeneous scheduler to plan
per-island placement, but exposed here as a standalone typed call so
non-QNN flows (saturn_opu, gemmini, spacemit, …) can reuse the same
output shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bridge import MerlinBridge, MerlinCallResult


@dataclass(frozen=True)
class MerlinDispatchMatrix:
    """Typed view of ``compile_dispatch_matrix.py``'s output."""

    matrix_path: Path
    targets: tuple[str, ...]
    out_dir: Path
    returncode: int
    payload: dict[str, Any]
    call_result: MerlinCallResult

    @property
    def per_target_dispatches(self) -> dict[str, list[str]]:
        """``{target: [dispatch_id, …]}`` projection."""
        out: dict[str, list[str]] = {t: [] for t in self.targets}
        for entry in self.payload.get("dispatches", []) or []:
            target = entry.get("target")
            disp = entry.get("dispatch") or entry.get("name")
            if target in out and disp:
                out[target].append(str(disp))
        return out


def compile_dispatch_matrix(
    bridge: MerlinBridge,
    *,
    source: Path,
    targets: list[str],
    out_dir: Path,
    extra_args: list[str] | None = None,
) -> MerlinDispatchMatrix:
    """Sweep merlin's compile over ``targets`` for one ``source``.

    Returns the parsed ``matrix.json`` plus the raw call result so
    callers can introspect tool stderr on failure.
    """
    source = Path(source).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cli: list[str] = [
        "--source", str(source),
        "--targets", ",".join(targets),
        "--out-dir", str(out_dir),
    ]
    if extra_args:
        cli.extend(extra_args)
    result = bridge.call("compile_dispatch_matrix", cli_args=cli)
    matrix_path = out_dir / "matrix.json"
    payload: dict[str, Any] = {}
    if matrix_path.is_file():
        try:
            payload = json.loads(matrix_path.read_text())
        except json.JSONDecodeError:
            payload = {}
    return MerlinDispatchMatrix(
        matrix_path=matrix_path,
        targets=tuple(targets),
        out_dir=out_dir,
        returncode=result.returncode,
        payload=payload,
        call_result=result,
    )
