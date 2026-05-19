"""Profile a compiled merlin dispatch matrix.

Two paths are supported:

* **Native merlin profiler** — delegates to
  ``tools/profile_dispatch_matrix.py`` via :class:`MerlinBridge`. This
  is the same code the QNN heterogeneous loop calls today; it pushes
  artifacts to a board over SSH and measures with
  ``merlin-dispatch-bench``.
* **Local runner** — runs a :class:`Runner` (host, spike, firesim,
  noop) over each dispatch listed in a previously-compiled
  ``matrix.json`` and writes a ``profiled_manifest.json`` in the same
  shape merlin's tool emits.

Both paths land at a ``profiled_manifest.json`` with the same key
shape so downstream consumers don't care which produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bridge import MerlinBridge, MerlinCallResult
from .runners import Runner


@dataclass(frozen=True)
class MerlinProfileResult:
    """Outcome of one profile pass."""

    manifest_path: Path
    via: str  # "merlin-tool" | "local-runner"
    returncode: int
    payload: dict[str, Any]
    call_result: MerlinCallResult | None

    @property
    def mean_us_by_dispatch(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for entry in self.payload.get("dispatches", []) or []:
            disp = entry.get("dispatch") or entry.get("name")
            mean = entry.get("mean_us")
            if disp and mean is not None:
                out[str(disp)] = float(mean)
        return out


def profile_dispatch_matrix(
    bridge: MerlinBridge,
    *,
    matrix_path: Path,
    out_path: Path,
    runner: Runner | None = None,
    ssh_host: str | None = None,
    ssh_identity: Path | None = None,
    board_bench: str | None = None,
    iterations: int = 10,
    extra_args: list[str] | None = None,
) -> MerlinProfileResult:
    """Profile every dispatch in ``matrix_path``.

    Resolution:

    * If ``runner`` is provided, run it locally and synthesise the
      manifest. The merlin bridge is unused on this path.
    * Otherwise delegate to merlin's
      ``tools/profile_dispatch_matrix.py`` and require board-side
      args (``ssh_host`` + ``board_bench``).
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path = Path(matrix_path).resolve()
    if runner is not None:
        return _profile_via_runner(
            runner=runner,
            matrix_path=matrix_path,
            out_path=out_path,
            iterations=iterations,
        )
    if not ssh_host or not board_bench:
        raise ValueError(
            "merlin profile_dispatch_matrix needs either runner= or "
            "(ssh_host=, board_bench=) to route to the on-board profiler"
        )
    cli: list[str] = [
        "--matrix", str(matrix_path),
        "--ssh-host", ssh_host,
        "--board-bench", board_bench,
        "--out", str(out_path),
        "--iterations", str(iterations),
    ]
    if ssh_identity:
        cli.extend(["--ssh-identity", str(ssh_identity)])
    if extra_args:
        cli.extend(extra_args)
    result = bridge.call("profile_dispatch_matrix", cli_args=cli)
    payload: dict[str, Any] = {}
    if out_path.is_file():
        try:
            payload = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            payload = {}
    return MerlinProfileResult(
        manifest_path=out_path,
        via="merlin-tool",
        returncode=result.returncode,
        payload=payload,
        call_result=result,
    )


def _profile_via_runner(
    *,
    runner: Runner,
    matrix_path: Path,
    out_path: Path,
    iterations: int,
) -> MerlinProfileResult:
    """Read matrix.json, dispatch each (target, dispatch) to ``runner``,
    write a merlin-shaped ``profiled_manifest.json``."""
    raw = json.loads(matrix_path.read_text()) if matrix_path.is_file() else {}
    by_target_artifact: dict[str, Path] = {}
    dispatch_groups: dict[str, list[str]] = {}
    for entry in raw.get("dispatches", []) or []:
        target = entry.get("target") or "unknown"
        disp = entry.get("dispatch") or entry.get("name")
        artifact = entry.get("vmfb") or entry.get("artifact")
        if disp is None:
            continue
        dispatch_groups.setdefault(target, []).append(str(disp))
        if artifact and target not in by_target_artifact:
            by_target_artifact[target] = Path(artifact)
    out_entries: list[dict[str, Any]] = []
    for target, dispatches in dispatch_groups.items():
        artifact = by_target_artifact.get(target, matrix_path.parent)
        measurements = runner.run(
            artifact=artifact, dispatch_ids=dispatches, iterations=iterations,
        )
        for d in dispatches:
            mean = measurements.get(d, float("nan"))
            out_entries.append({
                "target": target,
                "dispatch": d,
                "mean_us": mean,
                "runner": runner.kind,
            })
    payload = {
        "schema_version": "profiled_manifest_v1",
        "matrix_path": str(matrix_path),
        "runner": runner.kind,
        "iterations": iterations,
        "dispatches": out_entries,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return MerlinProfileResult(
        manifest_path=out_path,
        via="local-runner",
        returncode=0,
        payload=payload,
        call_result=None,
    )
