"""Golden replay for a graph compilation Stage 0 capture.

Reloads ``00_graph_capture/exported_program.pt2`` (when present) and the saved
goldens, runs the exported program, and compares to the saved outputs.

If ``exported_program.pt2`` is absent (e.g. the run is ``partial_success``
because ``torch.export`` failed but Dynamo captured partitions), this
module falls back to re-running the model from its config. That fallback
still proves the goldens are consistent with the model + seed; it does
not, however, prove the export round-trips.

Writes ``<run_dir>/validation/golden_replay.json`` with a clear
``mode`` field so the caller knows which path was exercised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from compgen.graph_compilation.capture import ModelConfig, _load_model_factory


@dataclass(frozen=True)
class ReplayResult:
    status: str  # "pass" | "fail"
    mode: str  # "exported_program" | "eager_from_config"
    max_abs_error: float
    max_rel_error: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "golden_replay_report_v1",
            "status": self.status,
            "mode": self.mode,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "detail": self.detail,
        }


def _coerce_to_tuple(obj: Any) -> tuple[Any, ...]:
    if isinstance(obj, tuple):
        return obj
    if isinstance(obj, list):
        return tuple(obj)
    return (obj,)


def _max_diffs(actual: tuple[Any, ...], expected: tuple[Any, ...]) -> tuple[float, float]:
    max_abs = 0.0
    max_rel = 0.0
    for a, e in zip(actual, expected):
        if not (isinstance(a, torch.Tensor) and isinstance(e, torch.Tensor)):
            continue
        if a.shape != e.shape:
            return float("inf"), float("inf")
        diff = (a.detach() - e.detach()).abs()
        if diff.numel() == 0:
            continue
        max_abs = max(max_abs, float(diff.max().item()))
        denom = e.detach().abs().clamp_min(1e-12)
        max_rel = max(max_rel, float((diff / denom).max().item()))
    return max_abs, max_rel


def replay_goldens(run_dir: Path, model_config: Path | None = None) -> ReplayResult:
    """Replay goldens and return the result.

    ``model_config`` is the path to the model config, needed only for
    the eager fallback when no exported program is present. If omitted,
    the fallback raises rather than fabricate an answer.
    """
    capture_dir = run_dir / "00_graph_capture"
    inputs_path = capture_dir / "golden_inputs.pt"
    outputs_path = capture_dir / "golden_outputs.pt"
    if not inputs_path.exists() or not outputs_path.exists():
        return ReplayResult(
            status="fail",
            mode="exported_program",
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            detail=f"goldens missing under {capture_dir}",
        )

    sample_inputs = torch.load(inputs_path, weights_only=False)
    expected = _coerce_to_tuple(torch.load(outputs_path, weights_only=False))

    ep_path = capture_dir / "exported_program.pt2"
    if ep_path.exists():
        try:
            ep = torch.export.load(str(ep_path))
        except Exception as exc:
            return ReplayResult(
                status="fail",
                mode="exported_program",
                max_abs_error=float("inf"),
                max_rel_error=float("inf"),
                detail=f"torch.export.load failed: {type(exc).__name__}: {exc}",
            )
        try:
            with torch.no_grad():
                actual_obj = ep.module()(*sample_inputs)
        except Exception as exc:
            return ReplayResult(
                status="fail",
                mode="exported_program",
                max_abs_error=float("inf"),
                max_rel_error=float("inf"),
                detail=f"running exported program failed: {type(exc).__name__}: {exc}",
            )
        actual = _coerce_to_tuple(actual_obj)
        max_abs, max_rel = _max_diffs(actual, expected)
        # Strict equality for replay-from-export: same code, same inputs.
        ok = max_abs == 0.0 and max_rel == 0.0
        return ReplayResult(
            status="pass" if ok else "fail",
            mode="exported_program",
            max_abs_error=max_abs,
            max_rel_error=max_rel,
            detail="exported program matched goldens" if ok else "exported program output differs from goldens",
        )

    # Fallback: eager replay from config.
    if model_config is None:
        return ReplayResult(
            status="fail",
            mode="eager_from_config",
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            detail="no exported_program.pt2 and no --model-config supplied for eager fallback",
        )
    cfg = ModelConfig.load(model_config)
    torch.manual_seed(cfg.seed)
    factory = _load_model_factory(cfg.model_path, cfg.factory)
    model, _config_inputs = factory()
    with torch.no_grad():
        actual = _coerce_to_tuple(model(*sample_inputs))
    max_abs, max_rel = _max_diffs(actual, expected)
    ok = max_abs == 0.0 and max_rel == 0.0
    return ReplayResult(
        status="pass" if ok else "fail",
        mode="eager_from_config",
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        detail="eager-from-config matched goldens" if ok else "eager-from-config output differs from goldens",
    )


def write_replay_report(run_dir: Path, result: ReplayResult) -> Path:
    out_dir = run_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "golden_replay.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
