"""Pass-pool ablation harness (M-36.1).

Compares CompGen pipeline behavior across selection modes on the same
set of models. The headline question for Section 20:

    Does Claude Code's candidate / pass selection differ from the
    deterministic greedy resolver, and when does that difference
    matter?

The harness runs each model under each mode, captures the
``agent_decision_request.json`` + ``agent_decision_response.json`` +
``agent_decision_validation.json`` artifacts, and emits a typed
``AblationResult`` per (model, mode) cell.

Modes today:

- ``greedy``    — deterministic; no agent involved.
- ``agent-file`` — the agent's response is read from a pre-authored
                   file. The harness either uses an operator-supplied
                   response or synthesizes the same selection greedy
                   would have made (the "trivial agent" baseline).

The pipeline does not yet *execute* multi-step pass plans — M-34
ships the *validator*, not the executor. So this harness compares
single-step decisions: which candidate did each mode pick, and were
the validation rows the same?

Honest residual: a real LLM-driven ``llm-live`` mode requires API
keys + network. We don't run it from the harness; the operator can
record an llm-live result and pass it in as ``agent-file``.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Per-cell result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AblationResult:
    """One (model, mode) cell."""

    model_id: str
    target_id: str
    mode: str
    selected_candidate_id: str
    candidate_kind: str  # set_tile_params / fuse_producer_consumer / ...
    pass_id: str  # alias of candidate_kind for cards-aware reporting
    validation_overall: str  # pass | fail | unknown
    validation_failures: tuple[str, ...]
    decision_seconds: float
    typed_outcome: str  # verified | typed_blocked | error
    error: str = ""
    promoted_candidates_count: int = 0  # M-37.2: M-28 candidates surfaced in the request
    promoted_hit: bool = False  # M-37.2: agent's pick matched a promoted candidate

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "target_id": self.target_id,
            "mode": self.mode,
            "selected_candidate_id": self.selected_candidate_id,
            "candidate_kind": self.candidate_kind,
            "pass_id": self.pass_id,
            "validation_overall": self.validation_overall,
            "validation_failures": list(self.validation_failures),
            "decision_seconds": self.decision_seconds,
            "typed_outcome": self.typed_outcome,
            "error": self.error,
            "promoted_candidates_count": self.promoted_candidates_count,
            "promoted_hit": self.promoted_hit,
        }


# --------------------------------------------------------------------------- #
# Per-model runner
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _classify_outcome(run_dir: Path, raised: Exception | None) -> tuple[str, str]:
    """Return (typed_outcome, error_text)."""
    if raised is not None:
        msg = str(raised)
        type_name = type(raised).__name__
        if "M-15B" in msg or "downstream" in msg.lower() or "Unsupported" in type_name:
            return "typed_blocked", f"{type_name}: {msg[:200]}"
        return "error", f"{type_name}: {msg[:200]}"

    verification = run_dir / "verification_report.json"
    if verification.exists():
        return "verified", ""
    # Otherwise the run reached a stage but we don't know the verify
    # outcome; classify as typed_blocked when honest-blocked, else
    # unknown. Without verification_report we conservatively say
    # typed_blocked-or-unverified.
    return "typed_blocked", ""


def _extract_response(run_dir: Path) -> tuple[str, str]:
    """Return (selected_candidate_id, candidate_kind) from response."""
    resp = _read_json(
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_response.json"
    )
    if not resp:
        # Greedy mode does not always emit a response file — fall back
        # to candidate_selection.json which always exists when
        # recipe_planning ran. The schema is flat (single decision per
        # file): candidate_kind + label + region_id + rationale.
        sel = _read_json(run_dir / "03_recipe_planning" / "candidate_selection.json")
        if sel:
            # The candidate id is encoded as ``label`` (e.g.
            # "tile_M16_N16_K16"); ``candidate_kind`` is the pass id.
            return (
                sel.get("label", "") or sel.get("candidate_id", ""),
                sel.get("candidate_kind", ""),
            )
        return ("", "")
    cid = resp.get("selected_candidate_id", "") or ""
    # candidate_kind comes from candidate_actions.json by lookup;
    # fall back to the rationale or pass_plan when present.
    cand_actions = _read_json(run_dir / "02_graph_analysis" / "candidate_actions.json")
    kind = ""
    for c in cand_actions.get("candidates") or []:
        if c.get("candidate_id") == cid:
            kind = c.get("kind", "") or c.get("candidate_kind", "")
            break
    return cid, kind


def _extract_validation(run_dir: Path) -> tuple[str, tuple[str, ...]]:
    val = _read_json(
        run_dir / "03_recipe_planning" / "agent_decision"
        / "agent_decision_validation.json"
    )
    if not val:
        return ("unknown", ())
    overall = val.get("overall", "unknown")
    failures = tuple(
        c["name"] for c in val.get("checks") or []
        if c.get("status") != "pass"
    )
    return (str(overall), failures)


def run_one_cell(
    *,
    model_yaml: Path,
    target_yaml: Path,
    out_dir: Path,
    mode: str,
    agent_response_path: Path | None = None,
) -> AblationResult:
    """Run one (model, mode) cell and return the typed result.

    ``mode`` is passed to ``run_graph_compilation`` as
    ``selection_mode``. ``agent_response_path`` is required for
    ``agent-file`` mode.
    """
    from compgen.graph_compilation.run import run_graph_compilation

    model_yaml = Path(model_yaml).resolve()
    target_yaml = Path(target_yaml).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)

    model_id = model_yaml.stem
    target_id = target_yaml.stem

    raised: Exception | None = None
    started = time.perf_counter()
    try:
        kwargs: dict[str, Any] = dict(
            model_config_path=model_yaml,
            target_config_path=target_yaml,
            out_dir=out_dir,
            stop_after="agent-decision-request",
            selection_mode=mode,
        )
        if mode == "agent-file" and agent_response_path is not None:
            kwargs["agent_decision_response_path"] = agent_response_path
        run_graph_compilation(**kwargs)
    except Exception as exc:  # noqa: BLE001 - classify
        raised = exc
    finally:
        elapsed = time.perf_counter() - started

    typed_outcome, error_text = _classify_outcome(out_dir, raised)
    cid, kind = _extract_response(out_dir) if raised is None else ("", "")
    overall, failures = _extract_validation(out_dir) if raised is None else ("unknown", ())

    # M-37.2: promoted-candidates count + promoted-hit detection. The
    # request carries a promoted_candidates list; if the agent's pick
    # matches one (by candidate_id or recipe_id), record a hit. This
    # is the warm-cache effectiveness metric the M-30 efficiency
    # report measured at run-level; M-37.2 measures it at decision-
    # level (per cell) so the ablation can attribute hits to modes.
    promoted_count = 0
    promoted_hit = False
    if raised is None:
        request_path = (
            out_dir / "03_recipe_planning" / "agent_decision"
            / "agent_decision_request.json"
        )
        if request_path.exists():
            try:
                req = json.loads(request_path.read_text(encoding="utf-8"))
                promoted = req.get("promoted_candidates") or []
                promoted_count = len(promoted)
                if cid:
                    for pc in promoted:
                        if (
                            pc.get("candidate_id") == cid
                            or pc.get("recipe_id") == cid
                        ):
                            promoted_hit = True
                            break
            except (json.JSONDecodeError, OSError):
                pass

    return AblationResult(
        model_id=model_id,
        target_id=target_id,
        mode=mode,
        selected_candidate_id=cid,
        candidate_kind=kind,
        pass_id=kind,  # candidate_kind is the pass_id today
        validation_overall=overall,
        validation_failures=failures,
        decision_seconds=elapsed,
        typed_outcome=typed_outcome,
        error=error_text,
        promoted_candidates_count=promoted_count,
        promoted_hit=promoted_hit,
    )


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #


@dataclass
class AblationPack:
    """Aggregate ablation report."""

    schema_version: str = "ablation_pack_v1"
    generated_at_utc: str = field(default_factory=_utc_now)
    commit: str = ""
    cells: list[AblationResult] = field(default_factory=list)

    def by_model(self) -> dict[str, list[AblationResult]]:
        out: dict[str, list[AblationResult]] = {}
        for c in self.cells:
            out.setdefault(c.model_id, []).append(c)
        return out

    def divergences(self) -> list[dict[str, Any]]:
        """Per-model rows where two modes disagreed on selected_candidate_id."""
        rows: list[dict[str, Any]] = []
        for model_id, cells in self.by_model().items():
            picks = {c.mode: c.selected_candidate_id for c in cells}
            distinct = {p for p in picks.values() if p}
            if len(distinct) > 1:
                rows.append({
                    "model_id": model_id,
                    "picks_by_mode": picks,
                    "distinct_pick_count": len(distinct),
                })
        return rows

    def summary(self) -> dict[str, Any]:
        cells = self.cells
        modes = sorted({c.mode for c in cells})
        models = sorted({c.model_id for c in cells})
        per_mode: dict[str, dict[str, Any]] = {}
        for m in modes:
            mode_cells = [c for c in cells if c.mode == m]
            per_mode[m] = {
                "cell_count": len(mode_cells),
                "verified": sum(1 for c in mode_cells if c.typed_outcome == "verified"),
                "typed_blocked": sum(
                    1 for c in mode_cells if c.typed_outcome == "typed_blocked"
                ),
                "error": sum(1 for c in mode_cells if c.typed_outcome == "error"),
                "validation_pass": sum(
                    1 for c in mode_cells if c.validation_overall == "pass"
                ),
                "validation_fail": sum(
                    1 for c in mode_cells if c.validation_overall == "fail"
                ),
                "mean_decision_seconds": (
                    sum(c.decision_seconds for c in mode_cells) / len(mode_cells)
                    if mode_cells else 0.0
                ),
                # M-37.2: warm-cache effectiveness per mode.
                "promoted_candidates_total": sum(
                    c.promoted_candidates_count for c in mode_cells
                ),
                "promoted_hit_count": sum(
                    1 for c in mode_cells if c.promoted_hit
                ),
            }
        return {
            "modes": modes,
            "models": models,
            "cell_count": len(cells),
            "per_mode": per_mode,
            "divergence_count": len(self.divergences()),
            # M-37.2: rolled-up across all cells.
            "promoted_hit_count_total": sum(1 for c in cells if c.promoted_hit),
            "promoted_hit_rate": (
                sum(1 for c in cells if c.promoted_hit) / len(cells)
                if cells else 0.0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "commit": self.commit,
            "summary": self.summary(),
            "divergences": self.divergences(),
            "cells": [c.to_dict() for c in self.cells],
        }


def emit_pack(pack: AblationPack, *, out_path: Path) -> Path:
    """Write the ablation pack as JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(pack.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


# --------------------------------------------------------------------------- #
# Suite runner
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AblationCellSpec:
    """One cell of the ablation matrix."""

    model_yaml: Path
    target_yaml: Path
    mode: str
    agent_response_path: Path | None = None


def run_suite(
    cells: Iterable[AblationCellSpec],
    *,
    out_root: Path,
    commit: str = "",
) -> AblationPack:
    """Run every cell, aggregate results, return the pack.

    Each cell gets its own subdirectory under ``out_root`` named
    ``<model>_<mode>``.
    """
    pack = AblationPack(commit=commit)
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    for spec in cells:
        cell_dir = out_root / f"{spec.model_yaml.stem}_{spec.mode}"
        result = run_one_cell(
            model_yaml=spec.model_yaml,
            target_yaml=spec.target_yaml,
            out_dir=cell_dir,
            mode=spec.mode,
            agent_response_path=spec.agent_response_path,
        )
        pack.cells.append(result)
    return pack
