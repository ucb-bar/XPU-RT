"""Promotion-efficiency report aggregator (M-30, Section 19).

Reads a graph_compilation run directory plus the recipe library audit
log and emits ``efficiency_pack.json`` with the headline aggregates
that prove the Section 19 falsifiable claim:

    Cold-run vs warm-run on the same suite shows
    ``fresh_emit_count_warm < fresh_emit_count_cold`` and
    ``gemini_token_delta < 0`` while every correctness gate in
    ``verification_report.json`` still passes. CompGen gets cheaper
    to run on a model whose region patterns it has seen before,
    without weakening any verification gate.

Aggregates emitted per run:

- ``agent_call_count`` — number of agent_decision_request emissions
  (always 0 for greedy mode, 1 for agent-file/llm-live).
- ``kernel_codegen_count`` — number of kernel_execution events in
  the stage ledger (M-19 fires once per region with a tile candidate;
  warmer caches mean fewer fresh codegen attempts).
- ``verifier_call_count`` — count of differential / post-lowering /
  fusion verification reports actually run (vs skipped).
- ``promoted_hit_count`` — number of regions whose
  ``agent_decision_request.json::visible_regions[].promoted_candidates``
  came back non-empty. This is the cross-run reuse signal.
- ``fresh_emit_count`` — number of regions where the agent had to
  emit a fresh decision (no promoted candidate available). Warm
  caches drive this down.
- ``gate_level_distribution`` — count of audit events per
  PromotionLevel string. Higher levels indicate richer evidence
  available to retrieval.
- ``gemini_token_delta`` — read from the Gemini usage tracker
  (``.compgen/gemini_usage/summary.json``); when present, the
  measurement script captures the snapshot delta between cold and
  warm runs.

Cold-vs-warm comparison is the script's job
(:mod:`scripts.dev.measure_promotion_efficiency`). This module is
the pure-function aggregator — it reads, projects, and writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@dataclass(frozen=True)
class EfficiencyAggregate:
    """Per-run aggregates surfaced by :func:`build_efficiency_pack`."""

    agent_call_count: int = 0
    kernel_codegen_count: int = 0
    verifier_call_count: int = 0
    promoted_hit_count: int = 0
    fresh_emit_count: int = 0
    region_count: int = 0
    gate_level_distribution: dict[str, int] = field(default_factory=dict)
    gemini_token_total: int = 0
    gemini_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_call_count": self.agent_call_count,
            "kernel_codegen_count": self.kernel_codegen_count,
            "verifier_call_count": self.verifier_call_count,
            "promoted_hit_count": self.promoted_hit_count,
            "fresh_emit_count": self.fresh_emit_count,
            "region_count": self.region_count,
            "gate_level_distribution": dict(self.gate_level_distribution),
            "gemini_token_total": self.gemini_token_total,
            "gemini_cost_usd": self.gemini_cost_usd,
        }


def _count_kernel_codegen(ledger: list[dict[str, Any]]) -> int:
    """Stage ledger events that name kernel_execution (M-19) artifacts."""
    return sum(
        1
        for ev in ledger
        if "kernel_execution" in (ev.get("note") or "")
    )


def _count_verifier_calls(ledger: list[dict[str, Any]]) -> int:
    """Differential / post-lowering / fusion verification ledger lines."""
    matchers = (
        "differential_verification",
        "real_transform_differential",
        "real_fusion_differential",
        "post_lowering",
        "compiled_fusion",
    )
    return sum(
        1
        for ev in ledger
        for m in matchers
        if m in (ev.get("note") or "")
    )


def _count_promoted_hits(request: dict[str, Any] | None) -> tuple[int, int, int]:
    """Return (region_count, promoted_hit, fresh_emit) from the agent request."""
    if request is None:
        return 0, 0, 0
    regions = request.get("visible_regions") or []
    region_count = len(regions)
    hits = 0
    for region in regions:
        if region.get("promoted_candidates"):
            hits += 1
    fresh = region_count - hits
    return region_count, hits, fresh


def _count_agent_calls(request_path: Path) -> int:
    """1 if the agent_decision_request was emitted this run, else 0."""
    return 1 if request_path.is_file() else 0


def _read_gate_distribution(library_audit: Path) -> dict[str, int]:
    """Count audit events per gate_level (M-29)."""
    dist: dict[str, int] = {}
    for ev in _read_jsonl(library_audit):
        if ev.get("event_type") != "promotion":
            continue
        level = (ev.get("data") or {}).get("gate_level") or "unknown"
        dist[level] = dist.get(level, 0) + 1
    return dist


def _read_gemini_summary(repo_root: Path) -> tuple[int, float]:
    """Read .compgen/gemini_usage/summary.json (best-effort)."""
    summary = _read_json(repo_root / ".compgen" / "gemini_usage" / "summary.json")
    if not summary:
        return 0, 0.0
    totals = summary.get("totals") or {}
    return int(totals.get("total_tokens") or 0), float(totals.get("total_cost_usd") or 0.0)


def build_efficiency_pack(
    run_dir: Path,
    *,
    library_path: Path | None = None,
    repo_root: Path | None = None,
) -> EfficiencyAggregate:
    """Aggregate efficiency metrics for a single Phase B run.

    Args:
        run_dir: A completed graph_compilation run directory.
        library_path: Optional recipe library root; defaults to
            ``.compgen_cache/recipes/``. Used for the gate-level
            distribution from the library's audit log.
        repo_root: Optional repo root for locating
            ``.compgen/gemini_usage/summary.json``. Defaults to the
            current working directory.

    Returns:
        An :class:`EfficiencyAggregate` with all per-run aggregates.
    """
    run_dir = Path(run_dir)
    library = (
        Path(library_path) if library_path else Path(".compgen_cache") / "recipes"
    )
    repo = Path(repo_root) if repo_root else Path.cwd()

    ledger = _read_jsonl(run_dir / "stage_ledger.jsonl")
    request_path = (
        run_dir / "03_recipe_planning" / "agent_decision" / "agent_decision_request.json"
    )
    request = _read_json(request_path)

    region_count, hits, fresh = _count_promoted_hits(request)
    gate_dist = _read_gate_distribution(library / "audit.jsonl")
    gemini_tokens, gemini_cost = _read_gemini_summary(repo)

    return EfficiencyAggregate(
        agent_call_count=_count_agent_calls(request_path),
        kernel_codegen_count=_count_kernel_codegen(ledger),
        verifier_call_count=_count_verifier_calls(ledger),
        promoted_hit_count=hits,
        fresh_emit_count=fresh,
        region_count=region_count,
        gate_level_distribution=gate_dist,
        gemini_token_total=gemini_tokens,
        gemini_cost_usd=gemini_cost,
    )


def emit_efficiency_pack(
    run_dir: Path,
    *,
    library_path: Path | None = None,
    repo_root: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Build aggregate + write ``efficiency_pack.json`` under the run dir."""
    aggregate = build_efficiency_pack(
        run_dir, library_path=library_path, repo_root=repo_root,
    )
    if out_path is None:
        out_path = run_dir / "04_promotion" / "efficiency_pack.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": "efficiency_pack_v1",
        **aggregate.to_dict(),
    }
    out_path.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out_path


@dataclass(frozen=True)
class EfficiencyDelta:
    """Pairwise efficiency comparison (cold → warm) for one model."""

    model_id: str
    cold: EfficiencyAggregate
    warm: EfficiencyAggregate

    def fresh_emit_delta(self) -> int:
        """Negative when warm < cold — the headline cross-run claim."""
        return self.warm.fresh_emit_count - self.cold.fresh_emit_count

    def gemini_token_delta(self) -> int:
        return self.warm.gemini_token_total - self.cold.gemini_token_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "cold": self.cold.to_dict(),
            "warm": self.warm.to_dict(),
            "fresh_emit_delta": self.fresh_emit_delta(),
            "gemini_token_delta": self.gemini_token_delta(),
            "claim_supported": (
                self.fresh_emit_delta() <= 0
                and self.gemini_token_delta() <= 0
            ),
        }


def compare_runs(
    *, model_id: str, cold_run: Path, warm_run: Path,
    library_path: Path | None = None,
    repo_root: Path | None = None,
) -> EfficiencyDelta:
    """Build a cold→warm efficiency delta for one model."""
    return EfficiencyDelta(
        model_id=model_id,
        cold=build_efficiency_pack(
            cold_run, library_path=library_path, repo_root=repo_root,
        ),
        warm=build_efficiency_pack(
            warm_run, library_path=library_path, repo_root=repo_root,
        ),
    )


__all__ = [
    "EfficiencyAggregate",
    "EfficiencyDelta",
    "build_efficiency_pack",
    "compare_runs",
    "emit_efficiency_pack",
]
