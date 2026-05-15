"""Promotion gates ladder (, Section 19).

Six-level promotion ladder that replaces the all-or-nothing
verification gate from
:func:`compgen.promotion.promote._inspect_verification`. Each level
demands strictly more evidence than the one below; promotion at level
``promoted`` (the default cutoff) requires the full readiness pack.

Levels (low → high):

- ``observed``: a candidate ran. The action was generated and selected
  but no verification has been discharged. Useful for record-keeping;
  not safe to apply without further checks.
``verified_fx``: graph-level differential evidence passed. /
  reported ``status=pass`` (or ``tolerance_eps``/``bit_equality``)
  on the FX-level transformation. The recipe is *correct* at the graph
  level but kernel codegen has not been validated yet.
- ``verified_kernel``: compiled-kernel differential evidence passed.
   confirm the lowered kernel agrees with the eager
  reference within tolerance. The recipe is *correct end-to-end*.
``characterized``: measured cost evidence recorded. analytical
  cost AND (OR ) measured cost both present so the recipe
  carries usable performance characterization, not just correctness.
``promoted``: full readiness pack passes. readiness
  matrices report ``overall=pass`` AND certificate hashes are
  recorded. This is the default minimum to promote — the recipe is
  correct, characterized, and audit-traceable.
- ``portable``: at least two distinct ``target_class`` values have
  observed promotions for the same ``region_signature``. The recipe
  has demonstrated cross-target reuse — the strongest claim Section
  19 makes.

The evaluator is pure-function: it inspects the bundle's evidence
artifacts and returns the highest level that holds. Stripping
evidence demotes the level monotonically — never the reverse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class PromotionLevel(Enum):
    """Promotion-gate level — ordered low → high.

    The numeric value encodes the level so callers can compare with
    ``<`` / ``>=``. Do not reorder these values or external storage
    that records ``gate_level`` will silently drift.
    """

    OBSERVED = 0
    VERIFIED_FX = 1
    VERIFIED_KERNEL = 2
    CHARACTERIZED = 3
    PROMOTED = 4
    PORTABLE = 5

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def from_string(cls, value: str) -> "PromotionLevel":
        """Decode the lowercase-string form back to an enum value."""
        upper = value.upper()
        try:
            return cls[upper]
        except KeyError as exc:
            raise ValueError(
                f"unknown PromotionLevel string {value!r}; "
                f"valid: {[level.name.lower() for level in cls]}"
            ) from exc


@dataclass(frozen=True)
class GateEvaluation:
    """Outcome of :func:`evaluate_gate`.

    Attributes:
        level: Highest :class:`PromotionLevel` the evidence supports.
        reasons: Per-level reason map — for each level *not* attained,
            a short string explaining what evidence was missing or
            failed. For levels attained, the entry says ``"ok"``.
        evidence_summary: Compact projection of the evidence used
            (artifact names + their pass/fail status). Mirrors the
            shape of ``PromotedRecipe.evidence_summary`` so the
            same dict can be persisted without further translation.
    """

    level: PromotionLevel
    reasons: dict[str, str] = field(default_factory=dict)
    evidence_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": str(self.level),
            "reasons": dict(self.reasons),
            "evidence_summary": dict(self.evidence_summary),
        }


# --------------------------------------------------------------------------- #
# Helpers — read individual evidence items
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


_PASS_STATUSES: frozenset[str] = frozenset({
    "pass",
    "passed",
    "tolerance_eps",
    "bit_equality",
    "ok",
})


def _status_passed(status: str | None) -> bool:
    """Treat the canonical pass-like statuses as success."""
    if status is None:
        return False
    return str(status).lower() in _PASS_STATUSES


# --------------------------------------------------------------------------- #
# Per-level evidence checks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _LevelChecks:
    """Per-level boolean predicates over a run dir.

    Each closure returns (passed: bool, reason: str). ``reason`` is
    short and descriptive — surfaced via :class:`GateEvaluation`.
    """

    observed: bool
    observed_reason: str
    verified_fx: bool
    verified_fx_reason: str
    verified_kernel: bool
    verified_kernel_reason: str
    characterized: bool
    characterized_reason: str
    promoted: bool
    promoted_reason: str
    evidence_summary: dict[str, Any]


def _check_observed(rp: Path) -> tuple[bool, str]:
    sel = _read_json(rp / "candidate_selection.json")
    if sel is None:
        return False, "candidate_selection.json missing"
    cid = sel.get("selected_candidate_id")
    if not cid:
        return False, "no candidate selected"
    return True, "ok"


def _check_verified_fx(rp: Path) -> tuple[bool, str, dict[str, Any]]:
    """(real_transform_differential) or (real_fusion) pass.

    Phase B writes these reports under per-stage subdirs
    (``real_verification/``, ``real_fusion_verification/``,
    ``differential_verification/``); legacy fixtures used the flat
    layout. Each candidate lists both locations.
    """
    summary: dict[str, Any] = {}
    candidates: tuple[tuple[str, tuple[Path, ...]], ...] = (
        ("real_transform", (
            rp / "real_verification" / "real_differential_report.json",
            rp / "real_transform_differential_report.json",
        )),
        ("real_fusion", (
            rp / "real_fusion_verification" / "real_fusion_differential_report.json",
            rp / "real_fusion_differential_report.json",
        )),
        ("differential", (
            rp / "differential_verification" / "differential_verification_report.json",
            rp / "differential_verification_report.json",
        )),
    )
    for label, paths in candidates:
        for p in paths:
            report = _read_json(p)
            if report is None:
                continue
            status = report.get("status") or report.get("overall")
            summary[f"fx_{label}"] = status
            if _status_passed(status):
                return True, f"{label}={status}", summary
            break  # first found wins; don't fall through to legacy on miss
    return False, "no FX-level differential pass found", summary


def _check_verified_kernel(run_dir: Path) -> tuple[bool, str, dict[str, Any]]:
    """ compiled-kernel differential pass."""
    ga = run_dir / "02_graph_analysis"
    summary: dict[str, Any] = {}

    for label, rel in (
        ("kernel_execution",
         "kernel_execution/kernel_execution_report.json"),
        ("region_compiled",
         "kernel_execution/region_compiled_differential_report.json"),
        ("compiled_fusion",
         "compiled_fusion/compiled_fusion_report.json"),
    ):
        report = _read_json(ga / rel)
        if report is None:
            continue
        status = report.get("status") or report.get("overall")
        summary[f"kernel_{label}"] = status
        if _status_passed(status):
            return True, f"{label}={status}", summary

    return False, "no compiled-kernel differential pass found", summary


def _nested_count(report: dict[str, Any] | None, key: str) -> int:
    """Read a count from a Phase B report — top-level OR nested under summary.

     reports keep their counts in a ``summary``
    sub-block (e.g. ``summary.candidates_modeled = 15`` instead of
    ``candidates_modeled = 15`` at top-level). Real-workload smoke
    test caught the gate evaluator reading the wrong layer.
    """
    if report is None:
        return 0
    val = report.get(key)
    if val is None:
        val = (report.get("summary") or {}).get(key)
    return int(val or 0)


def _check_characterized(run_dir: Path) -> tuple[bool, str, dict[str, Any]]:
    """analytical cost AND (OR ) measured cost present."""
    ga = run_dir / "02_graph_analysis"
    summary: dict[str, Any] = {}

    analytical = _read_json(ga / "analytical_cost" / "per_candidate_analytical_cost.json")
    has_analytical = _nested_count(analytical, "candidates_modeled") > 0
    summary["m21_analytical"] = "present" if has_analytical else "missing"

    bottleneck = _read_json(ga / "compiled_bottleneck" / "compiled_bottleneck_report.json")
    profiler = _read_json(ga / "profiler_evidence" / "profiler_evidence_report.json")
    has_bottleneck = _nested_count(bottleneck, "region_count_with_evidence") > 0
    has_profiler = (
        _nested_count(profiler, "gpu_collected_count") > 0
        or _nested_count(profiler, "cpu_collected_count") > 0
    )
    has_measured = has_bottleneck or has_profiler
    summary["m22_or_m221_measured"] = "present" if has_measured else "missing"

    if not has_analytical:
        return False, "M-21 analytical cost not present", summary
    if not has_measured:
        return False, "M-22 / M-22.1 measured cost not present", summary
    return True, "analytical+measured both present", summary


def _check_promoted(run_dir: Path) -> tuple[bool, str, dict[str, Any]]:
    """ readiness ``overall=pass`` AND certificates recorded."""
    ga = run_dir / "02_graph_analysis"
    summary: dict[str, Any] = {}

    fx_matrix = _read_json(ga / "readiness" / "graph_analysis_readiness_matrix.json")
    fx_overall = (fx_matrix or {}).get("overall") or "missing"
    summary["m17_1_fx_readiness"] = fx_overall

    # readiness matrix has lived at two paths historically; check both.
    kernel_matrix = (
        _read_json(ga / "kernel_readiness" / "kernel_section_readiness_matrix.json")
        or _read_json(ga / "kernel_section_readiness" / "kernel_section_readiness_matrix.json")
    )
    kernel_overall = (kernel_matrix or {}).get("overall") or "missing"
    summary["m24_kernel_readiness"] = kernel_overall

    # Certificate hashes are recorded by the bridge in the
    # synthesised verification_report.json under 04_promotion/.
    verify = _read_json(run_dir / "04_promotion" / "verification_report.json")
    has_certs = bool(verify and verify.get("passed"))
    summary["promotion_certificates"] = "present" if has_certs else "missing"

    if fx_overall != "pass":
        return False, f"FX readiness matrix overall={fx_overall!r}", summary
    if kernel_overall not in {"pass", "ready", "ready_for_m24_1"}:
        return False, f"kernel readiness matrix overall={kernel_overall!r}", summary
    if not has_certs:
        return False, "promotion certificates not recorded", summary
    return True, "all readiness rows pass + certificates recorded", summary


# --------------------------------------------------------------------------- #
# Public evaluator
# --------------------------------------------------------------------------- #


def evaluate_gate(
    run_dir: Path,
    *,
    region_signature: str = "",
    target_class: str = "",
    library_path: Path | None = None,
) -> GateEvaluation:
    """Return the highest :class:`PromotionLevel` the evidence supports.

    The function is monotonic by construction: each level's check
    only widens the evidence requirement; a passing level always
    implies all lower levels also pass. Reasons for failed levels
    are recorded in :attr:`GateEvaluation.reasons` for audit /
    debugging.

    The ``portable`` level needs cross-run information (≥2 distinct
    ``target_class`` values for the same ``region_signature``) — the
    caller must pass ``library_path`` so the evaluator can scan the
    recipe library. Without that, ``portable`` is unreachable and
    capped at ``promoted``.

    Args:
        run_dir: Root of a Phase B graph_compilation run.
        region_signature: Two-tier cache key; required for the
            ``portable`` check.
        target_class: This run's target_class — used to count how
            many *distinct* targets have promotions for the given
            ``region_signature``.
        library_path: Recipe library root. Defaults to
            ``.compgen_cache/recipes/``.
    """
    rp = run_dir / "03_recipe_planning"
    reasons: dict[str, str] = {}
    evidence: dict[str, Any] = {}
    level = PromotionLevel.OBSERVED  # presumed; checked below.

    obs_ok, obs_reason = _check_observed(rp)
    reasons["observed"] = obs_reason
    if not obs_ok:
        # No candidate selected — bundle has nothing to promote.
        return GateEvaluation(
            level=PromotionLevel.OBSERVED,
            reasons=reasons,
            evidence_summary=evidence,
        )

    # The ladder: check each level in order; first failure caps it.
    fx_ok, fx_reason, fx_evidence = _check_verified_fx(rp)
    evidence.update(fx_evidence)
    reasons["verified_fx"] = fx_reason
    if not fx_ok:
        return GateEvaluation(level=level, reasons=reasons, evidence_summary=evidence)
    level = PromotionLevel.VERIFIED_FX

    kernel_ok, kernel_reason, kernel_evidence = _check_verified_kernel(run_dir)
    evidence.update(kernel_evidence)
    reasons["verified_kernel"] = kernel_reason
    if not kernel_ok:
        return GateEvaluation(level=level, reasons=reasons, evidence_summary=evidence)
    level = PromotionLevel.VERIFIED_KERNEL

    char_ok, char_reason, char_evidence = _check_characterized(run_dir)
    evidence.update(char_evidence)
    reasons["characterized"] = char_reason
    if not char_ok:
        return GateEvaluation(level=level, reasons=reasons, evidence_summary=evidence)
    level = PromotionLevel.CHARACTERIZED

    prom_ok, prom_reason, prom_evidence = _check_promoted(run_dir)
    evidence.update(prom_evidence)
    reasons["promoted"] = prom_reason
    if not prom_ok:
        return GateEvaluation(level=level, reasons=reasons, evidence_summary=evidence)
    level = PromotionLevel.PROMOTED

    # Portable — needs cross-run data.
    if region_signature and target_class:
        portable_ok, portable_reason = _check_portable(
            region_signature=region_signature,
            this_target_class=target_class,
            library_path=library_path,
        )
        reasons["portable"] = portable_reason
        if portable_ok:
            level = PromotionLevel.PORTABLE
    else:
        reasons["portable"] = (
            "skipped — region_signature or target_class not provided"
        )

    return GateEvaluation(
        level=level, reasons=reasons, evidence_summary=evidence,
    )


def _check_portable(
    *,
    region_signature: str,
    this_target_class: str,
    library_path: Path | None,
) -> tuple[bool, str]:
    """≥2 distinct ``target_class`` values for the same region signature."""
    if library_path is None:
        library_path = Path(".compgen_cache") / "recipes"
    if not library_path.is_dir():
        return False, "recipe library does not exist yet"

    target_classes: set[str] = set()
    for recipe_dir in library_path.iterdir():
        if not recipe_dir.is_dir() or recipe_dir.name.endswith(".invalid"):
            continue
        sidecar = _read_json(recipe_dir / "promoted_recipe.json")
        if not sidecar:
            continue
        key = sidecar.get("key") or {}
        if str(key.get("region_signature", "")) != region_signature:
            continue
        recipe = sidecar.get("recipe") or {}
        tc = str((recipe.get("validity") or {}).get("target_class", ""))
        if tc:
            target_classes.add(tc)

    target_classes.add(this_target_class)  # include the run we're evaluating.
    if len(target_classes) < 2:
        return False, (
            f"only {len(target_classes)} distinct target_class observed for "
            f"region_signature={region_signature!r}: {sorted(target_classes)}"
        )
    return True, (
        f"observed {len(target_classes)} target_classes for "
        f"region_signature={region_signature!r}: {sorted(target_classes)}"
    )


__all__ = ["GateEvaluation", "PromotionLevel", "evaluate_gate"]
