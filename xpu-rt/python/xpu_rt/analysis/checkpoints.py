"""Analysis-summary registry + index (M-32).

Every analysis the pipeline emits is registered in :data:`KNOWN_SUMMARIES`
with:

- ``id``               — short stable name (e.g. ``graph_dossier_v3``)
- ``level``            — one of :class:`AnalysisLevel`
- ``relative_path``    — path under ``<run_dir>/`` (canonical location)
- ``dependencies``     — other summary ids whose state must hold for
                         this one to be valid (transitive invalidation)
- ``description``      — one-liner for the agent

Pass cards' ``reads`` / ``invalidates`` fields reference these ids, and
:func:`assert_resolvable` rejects any reference that is not registered.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class AnalysisLevel(str, enum.Enum):
    """The eight IR / analysis lenses CompGen exposes."""

    FX = "fx"  # A0 — after capture
    PAYLOAD = "payload"  # A1 — after FX→MLIR + after every accepted transform
    GRAPH = "graph"  # A1.5 — payload-derived graph dossier (legacy bucket)
    RECIPE = "recipe"  # A2 — after candidate selection / Recipe IR commits
    SEMANTIC = "semantic"  # A3 — verification obligations + counterexamples
    TILE = "tile"  # A4 — after lowering to tile/kernel level
    KERNEL = "kernel"  # A5 — after Triton / C / Exo generation
    PLAN = "plan"  # A6 — after scheduling / memory planning
    RUNTIME = "runtime"  # A7 — after bench / profile


ANALYSIS_LEVELS: tuple[str, ...] = tuple(level.value for level in AnalysisLevel)


class AnalysisSummaryError(RuntimeError):
    """Raised when a known-summary reference cannot be resolved."""


@dataclass(frozen=True)
class KnownSummary:
    """Static registry entry for one analysis summary the pipeline emits."""

    id: str
    level: AnalysisLevel
    relative_path: str
    dependencies: tuple[str, ...]
    description: str
    optional: bool = False  # True for opt-in artifacts (e.g. kernel evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level.value,
            "relative_path": self.relative_path,
            "dependencies": list(self.dependencies),
            "description": self.description,
            "optional": self.optional,
        }


# --------------------------------------------------------------------------- #
# Static registry
# --------------------------------------------------------------------------- #

# Order: increasing IR level. Two summaries can share a level (e.g.
# multiple FX-level summaries from one capture). ``dependencies`` are
# intentionally *direct only* — transitive closure is computed by
# :meth:`AnalysisIndex.transitively_invalidated_by`.
KNOWN_SUMMARIES: tuple[KnownSummary, ...] = (
    # ---- A0 FX ------------------------------------------------------------
    KnownSummary(
        id="capture_report",
        level=AnalysisLevel.FX,
        relative_path="00_graph_capture/capture_report.json",
        dependencies=(),
        description="Top-level FX capture report (model + targets + status)",
    ),
    KnownSummary(
        id="dynamo_summary",
        level=AnalysisLevel.FX,
        relative_path="00_graph_capture/dynamo_summary.json",
        dependencies=("capture_report",),
        description="torch.compile dynamo trace + graph-break summary",
    ),
    KnownSummary(
        id="export_graph",
        level=AnalysisLevel.FX,
        relative_path="00_graph_capture/export_graph.json",
        dependencies=("capture_report",),
        description="torch.export graph signature + node listing",
    ),
    KnownSummary(
        id="graph_breaks",
        level=AnalysisLevel.FX,
        relative_path="00_graph_capture/graph_breaks.json",
        dependencies=("dynamo_summary",),
        description="dynamo graph-break diagnostics",
    ),
    KnownSummary(
        id="compile_baseline",
        level=AnalysisLevel.FX,
        relative_path="00_graph_capture/compile_baseline.json",
        dependencies=("capture_report",),
        description="torch.compile baseline timing + outcome",
    ),
    # ---- A1 Payload -------------------------------------------------------
    KnownSummary(
        id="payload_summary",
        level=AnalysisLevel.PAYLOAD,
        relative_path="01_payload_lowering/lowering_summary.json",
        dependencies=("capture_report",),
        description="Payload-IR canonical lowering summary (op counts + status)",
    ),
    KnownSummary(
        id="dialect_coverage",
        level=AnalysisLevel.PAYLOAD,
        relative_path="01_payload_lowering/dialect_coverage.json",
        dependencies=("payload_summary",),
        description="Per-dialect op coverage of the lowered Payload IR",
    ),
    KnownSummary(
        id="unsupported_ops",
        level=AnalysisLevel.PAYLOAD,
        relative_path="01_payload_lowering/unsupported_ops.json",
        dependencies=("payload_summary",),
        description="Ops the lowering does not handle (typed-blocked surface)",
    ),
    KnownSummary(
        id="strict_gate_report",
        level=AnalysisLevel.PAYLOAD,
        relative_path="01_payload_lowering",
        # The report file is per-model: <model_id>_strict_gate_report.json
        # AnalysisIndex resolves this via a glob.
        dependencies=("payload_summary", "unsupported_ops"),
        description="M-16.1 typed strict-gate report (overall lowering status)",
    ),
    KnownSummary(
        id="canonical_pass_trace",
        level=AnalysisLevel.PAYLOAD,
        relative_path="01_payload_lowering/canonical_pass_trace.json",
        dependencies=("payload_summary",),
        description="Trace of every canonical pass applied during lowering",
    ),
    KnownSummary(
        id="fx_to_payload_accounting",
        level=AnalysisLevel.PAYLOAD,
        relative_path="01_payload_lowering/fx_to_payload_accounting.json",
        dependencies=("payload_summary", "export_graph"),
        description="Per-FX-node accounting of how each node lowered (or didn't)",
    ),
    KnownSummary(
        id="lowering_diagnostics",
        level=AnalysisLevel.PAYLOAD,
        relative_path="01_payload_lowering/lowering_diagnostics.json",
        dependencies=("payload_summary",),
        description="Diagnostics emitted during payload lowering",
    ),
    # ---- A1.5 Graph (payload-derived) ------------------------------------
    KnownSummary(
        id="graph_dossier_v3",
        level=AnalysisLevel.GRAPH,
        relative_path="02_graph_analysis/graph_dossier_v3.json",
        dependencies=("payload_summary",),
        description="M-09 graph dossier v3 (region-aware, calibration-ready)",
    ),
    KnownSummary(
        id="region_graph",
        level=AnalysisLevel.GRAPH,
        relative_path="02_graph_analysis/region_graph.json",
        dependencies=("graph_dossier_v3",),
        description="Region-level connectivity graph",
    ),
    KnownSummary(
        id="region_map",
        level=AnalysisLevel.GRAPH,
        relative_path="02_graph_analysis/region_map.json",
        dependencies=("graph_dossier_v3",),
        description="Mapping from FX nodes to logical regions",
    ),
    KnownSummary(
        id="tensor_use_def_graph",
        level=AnalysisLevel.GRAPH,
        relative_path="02_graph_analysis/tensor_use_def_graph.json",
        dependencies=("graph_dossier_v3",),
        description="Per-tensor use-def graph (consumer counts, lifetimes)",
    ),
    KnownSummary(
        id="candidate_actions",
        level=AnalysisLevel.GRAPH,
        relative_path="02_graph_analysis/candidate_actions.json",
        dependencies=("graph_dossier_v3", "region_map"),
        description="Full candidate-action space for every region",
    ),
    KnownSummary(
        id="cost_preview",
        level=AnalysisLevel.GRAPH,
        relative_path="02_graph_analysis/cost_preview_v2.json",
        dependencies=("candidate_actions",),
        description="Per-candidate cost preview (analytical + measured overlays)",
    ),
    KnownSummary(
        id="llm_action_space",
        level=AnalysisLevel.GRAPH,
        relative_path="02_graph_analysis/llm_action_space.json",
        dependencies=("candidate_actions", "cost_preview"),
        description="Bounded, legal-only LLM-facing view of the action space",
    ),
    KnownSummary(
        id="llm_graph_view",
        level=AnalysisLevel.GRAPH,
        relative_path="02_graph_analysis/llm_graph_view.json",
        dependencies=("llm_action_space",),
        description="LLM-facing graph view (per-region summary + candidates)",
    ),
    KnownSummary(
        id="numerical_sensitivity_audit",
        level=AnalysisLevel.GRAPH,
        relative_path="02_graph_analysis/numerical_sensitivity_audit.json",
        dependencies=("graph_dossier_v3",),
        description="Per-op numerical-sensitivity classification",
    ),
    # ---- A2 Recipe --------------------------------------------------------
    KnownSummary(
        id="candidate_selection",
        level=AnalysisLevel.RECIPE,
        relative_path="03_recipe_planning/candidate_selection.json",
        dependencies=("llm_action_space",),
        description="Per-region candidate selected by the agent (or greedy)",
    ),
    KnownSummary(
        id="recipe_summary",
        level=AnalysisLevel.RECIPE,
        relative_path="03_recipe_planning/recipe_summary.json",
        dependencies=("candidate_selection",),
        description="Committed Recipe IR summary (op counts, region IDs)",
    ),
    KnownSummary(
        id="recipe_validation",
        level=AnalysisLevel.RECIPE,
        relative_path="03_recipe_planning/recipe_validation.json",
        dependencies=("recipe_summary",),
        description="Recipe IR validation report (well-formedness)",
    ),
    KnownSummary(
        id="recipe_gate_verdict",
        level=AnalysisLevel.RECIPE,
        relative_path="03_recipe_planning/recipe_gate_verdict.json",
        dependencies=("recipe_validation",),
        description="Aggregate recipe-stage gate verdict",
    ),
    KnownSummary(
        id="real_transform_eligibility",
        level=AnalysisLevel.RECIPE,
        relative_path="03_recipe_planning/real_transform_eligibility.json",
        dependencies=("candidate_selection", "candidate_actions"),
        description="Per-candidate eligibility for real-transform lowering",
    ),
    KnownSummary(
        id="transform_lowering_report",
        level=AnalysisLevel.RECIPE,
        relative_path="03_recipe_planning/transform_lowering_report.json",
        dependencies=("real_transform_eligibility",),
        description="Outcome of real-transform lowering passes",
    ),
    KnownSummary(
        id="lowering_artifact_manifest",
        level=AnalysisLevel.RECIPE,
        relative_path="03_recipe_planning/lowering_artifact_manifest.json",
        dependencies=("transform_lowering_report",),
        description="Manifest of all real-transform-emitted artifacts",
    ),
    KnownSummary(
        id="kernel_contracts",
        level=AnalysisLevel.RECIPE,
        relative_path="03_recipe_planning/kernel_contracts",
        dependencies=("recipe_summary", "graph_dossier_v3"),
        description="Per-region kernel contracts (KernelContractV3)",
        optional=True,  # only emitted when contracts are derived
    ),
    # ---- A3 Semantic ------------------------------------------------------
    KnownSummary(
        id="semantic_obligations",
        level=AnalysisLevel.SEMANTIC,
        relative_path="03_recipe_planning/semantic_obligations.json",
        dependencies=("recipe_summary",),
        description="Semantic IR obligations (refinement, tolerance, formal)",
    ),
    KnownSummary(
        id="real_verification_report",
        level=AnalysisLevel.SEMANTIC,
        relative_path="03_recipe_planning/real_verification",
        dependencies=("semantic_obligations", "transform_lowering_report"),
        description="Real-transform differential verification report (per region)",
        optional=True,
    ),
    KnownSummary(
        id="real_fusion_verification_report",
        level=AnalysisLevel.SEMANTIC,
        relative_path="03_recipe_planning/real_fusion_verification",
        dependencies=("semantic_obligations", "transform_lowering_report"),
        description="Real-fusion differential verification report",
        optional=True,
    ),
    # ---- A5 Kernel --------------------------------------------------------
    KnownSummary(
        id="kernel_evidence_pack",
        level=AnalysisLevel.KERNEL,
        relative_path="02_graph_analysis/kernel_evidence_pack.json",
        dependencies=("recipe_summary",),
        description="Aggregate kernel evidence pack across compiled regions",
        optional=True,
    ),
    KnownSummary(
        id="kernel_lifetime_evidence",
        level=AnalysisLevel.KERNEL,
        relative_path="02_graph_analysis/kernel_lifetime/kernel_lifetime_evidence_report.json",
        dependencies=("kernel_evidence_pack",),
        description="M-24.1 register / spill / occupancy evidence per kernel",
        optional=True,
    ),
    KnownSummary(
        id="region_compiled_differential_report",
        level=AnalysisLevel.KERNEL,
        relative_path="02_graph_analysis/kernel_execution/region_compiled_differential_report.json",
        dependencies=("kernel_evidence_pack",),
        description="M-20 per-region compiled differential outcome",
        optional=True,
    ),
    KnownSummary(
        id="compiled_bottleneck_report",
        level=AnalysisLevel.KERNEL,
        relative_path="02_graph_analysis/compiled_bottleneck/compiled_bottleneck_report.json",
        dependencies=("region_compiled_differential_report",),
        description="M-22 compiled per-kernel bottleneck attribution",
        optional=True,
    ),
    # ---- A7 Runtime / profile --------------------------------------------
    KnownSummary(
        id="profiler_calibration_report",
        level=AnalysisLevel.RUNTIME,
        relative_path="02_graph_analysis/profiler_calibration/profiler_calibration_report.json",
        dependencies=("graph_dossier_v3",),
        description="M-18 profiler-calibrated cost overlay",
        optional=True,
    ),
)


def _by_id() -> dict[str, KnownSummary]:
    out: dict[str, KnownSummary] = {}
    for entry in KNOWN_SUMMARIES:
        if entry.id in out:
            raise AnalysisSummaryError(
                f"duplicate summary id {entry.id!r} in KNOWN_SUMMARIES"
            )
        out[entry.id] = entry
    return out


_KNOWN_BY_ID = _by_id()


def known_summary_ids() -> tuple[str, ...]:
    return tuple(sorted(_KNOWN_BY_ID))


def assert_resolvable(ids: list[str] | tuple[str, ...]) -> None:
    """Raise :class:`AnalysisSummaryError` if any id is not registered."""
    missing = [i for i in ids if i not in _KNOWN_BY_ID]
    if missing:
        raise AnalysisSummaryError(
            f"unknown analysis summary ids: {missing}; "
            f"known: {known_summary_ids()}"
        )


def summary_id_for_path(rel_path: str) -> str | None:
    """Reverse lookup: which summary id (if any) lives at this path?"""
    rel_path = rel_path.lstrip("/")
    # Exact match first.
    for entry in KNOWN_SUMMARIES:
        if entry.relative_path == rel_path:
            return entry.id
    # Path under a directory-typed summary (e.g. kernel_contracts/).
    for entry in KNOWN_SUMMARIES:
        if rel_path.startswith(entry.relative_path + "/"):
            return entry.id
    return None


# --------------------------------------------------------------------------- #
# Per-run AnalysisSummary
# --------------------------------------------------------------------------- #


def _sha256_short(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _content_hash(path: Path) -> str:
    """Content hash for either a single file or a directory.

    For directories we hash the sorted list of (relpath, content sha)
    so reordering files in the directory does not change the hash, but
    adding / removing / mutating a file does.
    """
    if not path.exists():
        return ""
    if path.is_file():
        return _sha256_short(path.read_bytes())
    h = hashlib.sha256()
    for sub in sorted(path.rglob("*")):
        if not sub.is_file():
            continue
        rel = sub.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(sub.read_bytes())
        h.update(b"\x01")
    return h.hexdigest()[:16]


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class AnalysisSummary:
    """One analysis summary as observed in a specific run dir.

    M-33: ``generation`` tracks how many invalidation events have been
    recorded against this summary id during the run. Default 0 keeps
    pre-M-33 callers honest — they observe the same shape, just
    without monotonic-bump semantics.
    """

    id: str
    level: str
    relative_path: str
    content_hash: str
    dependencies: tuple[str, ...]
    available: bool
    last_modified_utc: str
    description: str
    optional: bool
    generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "relative_path": self.relative_path,
            "content_hash": self.content_hash,
            "dependencies": list(self.dependencies),
            "available": self.available,
            "last_modified_utc": self.last_modified_utc,
            "description": self.description,
            "optional": self.optional,
            "generation": self.generation,
        }


def _resolve_path(run_dir: Path, entry: KnownSummary) -> Path:
    """Resolve the on-disk path for an entry, handling glob cases."""
    # ``strict_gate_report`` is per-model-id: <run_dir>/01_payload_lowering/<model>_strict_gate_report.json.
    # Glob first; the parent directory always exists when payload
    # lowering ran, but it is not the artifact we want.
    if entry.id == "strict_gate_report":
        for match in (run_dir / "01_payload_lowering").glob("*_strict_gate_report.json"):
            return match
        return run_dir / entry.relative_path  # nonexistent; available=False
    candidate = run_dir / entry.relative_path
    return candidate  # may not exist; AnalysisSummary will record available=False


def _materialize_entry(run_dir: Path, entry: KnownSummary) -> AnalysisSummary:
    path = _resolve_path(run_dir, entry)
    available = path.exists()
    rel_path = path.relative_to(run_dir).as_posix() if available else entry.relative_path
    if available:
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            last_modified = mtime.strftime("%Y-%m-%dT%H:%M:%SZ")
        except OSError:
            last_modified = ""
    else:
        last_modified = ""
    return AnalysisSummary(
        id=entry.id,
        level=entry.level.value,
        relative_path=rel_path,
        content_hash=_content_hash(path) if available else "",
        dependencies=entry.dependencies,
        available=available,
        last_modified_utc=last_modified,
        description=entry.description,
        optional=entry.optional,
    )


@dataclass
class AnalysisIndex:
    """Indexed analysis summaries observed in one run dir."""

    run_dir: Path
    summaries: dict[str, AnalysisSummary] = field(default_factory=dict)
    generated_at_utc: str = ""

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> AnalysisIndex:
        run_dir = Path(run_dir).resolve()
        idx = cls(run_dir=run_dir, generated_at_utc=_utc_now())
        for entry in KNOWN_SUMMARIES:
            idx.summaries[entry.id] = _materialize_entry(run_dir, entry)
        return idx

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def __contains__(self, summary_id: str) -> bool:
        return summary_id in self.summaries

    def __iter__(self) -> Iterator[AnalysisSummary]:
        for sid in sorted(self.summaries):
            yield self.summaries[sid]

    def __len__(self) -> int:
        return len(self.summaries)

    def get(self, summary_id: str) -> AnalysisSummary | None:
        return self.summaries.get(summary_id)

    def require(self, summary_id: str) -> AnalysisSummary:
        s = self.summaries.get(summary_id)
        if s is None:
            raise AnalysisSummaryError(
                f"summary id {summary_id!r} not registered "
                f"(known: {known_summary_ids()})"
            )
        return s

    def available_summaries(self) -> tuple[AnalysisSummary, ...]:
        return tuple(s for s in self if s.available)

    def by_level(self, level: AnalysisLevel | str) -> tuple[AnalysisSummary, ...]:
        level_str = level.value if isinstance(level, AnalysisLevel) else str(level)
        return tuple(s for s in self if s.level == level_str)

    def transitively_invalidated_by(
        self, invalidated_ids: list[str] | tuple[str, ...]
    ) -> tuple[str, ...]:
        """Return every summary id whose state depends (transitively)
        on any id in ``invalidated_ids``.

        M-33's invalidation enforcement consumes this.
        """
        assert_resolvable(list(invalidated_ids))
        seed = set(invalidated_ids)
        # Build reverse-dependency map: id → ids that depend on it
        deps_rev: dict[str, set[str]] = {sid: set() for sid in _KNOWN_BY_ID}
        for entry in KNOWN_SUMMARIES:
            for dep in entry.dependencies:
                if dep in deps_rev:
                    deps_rev[dep].add(entry.id)
        affected: set[str] = set(seed)
        worklist = list(seed)
        while worklist:
            current = worklist.pop()
            for downstream in deps_rev.get(current, set()):
                if downstream not in affected:
                    affected.add(downstream)
                    worklist.append(downstream)
        return tuple(sorted(affected))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "analysis_index_v1",
            "run_dir": str(self.run_dir),
            "generated_at_utc": self.generated_at_utc,
            "summary_count": len(self.summaries),
            "available_count": len(self.available_summaries()),
            "summaries": [s.to_dict() for s in self],
        }

    # ------------------------------------------------------------------ #
    # M-33: diff against another index
    # ------------------------------------------------------------------ #

    def diff(self, other: AnalysisIndex) -> dict[str, tuple[str, ...]]:
        """Diff against ``other`` (typically a later snapshot of the same run).

        Returns a mapping with three keys:

        - ``mutated``  — summary ids whose ``content_hash`` differs and
                         both indices report the summary as available.
        - ``appeared`` — summary ids that were unavailable in ``self`` but
                         are available in ``other``.
        - ``removed``  — summary ids that were available in ``self`` but
                         are unavailable in ``other``.

        The full ``InvalidationDiff`` view (ordered tuples per axis) is
        provided by :func:`compgen.analysis.invalidation.compute_invalidation_diff`;
        this method is the lower-level primitive.
        """
        mutated: list[str] = []
        appeared: list[str] = []
        removed: list[str] = []
        for sid in sorted(self.summaries):
            before = self.summaries[sid]
            after = other.summaries.get(sid)
            if after is None:
                continue
            if before.available and after.available:
                if before.content_hash != after.content_hash:
                    mutated.append(sid)
            elif before.available and not after.available:
                removed.append(sid)
            elif (not before.available) and after.available:
                appeared.append(sid)
        return {
            "mutated": tuple(mutated),
            "appeared": tuple(appeared),
            "removed": tuple(removed),
        }
