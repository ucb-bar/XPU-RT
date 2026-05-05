"""Capture/lower artifact contract & independent validator (graph_compilation artifact contract).

This package owns the on-disk contract for capture / lower / gap-discovery
runs and the independent validator that audits a run directory without
trusting its own manifest.

It is intentionally thin: it does not perform capture, lowering, or
analysis. Those responsibilities live in:

- ``compgen.capture.torch_export.capture_frontend_artifact`` (Stage 0)
- ``compgen.capture.torch_mlir_bridge.bridge_fx_graph`` /
  ``compgen.ir.payload.import_fx.FXImporter`` (Stage 1)
- ``compgen.pipeline.driver.compile_through_pipeline`` (deterministic
  pass ordering)
- ``compgen.analysis.graph_digest`` and
  ``compgen.agent.analyzer.NetworkAnalyzer`` (Stage 2)

Later graph compilation tasks wrap those modules and write the staged
``00_graph_capture/`` / ``01_payload_lowering/`` / ``02_graph_analysis/``
/ ``03_gap_discovery/`` / ``04_gap_closure/`` layout that this validator
enforces. graph_compilation artifact contract only defines and validates
the contract.

Schema location note: schemas live under
``compgen/graph_compilation/schemas/v1/`` so the contract owner owns its
schema, rather than the repo-wide ``compgen/schemas/v1/`` directory.
"""

from __future__ import annotations

from compgen.graph_compilation.artifacts import (
    CANONICAL_STAGE_ORDER,
    LEGACY_STAGE_DIR_PREFIXES,
    STAGE_DIR_PREFIXES,
    ArtifactRef,
    ModelRef,
    RuleResult,
    RunManifest,
    StageEvent,
    StageRecord,
    TargetRef,
    ValidationReport,
    stage_dir,
    stage_dir_canonical,
)
from compgen.graph_compilation.hashing import sha256_file, sha256_tree
from compgen.graph_compilation.validate import validate_run

__all__ = [
    "ArtifactRef",
    "CANONICAL_STAGE_ORDER",
    "LEGACY_STAGE_DIR_PREFIXES",
    "ModelRef",
    "RuleResult",
    "RunManifest",
    "STAGE_DIR_PREFIXES",
    "StageEvent",
    "StageRecord",
    "TargetRef",
    "ValidationReport",
    "sha256_file",
    "sha256_tree",
    "stage_dir",
    "stage_dir_canonical",
    "validate_run",
]
