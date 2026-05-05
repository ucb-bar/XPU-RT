"""Runtime import provenance (M-31A.2).

Snapshots ``sys.modules`` at well-defined points around a production run
and records which modules were imported. Production runs that import a
forbidden module (e.g. ``compgen.llm.mock_client``) fail the audit — that
is what tells a reader the run is real.

The provenance file (``<run_dir>/import_provenance.json``) is written at
the same seam as ``run_manifest.json`` and references the same
``run_id`` so cross-checks are trivial.

Cache mode is one of:

- ``cold``     — the run was directed to a clean output dir; no recipe
                 cache, kernel cache, or memory store was consulted
                 (assumed; not directly observable here)
- ``warm``     — the recipe cache may have been consulted; the default
- ``disabled`` — at least one of ``COMPGEN_DISABLE_RECIPE_MEMORY`` /
                 ``COMPGEN_DISABLE_KERNEL_CACHE`` is set
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compgen.audit.errors import ForbiddenImportError

# Modules that are forbidden on production paths. If any of these appears
# in the post-run snapshot of a production-classified run, the audit
# fails.
DEFAULT_FORBIDDEN_MODULES: tuple[str, ...] = (
    "compgen.llm.mock_client",
    "compgen.runtime.transport",  # StubNetworkTransport lives here
    "compgen.ir.payload.passes.runtime_stubs",
)

# Modules that are categorically "mock" (regardless of where they live).
# Importing one is not a hard fail by itself — many tests do — but it
# bumps the ``evidence_mode`` to ``mocked`` or ``mixed``.
DEFAULT_MOCK_MODULES: tuple[str, ...] = (
    "compgen.llm.mock_client",
    "compgen.memory.embeddings",  # MockEmbeddingProvider may be re-exported
    "compgen.kernels.providers.claude_code_default",  # StubCodegen
    "compgen.capture.unsupported.synthesize_fake",
)


@dataclass(frozen=True)
class ImportSnapshot:
    """Snapshot of relevant sys.modules keys at a moment in time."""

    label: str
    modules: tuple[str, ...]

    @classmethod
    def take(cls, label: str, *, prefixes: tuple[str, ...] = ("compgen", "torch")) -> ImportSnapshot:
        keys = sorted(
            name
            for name in sys.modules
            if any(name == p or name.startswith(p + ".") for p in prefixes)
        )
        return cls(label=label, modules=tuple(keys))


def _classify_cache_mode() -> str:
    disabled_recipe = os.environ.get("COMPGEN_DISABLE_RECIPE_MEMORY") == "1"
    disabled_kernel = os.environ.get("COMPGEN_DISABLE_KERNEL_CACHE") == "1"
    if disabled_recipe or disabled_kernel:
        return "disabled"
    return "warm"


def _classify_evidence_mode(mock_modules_imported: list[str]) -> str:
    if not mock_modules_imported:
        return "real"
    return "mocked"


@dataclass
class ImportProvenance:
    """Provenance dossier written next to ``run_manifest.json``."""

    schema_version: str = "import_provenance_v1"
    run_id: str = ""
    selection_mode: str = ""
    source_commit: str = ""
    cache_mode: str = "warm"
    evidence_mode: str = "real"
    production_modules_imported: list[str] = field(default_factory=list)
    mock_modules_imported: list[str] = field(default_factory=list)
    forbidden_modules_imported: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "selection_mode": self.selection_mode,
            "source_commit": self.source_commit,
            "cache_mode": self.cache_mode,
            "evidence_mode": self.evidence_mode,
            "production_modules_imported": list(self.production_modules_imported),
            "mock_modules_imported": list(self.mock_modules_imported),
            "forbidden_modules_imported": list(self.forbidden_modules_imported),
            "env_overrides": dict(self.env_overrides),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImportProvenance:
        return cls(
            schema_version=str(data.get("schema_version", "import_provenance_v1")),
            run_id=str(data.get("run_id", "")),
            selection_mode=str(data.get("selection_mode", "")),
            source_commit=str(data.get("source_commit", "")),
            cache_mode=str(data.get("cache_mode", "warm")),
            evidence_mode=str(data.get("evidence_mode", "real")),
            production_modules_imported=list(
                data.get("production_modules_imported") or []
            ),
            mock_modules_imported=list(data.get("mock_modules_imported") or []),
            forbidden_modules_imported=list(
                data.get("forbidden_modules_imported") or []
            ),
            env_overrides=dict(data.get("env_overrides") or {}),
        )


def compute_provenance(
    *,
    before: ImportSnapshot,
    after: ImportSnapshot,
    run_id: str,
    selection_mode: str,
    source_commit: str,
    forbidden_modules: tuple[str, ...] = DEFAULT_FORBIDDEN_MODULES,
    mock_modules: tuple[str, ...] = DEFAULT_MOCK_MODULES,
) -> ImportProvenance:
    """Diff two snapshots and classify.

    Forbidden / mock imports are scored against the *newly-loaded*
    modules (after − before), not the cumulative ``sys.modules``.
    A previously-loaded mock that this run did not trigger is not
    this run's failure — that would make the audit hostage to whatever
    a prior pytest case happened to import.
    """
    new_modules = sorted(set(after.modules) - set(before.modules))
    forbidden_imported = [
        m for m in new_modules
        if any(m == f or m.startswith(f + ".") for f in forbidden_modules)
    ]
    mock_imported = [
        m for m in new_modules
        if any(m == f or m.startswith(f + ".") for f in mock_modules)
    ]
    env_overrides = {
        k: v for k, v in os.environ.items()
        if k.startswith("COMPGEN_") and k in {
            "COMPGEN_DISABLE_RECIPE_MEMORY",
            "COMPGEN_DISABLE_KERNEL_CACHE",
            "COMPGEN_FORCE_REBUILD",
            "COMPGEN_RUN_KERNELS",
            "COMPGEN_CALIBRATE_PROFILER",
            "COMPGEN_CALIBRATE_CANDIDATES",
        }
    }
    return ImportProvenance(
        run_id=run_id,
        selection_mode=selection_mode,
        source_commit=source_commit,
        cache_mode=_classify_cache_mode(),
        evidence_mode=_classify_evidence_mode(mock_imported),
        production_modules_imported=new_modules,
        mock_modules_imported=sorted(set(mock_imported)),
        forbidden_modules_imported=sorted(set(forbidden_imported)),
        env_overrides=env_overrides,
    )


def write_provenance(provenance: ImportProvenance, *, run_dir: Path) -> Path:
    out_path = run_dir / "import_provenance.json"
    out_path.write_text(
        json.dumps(provenance.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def assert_no_forbidden(
    provenance: ImportProvenance,
    *,
    additional_forbidden: tuple[str, ...] = (),
) -> None:
    """Raise :class:`ForbiddenImportError` if any forbidden module loaded."""
    bad = list(provenance.forbidden_modules_imported)
    for extra in additional_forbidden:
        bad.extend(
            m for m in provenance.production_modules_imported
            if m == extra or m.startswith(extra + ".")
        )
    if bad:
        raise ForbiddenImportError(
            f"production run {provenance.run_id} imported forbidden modules: "
            f"{sorted(set(bad))}"
        )


def load_provenance(path: Path) -> ImportProvenance:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ImportProvenance.from_dict(raw)
