"""Registry: load and cross-validate every model + slice + suite YAML.

The registry walks ``configs/model_admission/``, ``configs/models/``, and
``configs/slices/`` (or any directories the caller passes in), validates
each YAML against its schema dataclass, and verifies cross-references:

- every ``model_registry.yaml`` entry has a corresponding model config file,
- every slice's ``parent_model_id`` resolves to a model config,
- every suite entry's ``model_id`` (and ``slice_id`` when set) resolves.

Failed validation raises :class:`RegistryError` with a precise pointer to
the offending file and field. No silent skips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog
import yaml

from compgen.model_admission.schemas import (
    REGISTRY_SCHEMA,
    ModelConfig,
    SliceConfig,
    SuiteConfig,
    _expect_schema,
)

log = structlog.get_logger(__name__)

DEFAULT_REGISTRY_PATH = Path("configs/model_admission/model_registry.yaml")
DEFAULT_MODELS_DIR = Path("configs/models")
DEFAULT_SLICES_DIR = Path("configs/slices")
DEFAULT_SUITES_DIR = Path("configs/model_admission")


class RegistryError(ValueError):
    """Raised when registry validation fails. Always names the offending file."""


@dataclass(frozen=True)
class RegistryEntry:
    """One row of the top-level model_registry.yaml."""

    model_id: str
    family: str
    scale_class: str
    support_mode: str
    blocking: bool
    requires_online_verification: bool


@dataclass
class Registry:
    """Loaded, cross-validated registry."""

    entries: dict[str, RegistryEntry] = field(default_factory=dict)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    slices: dict[str, SliceConfig] = field(default_factory=dict)
    suites: dict[str, SuiteConfig] = field(default_factory=dict)
    registry_path: Path | None = None
    models_dir: Path | None = None
    slices_dir: Path | None = None
    suites_dir: Path | None = None

    def get_model(self, model_id: str) -> ModelConfig:
        if model_id not in self.models:
            raise RegistryError(f"model_id={model_id!r} not in registry")
        return self.models[model_id]

    def get_slice(self, slice_id: str) -> SliceConfig:
        if slice_id not in self.slices:
            raise RegistryError(f"slice_id={slice_id!r} not in registry")
        return self.slices[slice_id]


def _load_registry_top(path: Path) -> dict[str, RegistryEntry]:
    if not path.exists():
        raise RegistryError(f"registry file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RegistryError(f"{path}: top-level must be a mapping")
    _expect_schema(raw, REGISTRY_SCHEMA, str(path))
    out: dict[str, RegistryEntry] = {}
    for row in raw.get("models", []) or []:
        if not isinstance(row, dict):
            raise RegistryError(f"{path}: each model row must be a mapping, got {type(row).__name__}")
        try:
            entry = RegistryEntry(
                model_id=str(row["model_id"]),
                family=str(row["family"]),
                scale_class=str(row["scale_class"]),
                support_mode=str(row["support_mode"]),
                blocking=bool(row["blocking"]),
                requires_online_verification=bool(row.get("requires_online_verification", True)),
            )
        except KeyError as exc:
            raise RegistryError(f"{path}: registry row missing required field {exc.args[0]!r}") from exc
        if entry.model_id in out:
            raise RegistryError(f"{path}: duplicate model_id={entry.model_id!r}")
        out[entry.model_id] = entry
    return out


def _load_dir(directory: Path, suffix: str = ".yaml") -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix == suffix and p.is_file())


def load_registry(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    slices_dir: Path = DEFAULT_SLICES_DIR,
    suites_dir: Path = DEFAULT_SUITES_DIR,
    *,
    require_registry_entries_have_configs: bool = True,
) -> Registry:
    """Discover and validate every YAML config; return a :class:`Registry`.

    Args:
        registry_path: Top-level model_registry.yaml.
        models_dir: Directory of per-model YAML configs (one per file).
        slices_dir: Directory of per-slice YAML configs (one per file).
        suites_dir: Directory containing suite YAMLs (always_test_models, etc.).
        require_registry_entries_have_configs: If true (default), every entry
            in the top-level registry must have a model config in models_dir.

    Raises:
        RegistryError: on any validation or cross-reference failure.
    """

    registry_path = registry_path.resolve()
    models_dir = models_dir.resolve()
    slices_dir = slices_dir.resolve()
    suites_dir = suites_dir.resolve()

    entries = _load_registry_top(registry_path)

    models: dict[str, ModelConfig] = {}
    for path in _load_dir(models_dir):
        # configs/models/ is a shared namespace (e.g. graph_compilation also writes
        # there with schema_version='graphcomp_model_config_v1'). Skip files that
        # don't claim to be ours rather than rejecting them.
        try:
            head = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RegistryError(f"{path}: invalid YAML: {exc}") from exc
        if not isinstance(head, dict):
            continue
        if head.get("schema_version") != "model_config_v1":
            continue
        try:
            cfg = ModelConfig.from_yaml(path)
        except Exception as exc:
            raise RegistryError(f"{path}: failed to parse model config: {exc}") from exc
        if cfg.model_id in models:
            raise RegistryError(f"{path}: duplicate model_id={cfg.model_id!r}")
        models[cfg.model_id] = cfg
        if cfg.loader.kind == "compgen_model_spec" and not cfg.loader.model_spec_id:
            raise RegistryError(f"{path}: loader.kind=compgen_model_spec requires model_spec_id")
        if cfg.loader.kind == "proxy" and not cfg.loader.proxy_module:
            raise RegistryError(f"{path}: loader.kind=proxy requires proxy_module")

    slices: dict[str, SliceConfig] = {}
    for path in _load_dir(slices_dir):
        try:
            head = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RegistryError(f"{path}: invalid YAML: {exc}") from exc
        if not isinstance(head, dict) or head.get("schema_version") != "slice_config_v1":
            continue
        try:
            sl = SliceConfig.from_yaml(path)
        except Exception as exc:
            raise RegistryError(f"{path}: failed to parse slice config: {exc}") from exc
        if sl.slice_id in slices:
            raise RegistryError(f"{path}: duplicate slice_id={sl.slice_id!r}")
        if sl.parent_model_id not in models:
            # Orphan slice: parent isn't a registered admission model. This
            # happens for in-progress work that uses slice_config_v1 to point
            # at parent models in other (e.g. payload_lowering) packages. We
            # skip the slice rather than hard-failing the registry, but a
            # suite entry that references it will fail loudly later.
            log.warning(
                "orphan_slice_skipped",
                path=str(path),
                slice_id=sl.slice_id,
                parent_model_id=sl.parent_model_id,
            )
            continue
        slices[sl.slice_id] = sl

    suites: dict[str, SuiteConfig] = {}
    for path in _load_dir(suites_dir):
        if path.name == registry_path.name:
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RegistryError(f"{path}: invalid YAML: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != "model_admission_suite_v1":
            continue
        try:
            suite = SuiteConfig.from_yaml(path)
        except Exception as exc:
            raise RegistryError(f"{path}: failed to parse suite config: {exc}") from exc
        for entry in suite.all_entries():
            if entry.model_id not in models:
                raise RegistryError(
                    f"{path}: suite entry references unknown model_id={entry.model_id!r}"
                )
            if entry.slice_id and entry.slice_id not in slices:
                raise RegistryError(
                    f"{path}: suite entry references unknown slice_id={entry.slice_id!r}"
                )
        suites[path.stem] = suite

    if require_registry_entries_have_configs:
        missing = [mid for mid in entries if mid not in models]
        if missing:
            raise RegistryError(
                f"{registry_path}: registry entries lack matching model configs in {models_dir}: "
                f"{missing}"
            )

    log.debug(
        "registry_loaded",
        registry=len(entries),
        models=len(models),
        slices=len(slices),
        suites=len(suites),
    )
    return Registry(
        entries=entries,
        models=models,
        slices=slices,
        suites=suites,
        registry_path=registry_path,
        models_dir=models_dir,
        slices_dir=slices_dir,
        suites_dir=suites_dir,
    )


__all__ = [
    "DEFAULT_MODELS_DIR",
    "DEFAULT_REGISTRY_PATH",
    "DEFAULT_SLICES_DIR",
    "DEFAULT_SUITES_DIR",
    "Registry",
    "RegistryEntry",
    "RegistryError",
    "load_registry",
]
