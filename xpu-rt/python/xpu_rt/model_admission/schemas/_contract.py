"""Frozen dataclasses for model admission configs and reports.

All on-disk shapes are owned here. Each report has a ``schema_version``
constant of the form ``<name>_v1`` matching the corresponding JSON Schema
under ``compgen/model_admission/schemas/v1/``.

Loaders intentionally accept ``dict[str, Any]`` (already YAML/JSON-decoded)
and validate field-by-field rather than introducing a runtime schema
dependency. JSON Schemas are still emitted for downstream tooling and for
contract documentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import yaml

# --------------------------------------------------------------------------- #
# Schema version constants (must match the const in the JSON Schemas).
# --------------------------------------------------------------------------- #

MODEL_CONFIG_SCHEMA = "model_config_v1"
SLICE_CONFIG_SCHEMA = "slice_config_v1"
SUITE_CONFIG_SCHEMA = "model_admission_suite_v1"
REGISTRY_SCHEMA = "model_registry_v1"
ADMISSION_REPORT_SCHEMA = "admission_report_v1"
EAGER_REPORT_SCHEMA = "eager_report_v1"
DYNAMO_REPORT_SCHEMA = "dynamo_report_v1"
FX_REPORT_SCHEMA = "fx_report_v1"
EXPORT_REPORT_SCHEMA = "export_report_v1"
TORCH_COMPILE_REPORT_SCHEMA = "torch_compile_report_v1"
SUITE_SUMMARY_SCHEMA = "admission_suite_summary_v1"

# --------------------------------------------------------------------------- #
# Status enums.
# --------------------------------------------------------------------------- #


class AdmissionStatus(StrEnum):
    """Top-level outcome of a (model, slice) admission probe.

    Exhaustive set; the probe must always emit exactly one of these.
    """

    AVAILABLE = "available"
    AVAILABLE_SLICE_ONLY = "available_slice_only"
    UNAVAILABLE_MISSING_WEIGHTS = "unavailable_missing_weights"
    UNAVAILABLE_GATED_ACCESS = "unavailable_gated_access"
    UNAVAILABLE_MISSING_DEPENDENCY = "unavailable_missing_dependency"
    UNAVAILABLE_TOO_LARGE = "unavailable_too_large"
    # The model loads + runs in principle, but this host's hardware (compute
    # capability / VRAM / dtype support / etc.) doesn't meet the model's
    # requirements. The model stays in the registry; its
    # ``hardware_requirements`` field records what it needs so a future
    # admission run on different hardware can flip it to ``available``.
    UNAVAILABLE_HARDWARE_CONSTRAINT = "unavailable_hardware_constraint"
    FAILED_EAGER = "failed_eager"
    FAILED_TORCH_COMPILE = "failed_torch_compile"


class StageStatus(StrEnum):
    """Per-stage probe outcome (eager / dynamo / torch.compile)."""

    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #


def _expect_schema(raw: Mapping[str, Any], expected: str, source: str) -> None:
    sv = raw.get("schema_version")
    if sv != expected:
        raise ValueError(
            f"{source}: expected schema_version={expected!r}, got {sv!r}"
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}")
    return raw


# --------------------------------------------------------------------------- #
# Config dataclasses (input).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSource:
    """Provenance metadata for a model. ``source_verified`` must remain false
    until a human has confirmed ``model_ref`` / ``repo_url`` against the live
    upstream. Web search is not a substitute.
    """

    provider: str = ""
    model_ref: str = "TO_BE_VERIFIED_ONLINE"
    repo_url: str = "TO_BE_VERIFIED_ONLINE"
    docs_url: str = "TO_BE_VERIFIED_ONLINE"
    revision: str | None = None
    verified_at: str | None = None
    verified_by: str | None = None
    source_verified: bool = False


@dataclass(frozen=True)
class ModelLoaderConfig:
    """Loader dispatch info.

    ``kind`` selects the loader implementation in ``loaders.py``:

    - ``compgen_model_spec`` - bridge to the existing Python ``ModelCatalog``;
      ``model_spec_id`` names the entry.
    - ``hf_transformers_vlm | hf_transformers_llm | hf_transformers_ocr |
      hf_transformers_vla`` - HuggingFace transformers loaders; the probe
      checks the local HF cache and never downloads.
    - ``proxy`` - importable tiny module under
      ``compgen.model_admission.proxies``.
    - ``unavailable`` - explicit no-op for declared-but-unsupported entries.
    """

    kind: str
    model_spec_id: str = ""
    proxy_module: str = ""
    trust_remote_code: bool = False
    dtype: str = "float32"
    device_policy: str = "auto"
    # Optional: importable Python module that builds a callable + sample
    # inputs for a model whose ``forward()`` doesn't fit the family default.
    # The module must expose ``build(model, processor) -> tuple[nn.Module,
    # tuple, dict]`` returning (wrapped_module_with_standard_forward, args,
    # kwargs). Used for Moondream2 (encode_image) and DeepSeek-OCR (custom
    # images-tuple format).
    adapter: str = ""


@dataclass(frozen=True)
class CompileConfig:
    """torch.compile settings for the admission probe."""

    mode: str = "torch_compile_admission"
    backend: str = "inductor"
    fullgraph: bool = False
    dynamic: bool = True
    # Optional: pin a specific ``transformers`` version for the probe
    # subprocess. Used when a model's bundled remote_code is only compatible
    # with an older transformers (e.g. OpenVLA targets 4.40, DeepSeek-OCR
    # targets 4.46, Moondream2 targets 4.52). The suite runner launches the
    # probe via ``uv run --with transformers==<pin>`` so the pin doesn't
    # touch the project venv. Empty string = use project transformers.
    transformers_pin: str = ""
    # Optional: extra pip-spec strings passed as additional ``--with`` flags
    # alongside ``transformers_pin``. Used when a model's remote_code also
    # needs a specific ``timm``, ``sentencepiece``, etc. Each entry must be
    # a complete pip requirement string (``timm<1.0``, ``num2words==0.5.14``).
    extra_pins: tuple[str, ...] = ()


@dataclass(frozen=True)
class SupportPolicy:
    """How aggressive the admission attempt should be."""

    mode: str = "full_or_slice_smoke"  # or slice_only / one_step_policy / admission_only
    full_model_blocking: bool = True
    reason: str = ""
    # Optional declaration of hardware needs. Populated when the YAML's
    # support section sets `hardware_requirements:`. Loader emits
    # UNAVAILABLE_HARDWARE_CONSTRAINT when set so admission output records
    # the model as "would work on different hardware" rather than dropping it.
    hardware_requirements: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExpectedOutcomes:
    """Best-effort baseline of what we expect from this entry."""

    can_eager_run: bool = True
    can_torch_compile: bool = True
    can_dynamo_capture: bool = True
    can_slice_compile: bool = True


@dataclass(frozen=True)
class InputsSpec:
    """Description of the slice's input shape family.

    The probe uses ``kind`` to dispatch to a synthetic-input builder; for
    proxy and ``compgen_model_spec`` entries the loader provides inputs
    directly and most of these fields are ignored.
    """

    kind: str = "tensor_only"
    processor_required: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    """Top-level model config. One per file under ``configs/models/``."""

    schema_version: str
    model_id: str
    family: str
    source: ModelSource
    loader: ModelLoaderConfig
    inputs: InputsSpec
    compile: CompileConfig
    support: SupportPolicy
    expected: ExpectedOutcomes
    notes: tuple[str, ...]
    raw_path: Path

    @classmethod
    def from_yaml(cls, path: Path) -> ModelConfig:
        path = path.resolve()
        raw = _read_yaml(path)
        _expect_schema(raw, MODEL_CONFIG_SCHEMA, str(path))
        src = raw.get("source", {}) or {}
        ldr = raw.get("loader", {}) or {}
        inp = raw.get("inputs", {}) or {}
        cmp = raw.get("compile", {}) or {}
        sup = raw.get("support", {}) or {}
        exp = raw.get("expected", {}) or {}
        return cls(
            schema_version=raw["schema_version"],
            model_id=str(raw["model_id"]),
            family=str(raw["family"]),
            source=ModelSource(
                provider=str(src.get("provider", "")),
                model_ref=str(src.get("model_ref", "TO_BE_VERIFIED_ONLINE")),
                repo_url=str(src.get("repo_url", "TO_BE_VERIFIED_ONLINE")),
                docs_url=str(src.get("docs_url", "TO_BE_VERIFIED_ONLINE")),
                revision=src.get("revision"),
                verified_at=src.get("verified_at"),
                verified_by=src.get("verified_by"),
                source_verified=bool(src.get("source_verified", False)),
            ),
            loader=ModelLoaderConfig(
                kind=str(ldr["kind"]),
                model_spec_id=str(ldr.get("model_spec_id", "")),
                proxy_module=str(ldr.get("proxy_module", "")),
                trust_remote_code=bool(ldr.get("trust_remote_code", False)),
                dtype=str(ldr.get("dtype", "float32")),
                device_policy=str(ldr.get("device_policy", "auto")),
                adapter=str(ldr.get("adapter", "")),
            ),
            inputs=InputsSpec(
                kind=str(inp.get("kind", "tensor_only")),
                processor_required=bool(inp.get("processor_required", False)),
                extra={k: v for k, v in inp.items() if k not in {"kind", "processor_required"}},
            ),
            compile=CompileConfig(
                mode=str(cmp.get("mode", "torch_compile_admission")),
                backend=str(cmp.get("backend", "inductor")),
                fullgraph=bool(cmp.get("fullgraph", False)),
                dynamic=bool(cmp.get("dynamic", True)),
                transformers_pin=str(cmp.get("transformers_pin", "")),
                extra_pins=tuple(str(p) for p in (cmp.get("extra_pins") or ())),
            ),
            support=SupportPolicy(
                mode=str(sup.get("mode", "full_or_slice_smoke")),
                full_model_blocking=bool(sup.get("full_model_blocking", True)),
                reason=str(sup.get("reason", "")),
                hardware_requirements=dict(sup.get("hardware_requirements") or {}),
            ),
            expected=ExpectedOutcomes(
                can_eager_run=bool(exp.get("can_eager_run", True)),
                can_torch_compile=bool(exp.get("can_torch_compile", True)),
                can_dynamo_capture=bool(exp.get("can_dynamo_capture", True)),
                can_slice_compile=bool(exp.get("can_slice_compile", True)),
            ),
            notes=tuple(str(n) for n in raw.get("notes", []) or []),
            raw_path=path,
        )


@dataclass(frozen=True)
class SliceConfig:
    """Slice declaration. One per file under ``configs/slices/``.

    A slice always references a parent model and may override loader fields
    (e.g. selecting one decoder block from a 235B parent).
    """

    schema_version: str
    slice_id: str
    parent_model_id: str
    description: str
    inputs: InputsSpec
    loader_override: ModelLoaderConfig | None
    expected: ExpectedOutcomes
    raw_path: Path

    @classmethod
    def from_yaml(cls, path: Path) -> SliceConfig:
        path = path.resolve()
        raw = _read_yaml(path)
        _expect_schema(raw, SLICE_CONFIG_SCHEMA, str(path))
        inp = raw.get("inputs", {}) or {}
        exp = raw.get("expected", {}) or {}
        lo_raw = raw.get("loader_override")
        lo: ModelLoaderConfig | None = None
        if lo_raw:
            lo = ModelLoaderConfig(
                kind=str(lo_raw["kind"]),
                model_spec_id=str(lo_raw.get("model_spec_id", "")),
                proxy_module=str(lo_raw.get("proxy_module", "")),
                trust_remote_code=bool(lo_raw.get("trust_remote_code", False)),
                dtype=str(lo_raw.get("dtype", "float32")),
                device_policy=str(lo_raw.get("device_policy", "auto")),
                adapter=str(lo_raw.get("adapter", "")),
            )
        return cls(
            schema_version=raw["schema_version"],
            slice_id=str(raw["slice_id"]),
            parent_model_id=str(raw["parent_model_id"]),
            description=str(raw.get("description", "")),
            inputs=InputsSpec(
                kind=str(inp.get("kind", "tensor_only")),
                processor_required=bool(inp.get("processor_required", False)),
                extra={k: v for k, v in inp.items() if k not in {"kind", "processor_required"}},
            ),
            loader_override=lo,
            expected=ExpectedOutcomes(
                can_eager_run=bool(exp.get("can_eager_run", True)),
                can_torch_compile=bool(exp.get("can_torch_compile", True)),
                can_dynamo_capture=bool(exp.get("can_dynamo_capture", True)),
                can_slice_compile=bool(exp.get("can_slice_compile", True)),
            ),
            raw_path=path,
        )


@dataclass(frozen=True)
class SuiteEntry:
    """One row in a suite file. ``slice_id`` may be empty for full-model probes."""

    model_id: str
    slice_id: str = ""


@dataclass(frozen=True)
class SuiteConfig:
    """Description of an admission run -- proxies, real_if_available, slice_only."""

    schema_version: str
    required_proxy: tuple[SuiteEntry, ...]
    required_real_if_available: tuple[SuiteEntry, ...]
    slice_only_stress: tuple[SuiteEntry, ...]
    raw_path: Path

    def all_entries(self) -> tuple[SuiteEntry, ...]:
        return self.required_proxy + self.required_real_if_available + self.slice_only_stress

    @classmethod
    def from_yaml(cls, path: Path) -> SuiteConfig:
        path = path.resolve()
        raw = _read_yaml(path)
        _expect_schema(raw, SUITE_CONFIG_SCHEMA, str(path))

        def _parse(rows: list[Any] | None) -> tuple[SuiteEntry, ...]:
            out: list[SuiteEntry] = []
            for r in rows or []:
                if not isinstance(r, dict):
                    raise ValueError(f"{path}: suite entry must be a mapping, got {type(r).__name__}")
                out.append(
                    SuiteEntry(
                        model_id=str(r["model_id"]),
                        slice_id=str(r.get("slice_id", "")),
                    )
                )
            return tuple(out)

        return cls(
            schema_version=raw["schema_version"],
            required_proxy=_parse(raw.get("required_proxy")),
            required_real_if_available=_parse(raw.get("required_real_if_available")),
            slice_only_stress=_parse(raw.get("slice_only_stress")),
            raw_path=path,
        )


# --------------------------------------------------------------------------- #
# Report dataclasses (output).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EagerReport:
    """Plain forward-pass result before any compile is attempted."""

    schema_version: str = EAGER_REPORT_SCHEMA
    model_id: str = ""
    slice_id: str = ""
    status: str = StageStatus.SKIPPED.value
    wall_time_s: float = 0.0
    output_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DynamoCaptureReport:
    """Result of ``torch._dynamo.export`` on the slice."""

    schema_version: str = DYNAMO_REPORT_SCHEMA
    model_id: str = ""
    slice_id: str = ""
    status: str = StageStatus.SKIPPED.value
    graph_count: int = 0
    op_count: int = 0
    graph_break_count: int = 0
    graph_breaks: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FxReport:
    """Result of ``torch.fx.symbolic_trace`` on the slice.

    FX is the oldest and least flexible tracing path. It rejects models with
    data-dependent control flow, dynamic shapes, or non-tensor leaf values --
    so a passing FX trace is a strong "graph is static" signal.
    """

    schema_version: str = FX_REPORT_SCHEMA
    model_id: str = ""
    slice_id: str = ""
    status: str = StageStatus.SKIPPED.value
    node_count: int = 0
    op_histogram: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExportReport:
    """Result of ``torch.export.export`` (the dynamo-export ATen graph)."""

    schema_version: str = EXPORT_REPORT_SCHEMA
    model_id: str = ""
    slice_id: str = ""
    status: str = StageStatus.SKIPPED.value
    graph_node_count: int = 0
    op_histogram: dict[str, int] = field(default_factory=dict)
    has_dynamic_shapes: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TorchCompileReport:
    """Result of ``torch.compile`` -- the headline admission metric."""

    schema_version: str = TORCH_COMPILE_REPORT_SCHEMA
    model_id: str = ""
    slice_id: str = ""
    attempted: bool = False
    status: str = StageStatus.SKIPPED.value
    backend: str = "inductor"
    fullgraph: bool = False
    dynamic: bool = True
    compile_time_s: float = 0.0
    first_run_time_s: float = 0.0
    second_run_time_s: float = 0.0
    graph_break_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HardwareRequirements:
    """What a model needs from a host to run (whether or not THIS host meets them).

    Empty/None values mean "unconstrained". Populated by the loader when the
    model declares constraints in its YAML or when a known incompatibility is
    detected at probe time (e.g. flash_attn requires CC >= 8.0).
    """

    min_compute_capability: str = ""    # e.g. "8.0" for Ampere+, "8.9" for Ada/Hopper
    min_vram_gb: float = 0.0
    required_dtypes: tuple[str, ...] = ()  # e.g. ("fp8",) for FP8-only models
    required_runtime_packages: tuple[str, ...] = ()  # e.g. ("flash_attn",)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdmissionReport:
    """Top-level admission outcome for one (model_id, slice_id) pair."""

    schema_version: str = ADMISSION_REPORT_SCHEMA
    model_id: str = ""
    slice_id: str = ""
    status: str = AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY.value
    reason: str = ""
    eager_report_path: str = ""
    fx_report_path: str = ""
    export_report_path: str = ""
    dynamo_report_path: str = ""
    torch_compile_report_path: str = ""
    environment_path: str = ""
    error_path: str | None = None
    recommended_next_step: str = ""
    # When status is UNAVAILABLE_HARDWARE_CONSTRAINT, this field records what
    # hardware would unblock the model. Empty for other statuses.
    hardware_requirements: HardwareRequirements | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SuiteSummaryRow:
    """One row of the always-test suite summary."""

    model_id: str
    slice_id: str
    family: str
    support_mode: str
    blocking: bool
    source_verified: bool
    weights_available: bool
    dependency_status: str
    eager_status: str
    fx_status: str
    export_status: str
    dynamo_status: str
    torch_compile_status: str
    graph_break_count: int
    compile_time_s: float
    recommended_next_step: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SuiteSummary:
    """Aggregated summary for the run-suite command."""

    schema_version: str = SUITE_SUMMARY_SCHEMA
    suite_path: str = ""
    out_dir: str = ""
    rows: list[SuiteSummaryRow] = field(default_factory=list)
    total: int = 0
    available: int = 0
    available_slice_only: int = 0
    unavailable: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_path": self.suite_path,
            "out_dir": self.out_dir,
            "total": self.total,
            "available": self.available,
            "available_slice_only": self.available_slice_only,
            "unavailable": self.unavailable,
            "failed": self.failed,
            "rows": [r.to_dict() for r in self.rows],
        }


__all__ = [
    "ADMISSION_REPORT_SCHEMA",
    "AdmissionReport",
    "AdmissionStatus",
    "CompileConfig",
    "DYNAMO_REPORT_SCHEMA",
    "DynamoCaptureReport",
    "EAGER_REPORT_SCHEMA",
    "EagerReport",
    "ExpectedOutcomes",
    "InputsSpec",
    "MODEL_CONFIG_SCHEMA",
    "ModelConfig",
    "ModelLoaderConfig",
    "ModelSource",
    "REGISTRY_SCHEMA",
    "SLICE_CONFIG_SCHEMA",
    "SUITE_CONFIG_SCHEMA",
    "SUITE_SUMMARY_SCHEMA",
    "SliceConfig",
    "StageStatus",
    "SuiteConfig",
    "SuiteEntry",
    "SuiteSummary",
    "SuiteSummaryRow",
    "SupportPolicy",
    "TORCH_COMPILE_REPORT_SCHEMA",
    "TorchCompileReport",
]
