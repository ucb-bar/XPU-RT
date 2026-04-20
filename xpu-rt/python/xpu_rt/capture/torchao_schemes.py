"""Enumerated TorchAO quantization schemes.

Per the approved wave-6 plan's user directive: coverage for **every**
TorchAO format — transformation paths, not runtime validation. This
module is the single source of truth for what schemes CompGen
recognizes. The capture pipeline (``torchao_pipeline.py``), the
FX-import decomposition table, the IR-pass frozensets, and the
coverage probe (``scripts/14_torchao_coverage_probe.py``) all read
from this catalog.

Entries are grouped by stability (stable / prototype / QAT) and tagged
with the hardware target so a caller can filter by what's available
locally. Every scheme's ``config_class_path`` is a dotted string
resolved lazily at runtime — missing TorchAO versions don't break the
import.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Literal

Stability = Literal["stable", "prototype", "qat", "compgen_custom"]
Granularity = Literal[
    "per_tensor",
    "per_channel",
    "per_group",
    "per_block",
    "per_token",
    "affine_fake",
]


@dataclass(frozen=True)
class TorchAOScheme:
    """Descriptor of one TorchAO quantization format.

    Attributes:
        name: Stable short identifier used by callers (e.g. "int4_weight_only").
        config_class_path: Dotted path to the TorchAO config class or QAT API
            (e.g. "torchao.quantization.Int4WeightOnlyConfig"). Resolved
            lazily; missing paths produce ``status="schema_only"``.
        weight_dtype: Dtype of the stored weight.
        activation_dtype: Dtype of the runtime activation, or None if weight-only.
        granularity: Quantization granularity.
        stability: stable | prototype | qat | compgen_custom
        target_hardware: Informational — where this format is expected to
            run (e.g. "cuda_hopper", "cpu", "any", "npu").
        params: Default parameters (group_size, bit_width, etc.) used
            when the capture pipeline instantiates the config.
        notes: Free-form. Surface additional caveats here.
    """

    name: str
    config_class_path: str
    weight_dtype: str
    granularity: Granularity
    stability: Stability
    target_hardware: str
    activation_dtype: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# Stable TorchAO configs (>=0.7)
# ---------------------------------------------------------------------------

_STABLE: tuple[TorchAOScheme, ...] = (
    TorchAOScheme(
        name="int8_weight_only",
        config_class_path="torchao.quantization.Int8WeightOnlyConfig",
        weight_dtype="int8",
        granularity="per_channel",
        stability="stable",
        target_hardware="any",
    ),
    TorchAOScheme(
        name="int4_weight_only",
        config_class_path="torchao.quantization.Int4WeightOnlyConfig",
        weight_dtype="int4",
        granularity="per_group",
        stability="stable",
        target_hardware="cuda",
        params={"group_size": 128},
        notes="TinyGEMM kernel — group_size commonly 32/64/128/256.",
    ),
    TorchAOScheme(
        name="float8_weight_only_e4m3",
        config_class_path="torchao.quantization.Float8WeightOnlyConfig",
        weight_dtype="float8_e4m3fn",
        granularity="per_tensor",
        stability="stable",
        target_hardware="cuda_hopper",
    ),
    TorchAOScheme(
        name="float8_weight_only_e5m2",
        config_class_path="torchao.quantization.Float8WeightOnlyConfig",
        weight_dtype="float8_e5m2",
        granularity="per_tensor",
        stability="stable",
        target_hardware="cuda_hopper",
        params={"weight_dtype": "float8_e5m2"},
    ),
    TorchAOScheme(
        name="float8_dyn_act_float8_weight",
        config_class_path="torchao.quantization.Float8DynamicActivationFloat8WeightConfig",
        weight_dtype="float8_e4m3fn",
        activation_dtype="float8_e4m3fn",
        granularity="per_tensor",
        stability="stable",
        target_hardware="cuda_hopper",
        notes="Hopper fast-accum FP8 MM.",
    ),
    TorchAOScheme(
        name="int8_static_act_int8_weight",
        config_class_path="torchao.quantization.Int8StaticActivationInt8WeightConfig",
        weight_dtype="int8",
        activation_dtype="int8",
        granularity="per_channel",
        stability="stable",
        target_hardware="any",
    ),
    TorchAOScheme(
        name="int8_dyn_act_int8_weight",
        config_class_path="torchao.quantization.Int8DynamicActivationInt8WeightConfig",
        weight_dtype="int8",
        activation_dtype="int8",
        granularity="per_channel",
        stability="stable",
        target_hardware="any",
    ),
    TorchAOScheme(
        name="float8_dyn_act_int4_weight",
        config_class_path="torchao.quantization.Float8DynamicActivationInt4WeightConfig",
        weight_dtype="int4",
        activation_dtype="float8_e4m3fn",
        granularity="per_group",
        stability="stable",
        target_hardware="cuda",
    ),
    TorchAOScheme(
        name="int8_dyn_act_intx_weight",
        config_class_path="torchao.quantization.Int8DynamicActivationIntxWeightConfig",
        weight_dtype="intx",
        activation_dtype="int8",
        granularity="per_group",
        stability="stable",
        target_hardware="cpu",
        notes="Intx bit widths 1..8; KLEIDIAI / ATen packing on ARM CPU.",
    ),
    TorchAOScheme(
        name="intx_weight_only",
        config_class_path="torchao.quantization.IntxWeightOnlyConfig",
        weight_dtype="intx",
        granularity="per_group",
        stability="stable",
        target_hardware="any",
    ),
)


# ---------------------------------------------------------------------------
# Prototype configs (schema-only, mostly H100+ or experimental)
# ---------------------------------------------------------------------------

_PROTOTYPE: tuple[TorchAOScheme, ...] = (
    TorchAOScheme(
        name="uintx_weight_only",
        config_class_path="torchao.prototype.quantization.quant_api.UIntxWeightOnlyConfig",
        weight_dtype="uint4",
        granularity="per_group",
        stability="prototype",
        target_hardware="cuda",
        notes="Triton Gemlite backend.",
    ),
    TorchAOScheme(
        name="int8_dyn_act_uintx_weight",
        config_class_path="torchao.prototype.quantization.quant_api.Int8DynamicActivationUIntxWeightConfig",
        weight_dtype="uint4",
        activation_dtype="int8",
        granularity="per_group",
        stability="prototype",
        target_hardware="cuda",
    ),
    TorchAOScheme(
        name="float8_static_act_float8_weight",
        config_class_path="torchao.prototype.quantization.quant_api.Float8StaticActivationFloat8WeightConfig",
        weight_dtype="float8_e4m3fn",
        activation_dtype="float8_e4m3fn",
        granularity="per_tensor",
        stability="prototype",
        target_hardware="cuda_hopper",
    ),
    TorchAOScheme(
        name="mx_dyn_act_mx_weight_mx4",
        config_class_path="torchao.prototype.mx_formats.inference_workflow.MXDynamicActivationMXWeightConfig",
        weight_dtype="mx4",
        activation_dtype="mx4",
        granularity="per_block",
        stability="prototype",
        target_hardware="cuda_blackwell",
        notes="Block-wise MX4 quantization.",
    ),
    TorchAOScheme(
        name="mx_dyn_act_mx_weight_mx6",
        config_class_path="torchao.prototype.mx_formats.inference_workflow.MXDynamicActivationMXWeightConfig",
        weight_dtype="mx6",
        activation_dtype="mx6",
        granularity="per_block",
        stability="prototype",
        target_hardware="cuda_blackwell",
    ),
    TorchAOScheme(
        name="mx_dyn_act_mx_weight_mx9",
        config_class_path="torchao.prototype.mx_formats.inference_workflow.MXDynamicActivationMXWeightConfig",
        weight_dtype="mx9",
        activation_dtype="mx9",
        granularity="per_block",
        stability="prototype",
        target_hardware="cuda_blackwell",
    ),
    TorchAOScheme(
        name="nvfp4_weight_only",
        config_class_path="torchao.prototype.mx_formats.inference_workflow.NVFP4WeightOnlyConfig",
        weight_dtype="nvfp4",
        granularity="per_block",
        stability="prototype",
        target_hardware="cuda_hopper_plus",
    ),
    TorchAOScheme(
        name="nvfp4_dyn_act_nvfp4_weight",
        config_class_path="torchao.prototype.mx_formats.inference_workflow.NVFP4DynamicActivationNVFP4WeightConfig",
        weight_dtype="nvfp4",
        activation_dtype="nvfp4",
        granularity="per_block",
        stability="prototype",
        target_hardware="cuda_hopper_plus",
    ),
)


# ---------------------------------------------------------------------------
# QAT (Quantization-Aware Training) configs
# ---------------------------------------------------------------------------

_QAT: tuple[TorchAOScheme, ...] = (
    TorchAOScheme(
        name="qat_intx_affine_fake",
        config_class_path="torchao.quantization.qat.api.intx_quantization_aware_training",
        weight_dtype="intx",
        granularity="affine_fake",
        stability="qat",
        target_hardware="any",
        notes="Affine fake-quantizer for int1..int8 weights.",
    ),
    TorchAOScheme(
        name="qat_mx_fake",
        config_class_path="torchao.prototype.qat.api",
        weight_dtype="mx4",
        granularity="affine_fake",
        stability="qat",
        target_hardware="cuda_blackwell",
        notes="MX fake-quant for QAT; prototype.",
    ),
    TorchAOScheme(
        name="qat_nvfp4_fake",
        config_class_path="torchao.prototype.qat.api",
        weight_dtype="nvfp4",
        granularity="affine_fake",
        stability="qat",
        target_hardware="cuda_hopper_plus",
        notes="NVFP4 fake-quant for QAT; prototype.",
    ),
)


# ---------------------------------------------------------------------------
# CompGen-custom (not TorchAO but ships in this project)
# ---------------------------------------------------------------------------

_COMPGEN_CUSTOM: tuple[TorchAOScheme, ...] = (
    TorchAOScheme(
        name="fp8_e4m3_po2_npu",
        config_class_path="compgen.quantization.smolvla_recipe.apply_smolvla_quantization",
        weight_dtype="float8_e4m3fn",
        granularity="per_tensor",
        stability="compgen_custom",
        target_hardware="npu",
        notes="Power-of-two-scaled FP8 E4M3 for NPU deployment (smolVLA).",
    ),
    TorchAOScheme(
        name="fp8_e4m3_po2",
        config_class_path="compgen.quantization.fp8_config.FP8E4M3Po2Config",
        weight_dtype="float8_e4m3fn",
        granularity="per_tensor",
        stability="compgen_custom",
        target_hardware="cuda_hopper",
    ),
)


# ---------------------------------------------------------------------------
# Public catalog
# ---------------------------------------------------------------------------

TORCHAO_SCHEMES: dict[str, TorchAOScheme] = {s.name: s for s in (*_STABLE, *_PROTOTYPE, *_QAT, *_COMPGEN_CUSTOM)}


def list_schemes(
    *,
    stability: Stability | None = None,
    target_hardware: str | None = None,
) -> list[TorchAOScheme]:
    """Return every scheme, optionally filtered by stability or target.

    No hardware detection: ``target_hardware`` is a string-match filter.
    """
    out: list[TorchAOScheme] = []
    for s in TORCHAO_SCHEMES.values():
        if stability is not None and s.stability != stability:
            continue
        if target_hardware is not None and s.target_hardware != target_hardware:
            continue
        out.append(s)
    return out


def resolve_config(scheme: TorchAOScheme) -> Any | None:
    """Lazy-resolve ``scheme.config_class_path`` to the Python object.

    Returns ``None`` if the path can't be resolved (TorchAO not
    installed, version too old, etc.). Callers treat ``None`` as
    ``status="schema_only"`` — the scheme is still in the catalog for
    transformation-path purposes, just not instantiable today.
    """
    path = scheme.config_class_path
    if "." not in path:
        return None
    module_path, _, attr = path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return None
    return getattr(module, attr, None)


def scheme_status(scheme: TorchAOScheme) -> str:
    """Return ``"ok"`` if the config class resolves, else ``"schema_only"``."""
    return "ok" if resolve_config(scheme) is not None else "schema_only"


__all__ = [
    "Granularity",
    "Stability",
    "TORCHAO_SCHEMES",
    "TorchAOScheme",
    "list_schemes",
    "resolve_config",
    "scheme_status",
]
