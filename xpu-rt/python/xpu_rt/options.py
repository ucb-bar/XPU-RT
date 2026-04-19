"""``CompGenOptions`` -- single dataclass of compilation knobs.

Mirrors hexagon-mlir's ``HexagonOptions``
(see `/scratch2/agustin/CompGen/tmp/hexagon-mlir/qcom_hexagon_backend/backend/hexagon_options.py`):
hash-able, serializable to/from a flat string-dict, with one flag per
optional pass or tunable. Waves 2+ passes accept a ``CompGenOptions``
argument and read their own switch from it.

Why a single top-level dataclass instead of per-pass configs:

- Reproducibility: the entire compilation of a model is characterized
  by ``(model, inputs, CompGenOptions)``. If those three hash to a
  known set, the output is cacheable.
- Search: the autotune loop mutates knobs on this object and
  re-runs compilation. A single schema makes the search space
  explicit.
- Serialization: dumping the options dict alongside the bundle
  means anyone can reproduce a run.

The defaults are **conservative**: every Wave 1+ pass is off by
default. Callers opt in per objective.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any


_DEFAULT_NUMERICS_POLICY = "preserve_f32"
_VALID_NUMERICS_POLICIES: frozenset[str] = frozenset(
    {"preserve_f32", "demote_activations_to_bf16", "demote_activations_to_f16",
     "normalize_to_target"}
)

_DEFAULT_SCHEDULING_POLICY = "static"
_VALID_SCHEDULING_POLICIES: frozenset[str] = frozenset(
    {"static", "dynamic", "hybrid"}
)


@dataclass(frozen=True)
class CompGenOptions:
    """Compilation knobs driving the 22-pass pipeline.

    Each flag is off (or set to its safe default) by default; callers
    opt in when they want a specific pass or optimization to fire.

    Groups:

    - **Wave 1 -- structural / numerics**:
        ``enable_decompose_concat``, ``enable_fold_transposes_into_dots``,
        ``enable_demote_contraction_inputs``, ``enable_set_numerics_policy``,
        ``numerics_policy``, ``demote_target_type``.

    - **Wave 2 -- semantic detection / Triton fusion**:
        ``enable_raise_special_ops``,
        ``enable_fuse_softmax_to_triton``,
        ``kernel_family_allowlist`` (e.g. ``frozenset({"triton", "cublas"})``).

    - **Wave 3 -- layout / algebraic**:
        ``enable_propagate_transposes``, ``transpose_aggressiveness``
        (``conservative`` / ``through_elementwise`` / ``through_conv`` /
        ``through_pad``), ``enable_plan_reduction``, ``reduction_policy``
        (``group`` / ``split`` / ``tree_reduce``).

    - **Wave 4 -- quantization**:
        ``enable_lower_quantized_matmul``, ``enable_lower_quantized_conv``,
        ``enable_fuse_dequant_matmul``, ``enable_normalize_subbyte``,
        ``quantized_matmul_policy`` (``always`` / ``zp_zero_only`` / ``skip``).

    - **Wave 5 -- large structural**:
        ``enable_lower_conv_to_img2col``, ``enable_match_library_call``,
        ``library_allowlist``.

    - **Wave 6 -- runtime / memory / streams**:
        ``enable_plan_buffers``, ``enable_insert_copies``,
        ``enable_dma_overlap``, ``enable_alias_io_buffers``,
        ``enable_assign_memory_space``, ``enable_assign_queue``,
        ``enable_assign_streams``, ``enable_insert_host_offload``,
        ``enable_normalize_subbyte_post_layout``.

    - **Target / profile**: ``target_profile`` points at a YAML target
        spec; the driver reads capabilities from it.

    - **Diagnostics**: ``enable_smt_refinement``, ``enable_differential_test``,
        ``regression_tolerance_atol``, ``regression_tolerance_rtol``.
    """

    # --- Wave 1 --------------------------------------------------------

    enable_decompose_concat: bool = False
    enable_fold_transposes_into_dots: bool = False
    enable_demote_contraction_inputs: bool = False
    enable_set_numerics_policy: bool = False
    numerics_policy: str = _DEFAULT_NUMERICS_POLICY
    demote_target_type: str = "bf16"  # or "f16"

    # --- Wave 2 --------------------------------------------------------

    enable_raise_special_ops: bool = False
    enable_fuse_softmax_to_triton: bool = False
    kernel_family_allowlist: frozenset[str] = field(default_factory=frozenset)

    # --- Wave 3 --------------------------------------------------------

    enable_propagate_transposes: bool = False
    transpose_aggressiveness: str = "conservative"  # {conservative, through_elementwise, through_conv, through_pad}
    enable_plan_reduction: bool = False
    reduction_policy: str = "group"  # {group, split, tree_reduce}

    # --- Wave 4 --------------------------------------------------------

    enable_lower_quantized_matmul: bool = False
    enable_lower_quantized_conv: bool = False
    enable_fuse_dequant_matmul: bool = False
    enable_normalize_subbyte: bool = False
    quantized_matmul_policy: str = "always"  # {always, zp_zero_only, skip}
    fuse_dequant_reassoc_safe: bool = True

    # --- Wave 5 --------------------------------------------------------

    enable_lower_conv_to_img2col: bool = False
    enable_match_library_call: bool = False
    library_allowlist: frozenset[str] = field(default_factory=frozenset)

    # --- Wave 6 --------------------------------------------------------

    enable_plan_buffers: bool = False
    enable_insert_copies: bool = False
    enable_dma_overlap: bool = False
    enable_alias_io_buffers: bool = False
    enable_assign_memory_space: bool = False
    enable_assign_queue: bool = False
    enable_assign_streams: bool = False
    enable_insert_host_offload: bool = False
    enable_normalize_subbyte_post_layout: bool = False
    scheduling_policy: str = _DEFAULT_SCHEDULING_POLICY

    # --- target / profile ---------------------------------------------

    target_profile: str = ""  # path to YAML target spec
    dma_line_bytes: int = 64
    vtcm_bytes: int = 0  # 0 means "unused"

    # --- diagnostics --------------------------------------------------

    enable_smt_refinement: bool = False
    enable_differential_test: bool = True
    regression_tolerance_atol: float = 1e-3
    regression_tolerance_rtol: float = 1e-3

    # --- misc ---------------------------------------------------------

    apply_recursively: bool = False
    restrict_to_region_ids: frozenset[str] = field(default_factory=frozenset)

    # --- validation + serialization ----------------------------------

    def __post_init__(self) -> None:
        if self.numerics_policy not in _VALID_NUMERICS_POLICIES:
            raise ValueError(
                f"numerics_policy must be one of "
                f"{sorted(_VALID_NUMERICS_POLICIES)}; got {self.numerics_policy!r}"
            )
        if self.scheduling_policy not in _VALID_SCHEDULING_POLICIES:
            raise ValueError(
                f"scheduling_policy must be one of "
                f"{sorted(_VALID_SCHEDULING_POLICIES)}; got {self.scheduling_policy!r}"
            )
        if self.transpose_aggressiveness not in {
            "conservative",
            "through_elementwise",
            "through_conv",
            "through_pad",
        }:
            raise ValueError(
                f"transpose_aggressiveness invalid: {self.transpose_aggressiveness!r}"
            )
        if self.reduction_policy not in {"group", "split", "tree_reduce"}:
            raise ValueError(f"reduction_policy invalid: {self.reduction_policy!r}")
        if self.quantized_matmul_policy not in {"always", "zp_zero_only", "skip"}:
            raise ValueError(
                f"quantized_matmul_policy invalid: {self.quantized_matmul_policy!r}"
            )
        if self.demote_target_type not in {"bf16", "f16"}:
            raise ValueError(
                f"demote_target_type must be bf16 or f16, got {self.demote_target_type!r}"
            )
        if self.regression_tolerance_atol < 0:
            raise ValueError("regression_tolerance_atol must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Flat string-keyed dict suitable for YAML / JSON dump.

        ``frozenset`` fields are serialized as sorted lists.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, frozenset):
                out[f.name] = sorted(value)
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompGenOptions:
        """Build options from the dict form emitted by :meth:`to_dict`."""
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            # Normalize lists back to frozenset when the field type is frozenset.
            default_is_set = isinstance(getattr(cls(), f.name), frozenset)
            if default_is_set and isinstance(value, (list, tuple, set)):
                value = frozenset(value)
            kwargs[f.name] = value
        return cls(**kwargs)

    def stable_key(self) -> tuple[tuple[str, Any], ...]:
        """Deterministic hashable key for caching.

        Converts list-valued (former frozenset) entries back to tuples
        so the resulting key is fully hashable.
        """
        def _hashable(v: Any) -> Any:
            if isinstance(v, (list, set, frozenset)):
                return tuple(sorted(v))
            return v

        return tuple(sorted((k, _hashable(v)) for k, v in self.to_dict().items()))

    def replace(self, **changes: Any) -> CompGenOptions:
        """Return a copy of ``self`` with ``changes`` applied.

        Mirrors ``dataclasses.replace`` but with validation.
        """
        current = asdict(self)
        # dataclasses.asdict collapses frozensets to sorted lists;
        # restore them before applying changes.
        for f in fields(self):
            if isinstance(getattr(self, f.name), frozenset):
                current[f.name] = getattr(self, f.name)
        current.update(changes)
        return CompGenOptions(**current)


# --- preset profiles ---------------------------------------------------


def cuda_a100_defaults() -> CompGenOptions:
    """Preset suitable for CUDA A100: BF16 GEMM, Triton kernels."""
    return CompGenOptions(
        enable_fold_transposes_into_dots=True,
        enable_demote_contraction_inputs=True,
        demote_target_type="bf16",
        enable_raise_special_ops=True,
        enable_fuse_softmax_to_triton=True,
        kernel_family_allowlist=frozenset({"triton", "cublas"}),
        enable_match_library_call=True,
        library_allowlist=frozenset({"cublas", "cudnn", "triton"}),
        enable_plan_buffers=True,
        enable_insert_copies=True,
        enable_assign_memory_space=True,
        enable_assign_queue=True,
        enable_assign_streams=True,
        scheduling_policy="dynamic",
        numerics_policy="demote_activations_to_bf16",
    )


def cuda_h100_defaults() -> CompGenOptions:
    """Preset suitable for CUDA H100: FP8 GEMM, Triton kernels, dma_overlap."""
    return cuda_a100_defaults().replace(
        demote_target_type="f16",
        enable_dma_overlap=True,
    )


def npu_fp8_defaults() -> CompGenOptions:
    """Preset for an FP8 NPU like the one in the smolVLA pipeline."""
    return CompGenOptions(
        enable_raise_special_ops=True,
        enable_lower_quantized_matmul=True,
        enable_lower_quantized_conv=True,
        enable_fuse_dequant_matmul=True,
        enable_normalize_subbyte=True,
        enable_plan_buffers=True,
        enable_insert_copies=True,
        enable_assign_memory_space=True,
        enable_assign_queue=True,
        enable_assign_streams=True,
        enable_dma_overlap=True,
        enable_insert_host_offload=True,
        enable_normalize_subbyte_post_layout=True,
        quantized_matmul_policy="zp_zero_only",
        fuse_dequant_reassoc_safe=True,
        scheduling_policy="static",
        numerics_policy="normalize_to_target",
        kernel_family_allowlist=frozenset({"qnn", "ukernel"}),
        library_allowlist=frozenset({"qnn"}),
    )


__all__ = [
    "CompGenOptions",
    "cuda_a100_defaults",
    "cuda_h100_defaults",
    "npu_fp8_defaults",
]
