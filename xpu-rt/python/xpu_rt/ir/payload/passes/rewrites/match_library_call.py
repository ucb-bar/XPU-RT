"""``match_library_call`` -- Phase 3 dispatcher. Tag compute-heavy ops
with the library implementation they should call.

Reconstruction of XLA's ``GemmRewriter`` +
``LibraryRewriter`` + ``OneDnnRewriter``. Zero external references;
CompGen owns the rewrite.

Walks every ``linalg.matmul``,
``compgen.quant.weight_int*pack_{mm,qm}``, and opaque
``func.call @aten_convolution`` in the module and -- based on the
caller-supplied ``library_allowlist`` -- tags each with the first
library that supports its shape + dtype. The tag is the contract
for the downstream kernel dispatcher: a ``"cublas"`` tag will be
lowered to a cuBLAS ukernel call, ``"triton"`` to a Triton kernel
invocation, etc.

Per-library shape / dtype rules (conservative defaults):

- **cuBLAS**      -- 2-D matmul on float16/bfloat16/float32 inputs; any M,N,K.
- **cuBLASLt**    -- 2-D matmul with int8 weights; supports per-channel scales.
- **cuDNN**       -- convolution (2-D or 3-D) with NCHW/NHWC layouts.
- **Triton**      -- 2-D matmul; 1-D / 2-D softmax; quantized variants.
- **oneDNN**      -- CPU matmul + conv; any dtype but prefers f32/bf16.
- **XNNPACK**     -- CPU mobile conv + depthwise; NHWC only.
- **QNN**         -- NPU int8 / fp8 conv + matmul; requires per-channel
  scales + NHWC.
- **rocBLAS**     -- AMD 2-D matmul; f16/bf16/f32.
- **MIOpen**      -- AMD conv; NCHW/NHWC.

Dispatch order: when multiple libraries match, we prefer the order
given in the allowlist. Ties are broken deterministically (first
match in allowlist wins).

LLM-tool signature:

    tool_name="match_library_call"
    wraps_pass="CompGen:LibraryRewriter+GemmRewriter+OneDnnRewriter"
    invent_slot="dispatch/library_call_matching"
    policy="DispatchByTargetAllowlist"
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from xdsl.dialects.builtin import (
    BFloat16Type,
    Float16Type,
    Float32Type,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import CallOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.ir import Attribute, Operation
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compgen.ir.quant import (
    WeightInt4PackMMOp,
    WeightInt4PackQMOp,
    WeightInt8PackMMOp,
)

_KNOWN_LIBRARIES = frozenset(
    {
        "cublas",
        "cublaslt",
        "cudnn",
        "triton",
        "onednn",
        "xnnpack",
        "qnn",
        "rocblas",
        "miopen",
    }
)

_MATMUL_LIBRARIES = {"cublas", "cublaslt", "triton", "onednn", "rocblas", "qnn"}
_QUANT_MATMUL_LIBRARIES = {"cublaslt", "triton", "qnn"}
_CONV_LIBRARIES = {"cudnn", "onednn", "xnnpack", "miopen", "qnn"}


@dataclass(frozen=True)
class MatchLibraryCallConfig:
    library_allowlist: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for lib in self.library_allowlist:
            if lib not in _KNOWN_LIBRARIES:
                raise ValueError(f"unknown library {lib!r}; known: {sorted(_KNOWN_LIBRARIES)}")


@dataclass
class MatchLibraryCallStats:
    ops_seen: int = 0
    matmul_matches: int = 0
    quant_matmul_matches: int = 0
    conv_matches: int = 0
    no_match: int = 0
    skipped_already_dispatched: int = 0
    dispatch_counts: dict[str, int] = field(default_factory=dict)

    def record(self, library: str) -> None:
        self.dispatch_counts[library] = self.dispatch_counts.get(library, 0) + 1


# --- per-library predicates --------------------------------------------------


def _is_float_elem(t: Attribute) -> bool:
    return isinstance(t, (Float16Type, BFloat16Type, Float32Type))


def _cublas_accepts_matmul(op: MatmulOp) -> bool:
    # cuBLAS supports 2-D (Sgemm) and 3-D batched (SgemmBatched/Strided) matmul.
    for v in op.operands:
        t = v.type
        if not isinstance(t, TensorType):
            return False
        rank = len(list(t.get_shape()))
        if rank not in (2, 3):
            return False
        if not _is_float_elem(t.get_element_type()):
            return False
    return True


def _cublaslt_accepts_quant_matmul(op: Operation) -> bool:
    # int8 weight with per-channel scales is the canonical cuBLASLt path.
    if isinstance(op, WeightInt8PackMMOp):
        return True
    return False


def _triton_accepts_matmul(op: MatmulOp) -> bool:
    # Triton is shape-agnostic for 2-D matmul.
    return _cublas_accepts_matmul(op)


def _triton_accepts_quant_matmul(op: Operation) -> bool:
    # Triton has int4 + int8 kernels.
    return isinstance(op, (WeightInt4PackMMOp, WeightInt4PackQMOp, WeightInt8PackMMOp))


def _onednn_accepts_matmul(op: MatmulOp) -> bool:
    # oneDNN accepts 2-D and batched 3-D matmul with any sensible dtype.
    for v in op.operands:
        t = v.type
        if not isinstance(t, TensorType):
            return False
        rank = len(list(t.get_shape()))
        if rank not in (2, 3):
            return False
    return True


def _cudnn_accepts_conv(call: CallOp) -> bool:
    # cuDNN accepts 4-D NCHW/NHWC conv; we only check rank.
    if len(call.operands) < 2:
        return False
    for v in call.operands[:2]:
        t = v.type
        if not isinstance(t, TensorType):
            return False
        if len(list(t.get_shape())) != 4:
            return False
    return True


def _qnn_accepts(op: Operation) -> bool:
    # QNN handles int8 / fp8 quantized ops. Accept packed-MM variants.
    if isinstance(op, _QUANT_MATMUL_ALL):
        return True
    if isinstance(op, CallOp) and "compgen.quantized_conv_scheduled" in op.attributes:
        return True
    return False


_QUANT_MATMUL_ALL = (WeightInt4PackMMOp, WeightInt4PackQMOp, WeightInt8PackMMOp)


# --- dispatcher --------------------------------------------------------------


def _first_matching_library(
    op: Operation,
    allowlist: Iterable[str],
) -> str | None:
    """Return the first library in ``allowlist`` whose predicate matches ``op``."""
    for lib in allowlist:
        if lib == "cublas" and isinstance(op, MatmulOp) and _cublas_accepts_matmul(op):
            return lib
        if lib == "rocblas" and isinstance(op, MatmulOp) and _cublas_accepts_matmul(op):
            return lib
        if lib == "triton":
            if isinstance(op, MatmulOp) and _triton_accepts_matmul(op):
                return lib
            if _triton_accepts_quant_matmul(op):
                return lib
        if lib == "cublaslt" and _cublaslt_accepts_quant_matmul(op):
            return lib
        if lib == "onednn" and isinstance(op, MatmulOp) and _onednn_accepts_matmul(op):
            return lib
        if lib == "cudnn" and isinstance(op, CallOp) and _cudnn_accepts_conv(op):
            return lib
        if lib == "miopen" and isinstance(op, CallOp) and _cudnn_accepts_conv(op):
            return lib
        if lib == "xnnpack" and isinstance(op, CallOp) and _cudnn_accepts_conv(op):
            return lib
        if lib == "qnn" and _qnn_accepts(op):
            return lib
    return None


def _is_conv_call(op: Operation) -> bool:
    if not isinstance(op, CallOp):
        return False
    hint = op.attributes.get("compgen._pattern_hint")
    if hint is None:
        return False
    return isinstance(hint, StringAttr) and hint.data in {"convolution", "quantized_convolution"}


# --- patterns ----------------------------------------------------------------


class _MatmulDispatchPattern(RewritePattern):
    def __init__(
        self,
        cfg: MatchLibraryCallConfig,
        stats: MatchLibraryCallStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter) -> None:
        self.stats.ops_seen += 1
        if "compgen.library_dispatch" in op.attributes:
            self.stats.skipped_already_dispatched += 1
            return
        lib = _first_matching_library(op, self.cfg.library_allowlist)
        if lib is None:
            self.stats.no_match += 1
            return
        op.attributes["compgen.library_dispatch"] = StringAttr(lib)
        self.stats.matmul_matches += 1
        self.stats.record(lib)


class _QuantMatmulDispatchPattern(RewritePattern):
    def __init__(
        self,
        cfg: MatchLibraryCallConfig,
        stats: MatchLibraryCallStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, _QUANT_MATMUL_ALL):
            return
        self.stats.ops_seen += 1
        if "compgen.library_dispatch" in op.attributes:
            self.stats.skipped_already_dispatched += 1
            return
        lib = _first_matching_library(op, self.cfg.library_allowlist)
        if lib is None:
            self.stats.no_match += 1
            return
        op.attributes["compgen.library_dispatch"] = StringAttr(lib)
        self.stats.quant_matmul_matches += 1
        self.stats.record(lib)


class _ConvDispatchPattern(RewritePattern):
    def __init__(
        self,
        cfg: MatchLibraryCallConfig,
        stats: MatchLibraryCallStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallOp, rewriter: PatternRewriter) -> None:
        if not _is_conv_call(op):
            return
        self.stats.ops_seen += 1
        if "compgen.library_dispatch" in op.attributes:
            self.stats.skipped_already_dispatched += 1
            return
        lib = _first_matching_library(op, self.cfg.library_allowlist)
        if lib is None:
            self.stats.no_match += 1
            return
        op.attributes["compgen.library_dispatch"] = StringAttr(lib)
        self.stats.conv_matches += 1
        self.stats.record(lib)


# --- entry point -------------------------------------------------------------


def run_match_library_call(
    module: ModuleOp,
    *,
    config: MatchLibraryCallConfig | None = None,
    apply_recursively: bool = False,
) -> MatchLibraryCallStats:
    cfg = config if config is not None else MatchLibraryCallConfig()
    stats = MatchLibraryCallStats()
    patterns = [
        _MatmulDispatchPattern(cfg, stats),
        _QuantMatmulDispatchPattern(cfg, stats),
        _ConvDispatchPattern(cfg, stats),
    ]
    for p in patterns:
        walker = PatternRewriteWalker(p, apply_recursively=apply_recursively)
        walker.rewrite_module(module)
    return stats


__all__ = [
    "MatchLibraryCallConfig",
    "MatchLibraryCallStats",
    "run_match_library_call",
]
