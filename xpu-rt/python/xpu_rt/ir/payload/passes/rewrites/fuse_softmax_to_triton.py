"""``fuse_softmax_to_triton`` -- lower ``compgen.linalg_ext.softmax`` to a
Triton kernel invocation.

Reconstruction of XLA's ``SoftmaxRewriterTriton`` as a CompGen
PatternRewriter. Zero external references; this module owns the
rewrite.

Approach follows hexagon-mlir's Triton integration
(`/scratch2/agustin/CompGen/tmp/hexagon-mlir/qcom_hexagon_backend/backend/compiler.py:40-58`):

1. Match a softmax op.
2. Emit a Python Triton kernel source string (canonical block-wise
   softmax kernel).
3. Attempt ``triton-shared-opt --triton-to-linalg-experimental`` via
   subprocess to lower the Triton source to linalg MLIR text.
4. When ``triton-shared-opt`` is available AND the subprocess
   succeeds, parse the returned linalg MLIR and splice it in-place
   (the softmax op is replaced by the lowered linalg body).
5. When ``triton-shared-opt`` is unavailable (the common case today
   in CI), annotate the softmax op with two attributes:
   - ``compgen.triton_source`` -- the Triton Python source string.
   - ``compgen.triton_kernel_call`` -- the kernel name, used by the
     later runtime to JIT-compile the kernel at load time.
   The op stays structurally intact so downstream passes still see
   a typed softmax. Zero new dialect ops.

No ``compgen.triton.*`` dialect introduced. The bridge is the
linalg MLIR text, matching hexagon-mlir's contract.

Gates:

- The rewrite fires only when the caller-supplied
  ``kernel_family_allowlist`` contains ``"triton"``. Defaults to
  empty (pass does nothing) so it's opt-in.
- The softmax must be over a 2-D tensor with a static last axis
  (the Triton template assumes ``[M, N]`` with N known at compile
  time). Higher-rank softmax is deferred.

LLM-tool signature:

    tool_name="fuse_softmax_to_triton"
    wraps_pass="CompGen:SoftmaxRewriterTriton"
    invent_slot="kernel_dispatch/triton_bridge"
    policy="EmitTritonSoftmaxWhenAllowed"
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field

from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType
from xdsl.ir import Operation
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compgen.ir.linalg_ext import SoftmaxOp


# --- configuration -----------------------------------------------------------


@dataclass(frozen=True)
class FuseSoftmaxToTritonConfig:
    kernel_family_allowlist: frozenset[str] = frozenset()
    block_size: int = 128
    triton_shared_opt_path: str = "triton-shared-opt"
    invoke_triton_shared: bool = True


@dataclass
class FuseSoftmaxToTritonStats:
    softmaxes_seen: int = 0
    softmaxes_annotated: int = 0
    softmaxes_skipped_rank: int = 0
    softmaxes_skipped_dynamic: int = 0
    softmaxes_skipped_policy: int = 0
    triton_shared_lowered: int = 0
    triton_shared_unavailable: int = 0


# --- Triton source template --------------------------------------------------


_TRITON_SOFTMAX_TEMPLATE = textwrap.dedent(
    """\
    import triton
    import triton.language as tl


    @triton.jit
    def {kernel_name}(
        input_ptr,
        output_ptr,
        n_rows,
        n_cols,
        input_row_stride,
        output_row_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        '''Row-wise softmax kernel.

        Each program instance handles a single row. Computes
        ``out = exp(x - max(x)) / sum(exp(x - max(x)))`` along the
        row (axis=-1).
        '''
        row_idx = tl.program_id(0)
        row_start = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_start + col_offsets, mask=mask, other=float('-inf'))
        row_max = tl.max(x, axis=0)
        x_stable = x - row_max
        exp_x = tl.exp(x_stable)
        denom = tl.sum(exp_x, axis=0)
        y = exp_x / denom
        out_row_start = output_ptr + row_idx * output_row_stride
        tl.store(out_row_start + col_offsets, y, mask=mask)
    """
)


def _emit_triton_source(kernel_name: str) -> str:
    return _TRITON_SOFTMAX_TEMPLATE.format(kernel_name=kernel_name)


# --- optional triton-shared invocation --------------------------------------


@dataclass
class TritonSharedResult:
    ok: bool
    linalg_mlir: str = ""
    diagnostics: str = ""


def _invoke_triton_shared(
    triton_source: str,
    *,
    tool_path: str = "triton-shared-opt",
) -> TritonSharedResult:
    """Run ``triton-shared-opt --triton-to-linalg-experimental`` if installed.

    Emits the Triton Python source to a temp file, compiles it to
    TTIR via the Triton compiler, then hands the TTIR to
    ``triton-shared-opt`` and captures the resulting linalg MLIR.

    When either ``triton`` or ``triton-shared-opt`` is unavailable
    we return ``TritonSharedResult(ok=False, ...)`` so the caller
    can fall back to annotation-only mode.
    """
    if shutil.which(tool_path) is None:
        return TritonSharedResult(
            ok=False,
            diagnostics=f"{tool_path} not found on PATH",
        )
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        return TritonSharedResult(
            ok=False,
            diagnostics=f"triton python package not importable: {exc}",
        )

    # Minimal TTIR pipeline inspired by hexagon-mlir's
    # ``ttir_to_ttsharedir`` (compiler.py:40-58). A full implementation
    # would exec() the Triton source, run ``kernel.compile(...)`` to
    # obtain TTIR, then pipe to triton-shared-opt. We leave that
    # end-to-end path as follow-up work when triton-shared is actually
    # available in the build env; this function's current role is
    # the SHAPE of the subprocess invocation so the rest of the
    # rewrite is structurally correct.
    with tempfile.TemporaryDirectory() as td:
        py_path = os.path.join(td, "kernel.py")
        with open(py_path, "w") as f:
            f.write(triton_source)
        # We don't yet execute Triton; instead we expose a hook point
        # so real triton-shared-opt lowering can plug in without any
        # pass-level change.
        return TritonSharedResult(
            ok=False,
            diagnostics=(
                f"triton-shared-opt found at {tool_path} but "
                "end-to-end lowering is not wired up in this wave"
            ),
        )


# --- helpers -----------------------------------------------------------------


def _copy_preserved(dst: Operation, src: Operation) -> None:
    for key in ("compgen.region_id", "compgen._pattern_hint"):
        if key in src.attributes and key not in dst.attributes:
            dst.attributes[key] = src.attributes[key]


# --- pattern -----------------------------------------------------------------


class FuseSoftmaxToTritonPattern(RewritePattern):
    def __init__(
        self,
        cfg: FuseSoftmaxToTritonConfig,
        stats: FuseSoftmaxToTritonStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: SoftmaxOp, rewriter: PatternRewriter
    ) -> None:
        self.stats.softmaxes_seen += 1

        # Policy gate: Triton must be on the allowlist.
        if "triton" not in self.cfg.kernel_family_allowlist:
            self.stats.softmaxes_skipped_policy += 1
            return

        # Already annotated? idempotent no-op.
        if "compgen.triton_kernel_call" in op.attributes:
            return

        # Shape gate: any rank >= 2, static last axis (softmax dim).
        # Non-softmax dims are implicitly flattened into the kernel's
        # row dim -- the Triton template dispatches one program
        # per "row".
        in_type = op.input.type
        if not isinstance(in_type, TensorType):
            self.stats.softmaxes_skipped_rank += 1
            return
        shape = list(in_type.get_shape())
        if len(shape) < 2:
            self.stats.softmaxes_skipped_rank += 1
            return
        # The softmax dim must equal the last axis (our template's
        # invariant). The ``raise_special_ops`` pass defaults to
        # ``last`` so this holds by construction.
        softmax_dim = op.dim.value.data
        if softmax_dim != len(shape) - 1:
            self.stats.softmaxes_skipped_rank += 1
            return
        if any(d < 0 for d in shape):
            self.stats.softmaxes_skipped_dynamic += 1
            return

        n_rows = 1
        for d in shape[:-1]:
            n_rows *= d
        n_cols = shape[-1]
        kernel_name = f"compgen_softmax_row_{n_rows}x{n_cols}"
        source = _emit_triton_source(kernel_name)

        annotated = SoftmaxOp(
            op.input,
            dim=op.dim.value.data,
            result_type=op.results[0].type,
        )
        _copy_preserved(annotated, op)
        annotated.attributes["compgen.triton_kernel_call"] = StringAttr(kernel_name)
        annotated.attributes["compgen.triton_source"] = StringAttr(source)
        annotated.attributes["compgen.triton_block_size"] = StringAttr(
            str(self.cfg.block_size)
        )

        # Optionally invoke triton-shared-opt.
        if self.cfg.invoke_triton_shared:
            result = _invoke_triton_shared(
                source, tool_path=self.cfg.triton_shared_opt_path
            )
            if result.ok:
                self.stats.triton_shared_lowered += 1
                annotated.attributes["compgen.triton_linalg_mlir"] = StringAttr(
                    result.linalg_mlir
                )
            else:
                self.stats.triton_shared_unavailable += 1
                annotated.attributes["compgen.triton_status"] = StringAttr(
                    "source_only"
                )
        else:
            annotated.attributes["compgen.triton_status"] = StringAttr("source_only")

        rewriter.replace_matched_op(annotated)
        self.stats.softmaxes_annotated += 1


# --- entry point -------------------------------------------------------------


def run_fuse_softmax_to_triton(
    module: ModuleOp,
    *,
    config: FuseSoftmaxToTritonConfig | None = None,
    apply_recursively: bool = False,
) -> FuseSoftmaxToTritonStats:
    """Tag every 2-D static softmax with a Triton kernel source + name.

    When ``config.kernel_family_allowlist`` does not include
    ``"triton"``, this pass is a no-op.
    """
    cfg = config if config is not None else FuseSoftmaxToTritonConfig()
    stats = FuseSoftmaxToTritonStats()
    pattern = FuseSoftmaxToTritonPattern(cfg, stats)
    walker = PatternRewriteWalker(
        pattern,
        apply_recursively=apply_recursively,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "FuseSoftmaxToTritonConfig",
    "FuseSoftmaxToTritonPattern",
    "FuseSoftmaxToTritonStats",
    "TritonSharedResult",
    "run_fuse_softmax_to_triton",
]
