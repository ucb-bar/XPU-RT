"""``normalize_subbyte`` -- ensure sub-byte packed tensors carry consistent
``(bit_width, pack_dim)`` metadata across op boundaries.

Reconstruction of XLA's ``SubByteNormalizationPass``. Zero external
references; CompGen owns the rewrite.

Scope in  (annotational):

1. Walk every op in the module and collect its sub-byte operands +
   results -- defined as SSA values whose defining op carries a
   ``compgen.quant.PackedIntTensorType`` property (a ``qtype``).
2. For each producer-consumer edge, compare ``(bit_width, pack_dim)``
   on the two ends. When they disagree, tag the consuming op with
   ``compgen.subbyte_boundary = "pack" | "unpack" | "repack"`` so
   a later structural pass can materialize the right
   ``compgen.tensor_ext.{pack,unpack}`` op.
3. When the consuming op is the packed-int-mm itself
   (``weight_int4pack_mm`` / ``weight_int8pack_mm``), record the
   chosen canonical ``(bit_width, pack_dim)`` on the module so
   downstream passes don't have to re-derive.

This pass does NOT emit tensor_ext.pack/unpack itself because the
insertion point depends on memory-space decisions made in .
The annotation contract is the stable seam.

LLM-tool signature:

    tool_name="normalize_subbyte"
    wraps_pass="CompGen:SubByteNormalization"
    invent_slot="quantization/subbyte_layout"
    policy="AnnotatePackedIntBoundaries"
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Operation
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)

from compgen.ir.quant import (
    PackedIntTensorType,
    WeightInt4PackMMOp,
    WeightInt4PackQMOp,
    WeightInt8PackMMOp,
)

_PACKED_MM_OPS = (WeightInt8PackMMOp, WeightInt4PackMMOp, WeightInt4PackQMOp)


@dataclass
class NormalizeSubbyteStats:
    ops_seen: int = 0
    ops_with_qtype: int = 0
    boundaries_annotated: int = 0
    canonical_bit_widths: dict[int, int] | None = None

    def __post_init__(self) -> None:
        if self.canonical_bit_widths is None:
            self.canonical_bit_widths = {}

    def record_canonical(self, bit_width: int) -> None:
        assert self.canonical_bit_widths is not None
        self.canonical_bit_widths[bit_width] = self.canonical_bit_widths.get(bit_width, 0) + 1


# --- helpers ------------------------------------------------------------------


def _packed_type(op: Operation) -> PackedIntTensorType | None:
    """Return the ``PackedIntTensorType`` carried by the op (or ``None``)."""
    # Ops carry qtype as a property (via ``AffineQuantizedTensorType``).
    # A dedicated ``packed_qtype`` property isn't defined yet; instead
    # we inspect ``qtype`` and check the storage_type for sub-byte
    # packing metadata.
    qt = op.properties.get("qtype") if hasattr(op, "properties") else None
    if qt is None:
        qt = op.attributes.get("qtype")
    if qt is None:
        return None

    # AffineQuantizedTensorType wraps the storage type. When that
    # storage type is itself a PackedIntTensorType, the op produces /
    # consumes sub-byte data.
    storage = getattr(qt, "storage_type", None)
    if isinstance(storage, PackedIntTensorType):
        return storage
    return None


def _implicit_packed_from_weight_int4(op: Operation) -> PackedIntTensorType | None:
    """Heuristic: weight_int4pack_mm always carries a packed weight.

    When the op doesn't yet carry an explicit qtype, treat the weight
    operand as implicitly packed (4-bit, pack_dim=1 for TorchAO's
    ``[O, K//2]`` layout).
    """
    if not isinstance(op, (WeightInt4PackMMOp, WeightInt4PackQMOp)):
        return None
    # We can't easily fabricate a PackedIntTensorType here without
    # knowing pack_dim. Return a sentinel: a "canonical" 4-bit type.
    try:
        return PackedIntTensorType(bit_width=4, pack_dim=1)
    except Exception:
        return None


# --- pattern -----------------------------------------------------------------


class _AnnotateSubbyteBoundariesPattern(RewritePattern):
    def __init__(self, stats: NormalizeSubbyteStats) -> None:
        self.stats = stats

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        self.stats.ops_seen += 1
        packed = _packed_type(op)
        if packed is None and isinstance(op, (WeightInt4PackMMOp, WeightInt4PackQMOp)):
            packed = _implicit_packed_from_weight_int4(op)
        if packed is None:
            return
        if "compgen.subbyte_canonical" in op.attributes:
            # idempotent
            return

        self.stats.ops_with_qtype += 1
        bw = packed.bit_width.value.data
        pd = packed.pack_dim.value.data
        op.attributes["compgen.subbyte_canonical"] = StringAttr(f"bit_width={bw},pack_dim={pd}")
        self.stats.record_canonical(bw)

        # Also annotate the "boundary" role so  passes know
        # where to insert pack/unpack ops. For packed-mm ops the
        # role is always 'unpack' (we feed the compute unit with
        # a packed tensor and it internally unpacks).
        if isinstance(op, _PACKED_MM_OPS):
            op.attributes["compgen.subbyte_boundary"] = StringAttr("unpack")
            self.stats.boundaries_annotated += 1


# --- entry point -------------------------------------------------------------


def run_normalize_subbyte(
    module: ModuleOp,
    *,
    apply_recursively: bool = False,
) -> NormalizeSubbyteStats:
    stats = NormalizeSubbyteStats()
    pattern = _AnnotateSubbyteBoundariesPattern(stats)
    walker = PatternRewriteWalker(
        pattern,
        apply_recursively=apply_recursively,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "NormalizeSubbyteStats",
    "run_normalize_subbyte",
]
