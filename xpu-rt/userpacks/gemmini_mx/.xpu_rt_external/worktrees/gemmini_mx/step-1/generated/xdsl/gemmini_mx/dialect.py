"""Generated xDSL dialect scaffolding."""

from __future__ import annotations

from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def
from xdsl.traits import Pure

@irdl_op_definition
class GemminiMxDispatchOp(IRDLOperation):
    name = "gemmini_mx.gemmini_mx_dispatch"
    input = operand_def(AnyAttr())
    output = result_def(AnyAttr())
    traits = frozenset((Pure(),))
    __doc__ = "Generated entry op for gemmini_mx"

GemminiMx = Dialect(
    "gemmini_mx",
    [GemminiMxDispatchOp],
    [],
)

