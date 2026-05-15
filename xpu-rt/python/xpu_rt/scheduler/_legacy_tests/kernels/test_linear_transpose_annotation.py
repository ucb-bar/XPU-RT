"""Tests for REQ-023 — ``linalg.transpose`` becomes a first-class
dispatch region AND the consuming ``linalg.matmul`` advertises
``compgen.transposed_b="true"``.

Two paths are kept open intentionally so consumers pick what fits:

- Pack composer that wants a transpose-fused matmul kernel → reads
  ``compgen.transposed_b="true"`` on the matmul + emits a B^T kernel.
  The transpose region's source can be ignored.
- Pack composer that wants to materialize the transpose explicitly
  (e.g. distinct memory hierarchy) → walks the dispatch graph; the
  transpose carries ``region_id`` + ``dispatch_id`` and is reachable
  as a producer of the matmul's B operand.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from compgen.capture.torch_export import capture_model
from compgen.ir.payload.import_fx import fx_to_xdsl
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.linalg import MatmulOp, TransposeOp


def _module_for(model: nn.Module, *args: torch.Tensor):
    ep = capture_model(model, args)
    module, _ = fx_to_xdsl(ep)
    return module


def test_linear_emits_transpose_with_region_id_and_dispatch_id() -> None:
    module = _module_for(nn.Linear(8, 4).eval(), torch.randn(1, 8))
    transposes = [op for op in module.walk() if isinstance(op, TransposeOp)]
    assert len(transposes) == 1, len(transposes)

    rid = transposes[0].attributes.get("compgen.region_id")
    did = transposes[0].attributes.get("compgen.dispatch_id")
    assert isinstance(rid, StringAttr) and rid.data.startswith("transpose_"), rid
    assert isinstance(did, StringAttr) and did.data == rid.data, (rid, did)


def test_linear_matmul_carries_transposed_b_flag() -> None:
    module = _module_for(nn.Linear(8, 4).eval(), torch.randn(1, 8))
    matmuls = [op for op in module.walk() if isinstance(op, MatmulOp)]
    assert len(matmuls) == 1
    flag = matmuls[0].attributes.get("compgen.transposed_b")
    assert isinstance(flag, StringAttr) and flag.data == "true", flag


def test_standalone_transpose_also_gets_region_and_dispatch_ids() -> None:
    """Generalised REQ-023: ``aten.t.default`` (no surrounding matmul)
    also gets ``region_id`` + ``dispatch_id``. Without the
    generalised pass, only transposes from ``decompose_linear`` got
    the annotations — standalone transposes from
    ``decompose_transpose`` were missing ``dispatch_id``."""

    class Standalone(nn.Module):
        def forward(self, w):
            return w.t()

    module = _module_for(Standalone(), torch.randn(4, 8))
    transposes = [op for op in module.walk() if isinstance(op, TransposeOp)]
    assert len(transposes) == 1
    rid = transposes[0].attributes.get("compgen.region_id")
    did = transposes[0].attributes.get("compgen.dispatch_id")
    assert isinstance(rid, StringAttr)
    assert isinstance(did, StringAttr)
    assert rid.data == did.data, (rid, did)


def test_no_unannotated_transpose_anywhere_in_module() -> None:
    """Strong invariant: every ``linalg.transpose`` in the IR carries
    both ``region_id`` and ``dispatch_id`` after FX import. No
    consumer-side dispatch-graph walker should ever see an orphan
    transpose with a missing producer."""

    class Composite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(x).t()

    module = _module_for(Composite().eval(), torch.randn(1, 8))
    for op in module.walk():
        if isinstance(op, TransposeOp):
            assert op.attributes.get("compgen.region_id") is not None, op
            assert op.attributes.get("compgen.dispatch_id") is not None, op


def test_dispatch_graph_resolves_matmul_b_operand_to_transpose_region() -> None:
    """The matmul's B operand SSA traces back to the transpose op,
    and that transpose carries a region_id — so a dispatch graph
    parser can resolve the producer node."""
    module = _module_for(nn.Linear(8, 4).eval(), torch.randn(1, 8))

    matmul = next(op for op in module.walk() if isinstance(op, MatmulOp))
    # Matmul has two ``ins`` operands; B is the second.
    b_operand = matmul.operands[1]
    producer = b_operand.owner
    assert isinstance(producer, TransposeOp), producer
    assert "compgen.region_id" in producer.attributes
