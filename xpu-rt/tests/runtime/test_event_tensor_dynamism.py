"""Phase-1 dynamism tests — UpdateOp / TriggerOp / MaterializeViewOp.

Closes the ROADMAP gap the v1 honest-state doc flagged: the three
symbolic-shape / data-dependent Event-Tensor ops used to raise
``SymbolicShapeUnsupportedError``. Phase 1 wires them through the
Python reference runtime. These tests pin the paper Fig. 4 / Fig. 5
semantics at the runtime layer; companion tests in
``tests/ir/event/test_lower.py`` pin the dialect-level lowering.
"""

from __future__ import annotations

import torch
from compgen.runtime.event_tensor import EventTensor, materialize_view

# ---------------------------------------------------------------------------
# EventTensor.update / EventTensor.trigger
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_writes_new_count(self) -> None:
        et = EventTensor((4,), wait_count_default=10, sym_name="E")
        et.update((2,), 3)
        assert et.load((2,)) == 3
        # Other cells untouched.
        assert et.load((0,)) == 10
        assert et.load((3,)) == 10

    def test_update_to_zero_unblocks_waiter(self) -> None:
        """Post-Phase-1 semantics: `update` must wake any parked
        waiter. Zero is the common case (producer says "no-op, skip")."""
        et = EventTensor((1,), wait_count_default=5, sym_name="E")
        et.update((0,), 0)
        # Wait returns immediately since counter is <= 0.
        et.wait((0,), timeout_s=1.0)

    def test_update_to_negative_is_legal(self) -> None:
        """Matches over-notify semantics — counter can go below zero."""
        et = EventTensor((1,), wait_count_default=3, sym_name="E")
        et.update((0,), -1)
        assert et.load((0,)) == -1
        et.wait((0,), timeout_s=1.0)


class TestTrigger:
    def test_trigger_sets_consumer_count(self) -> None:
        et = EventTensor((3,), wait_count_default=0, sym_name="E")
        et.trigger((1,), 7)
        assert et.load((1,)) == 7

    def test_trigger_rejects_negative(self) -> None:
        et = EventTensor((1,), wait_count_default=0, sym_name="E")
        import pytest

        with pytest.raises(ValueError, match="non-negative"):
            et.trigger((0,), -1)

    def test_trigger_zero_immediately_unblocks(self) -> None:
        et = EventTensor((1,), wait_count_default=0, sym_name="E")
        et.trigger((0,), 0)
        et.wait((0,), timeout_s=1.0)


# ---------------------------------------------------------------------------
# materialize_view module helper
# ---------------------------------------------------------------------------


class TestMaterializeView:
    def test_concretises_symbolic_dim(self) -> None:
        view = materialize_view((-1,), (4,), wait_count_default=3, sym_name="E__mat")
        assert view.shape == (4,)
        assert view.sym_name == "E__mat"
        assert view.load((0,)) == 3
        assert view.load((3,)) == 3

    def test_partial_concrete_mixed_shape(self) -> None:
        view = materialize_view((4, -1), (4, 8), wait_count_default=1)
        assert view.shape == (4, 8)

    def test_rank_mismatch_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="rank"):
            materialize_view((-1,), (4, 8))

    def test_non_positive_concrete_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="positive"):
            materialize_view((-1,), (0,))

    def test_concrete_mismatch_rejected(self) -> None:
        """Template dim is concrete (4); caller asks for 8 — ambiguous."""
        import pytest

        with pytest.raises(ValueError, match="disagrees"):
            materialize_view((4,), (8,))


# ---------------------------------------------------------------------------
# Paper Fig. 5b (MoE routing) — UpdateOp + TriggerOp end-to-end
# ---------------------------------------------------------------------------


class TestMoePattern:
    """The MoE case in the paper (§2.4, Fig. 5b): a ``topk`` tensor says
    how many tokens each expert receives; an ``exp_indptr`` CSR-style
    prefix sum says how many GroupGEMM tiles each expert triggers.
    Phase 1 lets the Python reference model this exactly."""

    def test_update_then_trigger_matches_paper_fig_5b(self) -> None:
        from compgen.ir.event.lower import lower_graph_op
        from compgen.ir.event.ops import (
            MaterializeViewOp,  # noqa: F401  (imported for type check completeness)
            TriggerOp,
            UpdateOp,
        )
        from xdsl.dialects.builtin import StringAttr

        # Import helpers from the dialect-level test file. This is
        # intentional — mirrors the MoE-pattern path end-to-end through
        # the IR, not a separate mock.
        from tests.ir.event.test_lower import (
            _make_call_device_op,
            _make_edge,
            _make_event_tensor_op,
            _make_graph_op,
        )

        # 4 experts. topk[i] = tokens routed to expert i.
        # exp_indptr is the prefix sum of GroupGEMM tiles per expert.
        topk = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
        exp_indptr = torch.tensor([0, 2, 2, 5, 6], dtype=torch.int64)

        # Token-side event tensor: counter per expert's token budget.
        token_et = _make_event_tensor_op("token_budget", shape=[4], wait_count=0)
        update = UpdateOp.create(
            properties={
                "target": _make_edge("token_budget", ["i"]),
                "source_tensor": StringAttr("topk"),
                "index_expr": StringAttr("i"),
            }
        )
        # GroupGEMM-tile-side event tensor.
        tile_et = _make_event_tensor_op("tile_count", shape=[4], wait_count=0)
        trigger = TriggerOp.create(
            properties={
                "target": _make_edge("tile_count", ["i"]),
                "trigger_range": StringAttr("exp_indptr"),
            }
        )
        cd = _make_call_device_op("f", task_shape=[1])
        g_op = _make_graph_op("g", [token_et, update, tile_et, trigger, cd])

        _graph, tensors = lower_graph_op(
            g_op,
            device_funcs={"f": lambda _c: None},
            index_env={"topk": topk, "exp_indptr": exp_indptr},
        )
        tb = tensors["token_budget"]
        tc = tensors["tile_count"]
        # Token budgets match topk.
        assert [tb.load((i,)) for i in range(4)] == [2, 0, 3, 1]
        # Tile counts match indptr deltas: [2, 0, 3, 1].
        assert [tc.load((i,)) for i in range(4)] == [2, 0, 3, 1]
