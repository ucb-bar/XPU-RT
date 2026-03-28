"""Tests for the rewrite verification pipeline."""

from __future__ import annotations

import pytest

from compgen.rewrite.verify_pdl import verify_rewrite_family

z3 = pytest.importorskip("z3")


class TestVerifyRewriteFamily:
    """Test end-to-end rewrite verification."""

    def test_commutativity_sound(self) -> None:
        """Commutativity of addition is sound."""
        result = verify_rewrite_family(
            pattern=lambda ops: ops[0] + ops[1],
            replacement=lambda ops: ops[1] + ops[0],
            num_operands=2,
            max_bitwidth=8,
        )
        assert result.sound

    def test_associativity_sound(self) -> None:
        """Reassociation of addition is sound."""
        result = verify_rewrite_family(
            pattern=lambda ops: (ops[0] + ops[1]) + ops[2],
            replacement=lambda ops: ops[0] + (ops[1] + ops[2]),
            num_operands=3,
            max_bitwidth=8,
        )
        assert result.sound

    def test_wrong_rewrite_detected(self) -> None:
        """Incorrect rewrite is detected."""
        result = verify_rewrite_family(
            pattern=lambda ops: ops[0] + ops[1],
            replacement=lambda ops: ops[0] * ops[1],
            num_operands=2,
            max_bitwidth=4,
        )
        assert not result.sound
