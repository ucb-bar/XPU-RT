"""Tests for the PDL rewrite verification backend."""

from __future__ import annotations

import pytest
from compgen.semantic.backends.xdsl_smt.pdl_backend import PDLVerificationBackend

z3 = pytest.importorskip("z3")


class TestPDLVerificationBackend:
    """Test PDL verification with callable patterns."""

    def test_commutativity_addi_sound(self) -> None:
        """add(a, b) == add(b, a) is sound for all bitwidths."""
        backend = PDLVerificationBackend()
        result = backend.verify_arith_rewrite(
            build_pattern=lambda ops: ops[0] + ops[1],
            build_replacement=lambda ops: ops[1] + ops[0],
            num_operands=2,
            max_bitwidth=8,
        )
        assert result.sound
        assert result.status == "sound"
        assert len(result.unsound_bitwidths) == 0

    def test_wrong_rewrite_unsound(self) -> None:
        """add(a, b) != sub(a, b) in general."""
        backend = PDLVerificationBackend()
        result = backend.verify_arith_rewrite(
            build_pattern=lambda ops: ops[0] + ops[1],
            build_replacement=lambda ops: ops[0] - ops[1],
            num_operands=2,
            max_bitwidth=4,
        )
        assert not result.sound
        assert result.status == "unsound"
        assert len(result.unsound_bitwidths) > 0

    def test_identity_sound(self) -> None:
        """x + 0 == x is sound."""
        backend = PDLVerificationBackend()
        result = backend.verify_arith_rewrite(
            build_pattern=lambda ops: ops[0] + z3.BitVecVal(0, ops[0].size()),
            build_replacement=lambda ops: ops[0],
            num_operands=1,
            max_bitwidth=16,
        )
        assert result.sound

    def test_double_negation_sound(self) -> None:
        """~~x == x for bitvectors."""
        backend = PDLVerificationBackend()
        result = backend.verify_arith_rewrite(
            build_pattern=lambda ops: ~(~ops[0]),
            build_replacement=lambda ops: ops[0],
            num_operands=1,
            max_bitwidth=16,
        )
        assert result.sound
