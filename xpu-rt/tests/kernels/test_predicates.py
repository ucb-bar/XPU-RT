"""M-61 — Typed predicate DSL unit tests.

Coverage:

- Round-trip every predicate kind through to_dict / from_dict.
- predicate_kind + predicate_plan_violation_suffix maps.
- predicate_from_dict rejects unknown kinds with typed ValueError.
- predicates_to_list / predicates_from_list iterate.
"""

from __future__ import annotations

import pytest


class TestPredicateRoundTrip:
    def test_mod_eq(self) -> None:
        from compgen.kernels.predicates import (
            ModEq,
            predicate_from_dict,
            predicate_to_dict,
        )

        p = ModEq(arg_dim="K", k=16)
        body = predicate_to_dict(p)
        assert body == {"kind": "mod_eq", "arg_dim": "K", "k": 16}
        assert predicate_from_dict(body) == p

    def test_byte_size_le(self) -> None:
        from compgen.kernels.predicates import (
            ByteSizeLe,
            predicate_from_dict,
            predicate_to_dict,
        )

        p = ByteSizeLe(arg="Y", max_bytes=4096)
        assert predicate_from_dict(predicate_to_dict(p)) == p

    def test_no_alias(self) -> None:
        from compgen.kernels.predicates import (
            NoAlias,
            predicate_from_dict,
            predicate_to_dict,
        )

        p = NoAlias(arg_a="A", arg_b="Y")
        assert predicate_from_dict(predicate_to_dict(p)) == p

    def test_dtype_in(self) -> None:
        from compgen.kernels.predicates import (
            DtypeIn,
            predicate_from_dict,
            predicate_to_dict,
        )

        p = DtypeIn(arg="A", dtype_set=("f32", "f16"))
        assert predicate_from_dict(predicate_to_dict(p)) == p

    def test_numerical_within_eps(self) -> None:
        from compgen.kernels.predicates import (
            NumericalWithinEps,
            predicate_from_dict,
            predicate_to_dict,
        )

        p = NumericalWithinEps(out="Y", ref="reference", eps=1e-3)
        assert predicate_from_dict(predicate_to_dict(p)) == p


class TestKindSuffix:
    def test_kind_strings(self) -> None:
        from compgen.kernels.predicates import (
            ByteSizeLe,
            DtypeIn,
            ModEq,
            NoAlias,
            NumericalWithinEps,
            predicate_kind,
        )

        assert predicate_kind(ModEq("K", 1)) == "mod_eq"
        assert predicate_kind(ByteSizeLe("Y", 1)) == "byte_size_le"
        assert predicate_kind(NoAlias("A", "B")) == "no_alias"
        assert predicate_kind(DtypeIn("A", ("f32",))) == "dtype_in"
        assert predicate_kind(NumericalWithinEps("Y", "ref", 0.0)) == "numerical_within_eps"

    def test_plan_violation_suffix(self) -> None:
        from compgen.kernels.predicates import (
            ModEq,
            NumericalWithinEps,
            predicate_plan_violation_suffix,
        )

        assert predicate_plan_violation_suffix(ModEq("K", 1)) == "MOD_EQ"
        assert (
            predicate_plan_violation_suffix(NumericalWithinEps("Y", "ref", 0.0))
            == "NUMERICAL_WITHIN_EPS"
        )


class TestRejectUnknown:
    def test_unknown_kind_raises(self) -> None:
        from compgen.kernels.predicates import predicate_from_dict

        with pytest.raises(ValueError, match="unknown predicate kind"):
            predicate_from_dict({"kind": "invent_new_pattern"})


class TestListHelpers:
    def test_round_trip_list(self) -> None:
        from compgen.kernels.predicates import (
            ByteSizeLe,
            ModEq,
            predicates_from_list,
            predicates_to_list,
        )

        ps = (ModEq("K", 16), ByteSizeLe("Y", 1024))
        body = predicates_to_list(ps)
        assert predicates_from_list(body) == ps

    def test_empty_list(self) -> None:
        from compgen.kernels.predicates import (
            predicates_from_list,
            predicates_to_list,
        )

        assert predicates_to_list(()) == []
        assert predicates_from_list([]) == ()
