"""Typed predicate DSL for contract pre/post-conditions.

Section 7 of the dream lists predicate verification as a load-bearing
verifier obligation: "M * N * 2 ≤ output_buffer_size", "K mod 16 == 0",
"Y[i,j] = relu(...) within epsilon_max_rel". lands the typed DSL
that backs those obligations.

Five predicate kinds, each a frozen dataclass, all with ``to_dict`` /
``from_dict`` for round-trip stability through the contract
serializer:

* :class:`ModEq` — ``arg_dim`` divisible by ``k`` (e.g. ``ModEq("K", 16)``).
* :class:`ByteSizeLe` — ``arg`` tensor's total bytes ≤ ``max_bytes``.
* :class:`NoAlias` — two tensors do not alias (``arg_a`` ≠ ``arg_b`` storage).
* :class:`DtypeIn` — ``arg`` tensor's dtype in the allowlist.
* :class:`NumericalWithinEps` — output within ``eps`` of reference
  (postcondition; the verifier lifts to a Higham-bounded differential).

The ``Predicate`` type is a :class:`typing.Union` over these five.
``predicate_kind(p)`` returns a stable string used as the
``PLAN_VIOLATION_PRECONDITION_<KIND>`` /
``PLAN_VIOLATION_POSTCONDITION_<KIND>`` typed-error suffix in the
emitted glue (extension).

The DSL is deliberately small. Sections that need richer predicates
(e.g. relational constraints between dims) can extend the union; the
classifier in :func:`predicate_from_dict` rejects unknown kinds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


# --------------------------------------------------------------------------- #
# Predicate dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModEq:
    """``arg_dim`` is divisible by ``k``.

    The ``arg_dim`` is a free-text reference to a contract dimension.
    Convention: ``"<dim_name>"`` (e.g. ``"K"``, ``"M"``) names a
    contract-level dimension; ``"<arg>.dims[i]"`` names a tensor's
    i-th dim. The plan-assertion emitter resolves the reference
    against the ``io`` contract.
    """

    arg_dim: str
    k: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "mod_eq", "arg_dim": self.arg_dim, "k": int(self.k)}


@dataclass(frozen=True)
class ByteSizeLe:
    """``arg`` tensor's ``numel * element_size`` ≤ ``max_bytes``."""

    arg: str
    max_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "byte_size_le",
            "arg": self.arg,
            "max_bytes": int(self.max_bytes),
        }


@dataclass(frozen=True)
class NoAlias:
    """Two contract args do not share storage."""

    arg_a: str
    arg_b: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "no_alias", "arg_a": self.arg_a, "arg_b": self.arg_b}


@dataclass(frozen=True)
class DtypeIn:
    """``arg`` tensor's dtype is in the allowlist."""

    arg: str
    dtype_set: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "dtype_in",
            "arg": self.arg,
            "dtype_set": list(self.dtype_set),
        }


@dataclass(frozen=True)
class NumericalWithinEps:
    """Postcondition: output within ``eps`` relative-error of reference.

    The ``ref`` field names the reference flavour: ``"reference"`` for
    the differential-against-eager check; an explicit name when
    the contract pins a specific reference function.
    """

    out: str
    ref: str
    eps: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "numerical_within_eps",
            "out": self.out,
            "ref": self.ref,
            "eps": float(self.eps),
        }


Predicate = Union[ModEq, ByteSizeLe, NoAlias, DtypeIn, NumericalWithinEps]


# --------------------------------------------------------------------------- #
# Round-trip + typed error suffix
# --------------------------------------------------------------------------- #


_KIND_TO_PLAN_VIOLATION_SUFFIX: dict[str, str] = {
    "mod_eq": "MOD_EQ",
    "byte_size_le": "BYTE_SIZE_LE",
    "no_alias": "NO_ALIAS",
    "dtype_in": "DTYPE_IN",
    "numerical_within_eps": "NUMERICAL_WITHIN_EPS",
}


def predicate_kind(p: Predicate) -> str:
    """Return the wire-format ``kind`` string for a predicate.

    Stable across versions — the emitter and verifier both grep on
    these values.
    """
    return p.to_dict()["kind"]


def predicate_plan_violation_suffix(p: Predicate) -> str:
    """The ``PLAN_VIOLATION_*_<SUFFIX>`` token for a predicate.

    Used by the plan-assertion emitter to construct subclass names
    (``PLAN_VIOLATION_PRECONDITION_MOD_EQ``,
    ``PLAN_VIOLATION_POSTCONDITION_NUMERICAL_WITHIN_EPS``, …).
    """
    return _KIND_TO_PLAN_VIOLATION_SUFFIX[predicate_kind(p)]


def predicate_to_dict(p: Predicate) -> dict[str, Any]:
    """Serialize a predicate to its on-disk JSON form."""
    return p.to_dict()


def predicate_from_dict(body: dict[str, Any]) -> Predicate:
    """Inverse of :func:`predicate_to_dict`. Rejects unknown kinds."""
    kind = body.get("kind", "")
    if kind == "mod_eq":
        return ModEq(arg_dim=str(body["arg_dim"]), k=int(body["k"]))
    if kind == "byte_size_le":
        return ByteSizeLe(arg=str(body["arg"]), max_bytes=int(body["max_bytes"]))
    if kind == "no_alias":
        return NoAlias(arg_a=str(body["arg_a"]), arg_b=str(body["arg_b"]))
    if kind == "dtype_in":
        return DtypeIn(
            arg=str(body["arg"]),
            dtype_set=tuple(str(d) for d in (body.get("dtype_set") or ())),
        )
    if kind == "numerical_within_eps":
        return NumericalWithinEps(
            out=str(body["out"]),
            ref=str(body["ref"]),
            eps=float(body["eps"]),
        )
    raise ValueError(f"unknown predicate kind: {kind!r}")


def predicates_to_list(preds: tuple[Predicate, ...]) -> list[dict[str, Any]]:
    """Serialize a tuple of predicates."""
    return [predicate_to_dict(p) for p in preds]


def predicates_from_list(body: list[dict[str, Any]]) -> tuple[Predicate, ...]:
    """Deserialize a list of predicate dicts."""
    return tuple(predicate_from_dict(d) for d in (body or ()))


__all__ = [
    "ByteSizeLe",
    "DtypeIn",
    "ModEq",
    "NoAlias",
    "NumericalWithinEps",
    "Predicate",
    "predicate_from_dict",
    "predicate_kind",
    "predicate_plan_violation_suffix",
    "predicate_to_dict",
    "predicates_from_list",
    "predicates_to_list",
]
