"""Closed guard-expression language used for synthesized predicates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class CmpOp(str, Enum):
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="
    GE = ">="
    GT = ">"


class BoolOp(str, Enum):
    AND = "and"
    OR = "or"


@dataclass(frozen=True)
class Expr:
    """Base class for synthesized guard expressions."""


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Const(Expr):
    value: Any


@dataclass(frozen=True)
class Cmp(Expr):
    op: CmpOp
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class BoolN(Expr):
    op: BoolOp
    terms: tuple[Expr, ...]


@dataclass(frozen=True)
class Not(Expr):
    term: Expr


@dataclass(frozen=True)
class Add(Expr):
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class Sub(Expr):
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class Mul(Expr):
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class Div(Expr):
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class ModEq(Expr):
    lhs: Expr
    mod: int
    rem: int


def _eval(expr: Expr, env: Mapping[str, Any]) -> Any:
    if isinstance(expr, Var):
        if expr.name not in env:
            raise KeyError(f"unknown guard variable: {expr.name}")
        return env[expr.name]
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Add):
        return _eval(expr.lhs, env) + _eval(expr.rhs, env)
    if isinstance(expr, Sub):
        return _eval(expr.lhs, env) - _eval(expr.rhs, env)
    if isinstance(expr, Mul):
        return _eval(expr.lhs, env) * _eval(expr.rhs, env)
    if isinstance(expr, Div):
        rhs = _eval(expr.rhs, env)
        if rhs == 0:
            raise ZeroDivisionError("guard expression division by zero")
        return _eval(expr.lhs, env) / rhs
    if isinstance(expr, ModEq):
        mod = int(expr.mod)
        if mod == 0:
            raise ZeroDivisionError("guard expression modulo by zero")
        return (_eval(expr.lhs, env) % mod) == expr.rem
    if isinstance(expr, Not):
        return not _eval(expr.term, env)
    if isinstance(expr, BoolN):
        values = [_eval(term, env) for term in expr.terms]
        return all(values) if expr.op == BoolOp.AND else any(values)
    if isinstance(expr, Cmp):
        lhs = _eval(expr.lhs, env)
        rhs = _eval(expr.rhs, env)
        return {
            CmpOp.LT: lhs < rhs,
            CmpOp.LE: lhs <= rhs,
            CmpOp.EQ: lhs == rhs,
            CmpOp.NE: lhs != rhs,
            CmpOp.GE: lhs >= rhs,
            CmpOp.GT: lhs > rhs,
        }[expr.op]
    raise TypeError(f"unsupported guard expression type: {type(expr)!r}")


def eval_guard(expr: Expr, env: Mapping[str, Any]) -> bool:
    """Evaluate a guard expression against a concrete environment."""

    value = _eval(expr, env)
    if not isinstance(value, bool):
        raise TypeError(f"guard expression must evaluate to bool, got {type(value)!r}")
    return value


def and_(*terms: Expr) -> Expr:
    """Build a flattened conjunction."""

    if not terms:
        return Const(True)
    flat: list[Expr] = []
    for term in terms:
        if isinstance(term, BoolN) and term.op == BoolOp.AND:
            flat.extend(term.terms)
        else:
            flat.append(term)
    return BoolN(BoolOp.AND, tuple(flat))


def or_(*terms: Expr) -> Expr:
    """Build a flattened disjunction."""

    if not terms:
        return Const(False)
    flat: list[Expr] = []
    for term in terms:
        if isinstance(term, BoolN) and term.op == BoolOp.OR:
            flat.extend(term.terms)
        else:
            flat.append(term)
    return BoolN(BoolOp.OR, tuple(flat))


def expr_to_json(expr: Expr) -> dict[str, Any]:
    """Serialize an expression to a JSON-friendly dictionary."""

    if isinstance(expr, Var):
        return {"kind": "Var", "name": expr.name}
    if isinstance(expr, Const):
        return {"kind": "Const", "value": expr.value}
    if isinstance(expr, Cmp):
        return {
            "kind": "Cmp",
            "op": expr.op.value,
            "lhs": expr_to_json(expr.lhs),
            "rhs": expr_to_json(expr.rhs),
        }
    if isinstance(expr, BoolN):
        return {
            "kind": "BoolN",
            "op": expr.op.value,
            "terms": [expr_to_json(term) for term in expr.terms],
        }
    if isinstance(expr, Not):
        return {"kind": "Not", "term": expr_to_json(expr.term)}
    if isinstance(expr, Add):
        return {"kind": "Add", "lhs": expr_to_json(expr.lhs), "rhs": expr_to_json(expr.rhs)}
    if isinstance(expr, Sub):
        return {"kind": "Sub", "lhs": expr_to_json(expr.lhs), "rhs": expr_to_json(expr.rhs)}
    if isinstance(expr, Mul):
        return {"kind": "Mul", "lhs": expr_to_json(expr.lhs), "rhs": expr_to_json(expr.rhs)}
    if isinstance(expr, Div):
        return {"kind": "Div", "lhs": expr_to_json(expr.lhs), "rhs": expr_to_json(expr.rhs)}
    if isinstance(expr, ModEq):
        return {
            "kind": "ModEq",
            "lhs": expr_to_json(expr.lhs),
            "mod": expr.mod,
            "rem": expr.rem,
        }
    raise TypeError(f"unsupported expression type: {type(expr)!r}")


def expr_from_json(data: Mapping[str, Any]) -> Expr:
    """Deserialize an expression from JSON-friendly data."""

    kind = data["kind"]
    if kind == "Var":
        return Var(str(data["name"]))
    if kind == "Const":
        return Const(data["value"])
    if kind == "Cmp":
        return Cmp(
            CmpOp(str(data["op"])),
            expr_from_json(data["lhs"]),
            expr_from_json(data["rhs"]),
        )
    if kind == "BoolN":
        return BoolN(
            BoolOp(str(data["op"])),
            tuple(expr_from_json(item) for item in data.get("terms", [])),
        )
    if kind == "Not":
        return Not(expr_from_json(data["term"]))
    if kind == "Add":
        return Add(expr_from_json(data["lhs"]), expr_from_json(data["rhs"]))
    if kind == "Sub":
        return Sub(expr_from_json(data["lhs"]), expr_from_json(data["rhs"]))
    if kind == "Mul":
        return Mul(expr_from_json(data["lhs"]), expr_from_json(data["rhs"]))
    if kind == "Div":
        return Div(expr_from_json(data["lhs"]), expr_from_json(data["rhs"]))
    if kind == "ModEq":
        return ModEq(expr_from_json(data["lhs"]), int(data["mod"]), int(data["rem"]))
    raise ValueError(f"unknown expression kind: {kind}")


__all__ = [
    "Add",
    "BoolN",
    "BoolOp",
    "Cmp",
    "CmpOp",
    "Const",
    "Div",
    "Expr",
    "ModEq",
    "Mul",
    "Not",
    "Sub",
    "Var",
    "and_",
    "eval_guard",
    "expr_from_json",
    "expr_to_json",
    "or_",
]
