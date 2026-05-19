"""Base class for Python-based eqsat rewrite rules.

These rules operate on xDSL IR that has been wrapped with equivalence.class
ops.  Instead of destructively rewriting (like a normal RewritePattern),
they **add equivalent alternatives** to e-classes.

The pattern is:
    1. Walk the module looking for pattern matches
    2. For each match, create the alternative ops
    3. Add the new op's result as an additional operand to the existing e-class

This is the non-destructive rewriting at the heart of equality saturation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from xdsl.dialects import equivalence
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, OpResult


class EqSatRewriteRule(ABC):
    """Base class for equality saturation rewrite rules.

    Subclasses implement ``match_and_add`` which finds pattern matches
    and adds equivalent alternatives to e-classes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this rule."""

    @abstractmethod
    def match_and_add(self, module: ModuleOp) -> int:
        """Find matches and add alternatives. Return match count."""

    def apply(self, module: ModuleOp) -> int:
        """Apply this rule to the module. Returns number of matches."""
        return self.match_and_add(module)


def add_alternative_to_eclass(
    eclass_op: equivalence.ClassOp,
    new_result: OpResult,
) -> None:
    """Add a new equivalent alternative to an existing e-class.

    The new op must already be inserted into the block. This function
    adds its result as an additional operand of the e-class.

    Args:
        eclass_op: The existing equivalence.class operation.
        new_result: The result of the new equivalent operation.
    """
    eclass_op.operands = list(eclass_op.operands) + [new_result]


def find_defining_op_through_eclass(value: OpResult) -> Operation | None:
    """Follow a value through an equivalence.class to find its defining op.

    In e-graph form, values flow through eclasses:
        %x = some_op(...)
        %x_eq = equivalence.class %x
        %y = other_op(%x_eq, ...)

    This function, given %x_eq, returns some_op.
    Given %x directly, returns some_op.
    """
    owner = value.owner
    if isinstance(owner, equivalence.AnyClassOp):
        # The eclass has operands; return the first one's owner
        if owner.operands:
            first_operand = owner.operands[0]
            if isinstance(first_operand, OpResult):
                return first_operand.owner
    return owner


def get_eclass_for_result(result: OpResult) -> equivalence.ClassOp | None:
    """Find the equivalence.class that wraps this result, if any."""
    for use in result.uses:
        if isinstance(use.operation, equivalence.AnyClassOp):
            return use.operation
    return None
