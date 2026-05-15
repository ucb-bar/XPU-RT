"""Recipe IR Family H: Multi-plan dispatch (, Phase G — §12 Dream 5).

A ``recipe.plan_dispatch_table`` op authors a multi-plan dispatcher
over a feature vector (batch, seqlen, dtype, ...). Each plan binding
remains independently verified by the rest of the verification
ladder; the dispatcher itself is a small typed lookup the emitter
materialises into Python / C11 / C++ depending on the chosen Layer-1
emit target.

Concrete shape:

    %t = recipe.plan_dispatch_table {
        feature_keys = ["batch", "seqlen", "dtype"],
        entries = [
            #recipe.plan_dispatch_entry<features={batch=1},  plan_ref=@plan_b1>,
            #recipe.plan_dispatch_entry<features={batch=4},  plan_ref=@plan_b4>,
            #recipe.plan_dispatch_entry<features={batch=16}, plan_ref=@plan_b16>
        ],
        default_plan_ref = @plan_default
    }

The dispatcher selection is **deterministic**: entries are tried in
declaration order; the first whose feature-set matches the runtime
vector wins. The default plan fires only when no entry matches.

The verifier (D6) enforces:

- Every ``plan_ref`` resolves to a verified ``ExecutionPlan`` on disk
  (i.e. has a ``05_execution_plan/<plan_name>/execution_plan.yaml``).
- Every entry's feature key set is a subset of the table's declared
  ``feature_keys``.
- The default plan exists; without it the dispatcher would have to
  raise at runtime, which §12 D5 explicitly forbids ("dispatcher is
  trivial").
"""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    traits_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class PlanDispatchTableOp(IRDLOperation):
    """Multi-plan dispatcher over a runtime feature vector.

    Carries:
      - feature_keys : the runtime feature names (e.g. ["batch",
        "seqlen", "dtype"]) the dispatcher consults.
      - entries      : ordered list of {features, plan_ref} dicts.
      - default_plan : symbol ref the runtime falls back on when no
        entry matches.
    """

    name = "recipe.plan_dispatch_table"

    feature_keys = prop_def(ArrayAttr)
    entries = prop_def(ArrayAttr)
    default_plan_ref = prop_def(StringAttr)
    workload = opt_prop_def(StringAttr)
    target = opt_prop_def(StringAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        keys: set[str] = set()
        for k in self.feature_keys.data:
            if not isinstance(k, StringAttr):
                raise VerifyException(
                    f"feature_keys entries must be StringAttr; got {type(k)}"
                )
            keys.add(k.data)
        if not keys:
            raise VerifyException(
                "plan_dispatch_table requires at least one feature key"
            )
        if not self.entries.data:
            raise VerifyException(
                "plan_dispatch_table requires at least one entry; an empty "
                "table would degrade silently to the default plan"
            )
        for idx, entry in enumerate(self.entries.data):
            if not isinstance(entry, DictionaryAttr):
                raise VerifyException(
                    f"entry[{idx}] must be a DictionaryAttr, got {type(entry)}"
                )
            features = entry.data.get("features")
            plan_ref = entry.data.get("plan_ref")
            if features is None or not isinstance(features, DictionaryAttr):
                raise VerifyException(
                    f"entry[{idx}]: features must be a DictionaryAttr"
                )
            if plan_ref is None or not isinstance(plan_ref, StringAttr):
                raise VerifyException(
                    f"entry[{idx}]: plan_ref must be a StringAttr"
                )
            for fk in features.data.keys():
                if fk not in keys:
                    raise VerifyException(
                        f"entry[{idx}]: feature {fk!r} not in declared "
                        f"feature_keys {sorted(keys)!r}"
                    )
        if not self.default_plan_ref.data:
            raise VerifyException(
                "default_plan_ref must be a non-empty symbol; M-90 forbids "
                "dispatchers without a fallback"
            )


_DISPATCH_OPS = [PlanDispatchTableOp]


__all__ = ["PlanDispatchTableOp", "_DISPATCH_OPS"]
