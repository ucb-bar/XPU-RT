"""Custom extraction with non-additive cost model.

Extends xDSL's EqsatAddCostsPass + EqsatExtractPass with CompGen's
non-additive cost model that accounts for fusion, copies, and
backend matching.

Pipeline:
    1. Apply CostModel to assign per-op costs (non-additive)
    2. Run fixed-point propagation (via EqsatAddCostsPass internals)
    3. Extract cheapest subprogram (via EqsatExtractPass)
"""

from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp
from xdsl.transforms.eqsat_add_costs import add_eqsat_costs
from xdsl.transforms.eqsat_extract import eqsat_extract

from compgen.eqsat.cost_model import CostModel


def extract_with_cost_model(
    module: ModuleOp,
    cost_model: CostModel,
) -> None:
    """Assign non-additive costs and extract the cheapest subprogram.

    This is the preferred extraction path when using CompGen's cost model
    instead of the simple integer cost file.

    Args:
        module: Module with equivalence.class ops.
        cost_model: CompGen's non-additive cost model.
    """
    # Step 1: Assign per-op costs using our non-additive model
    cost_model.assign_costs(module)

    # Step 2: Run fixed-point propagation to compute e-class costs
    # We use xDSL's add_eqsat_costs on each block that has eclasses,
    # but with an empty cost_dict since we already assigned costs.
    from xdsl.dialects import equivalence

    eclass_parent_blocks = set(
        o.parent for o in module.walk() if o.parent is not None and isinstance(o, equivalence.AnyClassOp)
    )
    for block in eclass_parent_blocks:
        add_eqsat_costs(block, default=None, cost_dict={})

    # Step 3: Extract the cheapest subprogram
    eqsat_extract(module)
