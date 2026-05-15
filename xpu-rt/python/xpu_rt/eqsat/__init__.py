"""Equality saturation subsystem for CompGen.

Uses xDSL's native ``equivalence`` dialect to explore equivalent
computational forms and extract the best one via a global cost model.
The LLM proposes local equivalences and search bias; the e-graph
composes them globally; the extractor/solver picks a consistent result.
"""

from __future__ import annotations

from compgen.eqsat.config import EqSatConfig
from compgen.eqsat.pipeline import run_eqsat_pass

__all__ = ["EqSatConfig", "run_eqsat_pass"]
