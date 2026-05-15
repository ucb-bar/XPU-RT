"""Semantic verification execution layer.

This package contains the verification backends and executor that power
CompGen's Semantic IR (Layer 3). The IR definitions live in
``compgen.ir.semantic``; this package handles execution.

Backends:
    - ``xdsl_smt``: Uses xDSL's SMT dialect + Z3 for formal verification.
"""

from __future__ import annotations

__all__: list[str] = []
