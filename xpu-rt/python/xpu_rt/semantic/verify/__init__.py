"""Execution-backed verification for CompGen artifacts.

Provides numeric tensor comparison, a verification harness for comparing
reference and candidate callables, and helpers for building eager
references from ``nn.Module`` instances.

The verification package implements the functional layer of the
verification ladder (structural -> **functional** -> performance -> formal).
"""

from __future__ import annotations

from compgen.semantic.verify.compare import ComparisonConfig, NumericComparison, compare_tensors
from compgen.semantic.verify.eager_reference import EagerReference, build_eager_reference
from compgen.semantic.verify.harness import VerificationRun, verify_callable_against_reference

__all__ = [
    "ComparisonConfig",
    "EagerReference",
    "NumericComparison",
    "VerificationRun",
    "build_eager_reference",
    "compare_tensors",
    "verify_callable_against_reference",
]
