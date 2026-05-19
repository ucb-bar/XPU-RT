"""Declarative extension generators for xDSL dialects and LLVM forks."""

from __future__ import annotations

from xpu_rt.extensions.llvm_patchgen import LLVMIntrinsicSpec, LLVMPatchSpec, generate_llvm_patch_bundle
from xpu_rt.extensions.xdsl_generate import (
    DialectOperandSpec,
    DialectOpSpec,
    DialectResultSpec,
    DialectSpec,
    generate_xdsl_dialect,
)

__all__ = [
    "DialectOpSpec",
    "DialectOperandSpec",
    "DialectResultSpec",
    "DialectSpec",
    "LLVMIntrinsicSpec",
    "LLVMPatchSpec",
    "generate_llvm_patch_bundle",
    "generate_xdsl_dialect",
]
