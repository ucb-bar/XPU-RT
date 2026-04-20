"""Generate LLVM fork patch scaffolding from a declarative intrinsic spec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLVMIntrinsicSpec:
    name: str
    ret_type: str
    arg_types: tuple[str, ...]
    summary: str = ""


@dataclass(frozen=True)
class LLVMPatchSpec:
    dialect_name: str
    intrinsics: tuple[LLVMIntrinsicSpec, ...]


def _coerce_patch_spec(spec: LLVMPatchSpec | dict[str, Any]) -> LLVMPatchSpec:
    if isinstance(spec, LLVMPatchSpec):
        return spec
    intrinsics = tuple(
        LLVMIntrinsicSpec(
            name=str(item["name"]),
            ret_type=str(item["ret_type"]),
            arg_types=tuple(str(arg) for arg in item.get("arg_types", [])),
            summary=str(item.get("summary", "")),
        )
        for item in spec.get("intrinsics", [])
    )
    return LLVMPatchSpec(dialect_name=str(spec["dialect_name"]), intrinsics=intrinsics)


def generate_llvm_patch_bundle(spec: LLVMPatchSpec | dict[str, Any]) -> dict[str, str]:
    """Emit LLVM fork patch fragments for intrinsics and lowering."""

    spec = _coerce_patch_spec(spec)
    td_name = "".join(piece.capitalize() for piece in spec.dialect_name.split("_"))
    intrinsics_lines = [
        f"// Generated LLVM intrinsic declarations for {spec.dialect_name}",
        f'let TargetPrefix = "{spec.dialect_name}" in {{',
    ]
    lowering_lines = [
        f"// Generated lowering skeleton for {spec.dialect_name}",
        '#include "llvm/IR/IRBuilder.h"',
        "",
        f"namespace {spec.dialect_name} {{",
    ]
    test_lines = [
        "; RUN: opt -passes=instcombine %s -S | FileCheck %s",
        f"; Generated smoke test for {spec.dialect_name}",
        "",
    ]

    for intrinsic in spec.intrinsics:
        enum_name = intrinsic.name.replace(".", "_")
        args = ", ".join(intrinsic.arg_types) or "llvm_any_ty"
        intrinsics_lines.append(
            f"  def int_{enum_name} : Intrinsic<[ {intrinsic.ret_type} ], [ {args} ]>; // {intrinsic.summary}"
        )
        lowering_lines.append(f"llvm::Function *get_{enum_name}(llvm::Module &module);")
        test_lines.append(f"declare {intrinsic.ret_type} @{intrinsic.name}()")

    intrinsics_lines.append("}")
    lowering_lines.append("}")
    return {
        f"Intrinsics{td_name}.td": "\n".join(intrinsics_lines) + "\n",
        f"Lower{td_name}.cpp": "\n".join(lowering_lines) + "\n",
        f"{spec.dialect_name}_intrinsics.ll": "\n".join(test_lines) + "\n",
    }


__all__ = [
    "LLVMIntrinsicSpec",
    "LLVMPatchSpec",
    "generate_llvm_patch_bundle",
]
