"""Hand-authored Saturn TargetKnowledgeCard builder.

The universal ingestion seed at :mod:`xpu_rt.memory.seeds.saturn`
drives an LLM-routed pipeline that consumes the asciidoc reference
manual + Scala parameter sweep. That gives a richer card but costs
Gemini tokens.

For the cross-target comparison study we want a usable card now, at
$0 cost. This module assembles one from the same canonical inputs
(``configs/targets/saturn_opu_v128.yaml`` + the RVV-1.0 spec) but
without any LLM in the loop. Run via::

    uv run python -m xpu_rt.memory.seeds.saturn_hand_card

to write the card to
``xpu-rt/.knowledge/targets/saturn_opu_v128/target_card.json``.
"""

from __future__ import annotations

import argparse
import sys

from xpu_rt.memory import target_knowledge as tk
from xpu_rt.memory.target_knowledge import (
    DerivationRule,
    HardwareSpec,
    ISAInstruction,
    IntrinsicSignature,
    KernelExemplar,
    MemoryTierSpec,
    ParameterRange,
    TargetKnowledgeCard,
)


# ---------------------------------------------------------------------------
# Saturn OPU V128 facts — sourced from
# ``configs/targets/saturn_opu_v128.yaml`` + the Saturn reference
# manual (``chipyard/generators/saturn/docs/intro.adoc``, etc.).
# ---------------------------------------------------------------------------


_PARAMETERS = (
    ParameterRange(name="vLen", description="Vector register width (bits)", default=128),
    ParameterRange(name="dLen", description="Datapath width (bits)", default=64),
    ParameterRange(name="vrfBanking", description="Vector register file banks", default=2),
    ParameterRange(
        name="issStructure",
        description="Issue stage policy: single-lane vs paired-lane vmacc",
        default="paired",
    ),
)

_MEMORY_TIERS = (
    MemoryTierSpec(name="vrf", kind="registers", size_bytes=8 * 1024),
    MemoryTierSpec(name="l2", kind="l2", size_bytes=512 * 1024),
    MemoryTierSpec(name="dram", kind="dram"),
)


_INSTRUCTIONS = (
    ISAInstruction(
        mnemonic="vsetvli",
        signature="vsetvli rd, rs1, vtypei",
        summary="Configure vector unit (SEW, LMUL, mask policy) and read back the active vL.",
    ),
    ISAInstruction(
        mnemonic="vle8.v",
        signature="vle8.v vd, (rs1)",
        summary="Unit-stride load of 8-bit elements into vector register vd.",
    ),
    ISAInstruction(
        mnemonic="vle32.v",
        signature="vle32.v vd, (rs1)",
        summary="Unit-stride load of 32-bit elements into vector register vd.",
    ),
    ISAInstruction(
        mnemonic="vse32.v",
        signature="vse32.v vs3, (rs1)",
        summary="Unit-stride store of 32-bit elements from vector register.",
    ),
    ISAInstruction(
        mnemonic="vmacc.vx",
        signature="vmacc.vx vd, rs1, vs2",
        summary="Integer multiply-accumulate: vd[i] += rs1 * vs2[i].",
    ),
    ISAInstruction(
        mnemonic="vwmacc.vx",
        signature="vwmacc.vx vd, rs1, vs2",
        summary="Widening signed multiply-accumulate: vd[2i:2i+1] += rs1 * vs2[i] (8-bit→16-bit).",
    ),
    ISAInstruction(
        mnemonic="vmul.vv",
        signature="vmul.vv vd, vs1, vs2",
        summary="Vector-vector integer multiply, low half of result.",
    ),
    ISAInstruction(
        mnemonic="vadd.vv",
        signature="vadd.vv vd, vs1, vs2",
        summary="Vector-vector integer add.",
    ),
    ISAInstruction(
        mnemonic="vredsum.vs",
        signature="vredsum.vs vd, vs2, vs1",
        summary="Vector reduce sum into scalar element vd[0] (seed = vs1[0]).",
    ),
    ISAInstruction(
        mnemonic="vmv.v.x",
        signature="vmv.v.x vd, rs1",
        summary="Broadcast scalar rs1 to every element of vd.",
    ),
    ISAInstruction(
        mnemonic="vfmacc.vv",
        signature="vfmacc.vv vd, vs1, vs2",
        summary="Float fused multiply-accumulate: vd[i] += vs1[i] * vs2[i].",
    ),
)


_INTRINSICS = (
    IntrinsicSignature(
        name="__riscv_vsetvl_e8m1",
        c_signature="size_t __riscv_vsetvl_e8m1(size_t avl);",
        summary="Request vL for SEW=8 LMUL=1 — returns the granted vL (≤ vLen/8 = 16 for vLen=128).",
    ),
    IntrinsicSignature(
        name="__riscv_vsetvl_e32m1",
        c_signature="size_t __riscv_vsetvl_e32m1(size_t avl);",
        summary="Request vL for SEW=32 LMUL=1 — returns granted vL (≤ vLen/32 = 4 for vLen=128).",
    ),
    IntrinsicSignature(
        name="__riscv_vle8_v_i8m1",
        c_signature="vint8m1_t __riscv_vle8_v_i8m1(const int8_t *base, size_t vl);",
        summary="Unit-stride load i8.",
    ),
    IntrinsicSignature(
        name="__riscv_vle32_v_i32m1",
        c_signature="vint32m1_t __riscv_vle32_v_i32m1(const int32_t *base, size_t vl);",
        summary="Unit-stride load i32.",
    ),
    IntrinsicSignature(
        name="__riscv_vse32_v_i32m1",
        c_signature="void __riscv_vse32_v_i32m1(int32_t *base, vint32m1_t v, size_t vl);",
        summary="Unit-stride store i32.",
    ),
    IntrinsicSignature(
        name="__riscv_vmv_v_x_i32m1",
        c_signature="vint32m1_t __riscv_vmv_v_x_i32m1(int32_t value, size_t vl);",
        summary=(
            "BROADCAST scalar i32 to vector i32. CRITICAL: this is the "
            "integer-broadcast intrinsic. `__riscv_vfmv_v_f_i32m1` does "
            "NOT exist — `vfmv` is for floats only. For integer-zero "
            "initialisation use `__riscv_vmv_v_x_i32m1(0, vl)`."
        ),
    ),
    IntrinsicSignature(
        name="__riscv_vmv_v_i_i32m1",
        c_signature="vint32m1_t __riscv_vmv_v_i_i32m1(/* imm */ int imm, size_t vl);",
        summary="Broadcast 5-bit immediate — handy for zero-init via __riscv_vmv_v_i_i32m1(0, vl).",
    ),
    IntrinsicSignature(
        name="__riscv_vwmacc_vx_i32m1",
        c_signature="vint32m1_t __riscv_vwmacc_vx_i32m1(vint32m1_t acc, int16_t rs, vint16mf2_t vs2, size_t vl);",
        summary="Widening i16 → i32 fused multiply-accumulate.",
    ),
    IntrinsicSignature(
        name="__riscv_vwmacc_vv_i32m1",
        c_signature="vint32m1_t __riscv_vwmacc_vv_i32m1(vint32m1_t acc, vint16mf2_t vs1, vint16mf2_t vs2, size_t vl);",
        summary="Widening i16×i16 → i32 fused multiply-accumulate, vector-vector.",
    ),
    IntrinsicSignature(
        name="__riscv_vwmul_vv_i32m1",
        c_signature="vint32m1_t __riscv_vwmul_vv_i32m1(vint16mf2_t vs1, vint16mf2_t vs2, size_t vl);",
        summary="Widening i16×i16 → i32 vector-vector multiply.",
    ),
    IntrinsicSignature(
        name="__riscv_vsext_vf2_i16mf2",
        c_signature="vint16mf2_t __riscv_vsext_vf2_i16mf2(vint8mf4_t op, size_t vl);",
        summary="Sign-extend i8 vector to i16 (LMUL halves; the i8 LMUL is mf4 for an i16 LMUL=mf2).",
    ),
    IntrinsicSignature(
        name="__riscv_vredsum_vs_i32m1_i32m1",
        c_signature="vint32m1_t __riscv_vredsum_vs_i32m1_i32m1(vint32m1_t scratch, vint32m1_t vec, size_t vl);",
        summary="Reduce-sum vec[0..vl-1] into scratch[0] (use for dot-product style inner loops).",
    ),
    IntrinsicSignature(
        name="__riscv_vmv_x_s_i32m1_i32",
        c_signature="int32_t __riscv_vmv_x_s_i32m1_i32(vint32m1_t v);",
        summary="Extract element 0 of an i32 vector as a scalar i32. Companion to vredsum.",
    ),
)


_DERIVATION_RULES = (
    DerivationRule(
        name="active_vL_i8",
        symbolic="vL <= vLen / SEW = 128 / 8 = 16",
        concrete_value=16.0,
        unit="elements",
        derivation=(
            "vLen=128 (target preset), SEW=8 for i8. RVV 1.0 grants up to "
            "vLen/SEW elements per vsetvli call; the picker should request "
            "AVL ≤ 16 for the i8 GEMM inner loop and accept whatever vL "
            "comes back."
        ),
        applies_to="i8 matmul inner loop",
        how_to_apply="Call __riscv_vsetvl_e8m1(remaining_K) and use the returned vl.",
    ),
    DerivationRule(
        name="active_vL_i32_accumulator",
        symbolic="vL <= vLen / 32 = 128 / 32 = 4",
        concrete_value=4.0,
        unit="elements",
        derivation=(
            "i8→i32 widening multiply-accumulate (vwmacc) produces 4× the "
            "destination width, so the i32 accumulator vector can hold "
            "vLen/32 elements at LMUL=1."
        ),
        applies_to="i8 GEMM accumulator",
        how_to_apply="Allocate (M_tile / 4) × N_tile i32 accumulator strips.",
    ),
)


_EXEMPLARS = (
    KernelExemplar(
        name="vec_sgemm",
        op_family="matmul",
        path="vec_sgemm.c",
        language="c",
        tags=("rvv", "f32", "gemm"),
        source="chipyard/generators/saturn/benchmarks/vec-sgemm/main.c",
    ),
    KernelExemplar(
        name="vec_dotprod",
        op_family="reduce",
        path="vec_dotprod.c",
        language="c",
        tags=("rvv", "f32", "reduction"),
        source="chipyard/generators/saturn/benchmarks/vec-dotprod/main.c",
    ),
    KernelExemplar(
        name="opu_gemm_i8",
        op_family="matmul",
        path="opu_gemm.c",
        language="c",
        tags=("rvv", "i8", "outer-product"),
        source="chipyard/generators/saturn/benchmarks/opu-gemm/main.c",
    ),
)


_CONSTRAINTS = (
    "RVV 1.0 conformant; vLen=128 bits per the OPU V128 preset.",
    "DLEN (datapath) = vLen/2 = 64 bits — paired-lane vmacc is 2 cycles/element at full SEW.",
    "Call vsetvli before every vector inner loop; the returned vL may be < requested.",
    "i8×i8→i32 inner loops should use vwmacc.vx (widening) — vmacc.vx narrows to SEW.",
    "Loads are coalesced when the base pointer is element-aligned.",
    "Strided / indexed loads exist (vlse, vluxei) but are slower than unit-stride.",
    "CRITICAL: RVV vector types (vint8m1_t, vint32m1_t, ...) are SIZELESS. "
    "You CANNOT declare arrays of them ('error: array elements cannot have RVV type'). "
    "Instead use multiple named variables (vint32m1_t acc0, acc1, acc2, ...) for tile accumulators.",
    "CRITICAL: there is no `__riscv_vfmv_v_f_i32m1`. The `vfmv` family is for FLOATS only. "
    "For integer-broadcast use `__riscv_vmv_v_x_i32m1(0, vl)` or `__riscv_vmv_v_i_i32m1(0, vl)`.",
    "Canonical i8×i8 → i32 dot-product inner loop pattern (RVV 1.0 C intrinsics):\n"
    "    size_t vl = __riscv_vsetvl_e8m1(K_remaining);\n"
    "    vint8mf4_t a_i8 = __riscv_vle8_v_i8mf4(&A[m*K + k], vl);\n"
    "    vint8mf4_t b_i8 = __riscv_vle8_v_i8mf4(&B[k*N + n], vl);\n"
    "    vint16mf2_t a_i16 = __riscv_vsext_vf2_i16mf2(a_i8, vl);\n"
    "    vint16mf2_t b_i16 = __riscv_vsext_vf2_i16mf2(b_i8, vl);\n"
    "    acc = __riscv_vwmacc_vv_i32m1(acc, a_i16, b_i16, vl);\n"
    "Then reduce-sum with __riscv_vredsum_vs_i32m1_i32m1 and extract scalar with "
    "__riscv_vmv_x_s_i32m1_i32 to write into C[m*N+n].",
    "When in doubt, prefer a scalar reference loop — Saturn's stock spike with "
    "--isa=rv64gcv_zvl128b_zicntr will still run correctly without OPU asm macros, "
    "just slower. The harness reads cycles via the standard `mcycle` CSR.",
)


# ---------------------------------------------------------------------------
# Card assembly
# ---------------------------------------------------------------------------


def build_card() -> TargetKnowledgeCard:
    """Assemble the hand-authored Saturn TargetKnowledgeCard."""
    return TargetKnowledgeCard(
        target_id="saturn_opu_v128",
        target_profile_ref="configs/targets/saturn_opu_v128.yaml",
        hardware_spec=HardwareSpec(
            isa_family="riscv-rvv",
            parameters=_PARAMETERS,
            memory_tiers=_MEMORY_TIERS,
            instructions=_INSTRUCTIONS,
            intrinsics=_INTRINSICS,
            dataflow_modes=("rvv1.0", "opu-outer-product"),
            constraints=_CONSTRAINTS,
            derivation_rules=_DERIVATION_RULES,
        ),
        exemplars=_EXEMPLARS,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print but don't write.")
    args = parser.parse_args(argv)
    card = build_card()
    if args.dry_run:
        print(f"Card built for {card.target_id} with "
              f"{len(card.hardware_spec.instructions)} instructions, "
              f"{len(card.hardware_spec.intrinsics)} intrinsics, "
              f"{len(card.exemplars)} exemplars.", file=sys.stderr)
        return 0
    written = tk.save(card)
    print(f"wrote Saturn target card to {tk.target_dir(written.target_id)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_card", "main"]
