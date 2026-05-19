"""Hand-authored Gemmini TargetKnowledgeCard builder.

Parallel to :mod:`xpu_rt.memory.seeds.saturn_hand_card`. Writes
``xpu-rt/.xpu_rt/knowledge/targets/gemmini_mx/target_card.json`` from
canonical Chipyard Gemmini facts without any LLM call.

Run via::

    uv run python -m xpu_rt.memory.seeds.gemmini_hand_card
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


_PARAMETERS = (
    ParameterRange(name="pe_dim", description="Systolic mesh dimension (rows × cols)", default=16),
    ParameterRange(name="spad_size_kb", description="Scratchpad capacity (KiB)", default=256),
    ParameterRange(name="acc_size_kb", description="Accumulator capacity (KiB)", default=64),
    ParameterRange(name="dataflow", description="Output-stationary or weight-stationary", default="weight_stationary"),
)


_MEMORY_TIERS = (
    MemoryTierSpec(name="scratchpad", kind="scratchpad", size_bytes=256 * 1024),
    MemoryTierSpec(name="accumulator", kind="accumulator", size_bytes=64 * 1024),
    MemoryTierSpec(name="dram", kind="dram"),
)


_INSTRUCTIONS = (
    ISAInstruction(
        mnemonic="gemmini_mvin",
        signature="gemmini_mvin(dram_addr, sp_addr)",
        summary="DMA a tile from DRAM into scratchpad (16-row by configurable-col).",
        funct_code=2,
    ),
    ISAInstruction(
        mnemonic="gemmini_mvout",
        signature="gemmini_mvout(dram_addr, sp_addr)",
        summary="DMA a tile from accumulator out to DRAM.",
        funct_code=3,
    ),
    ISAInstruction(
        mnemonic="gemmini_preload",
        signature="gemmini_preload(weight_sp, out_acc)",
        summary="Stage the weight matrix tile + reserve an accumulator destination.",
        funct_code=6,
    ),
    ISAInstruction(
        mnemonic="gemmini_compute_preloaded",
        signature="gemmini_compute_preloaded(a_sp, b_sp)",
        summary="Run one mesh-sized MMA against the previously preloaded weight.",
        funct_code=4,
    ),
    ISAInstruction(
        mnemonic="gemmini_compute_accumulated",
        signature="gemmini_compute_accumulated(a_sp, b_sp)",
        summary="MMA accumulating into the existing accumulator entry.",
        funct_code=5,
    ),
    ISAInstruction(
        mnemonic="gemmini_loop_ws",
        signature="gemmini_loop_ws(M, N, K, …)",
        summary="High-level CISC instruction running a full tiled GEMM in weight-stationary mode.",
        funct_code=8,
    ),
    ISAInstruction(
        mnemonic="gemmini_fence",
        signature="gemmini_fence()",
        summary="Wait for all outstanding Gemmini commands to retire.",
    ),
    ISAInstruction(
        mnemonic="gemmini_config_st",
        signature="gemmini_config_st(stride)",
        summary="Configure the row stride for subsequent mvout operations.",
        funct_code=1,
    ),
    ISAInstruction(
        mnemonic="gemmini_config_ex",
        signature="gemmini_config_ex(dataflow, act, scale, …)",
        summary="Configure dataflow mode + activation + output scale.",
        funct_code=0,
    ),
    ISAInstruction(
        mnemonic="gemmini_extended_mvin",
        signature="gemmini_extended_mvin(dram, sp, cols, rows)",
        summary="mvin variant taking explicit (cols, rows) — for tiles smaller than the mesh.",
    ),
)


_INTRINSICS = (
    IntrinsicSignature(
        name="tiled_matmul_auto",
        c_signature=(
            "void tiled_matmul_auto(\n"
            "    size_t dim_I, size_t dim_J, size_t dim_K,\n"
            "    const elem_t* A, const elem_t* B,\n"
            "    const void* D, void* C,\n"
            "    size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,\n"
            "    scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,\n"
            "    int act, acc_scale_t scale, acc_scale_t bert_scale,\n"
            "    bool repeating_bias,\n"
            "    bool transpose_A, bool transpose_B,\n"
            "    bool full_C, bool low_D,\n"
            "    uint8_t weightA,\n"
            "    enum tiled_matmul_type_t tiled_matmul_type)"
        ),
        summary=(
            "One-line tiled GEMM with 21 arguments. ALL arguments are required. "
            "Common values for i8 GEMM: stride_*=dim_J or stride_*=dim_K as appropriate, "
            "A/B/D_scale_factor=MVIN_SCALE_IDENTITY (or 1.0f for f32), act=NO_ACTIVATION, "
            "scale=ACC_SCALE_IDENTITY (or 1.0), bert_scale=ACC_SCALE_IDENTITY, "
            "repeating_bias=false, transpose_A=false, transpose_B=false, full_C=false, "
            "low_D=false, weightA=3, tiled_matmul_type=WS. Pass D=NULL for no bias."
        ),
    ),
    IntrinsicSignature(
        name="gemmini_extended_mvin",
        c_signature="#define gemmini_extended_mvin(dram_addr, spad_addr, cols, rows)",
        summary="DMA cols×rows i8 elements (cols ≤ DIM=16, rows ≤ DIM) from DRAM to scratchpad.",
    ),
    IntrinsicSignature(
        name="gemmini_mvin",
        c_signature="#define gemmini_mvin(dram_addr, spad_addr)",
        summary="Shorthand for gemmini_extended_mvin with (DIM, DIM) — moves a full 16×16 i8 tile.",
    ),
    IntrinsicSignature(
        name="gemmini_extended_mvout",
        c_signature="#define gemmini_extended_mvout(dram_addr, spad_addr, cols, rows)",
        summary="DMA cols×rows accumulator entries out to DRAM.",
    ),
    IntrinsicSignature(
        name="gemmini_mvout",
        c_signature="#define gemmini_mvout(dram_addr, spad_addr)",
        summary="Shorthand for gemmini_extended_mvout with (DIM, DIM).",
    ),
    IntrinsicSignature(
        name="gemmini_extended_preload",
        c_signature="#define gemmini_extended_preload(BD, C, BD_cols, BD_rows, C_cols, C_rows)",
        summary="Stage BD weights and reserve C accumulator entry with explicit shapes.",
    ),
    IntrinsicSignature(
        name="gemmini_preload",
        c_signature="#define gemmini_preload(BD, C)",
        summary="Shorthand for gemmini_extended_preload with all shapes set to DIM.",
    ),
    IntrinsicSignature(
        name="gemmini_extended_compute_preloaded",
        c_signature="#define gemmini_extended_compute_preloaded(A, BD, A_cols, A_rows, BD_cols, BD_rows)",
        summary="MMA using preloaded BD; explicit operand shapes for boundary tiles.",
    ),
    IntrinsicSignature(
        name="gemmini_compute_preloaded",
        c_signature="#define gemmini_compute_preloaded(A, BD)",
        summary="MMA using preloaded BD; full DIM×DIM tiles.",
    ),
    IntrinsicSignature(
        name="gemmini_compute_accumulated",
        c_signature="#define gemmini_compute_accumulated(A, BD)",
        summary="MMA accumulating into the existing accumulator entry.",
    ),
    IntrinsicSignature(
        name="gemmini_extended_config_ex",
        c_signature=(
            "#define gemmini_extended_config_ex("
            "dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose)"
        ),
        summary="Configure dataflow + activation + shift + per-row stride + transpose flags.",
    ),
    IntrinsicSignature(
        name="gemmini_config_ex",
        c_signature="#define gemmini_config_ex(dataflow, sys_act, sys_shift)",
        summary="Shorthand — A_stride=1, no transpose.",
    ),
    IntrinsicSignature(
        name="gemmini_config_st",
        c_signature="#define gemmini_config_st(stride)",
        summary="Configure mvout row stride (elements between rows of the DRAM output).",
    ),
    IntrinsicSignature(
        name="gemmini_loop_ws",
        c_signature=(
            "#define gemmini_loop_ws("
            "I, J, K, pad_I, pad_J, pad_K, A, B, D, C, "
            "A_stride, B_stride, D_stride, C_stride, "
            "A_transpose, B_transpose, full_C, low_D, "
            "ex_accumulate, act, a_spad_id, b_spad_id, is_resadd)"
        ),
        summary=(
            "CISC loop running a full tiled WS GEMM over I × J × K tiles. 23 args; "
            "common: pad_*=0 when dims divide DIM, a_spad_id=0, b_spad_id=0, is_resadd=0, "
            "ex_accumulate=0 for first chunk."
        ),
    ),
    IntrinsicSignature(
        name="gemmini_flush",
        c_signature="#define gemmini_flush(skip)",
        summary="Flush the systolic array pipeline. Always pass 0; called at kernel start.",
    ),
    IntrinsicSignature(
        name="gemmini_fence",
        c_signature="static inline void gemmini_fence(void)",
        summary="Wait for all outstanding Gemmini commands to retire before reading C from DRAM.",
    ),
)


_DERIVATION_RULES = (
    DerivationRule(
        name="dim_tile_size",
        symbolic="DIM = pe_dim = 16; tile = DIM × DIM = 256 i8 = 256 B",
        concrete_value=256.0,
        unit="bytes",
        derivation=(
            "Gemmini's mesh is pe_dim × pe_dim = 16 × 16. The single-MMA "
            "tile is exactly that many i8 elements (256 B). All mvin/mvout "
            "address calculations work in DIM-aligned units."
        ),
        applies_to="i8 matmul",
        how_to_apply=(
            "Pad M / K / N to multiples of DIM=16, or use "
            "gemmini_extended_mvin/extended_compute to handle the boundary."
        ),
    ),
    DerivationRule(
        name="scratchpad_per_input_tile",
        symbolic=(
            "tile_K × tile_J × 1 B (B input)  +  tile_I × tile_K × 1 B (A input)  "
            "≤ spad_size_kb × 1024"
        ),
        concrete_value=256.0 * 1024,
        unit="bytes",
        derivation=(
            "Scratchpad holds A and B tiles + intermediates. Default 256 KiB. "
            "tiled_matmul_auto's incremental picker walks tile_I = tile_J = "
            "tile_K = 1 upward, doubling until the constraint binds — that's "
            "what makes it succeed where boundary-maxing fails."
        ),
        applies_to="tiled GEMM",
        how_to_apply="Call tiled_matmul_auto; trust its picker.",
    ),
    DerivationRule(
        name="accumulator_per_output_tile",
        symbolic="tile_I × tile_J × 4 B (i32 acc) ≤ acc_size_kb × 1024",
        concrete_value=64.0 * 1024,
        unit="bytes",
        derivation=(
            "Accumulator stores i32 partial sums; 64 KiB default. Bounds "
            "tile_I × tile_J ≤ 16,384 (= 64 KiB / 4 B). For M=64, N=720 → "
            "max tile_J ≈ 256."
        ),
        applies_to="tiled GEMM",
        how_to_apply=(
            "When using gemmini_loop_ws or hand-rolled tiling, ensure "
            "tile_I × tile_J × 4 ≤ acc_size_kb × 1024."
        ),
    ),
)


_EXEMPLARS = (
    KernelExemplar(
        name="tiled_matmul_baremetal",
        op_family="matmul",
        path="tiled_matmul_baremetal.c",
        language="c",
        tags=("gemmini", "i8", "i32", "tiled_matmul_auto"),
        source="chipyard/generators/gemmini/software/gemmini-rocc-tests/bareMetalC/tiled_matmul_baremetal.c",
    ),
    KernelExemplar(
        name="matmul_ws",
        op_family="matmul",
        path="matmul_ws.c",
        language="c",
        tags=("gemmini", "weight_stationary"),
        source="chipyard/generators/gemmini/software/gemmini-rocc-tests/bareMetalC/matmul_ws.c",
    ),
    KernelExemplar(
        name="resnet50_layer1",
        op_family="conv",
        path="resnet50_layer1.c",
        language="c",
        tags=("gemmini", "conv", "tiled"),
        source="chipyard/generators/gemmini/software/gemmini-rocc-tests/bareMetalC/resnet50_layer1.c",
    ),
)


_CONSTRAINTS = (
    "RoCC custom-3 opcode (XCUSTOM_ACC); use the gemmini.h intrinsics, not raw asm.",
    "Scratchpad + accumulator addresses are in DIM-aligned units (16-row granularity).",
    "Weight-stationary dataflow (default) loads weights once, streams activations. Output-stationary is opt-in via gemmini_config_ex.",
    "Always call gemmini_flush(0) at kernel start; gemmini_fence() before reading results.",
    "tiled_matmul_auto handles all the picker logic — prefer it over hand-tiling unless you have a specific reason.",
    "Tiles smaller than DIM=16 in any dim need gemmini_extended_* variants.",
    "i8 inputs → i32 accumulator (4× the byte width). Output dtype selected by gemmini_config_ex.",
    "tiled_matmul_auto has 21 required positional args — ALL MUST be passed. Skipping any causes 'too few arguments' compile errors.",
    "CRITICAL: when the C output buffer is int32_t* (not elem_t*/int8_t*), "
    "pass full_C=true so the systolic array writes raw i32 accumulator values "
    "instead of applying scale + activation + saturation to fit elem_t. Our "
    "harness uses int32_t C[M*N] and a scalar reference that computes "
    "C[m*N+n] = sum_k (int32_t)A[m*K+k] * (int32_t)B[k*N+n] — full_C=false "
    "with default scale produces wrong (truncated/scaled) results.",
    "Canonical i8 → i32 GEMM call for our int32_t output harness:\n"
    "    tiled_matmul_auto(\n"
    "        M, N, K,\n"
    "        (elem_t*)A, (elem_t*)B, NULL /*no bias*/, (elem_t*)C,\n"
    "        /*stride_A=*/K, /*stride_B=*/N, /*stride_D=*/0, /*stride_C=*/N,\n"
    "        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,\n"
    "        /*act=*/NO_ACTIVATION,\n"
    "        /*scale=*/ACC_SCALE_IDENTITY,\n"
    "        /*bert_scale=*/0,\n"
    "        /*repeating_bias=*/false,\n"
    "        /*transpose_A=*/false, /*transpose_B=*/false,\n"
    "        /*full_C=*/true,   /* MUST be true for int32_t C[] output */\n"
    "        /*low_D=*/false,\n"
    "        /*weightA=*/0,\n"
    "        /*tiled_matmul_type=*/WS);\n"
    "(cast (int32_t*)C → (elem_t*)C is required by the API signature; "
    "full_C=true tells the implementation to write i32 anyway.)",
    "gemmini_loop_ws has 23 required positional args.",
    "Scratchpad rows = BANK_NUM * BANK_ROWS / 2 = bank-partitioned; double-buffer aware.",
    "Accumulator rows = ACC_ROWS / DIM = number of DIM×DIM accumulator entries.",
    "elem_t is int8_t for default INT8 config; acc_t is int32_t; scale_t is scale_t_bits (typically int32_t).",
    "Canonical constant names: MVIN_SCALE_IDENTITY (=1 i8 / 1.0 f32), ACC_SCALE_IDENTITY (=1.0), NO_ACTIVATION (=0), WS (weight-stationary), OS (output-stationary).",
    "When using tiled_matmul_auto, do NOT call gemmini_mvin/preload/compute/mvout yourself — tiled_matmul_auto handles all of that internally.",
    "DECISION RULE for K dimension: if K <= 720, call tiled_matmul_auto ONCE with the canonical args above. If K >= 960, you MUST use the K-split workaround described next — do NOT apply K-split when K <= 720 (it is strictly slower and unnecessary).",
    "CRITICAL — K >= 960 limitation: tiled_matmul_auto silently corrupts output "
    "(produces ~10–30% wrong values starting at index 0) when the K dimension is "
    "960 or larger. This affects the I8→I32 GEMM path with full_C=true. "
    "Empirically: K=720 → 0/N mismatches, K=960 → 6144/N mismatches, K=2560 → "
    "garbage. WORKAROUND: split K into chunks of <= 720 (we use KSPLIT=480 for "
    "extra headroom) and accumulate via the D-bias path. Pattern:\n"
    "    #define KSPLIT 480\n"
    "    for (int64_t k0 = 0; k0 < K; k0 += KSPLIT) {\n"
    "        int64_t kc = (K - k0 < KSPLIT) ? (K - k0) : KSPLIT;\n"
    "        int8_t  *A_sub = A + k0;       // A stride K, offset k0\n"
    "        int8_t  *B_sub = B + k0 * N;   // B stride N, offset k0*N\n"
    "        const void *D_arg = (k0 == 0) ? NULL : (const void *)C;\n"
    "        tiled_matmul_auto(M, N, kc, A_sub, B_sub, D_arg, C,\n"
    "            /*stride_A=*/K, /*stride_B=*/N, /*stride_D=*/N, /*stride_C=*/N,\n"
    "            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, 1, NO_ACTIVATION,\n"
    "            ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,\n"
    "            /*repeating_bias=*/false, false, false,\n"
    "            /*full_C=*/true, /*low_D=*/false, /*weightA=*/0, WS);\n"
    "    }\n"
    "On chunk 0 (k0=0) D=NULL so C is overwritten from scratch; on subsequent "
    "chunks D=C feeds the previous partial-sum back as bias before mvout, "
    "implementing the K-direction reduction. Verified correct for K up to "
    "2560 with N up to 2560.",
)


def build_card() -> TargetKnowledgeCard:
    return TargetKnowledgeCard(
        target_id="gemmini",
        target_profile_ref="configs/targets/gemmini.yaml",
        hardware_spec=HardwareSpec(
            isa_family="rocc-systolic",
            parameters=_PARAMETERS,
            memory_tiers=_MEMORY_TIERS,
            instructions=_INSTRUCTIONS,
            intrinsics=_INTRINSICS,
            dataflow_modes=("weight_stationary", "output_stationary"),
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
        print(
            f"Card built for {card.target_id} with "
            f"{len(card.hardware_spec.instructions)} instructions, "
            f"{len(card.hardware_spec.intrinsics)} intrinsics, "
            f"{len(card.exemplars)} exemplars, "
            f"{len(card.hardware_spec.derivation_rules)} derivation_rules.",
            file=sys.stderr,
        )
        return 0
    written = tk.save(card)
    print(f"wrote Gemmini target card to {tk.target_dir(written.target_id)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_card", "main"]
