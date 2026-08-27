# What IME actually does on this board

Written after reading the official SpaceMiT IME specification
(github.com/spacemit-com/riscv-ime-extension-spec, rev 20240422) and
disassembling the real builds. Three previously-recorded claims were wrong.
This file is the corrected record.

## The micro-tile is 4x4x8, and that is forced by hardware

The spec's MAC-unit table is indexed by `vl*SEW`, not by VLEN. For VLEN=256,
SEW=8, LMUL=1, `vl=32`:

    vl*SEW = 256, SEW=8  ->  M x N x K = 4 x 4 x 8, Copies=1

`Copies = (sqrt(VLEN/64) is integral ? 1 : 2)`; for VLEN=256 that is 1, so the
`[x2]` replication cases do not arise here.

**K is not a tiling parameter.** It is pinned at 8 by `vl`+`SEW`. Deep
reductions are expressed as a loop of accumulating `vmadot`s, so a K=32 matmul
tiles into 4 of them exactly as cleanly as K=2048 tiles into 256. The only other
legal int8 tile at VLEN=256 is `2x2x4[x2]`; anything else raises an illegal
instruction, because "when the MAC unit selected by the configure instruction is
not supported by the hardware, an illegal instruction will occur".

Operand layout, for anyone writing a kernel:

* A in `vs1`: (M,K) row-major, K contiguous. 32 bytes = one 4x8 int8 tile.
* B in `vs2`: **pre-transposed** -- indexed `B[j*K + k]`, i.e. C = A x B^T.
* C in `vd`: (M,N) int32, 4x4 = 64 bytes = **two** registers, `vd` even-aligned.
* Accumulates (`C += A*B`); there is no overwrite form, so zero `vd` first.
* `vsetvli` must set `vl=32, e8, m1` immediately before the instruction. Setting
  `vl` for the e32 accumulator and leaving it (`vsetvli zero, zero`) leaves
  vl=16 and **SIGILLs**.

## CORRECTION 1: the discriminator is M, not K

Earlier in this project I wrote that "K, not M, is the discriminator", from the
observation that DroNet's only vmadot is in a K=2048 matmul while MLP's K=10 and
K=32 matmuls have none. **That inference was wrong.**

Every xsmtvdot lowering path in the vendored IREE requires **M0=4 and N0=4**:

* `VectorContractCustomKernels.cpp` registers exactly one int8 pattern,
  `m0=4, n0=4, k0=8` -- and no narrow-M variants, unlike the RVV path which
  registers m0 in {8,7,4,2,1} plus a matvec specialization.
* The ukernel dispatcher branches `if (M0 == 4 && N0 == 4)` for the native path
  and otherwise runs a pure-RVV widening-multiply fallback with zero vmadot.
* For M=1, `chooseMatmulTile` actively *prefers* `{1,4,8}` (penalty 0) over
  `{4,4,8}` (penalty 3), and `limitVectorTileSizes` independently clamps the M
  tile 4 -> 1.

So MLP gets no vmadot because **M=1**, and K is irrelevant to that decision.

## CORRECTION 2: the vmadot is real and executed -- not dead code

A reasonable hypothesis was that the single vmadot is dead code: the native
4x4x8 tile function is ALWAYS_INLINE and table-referenced, so it can survive
into an object even when the dispatcher never calls it. **Disassembly rules that
out for this build.**

`dronet$async_dispatch_16_matmul_1x1x2048_i8xi8xi32`, from
`files/module_dronet_q_int8_linked_embedded_elf_riscv_64.s`:

    .LBB17_1:                                  # loop head
            ...vle8.v loads...
            addi    a5, a5, 32                 # advance 32 B = one 4x8 tile
            vsetvli zero, t0, e8, m1, ta, ma   # t0=32 -> select the 4x4x8 unit
            smt.vmadot      v8, v13, v12       # INSIDE the loop
            ...vslidedown extracts...
            bne     a5, t2, .LBB17_1           # live back edge

It is inside a real loop with a taken back edge, and the `vsetvli` immediately
before it is exactly the spec's tile selection. This build sets
`enable-ukernels=none`, so no ukernel bitcode is linked at all -- the instruction
came from the vector-contract custom-kernel path, which inlines directly into
the dispatch body. There is no dead ukernel here to be confused with.

## CORRECTION 3: but 15/16 of that work is discarded

The interesting part is four instructions later:

    vsetivli        zero, 4, e32, mf2, ta, ma
    vmv.v.i         v0, 1
    vse32.v         v8, (a0), v0.t             # MASKED store, mask = 1

IREE padded a 1x1x2048 GEMV up to the 4x4 tile the instruction requires, ran it,
and then stored **one** of the sixteen int32 results. So the matrix engine does
fire on this board, on exactly one dispatch of one model, computing 16x more
arithmetic than the model asked for.

Measured consequence: dispatch 16 is 0.0926 ms under IME against 0.1200 ms under
RVV -- **23% faster** even while wasting 15/16 of the lanes. But that dispatch is
**0.075% of DroNet's 122.7 ms**, so it is worth 0.027 ms overall, against IME
being 7.9% slower on the model as a whole.

## CORRECTION 4: the "IME wins" in compile_advice.json are not IME wins

`artifacts/k1_run/compile_advice.json` recommends IME for dispatch 14
(`reduction_128x4x4x64`, -25.8%) and dispatch 7 (`elementwise_32x14x14`, -5.3%).
**Neither function contains a vmadot.** Measured here: dispatch 14 is 0.742x
under IME -- 26% faster -- with no matrix instruction anywhere in it. That is
incidental codegen variation from the differing data-tiling path, and
attributing it to the matrix engine is wrong. Dispatch 16, the one dispatch that
does use IME, is a 0.772x ratio -- indistinguishable from the no-vmadot case.

## CORRECTION 5: no convolution can ever use IME here

All 8 of DroNet's convolutions compiled to pure RVV. That is not a shape
accident: **every xsmtvdot hook in this IREE is matmul-only**
(`getMatmulSpacemiTVectorSizes`, and a tile query handling only
`OPERATION_MATMUL_I8I8I32`). No conv -> img2col/mmt4d path is wired to it, so no
convolution can reach the pattern regardless of its dimensions.

This is despite the spec devoting its entire worked example to convolution via
`vmadot`/`vmadot1`/`vmadot2` -- the sliding-window instructions exist precisely
for conv, and that lowering is simply not implemented.

## Consequence for this project

DroNet is 111 of its 122.7 ms in convolutions, and none of them can reach the
matrix engine. So **"IME is 7.9% slower on DroNet" is not a measurement of the
matrix engine** -- it is a measurement of what the `+xsmtvdot` data-tiling path
does to code that is 99.9% RVV either way. The honest statement is that IME is
untested on this workload, because the workload never reaches it.

Two routes would change that, in increasing order of work:

1. **Pad narrow-M matmuls to M0=4** instead of shrinking the tile to `{1,4,8}`.
   Only a win where the RVV fallback is more than 4x slower, since vmadot then
   wastes 3 of 4 M lanes -- but dispatch 16 shows it can be (0.772x while
   wasting 15/16).
2. **Wire conv through img2col/mmt4d so it can hit the matmul hooks.** This is
   where DroNet's time actually is, and it is the only route with access to
   111 ms of the 122.7.

## Reference facts worth keeping

* Extension name is `xsmtvdot`; depends on `Zve32f`, not merely `Zve32x`.
  `-mcpu=spacemit-x60` sets VLEN=256 + IME in one flag.
* LLVM spells every mnemonic `smt.`-prefixed (`smt.vmadot`). Bare `vmadot` is
  SpaceMiT-toolchain syntax only.
* Raw encoding, verified byte-exact against LLVM's MC tests:
  `.insn r 0x2b, 3, 0x71, v8, v0, v4` -- opcode 0x2b (Custom-1), funct3 3 (SS =
  signed x signed), funct7 0x71 (OPMMA).
* LLVM IR intrinsic exists and is usable:
  `llvm.riscv.smt.vmadot.nxv4i32.nxv8i8.nxv8i8(acc, a, b, vl)`. There are **no**
  clang C builtins and no intrinsics header, so kernels need `.insn` or the IR
  intrinsic.
* The X60 implements a strict subset: int8 x int8 -> int32 only. No int4, no
  bf16, no fp16 matrix path (those belong to the VLEN=1024 parts).
* LLVM has **no scheduling model** for `smt.vmadot` -- zero hits for it in
  `RISCVSchedSpacemitX60.td` -- so latency and throughput must be measured, not
  looked up. One vmadot = 128 int8 MACs, against 32 MACs per `vwmul`+`vwadd`
  pair on RVV, so the instruction-count ceiling is 8:1.
* `check_hotloop_asm.py`, referenced by
  `benchmarks/SpacemiTX60/compile_matmul_xsmt_i8_ukernel_all.sh`, does not exist
  in the tree. `runtime/scripts/verify_ime_build.sh` replaces it, and it is now
  wired into `compile_all_models.sh` with per-model expectations -- note that a
  presence check alone would pass on dead code, which is why the loop-membership
  check above had to be done by hand.

## The ukernel route: measured, and it does not work

The plan predicted that enabling the vendored ukernels was "the most likely
route to more than one vmadot", since `enable-ukernels=none` in the `generic:`
block makes 826 lines of hand-written RISC-V int8 microkernel -- including a
native 4x4x8 `smt.vmadot` tile -- dead code in every K1 build.

Two variants were added (`RVV_ukernel`, `IME_ukernel`, both with
`enable-ukernels=all` + `link-ukernel-bitcode=true` and
`enable-vector-contract-custom-kernels=false` so the two routes to the same
instruction cannot be confused), compiled, and profiled on the board.

**The prediction is refuted.** Measured, DroNet, cluster 0, core 0, 10 reps:

| variant | dronet | vs RVV | mlp | vs RVV | vmadot |
|---|---|---|---|---|---|
| RVV | 113.71 ms | 1.000 | 335.7 us | 1.000 | 0 |
| scalar | 137.85 ms | 1.212 | 357.4 us | 1.064 | 0 |
| IME | 122.73 ms | 1.079 | n/a | | **1** |
| RVV_ukernel | 113.64 ms | **0.999** | 335.8 us | 1.000 | 0 |
| IME_ukernel | 122.73 ms | **1.079** | 349.6 us | 1.041 | **0** |

Two things follow, and the second is the more useful.

**1. The ukernel path is unreachable for these models.** `RVV_ukernel` is
0.999x of `RVV` -- identical within noise -- and the disassembly says why: the
build contains **zero `iree_uk` symbols and zero `mmt4d` symbols**. The riscv_64
ukernel bitcode *is* built
(`build/host-vanilla-release/.../ukernel_bitcode_generic_riscv_64_mmt4d_tile_generic.c.bc`
exists), so this is not a missing artifact. No mmt4d op is ever formed for these
shapes, so there is nothing for an mmt4d ukernel to serve, and the flag changes
nothing. Reaching it would require the conv -> img2col/mmt4d lowering that
correction 5 above says is absent.

**2. The +7.9% IME penalty has nothing to do with the matrix instruction.**
`IME_ukernel` measures **122.73 ms, identical to `IME`'s 122.73 ms**, while
containing **no vmadot at all** (turning off custom-kernels removed the only
one). So the entire IME regression is the `+xsmtvdot` data-tiling path acting on
code that is pure RVV either way. The one real vmadot is worth 0.027 ms of
122.73 -- unmeasurable at model scale, consistent with dispatch 16 being 0.075%
of runtime.

That also means `IME_ukernel` is strictly the worst of the IME options: it pays
the data-tiling penalty and loses the instruction. It is kept in the yaml as the
recorded negative result rather than deleted, so nobody re-runs this experiment.
