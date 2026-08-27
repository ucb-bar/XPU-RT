# Codex-generated RVV `linear_s8` — correct, and rejected

Generated through ModelBlaster's own provider path (`LLM_PROVIDER=codex`), model
`gpt-5.6-sol`, prompt sha256[:16] `a10e56e72849432c`, 19 818 in / 9 803 out
tokens. Call logged in `../codex_calls.jsonl` with `provider: codex`.

## What it did

Uses the right primitives for an M=1 GEMV — `__riscv_vwmacc_vv_i32m4` widening
multiply-accumulate over the contiguous reduction, `__riscv_vredsum_vs_i32m4_i32m1`
to finish, `vsext_vf2` for the offset path. Compiles clean at
`-march=rv64gcv_zvl256b -mabi=lp64d`. Runs on the K1 **bit-exact**:
`max_abs_err=0 max_rel_err=0`.

## Measured on the board (core 0, rdtime ticks)

| `linear_s8` call | scalar ref | curated RVV | **Codex RVV** | codex/curated |
|---|---|---|---|---|
| M=1 K=16 N=256 | 1052 | 678 | 655 | 0.97x |
| M=1 K=256 N=128 | 5567 | **418** | 803 | **1.92x** |
| M=1 K=128 N=64 | 1414 | **164** | 324 | **1.98x** |
| M=1 K=64 N=4 | 62 | **20** | 25 | 1.25x |
| **linear_s8 total** | 8095 | **1280** | 1807 | **1.41x** |
| model total | 8965 | **2122** | 2651 | 1.25x |

## Verdict: REJECT

Against the scalar reference the Codex kernel is **4.48x faster**, which is the
number it would be tempting to report. Against the baseline that actually
matters — `kernels/rvv/rvv_linear_s8_direct.c`, which already exists in the tree
— it is **41% slower** on `linear_s8` and 25% slower end to end.

The accept criterion is "correctness passes **and** the selected metric
improves". Correctness passes; the metric does not. So the kernel is archived
here rather than promoted into `kernels/rvv/`, and the curated kernel stays.

It only loses on the two large-K calls (K=256 and K=128), and wins slightly at
K=16 — consistent with the curated kernel blocking the reduction better as K
grows. That is a concrete, testable direction for a second round, which is the
useful thing to take from a rejected candidate.
