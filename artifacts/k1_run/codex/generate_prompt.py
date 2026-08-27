import hashlib, os, sys
from modelblaster.pipeline import reference_kernels as rk
from modelblaster.pipeline.llm_client import make_llm_client
from modelblaster.pipeline.bedrock_client import extract_code_block

spec = rk.KERNEL_SPECS["linear_s8"]
system = ("You are writing a single C99 function for a RISC-V SpaceMiT X60 core. "
          "Output ONLY one ```c code block containing the complete function. No prose.")
user = f"""Optimise this kernel for RVV 1.0 on a SpaceMiT X60 (rv64gcv, VLEN=256, zvl256b).

HARD REQUIREMENTS
- Exact signature, unchanged:
{spec.signature}
- BIT-EXACT with the reference below for all inputs. The requantise tail
  (Q0.31 rounding multiply, then shift, then offset, then clamp) must produce
  identical results. Do not reassociate the int32 accumulation.
- C99 + RVV intrinsics from <riscv_vector.h> only. No inline asm, no libc calls.
- Must compile with: -march=rv64gcv_zvl256b -mabi=lp64d
- Handle any M, K, N including K not a multiple of the vector length.

WHAT ACTUALLY MATTERS HERE
This model calls it with M=1 in every case, i.e. it is a GEMV, not a GEMM.
Measured on the board, the dominant call is M=1, K=256, N=128 and it is 62% of
total inference time. The other calls are M=1 with (K=16,N=256),
(K=128,N=64), (K=64,N=4). Optimise for M=1 first; keep the general path correct.

Useful facts: int8 x int8 -> int32 widening multiply-accumulate is available
(vwmacc family). weight is [N, K] row-major, so weight[n] is contiguous in k,
and input is contiguous in k too -- both operands stride-1 along the reduction.

REFERENCE IMPLEMENTATION (semantics are normative):
```c
{spec.reference_impl}
```
"""
c = make_llm_client()
print("provider:", type(c).__name__, "model:", c.model_id, flush=True)
print("prompt_sha256_16:", hashlib.sha256((system+user).encode()).hexdigest()[:16], flush=True)
r = c.converse(user=user, system=system, timeout=2400.0, phase="synth:linear_s8")
code = extract_code_block(r.text, lang="c")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codex_linear_s8.c")
open(out,"w").write(code)
print("wrote", out, len(code), "bytes", flush=True)
print("tokens in/out:", r.input_tokens, r.output_tokens, flush=True)
