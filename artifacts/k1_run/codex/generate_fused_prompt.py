import hashlib, io, os, sys
from modelblaster.pipeline.llm_client import make_llm_client
from modelblaster.pipeline.bedrock_client import extract_code_block

ref = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fused_ref.c"), encoding="utf-8").read()
system = ("You are writing a single C99 function for a RISC-V SpaceMiT X60 core. "
          "Output ONLY one ```c code block containing the complete function. No prose.")
user = f"""Optimise this FUSED kernel for RVV 1.0 on a SpaceMiT X60 (rv64gcv, VLEN=256, zvl256b).

CONTEXT
ModelBlaster fused a linear_s8 with the elu_s8 that consumed its output, so the
intermediate int8 tensor is never materialised. That fusion was recommended
because, once the linear kernel was vectorised, the elementwise elu ops were
39.7% of measured runtime on the board and existed only to rewrite a tensor
between two matmuls.

HARD REQUIREMENTS
- Exact signature, unchanged (note the name is model-mangled):
  void kernel_linear_s8_elu_s8_mlp_control(const int8_t *input, const int8_t *weight,
      const int32_t *bias, int8_t *output, int M, int K, int N,
      int input_offset, int filter_offset, int linear_output_offset,
      int output_multiplier, int output_shift,
      int linear_activation_min, int linear_activation_max,
      float scale_linear_out, float scale_final_out,
      int activation_min, int activation_max, float alpha)
- Numerically equivalent to the reference for all inputs. The int32 accumulation
  and the Q0.31 requantise tail must not be reassociated. The elu tail goes
  through float exactly as written, including the int8 round-trip through
  `linear_int8` -- that truncation is observable.
- C99 + RVV intrinsics from <riscv_vector.h>. expf/roundf from <math.h> are
  allowed in the elu tail if you cannot vectorise it exactly.
- Compiles with: -march=rv64gcv_zvl256b -mabi=lp64d
- Any M, K, N, including K not a multiple of the vector length.

WHAT MATTERS
Every call has M=1, i.e. GEMV. Measured shapes: (K=16,N=256), (K=256,N=128),
(K=128,N=64). Vectorise the K reduction with the widening int8 multiply-
accumulate family (vwmacc); weight[n] and input are both contiguous in k.

REFERENCE IMPLEMENTATION (normative):
```c
{ref}
```
"""
c = make_llm_client()
print("provider:", type(c).__name__, "model:", c.model_id, flush=True)
print("prompt_sha256_16:", hashlib.sha256((system+user).encode()).hexdigest()[:16], flush=True)
r = c.converse(user=user, system=system, timeout=2400.0, phase="synth:linear_s8_elu_s8")
code = extract_code_block(r.text, lang="c")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codex_fused.c")
open(out,"w").write(code)
print("wrote", out, len(code), "bytes; tokens", r.input_tokens, r.output_tokens, flush=True)
