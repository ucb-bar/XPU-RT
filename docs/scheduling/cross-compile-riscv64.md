# Cross-compiling XPU-RT bundles for riscv64 + Spike

## What this proves

XPU-RT's compile pipeline emits per-region kernel C source via
provider artifacts (e.g. the XNNPACK provider's
`generated_kernels/xnnpack/<region>.c`). Historically those sources
were `cffi`-compiled on the same host that runs the pipeline. That
flow leaves an open question: is the integration real for a non-x86
target?

This document covers the **`riscv64_spike_rvv`** first-class target
and the **`cross_compile_riscv64`** bundle stage that closes the
question. The stage takes a finished bundle, cross-compiles every
generated kernel + a generated driver against the pre-built
`libxpu_rt_static.a`, and emits one `program.elf` that runs directly
on Chipyard's spike via HTIF (no proxy kernel, no Linux, no
`futex(2)`).

## Toolchain

| Component | Path | Role |
|---|---|---|
| **clang 18** | `/scratch2/agustin/merlin/build_tools/riscv-tools-iree/toolchain/clang/linux/RISCV/bin/clang` | Compile (correct RVV v1.0 intrinsics; matches the UCB-BAR XNNPACK fork's spelling). |
| **riscv64-unknown-elf-gcc 13.2** | `/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/` | Linker driver. Picked because `htif_nano.specs` cascades through newlib here. |
| **`htif_nano.specs`** | `<chipyard>/.conda-env/riscv-tools/riscv64-unknown-elf/lib/htif_nano.specs` | Newlib HTIF specs — gives us `printf` → `tohost` SYS_write and `_exit` → `tohost` exit-syscall. |
| **`htif.ld`** | `runtime/native/libxpu_rt/toolchains/../../../../scratch2/agustin/merlin/build_tools/firesim/htif.ld` (Merlin's) | Linker script. Reserves a `.htif` section + 1 GiB heap + TLS .tdata/.tbss layout. |
| **Spike** | `<chipyard>/toolchains/riscv-tools/riscv-isa-sim/build/spike` | Functional RISC-V simulator with VLEN=128 hardcoded. |

The CMake toolchain file at
`runtime/native/libxpu_rt/toolchains/riscv64-spike-rvv.cmake` wires
these together; `XPURT_TOOLCHAIN_ROOT` / `CHIPYARD_ROOT` /
`MERLIN_IREE_TOOLCHAIN_ROOT` env vars let you point at alternate
installations.

### Why this combo (after three pivots)

1. **Chipyard's `riscv64-unknown-elf-gcc 13.2` alone** failed: missing
   `std::round` / `std::expm1` / `std::erf` / `std::cbrt` /
   `std::copysign` in libstdc++ + ~250 XNNPACK RVV microkernel files
   miscompile due to GCC 13.2 RVV intrinsic drift (multi-vector tuple
   types, `__RISCV_FRM_*`).
2. **`riscv64-unknown-linux-gnu-gcc 13.2 -static` + chipyard pk** fixed
   the C++ stdlib but pk doesn't implement `futex(2)`, so glibc NPTL
   TLS init traps unconditionally.
3. **Zephyr SDK 1.0.0-beta1 (GCC 14.3 picolibc)** cleared both issues
   but was too slow on functional spike — XNN init didn't complete
   within 5 min wallclock.
4. **Clang 18 (Merlin IREE bundle) + chipyard's gcc-link via newlib
   htif_nano.specs**: works end-to-end. Same pattern Merlin uses for
   IREE on Chipyard. ~30 s wallclock for XNN init + one FC.

## Bare-metal compatibility files

Newlib (chipyard's `riscv64-unknown-elf` toolchain) ships pthread
declarations but not all implementations, and its libstdc++ doesn't
hoist C math functions into `std::`. `runtime/native/libxpu_rt/src/compat/picolibc-freestanding/`
provides:

| File | Role |
|---|---|
| `cmath` | `#include_next <cmath>` then `using ::round; using ::expm1; …` into namespace std::. Mandatory for XNNPACK's reference C++ kernels. |
| `time.h` | `#include_next <time.h>` + stub `clock_gettime` returning zero ts. Used by XNNPACK's optional profiling path. |
| `pthread_compat.c` | Single-threaded no-op `pthread_once` / `pthread_mutex_*` + a `posix_memalign` shim. **Critical**: newlib's `pthread_once_t = { int is_initialized; int init_executed; }` with `PTHREAD_ONCE_INIT = {1, 0}` — check field `[1]`, not `[0]`. |

## The `cross_compile_riscv64` stage

Implemented in `xpu_rt/runtime/cross_compile/riscv64_bare.py`.

```python
from xpu_rt.runtime.cross_compile import cross_compile_riscv64_bundle
from xpu_rt.graph_compilation.region_dossier import load_target_profile

profile = load_target_profile("xpu-rt/configs/targets/riscv64_spike_rvv.yaml")
result = cross_compile_riscv64_bundle(
    bundle_dir,
    target_id="riscv64_spike_rvv",
    cross=profile.cross_compile,        # CrossCompileConfig
    repo_root=Path("."),
    model_id="my_model",
)
# result.status == "ok"
# result.elf_path  → <bundle>/program.elf
```

The orchestrator:

1. Locates the pre-built `libxpu_rt_static.a` under
   `build/riscv-spike/`. Errors with typed
   `CrossCompileError(reason="libxpu_rt_missing", ...)` if missing.
2. Reads every `generated_kernels/xnnpack/*.c` plus the shared
   `kernel_metadata.json` for per-region shape info.
3. Renames each kernel's exported symbols (`xpu_rt_kernel_init` →
   `region{N}_init`, etc.) so multi-region bundles don't get linker
   collisions.
4. Renders a driver from `driver_template.c.tmpl`: stages
   `golden_inputs.pt` into region 0's input buffer, wires region N's
   output → region N+1's input, calls each region in execution-plan
   order, emits the final output's fp32 hex-bits + a 64-bit FNV-1a
   checksum to HTIF stdout.
5. Generates a top-level `CMakeLists.txt` that links everything via
   `--start-group / --end-group` so cross-archive references resolve
   in a single linker pass.
6. Runs `cmake -B build` + `cmake --build build` against the riscv
   toolchain file.
7. Copies the resulting `program.elf` to `<bundle>/program.elf`.

Every failure path raises a typed `CrossCompileError` with a `reason`
attribute (`cross_gcc_missing` / `libxpu_rt_missing` /
`no_kernels_to_compile` / `cmake_configure_failed` / `link_failed` /
`bridge_header_missing`). A `CrossCompileResult` dataclass carries
status + paths on success.

## Running the produced ELF

```bash
spike --isa=rv64gcv <bundle>/program.elf
```

Expected stdout shape:

```
xpu_rt-bundle <model> @ riscv64_spike_rvv: begin
output_count: <N>
output: <hex32_0> <hex32_1> ... <hex32_{N-1}>
checksum: 0x<16-hex-digits>
PASS
```

Each output token is a 32-bit fp32 bit pattern (no `0x` prefix,
lowercase). Decode with:

```python
import struct
floats = [
    struct.unpack("<f", struct.pack("<I", int(tok, 16)))[0]
    for tok in line.split()[1:]
]
```

## Agent-in-the-loop

`.claude/skills/xnnpack-on-spike.md` drives the whole flow via
Claude Code's MCP loop. Invoke:

```
/xnnpack-on-spike <model>
```

The skill calls `mcp__compgen__compgen_compile_torch_model` with
`target=riscv64_spike_rvv` and `selection_mode=agent-file`, then loops
on `compgen_inspect_pipeline_run` to read each pending
`agent_decision_request.json`, applies the policy
table in the skill body, submits via
`compgen_commit_agent_decision_response`, and surfaces a 4-line PASS
summary when the pipeline finishes.

Decisions covered: provider auction (xnnpack first on host_cpu),
tile params (smallest legal RVV-128 tile), layout (NHWC; accept
NCHW→NHWC transposes), decomp vs native (native for XNNPACK-
supported), fusion (pointwise pairs only).

## Verification + tests

| Test | What it asserts |
|---|---|
| `runtime/native/libxpu_rt/tests/test_xnnpack_bridge_riscv.c` | Bridge FC f32 directly on spike: `[3.01, 1.32, -1.47]`, PASS. |
| `xpu-rt/tests/cross_compile/test_riscv64_e2e.py` | End-to-end: synthesized bundle (provider.propose+extras) → cross-compile → spike run → output decode → analytical reference within 1e-5. |
| `xpu-rt/tests/cross_compile/test_riscv64_parity.py` | Real `torch.nn.Linear(8,5)` → eager reference → bundle → cross-compile → spike → `max_abs<1e-4`, `max_rel<1e-5` vs eager. |

Run the suite:

```bash
uv run pytest xpu-rt/tests/cross_compile/ -v
```

The tests skip cleanly when the cross-toolchain, spike, or pre-built
`libxpu_rt_static.a` aren't available.

## Honest non-claims (v1)

- **Single-FC v1**. The driver template wires regions sequentially; a
  full multi-op network needs (a) the pipeline to invoke the provider
  per region with real `request.extras["static_weights_f32"]` for each
  FC / conv / etc., and (b) the bridge cases for binary, unary,
  softmax, pool to be exercised. The bridge already has those cases
  wired (Phase B); only the pipeline-side weight extraction is the gap.
- **Single-threaded, no pk**. We use direct HTIF, no Linux, no
  multi-hart. Real silicon or qemu-system-virt is a different runner
  pattern (the target YAML's `runner:` block has the override knob).
- **VLEN=128 only**. Spike hardcodes it; a real target with VLEN≠128
  needs the XNNPACK build to be re-run with the matching `-march=`
  flag.
- **Functional spike is slow**. Expect ~30 s wallclock per FC call.
  qemu-system-riscv64 8.x (if procured) is ~100× faster and runs the
  same ELF unchanged.

## Rollback

Every Phase A/B/C/D/E file is either new or gated behind
`CMAKE_CROSSCOMPILING + CMAKE_SYSTEM_PROCESSOR=riscv64`. Reverting is
`git revert` per phase commit. Host x86_64 builds and all prior
milestones are unaffected.
