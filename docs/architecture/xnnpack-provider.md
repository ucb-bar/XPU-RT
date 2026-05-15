# XNNPACK Kernel Provider

XPU-RT ships an XNNPACK kernel provider so the auction can route CPU
operator kernels through Google/Meta's tuned NHWC microkernel library.
Kernels execute through libxpu_rt's CPU task driver via a stable C ABI
shim (the **xnnpack_bridge**) — no special-cased Python runtime, no
Python-side dlopen tricks. From the executor's point of view, an
XNNPACK kernel is the same kind of artifact as a Triton- or
cffi-C-generated one: a `.so` exposing `xpu_rt_kernel_init` /
`xpu_rt_kernel_run`.

## Source

The provider vendors the UCB-BAR fork of XNNPACK as a git submodule at
`third_party/XNNPACK/`, pinned to the `torch-1.0.0-bump` branch. That
branch carries the active UCB-BAR delta (~100 LOC of build/config glue
on top of upstream `google/XNNPACK`):

- ExecuTorch-1.0.0 baseline.
- Zephyr / bare-metal build options (force-static lib, optional
  pthreadpool stubbing, `XNN_HAS_MMAP=0` for embedded targets).
- RISC-V vector-length detection via `vsetvli` instead of
  `csrr vlenb`, so RVV is usable without `/proc/cpuinfo` / `getauxval` /
  `hwprobe` — important for the Chipyard / Saturn-vector targets.

Two small XPU-RT-local patches sit on top of the fork tip:

1. **`XNN_HAS_MMAP` is gated on `XNN_PLATFORM_LINUX`** instead of being
   hard-coded to `0`. The fork's `0` was correct for bare-metal but
   broke Linux host builds: `resize_buffer()` calls `mremap()` under
   `#if XNN_PLATFORM_LINUX` regardless of `XNN_HAS_MMAP`. The gate
   restores Linux's path while keeping the bare-metal stub.
2. **`XNN_ENABLE_CPUINFO` is a `CACHE STRING`** so an outer build can
   flip it to `1`. The fork hard-coded `0`, which suppresses
   `<cpuinfo.h>` but leaves `hardware-config.c`'s unconditional
   `cpuinfo_has_x86_*()` calls turning into implicit-decl external
   refs. The cache variable lets us turn it on for Linux host builds
   from `runtime/native/libxpu_rt/CMakeLists.txt` without forking the
   file.

Both patches are local-only, attributed to the XPU-RT integration. The
fork's branch tip is otherwise unchanged.

## Build

```bash
cmake -S runtime/native/libxpu_rt -B build/rt-cpu-xnn \
      -DCG_RT_WITH_CUDA=OFF \
      -DXPURT_WITH_XNNPACK=ON
cmake --build build/rt-cpu-xnn --parallel
```

When `XPURT_WITH_XNNPACK=ON`, the libxpu_rt CMakeLists:

- adds `third_party/XNNPACK` as a sub-project (force-static, no
  benchmarks, no tests, no kleidiai),
- fetches `pytorch/cpuinfo` via `FetchContent` (static, no tools),
- links `XNNPACK` + `cpuinfo` into both `libxpu_rt.so` and
  `libxpu_rt_static.a` inside a `--start-group/--end-group` pair so the
  single-pass linker resolves the cross-archive refs,
- defines `XPURT_HAS_XNNPACK=1` so `xnnpack_bridge.c` compiles the live
  path (it stubs to `-XPU_RT_XNN_ENOTSUP` otherwise).

When the flag is **off** (the default), the bridge translation unit
still compiles but every entry returns `-XPU_RT_XNN_ENOTSUP`. The
Python provider's `probe()` detects that and emits a typed `blocked`
result with `blocked_reason=build_flag_missing`.

## Bridge ABI

The bridge header at
`runtime/native/libxpu_rt/src/drivers/xnnpack/xnnpack_bridge.h`
exposes a small, **wire-stable** C ABI:

```c
typedef enum {
    XPU_RT_XNN_OP_UNKNOWN                              = 0,
    XPU_RT_XNN_OP_FULLY_CONNECTED_F32                  = 1,
    XPU_RT_XNN_OP_CONVOLUTION2D_NHWC_F32               = 2,
    XPU_RT_XNN_OP_DEPTHWISE_CONVOLUTION2D_NHWC_F32     = 3,
    /* ... ~30 op kinds, append-only ... */
} xpu_rt_xnn_op_kind;

int             xpu_rt_xnn_global_initialize(void);
xpu_rt_xnn_op*  xpu_rt_xnn_create(xpu_rt_xnn_op_kind, ...);
int             xpu_rt_xnn_reshape_setup(xpu_rt_xnn_op*, ...);
int             xpu_rt_xnn_run(xpu_rt_xnn_op*);
void            xpu_rt_xnn_destroy(xpu_rt_xnn_op*);
```

Kernel artifacts emitted by `XnnpackProvider.propose()` include this
header (not `<xnnpack.h>` directly) so they stay decoupled from
XNNPACK version drift.

## Auction position

`xpu-rt/python/xpu_rt/providers/cards/xnnpack.yaml` declares
`integration_level: promote`, the same rank as the deterministic
`cffi_c` correctness anchor. The
`xpu_rt.providers.provider_routing.KIND_PREFERENCE` table then puts
`xnnpack` first on `host_cpu` for:

| Family | Order |
|---|---|
| `matmul` | `xnnpack` → cutlass_cute → tilelang → autocomp → triton → cffi_c |
| `conv2d` | `xnnpack` → cffi_c → triton |
| `depthwise_conv2d` | `xnnpack` → cffi_c |
| `softmax`, `avg_pool`, `max_pool`, `global_avg_pool` | `xnnpack` first |
| `unary`, `binary`, `prelu`, `leaky_relu` | `xnnpack` → triton → cffi_c |
| `transpose`, `resize_bilinear`, `slice` | `xnnpack` first |

XNNPACK does **not** win `attention` or `fused_region` — those still
go through `triton` / `thunderkittens` / `autocomp` / `kernelblaster`.

## Layout constraint

XNNPACK is NHWC-only for spatial ops. The provider's `can_bid()`
declines non-NHWC contracts with `blocked_reason=requires_nhwc_layout`
and a low (but non-zero) confidence:

```python
bid = provider.can_bid(Contract(op_kind="conv2d", layout="NCHW"), Target())
# bid.kind_match == "declined_layout"
# bid.blocked_reason == "requires_nhwc_layout"
# bid.confidence ≈ 0.10
```

The layout-normalisation pass uses that reason to insert an
NCHW→NHWC transpose upstream and re-run the auction; on the second
pass XNNPACK matches and wins.

## Op coverage (v1)

The catalogue in `xpu-rt/python/xpu_rt/kernels/xnnpack_adapter.py`
declares which `(op_kind, dtype, layout)` triples the provider claims:

| Op family | f32 | f16 | qs8 | qu8 | qd8→f32 |
|---|:---:|:---:|:---:|:---:|:---:|
| fully_connected / matmul / linear | ✓ | ✓ | ✓ | — | ✓ |
| batch_matmul | ✓ | ✓ | — | — | ✓ |
| conv2d (NHWC) | ✓ | ✓ | ✓ | ✓ | ✓ |
| depthwise_conv2d (NHWC) | ✓ | — | ✓ | — | — |
| deconv2d (NHWC) | ✓ | — | — | — | — |
| pool (avg/max/global-avg) | ✓ | — | — | — | — |
| softmax (NC) | ✓ | — | — | — | — |
| unary (relu/sigmoid/tanh/hswish/gelu/...) | ✓ | — | — | — | — |
| binary (add/sub/mul/div/max/min) | ✓ | — | — | — | — |
| reduce (sum/mean/max/min) | ✓ | — | — | — | — |
| prelu / leaky_relu (NHWC) | ✓ | — | — | — | — |
| transpose / permute | ✓ | — | — | — | — |
| resize_bilinear (NHWC) | ✓ | — | — | — | — |
| slice / static_slice | ✓ | — | — | — | — |

Only the f32 fully-connected path is wired in the bridge today (the C
side has one `switch` case). The remaining slots route correctly but
return `-XPU_RT_XNN_ENOTSUP` until the corresponding case lands; each
addition is one bridge case + one provider catalogue line.

## Probe / fallback semantics

| libxpu_rt build | mosek.lic | Result of `XnnpackProvider().probe()` |
|---|---|---|
| `-DXPURT_WITH_XNNPACK=ON` | irrelevant | `available`, `version="xnnpack-runtime"` |
| `-DXPURT_WITH_XNNPACK=OFF` (default) | irrelevant | `blocked`, `blocked_reason=build_flag_missing` |
| libxpu_rt not found at all | irrelevant | `not_installed`, candidate paths in `detail` |

Set `XPURT_RUNTIME_DIR` to point the loader at a custom build dir
when developing.

## Tests

- `xpu-rt/tests/kernels/providers/test_xnnpack_provider.py` — adapter
  catalogue, probe, `can_bid`, `propose` (17 tests).
- `xpu-rt/tests/kernels/test_xnnpack_routing.py` — `KIND_PREFERENCE`
  contract + auction ordering (11 tests).
- `runtime/native/libxpu_rt/tests/test_*.c` — the existing 27 native
  tests must keep passing with `-DXPURT_WITH_XNNPACK=ON`; they
  validate libxpu_rt's CPU/event-tensor/command-buffer/semaphore
  semantics are unaffected by the XNNPACK link.

Run all of them with:

```bash
uv run pytest xpu-rt/tests/kernels/providers/test_xnnpack_provider.py \
              xpu-rt/tests/kernels/test_xnnpack_routing.py
cmake -S runtime/native/libxpu_rt -B build/rt-cpu-xnn -DXPURT_WITH_XNNPACK=ON
cmake --build build/rt-cpu-xnn --parallel
cd build/rt-cpu-xnn && \
    for t in test_*; do [ -x "./$t" ] && ./$t; done
```
