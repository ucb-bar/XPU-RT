# XPU-RT Runtime Tools

This directory contains scripts and tools for the compile/profile/run flow.
The runtime dispatch runners are built inside merlin and used directly.

## Pre-built runners (recommended)

Merlin's build produces two dispatch runner binaries. These are the primary
way to execute dispatch graphs on target hardware:

| Binary | Purpose |
|---|---|
| `merlin-baseline-async` | Baseline topo-order runner (sequential or parallel) |
| `merlin-dispatch-scheduler` | Two-cluster CPU_P+CPU_E scheduled runner |

After building merlin (`setup.sh` or `merlin build --profile spacemit`), the
binaries are in:

```
third_party/merlin/build/<profile>/runtime/plugins/merlin-samples/
```

### Example: run a dispatch graph on SpacemiT

```bash
# From the XPU-RT root, after setup.sh:
RUNNER=./third_party/merlin/build/spacemit-merlin-perf/runtime/plugins/merlin-samples/merlin-baseline-async

$RUNNER gen/vmfb/mlp/spacemit_x60/RVV/mlp.q.int8/mlp.q.int8_dispatch_graph.json \
  local-task 10 1 1
```

## Custom tools (advanced)

If you need a custom runner, you can link against the standalone archive
(`libxpurt_standalone.a`) which bundles the xpu-rt runtime objects and the
full IREE runtime in a single `.a`:

```bash
./runtime/build_runtime.sh --target spacemit \
  --xpurt-lib ./third_party/merlin/build/spacemit-merlin-perf/runtime/src/iree/runtime/libxpurt_standalone.a
```

The header for the C API is at `third_party/merlin/samples/common/xpu-rt/baseline_runner.h`.

### Zephyr integration

For Zephyr, cross-compile merlin with a Zephyr-compatible toolchain, then
link `libxpurt_standalone.a` into your Zephyr app using
`zephyr_library_import_from_static()` or `target_link_libraries()`.
The standalone archive is self-contained — no IREE build tree needed at
link time.

### Notes

- FlatBuffers verification in IREE VM needs FlatCC verifier helpers (symbols
  like `flatcc_verify_field`). In this build those live in
  `libflatcc_parsing.a`. `build_runtime.sh` will auto-detect it from the same
  Merlin build tree, or you can override via
  `XPURT_FLATCC_PARSING_LIB=/path/to/libflatcc_parsing.a`.
