# FreshScheduler runtime tools

This directory builds small runtime tools that use the Merlin `xpu-rt` wrapper.

## Build `json_dispatch_runner` out-of-tree (recommended)

Use the **standalone** archive produced by the Merlin/IREE build:
`libxpurt_iree_plugin_standalone.a` (this contains the xpu-rt plugin objects **and**
the IREE runtime objects in one `.a`).

### SpacemiT example

Assuming you already built Merlin with:

```bash
python3 tools/build.py --target spacemit --config perf --with-plugin
```

Build the runner in `FreshScheduler/runtime`:

```bash
./runtime/build_runtime.sh --target spacemit \
  --xpurt-lib ./merlin/build/spacemit-merlin-perf/runtime/src/iree/runtime/libxpurt_iree_plugin_standalone.a
```

The binary will be at `runtime/build-spacemit/json_dispatch_runner`.

### Notes

- If you pass `libxpurt_iree_plugin.a` (plugin-only), `build_runtime.sh` will try to
  locate/build the standalone archive in the same Merlin build tree and use it.
- FlatBuffers verification in IREE VM needs FlatCC verifier helpers (symbols like
  `flatcc_verify_field`). In this build those live in `libflatcc_parsing.a`.
  `build_runtime.sh` will auto-detect it from the same Merlin build tree, or you
  can override via `XPURT_FLATCC_PARSING_LIB=/path/to/libflatcc_parsing.a`.
- The runner’s public header comes from `merlin/xpu-rt/xpurt_scheduler_core.h`
  (it intentionally does **not** include IREE headers).

