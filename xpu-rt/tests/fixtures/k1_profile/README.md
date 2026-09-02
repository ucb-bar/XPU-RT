# `spacemit_x60` profile fixture

A committed, board-free copy of a **real** SpaceMiT K1 profile, so the K1 side of
the profile parser has test coverage without a checkout needing the untracked
`gen/profile_mb/` tree (which no `gen_root` can even address — its parent
directory is not named `profile`; see
`docs/K1/k1_modelblaster_xpurt_closed_loop.md` §"`PROFILE_OUT_ROOT` must end in
`profile`").

## Layout

```
gen/profile/<impl>/spacemit_x60/dronet/dronet.int8/<input_tag>/topo_0/{results.csv,profile.jsonl}
```

`<impl>` is `rvv_x60` and `scalar`. The `<input_tag>` level is present because
that is what ModelBlaster's profile writer actually emits;
`profile_loader.find_profile_csv` globs it, and `compile_advice.load_profiles`
now does too.

## Provenance

`results.csv` for both implementations is a **verbatim copy** of

```
gen/profile_mb/rvv_x60/spacemit_x60/dronet/dronet.int8/dronet_spacemit_x60_rvv_x60_dronet.int8/topo_0/results.csv
gen/profile_mb/scalar/spacemit_x60/dronet/dronet.int8/dronet_spacemit_x60_scalar_dronet.int8/topo_0/results.csv
```

measured on the K1 (`rdtime`, core 0, curated kernels, all dispatches
bit-exact). 21 dispatches each. Nothing was rescaled or rounded.

Two schema generations are deliberately kept, because the parser has to read
both:

* `rvv_x60/…/results.csv` has **14** columns — the 13 IREE/ModelBlaster columns
  plus **`implementation`**, which records the kernel that actually executed
  (`curated[rvv]/rvv_vsmul_vnclip`, `curated[rvv]/direct`, …). That column is the
  only thing in the tree that distinguishes a vector build from one silently
  running scalar code, so a fixture without it cannot catch that regression.
* `scalar/…/results.csv` has the older **13** columns and no `implementation`.
  Its `module_name`s carry the `scalar` backend tag instead.

## `profile.jsonl` is derived, not measured

The ModelBlaster writer emits only `results.csv`; `compile_advice` reads
`profile.jsonl` (the IREE-era profiler's format). The `.jsonl` here is generated
from the `results.csv` beside it and therefore carries **only fields that exist
in the CSV**: `dispatch_id`, `module_name`, `median_ms`, `cycles`, `op`, `shape`,
`n_cores`, `implementation`.

It deliberately omits `samples_ms`, `mean_ms`, `stdev_ms` and `cv_pct`: those
were never recorded for this run, and inventing a dispersion would let
`implementation_advice` report `confidence="high"` on made-up variance. With
`cv_pct` absent the advisor falls back to `"medium"`, which is the honest answer.
