#!/usr/bin/env python3
"""Emit an ime_x60 profile CSV for a conv net from its rvv_x60 profile + a
MEASURED IME-vs-RVV comparison, so the scheduler can place conv dispatches on
the K1 matrix engine — and only where IME is measured faster.

Two sources for the comparison, both MEASURED, neither modelled:

  --from-runs <rvv_stdout>:<ime_stdout>   (preferred where it exists)
      A matched PAIR of whole-net harness runs at the SAME hart count and from
      the SAME build flavour — e.g. the deployed RVV build and the all-IME build
      both on four harts of the sharded model. The per-dispatch ratio is taken
      from that pair, so it is a real per-dispatch number at the width the cell
      is for, not a per-shape single-hart bench extrapolated to four harts.
      The IME kernel is per-hart on cluster 0, so its win CHANGES with width
      (49/57 dispatches at one hart, 46/57 at four): a width-blind speedup would
      place half a dozen dispatches on the wrong engine.

  (default) the per-shape table for the op, via ModelBlaster's ime_cost —
      `conv2d_s8` against the standalone RVV conv, `conv2d_batchnorm2d_silu_s8`
      against the FUSED RVV kernel the deployed build actually runs. Reading one
      op's table for the other compares against the wrong baseline; ime_cost
      owns that mapping, so this tool keeps no copy of the table of its own.

For each dispatch in the rvv profile:
  * conv2d* with a MEASURED speedup > 1  -> cycles = round(rvv_cycles/speedup),
    implementation = curated[ime]/ime_vmadot_4x4x8, module_name rvv_x60->ime_x60.
  * everything else (conv losers, non-conv ops) -> copied verbatim from the rvv
    profile (same cost), so IME is never cheaper there and the solver keeps RVV.

The IME cost is the measured SPEEDUP applied to THIS profile's rvv baseline
(not the standalone bench's absolute cycles), so rvv and ime cells are on the
same clock and the per-dispatch min the solver takes is apples-to-apples.

Usage:
  python scripts/make_ime_profile.py --net dronet
  python scripts/make_ime_profile.py --net yolov8_nano
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "ModelBlaster"))
from pipeline import ime_cost  # noqa: E402  (the only-if-better rule, one copy)

MEASURED = os.path.join(REPO, "xpu-rt", "data", "ime_measured_conv.csv")


def _speedup(op, shape):
    """Measured IME speedup for (op, shape), or None when not KNOWN faster.

    Delegates to ime_cost so the op->table mapping lives in exactly one place:
    the fused conv is costed against the fused RVV kernel, the standalone conv
    against the standalone one.
    """
    sp, _prov = ime_cost.ime_speedup_for(op, shape)
    return sp


PROFILE_RE = re.compile(r"=== MODELBLASTER_PROFILE_BEGIN ===\n(.*?)(?:\n=== |\Z)", re.S)


def _run_cycles(path):
    """{dispatch_id: (name, op, cycles)} from a harness stdout's final profile block."""
    m = PROFILE_RE.search(open(path).read())
    if not m:
        raise SystemExit(f"{path}: no MODELBLASTER_PROFILE_BEGIN block")
    out = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.startswith("dispatch_id") or line.startswith("==="):
            continue
        f = line.split(",")
        try:
            out[int(f[0])] = (f[1], f[2], int(f[-1]))
        except (ValueError, IndexError):
            continue
    if not out:
        raise SystemExit(f"{path}: profile block has no rows")
    return out


def _ratios_from_runs(spec, winners_only: bool = True):
    """<rvv_stdout>:<ime_stdout> -> {dispatch_id: rvv_cycles/ime_cycles}.

    Both runs must be the same network at the same hart count; the ratio is
    per DISPATCH, so a shape that wins at one width and loses at another is
    recorded correctly for each.

    `winners_only` (the default) keeps only the dispatches the IME run ran
    faster, which is the only-if-better rule the table path enforces. A build
    whose op set came from its own picks has already chosen, and runs those
    dispatches on the engine whether they win or lose, so it asks for all of
    them: a ratio below 1 is then a cell that costs MORE than RVV, which is the
    honest price of that deployment.
    """
    rvv_p, ime_p = spec.split(":", 1)
    r, i = _run_cycles(rvv_p), _run_cycles(ime_p)
    common = sorted(set(r) & set(i))
    if not common:
        raise SystemExit(f"--from-runs {spec}: the two runs share no dispatch ids")
    out = {}
    for d in common:
        if r[d][1] != i[d][1]:
            raise SystemExit(f"--from-runs {spec}: dispatch {d} is {r[d][1]} in one run "
                             f"and {i[d][1]} in the other -- not the same graph")
        if i[d][2] > 0 and (r[d][2] > i[d][2] or not winners_only):
            out[d] = r[d][2] / i[d][2]
    return out, os.path.basename(rvv_p), os.path.basename(ime_p), len(common)


def _ime_ops_from_picks(path: str) -> set:
    """The ops a BUILD put on the matrix engine, from its own `kernel_picks.json`.

    The default gate is `ime_cost.ime_useful`: the measured table's answer to
    "does the engine deserve this op". That is the right question for a
    table-guided build and the wrong one for a forced (`MB_IME_FORCE=1`) build,
    which takes the IME kernel for every op that HAS one -- including ops the
    table has never measured and ops it measures as losses. Costing a forced
    build from the table's answer would leave its extra ops priced as RVV, and
    a schedule solved against that is a schedule for a build that was not made.

    `ime_cost` deliberately does not read MB_IME_FORCE outside the generator, so
    this asks the artifact instead of the environment: the picks file the build
    wrote next to its kernels.
    """
    d = json.load(open(path))
    ops = set(d.get("ime_forced_ops") or ())
    if not ops:
        for op, pick in (d.get("picks") or {}).items():
            k = pick if isinstance(pick, str) else json.dumps(pick)
            if "ime" in k.lower():
                ops.add(op)
    if not ops:
        raise SystemExit(
            f"{path}: names no op on the matrix engine (no `ime_forced_ops`, no "
            f"curated[ime] pick). A table-guided build needs no --ime-ops-from-picks.")
    return ops


def _parse_shape(shape: str) -> dict:
    return {k: int(v) for k, v in re.findall(r"([A-Za-z]+)=(\d+)", shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--variant", default="int8",
                    help="quant/variant tag, e.g. int8 | reffused.int8 | unfused.int8")
    ap.add_argument("--target", default="spacemit_x60")
    ap.add_argument("--topo", default="topo_0")
    ap.add_argument("--gen-root", default="gen/profile_mb",
                    help="profile tree root, as the workload spec names it "
                         "(gen/profile_mb, gen/mb_shard/profile, ...)")
    ap.add_argument("--from-runs", default=None, metavar="RVV_STDOUT:IME_STDOUT",
                    help="take the per-dispatch ratio from a matched pair of whole-net "
                         "harness runs at THIS topo's hart count instead of the per-shape table")
    ap.add_argument("--ime-ops-from-picks", default=None, metavar="KERNEL_PICKS_JSON",
                    help="ask THAT BUILD which ops it put on the engine, instead of asking the "
                         "measured table which ops deserve to be there. The two answers differ "
                         "for a build made with MB_IME_FORCE=1, which takes the IME kernel for "
                         "every op that has one; costing such a build from the table's answer "
                         "would leave its extra ops priced as RVV. Reads `ime_forced_ops`, else "
                         "the ops of `picks` whose kernel is a curated[ime] one.")
    args = ap.parse_args()

    v = args.variant
    rvv = (f"{args.gen_root}/rvv_x60/{args.target}/{args.net}/{args.net}.{v}/"
           f"{args.net}_{args.target}_rvv_x60_{args.net}.{v}/{args.topo}/results.csv")
    rvv_path = os.path.join(REPO, rvv)
    if not os.path.exists(rvv_path):
        raise SystemExit(f"rvv profile not found: {rvv_path}")
    out = rvv.replace("rvv_x60", "ime_x60")
    out_path = os.path.join(REPO, out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ratios, src_rvv, src_ime, n_common = (None, None, None, 0)
    if args.from_runs:
        ratios, src_rvv, src_ime, n_common = _ratios_from_runs(
            args.from_runs, winners_only=not args.ime_ops_from_picks)

    rows = list(csv.DictReader(open(rvv_path)))

    # WHICH OPS THE ime_x60 BUILD WILL ACTUALLY SWAP. A faster row in the IME
    # run is only an IME result for an op the picker gives an IME kernel to;
    # for any other op both runs executed the SAME rvv kernel and the ratio is
    # run-to-run noise (yolov8_nano_64x96's two `conv2d_s8` detect heads measured
    # "1.03x" that way). Offering those as ime cells makes the solver choose
    # between two measurements of one kernel. `ime_useful` is the picker's own
    # guard, so asking it here is asking the build what it will do.
    by_op = {}
    for r in rows:
        if (r.get("op") or "").startswith("conv2d"):
            by_op.setdefault(r["op"], []).append(_parse_shape(r.get("shape", "")))
    if args.ime_ops_from_picks:
        ime_ops = _ime_ops_from_picks(args.ime_ops_from_picks) & set(by_op)
        print(f"  ops the ime_x60 build swapped, per {args.ime_ops_from_picks}: "
              f"{sorted(ime_ops) or '(none)'}   kept on rvv: "
              f"{sorted(set(by_op) - ime_ops) or '(none)'}")
    else:
        ime_ops = {op for op, shapes in by_op.items() if ime_cost.ime_useful(op, shapes)[0]}
        print(f"  ops the ime_x60 build will swap: {sorted(ime_ops) or '(none)'}"
              f"   kept on rvv: {sorted(set(by_op) - ime_ops) or '(none)'}")
    fieldnames = rows[0].keys()
    n_ime = 0
    n_loss = 0
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            op = r.get("op", "")
            if op in ime_ops:
                if ratios is not None:
                    try:
                        sp = ratios.get(int(r["dispatch_id"]))
                    except (KeyError, ValueError):
                        sp = None
                else:
                    sp = _speedup(op, _parse_shape(r.get("shape", "")))
                # ONLY-IF-BETTER, unless the build has already chosen. When the op set
                # comes from a BUILD's picks, that build runs the IME kernel for these ops
                # whatever the ratio says, so its measured cost is the cost -- including
                # where the engine LOSES. Filtering losers out there would file a cell the
                # deployment cannot take. The table path keeps the only-if-better rule.
                if sp is not None and (sp > 1.0 or args.ime_ops_from_picks):
                    n_loss += (sp <= 1.0)
                    r = dict(r)
                    # scale EVERY cost column by the measured speedup — the loader
                    # reads `mean_time` (load_profiled_times L148), not `cycles`,
                    # so scaling only cycles left the IME cell at the RVV cost.
                    r["cycles"] = str(int(round(float(r["cycles"]) / sp)))
                    for col in ("mean_time", "mean_time_ns"):
                        if r.get(col):
                            r[col] = f"{float(r[col]) / sp:.6f}"
                    r["implementation"] = "curated[ime]/ime_vmadot_4x4x8"
                    r["module_name"] = r["module_name"].replace("rvv_x60", "ime_x60")
                    # PROVENANCE. This cell is a measured RVV cost divided by a measured
                    # speedup -- a prediction, filed in the tree the loader reads as board
                    # measurement. It must not keep the rvv row's `source=k1`, or a
                    # derived cost is indistinguishable from a per-dispatch measurement
                    # (and `--require-source k1` would wave it through).
                    if "source" in r:
                        r["source"] = f"ime_derived({r.get('source') or 'unknown'}/x{sp:g})"
                    n_ime += 1
            w.writerow(r)
    prov = {
        "schema": "ime_derived_profile/v1",
        "kind": "DERIVED, not measured per dispatch",
        "out": out,
        "derived_from": rvv,
        "derived_from_sha256": hashlib.sha256(open(rvv_path, "rb").read()).hexdigest(),
        "speedup_source": (f"matched board runs {src_rvv} (rvv) vs {src_ime} (ime), "
                           f"{n_common} shared dispatches, per-dispatch ratio"
                           if ratios is not None else
                           "ModelBlaster pipeline/ime_cost per-op measured tables"),
        "ime_ops_source": (args.ime_ops_from_picks or
                           "ModelBlaster pipeline/ime_cost.ime_useful (table-guided)"),
        "ime_ops": sorted(ime_ops),
        "speedup_table": (os.path.relpath(MEASURED, REPO) if ratios is None else None),
        "n_rows": len(rows),
        "n_rows_derived": n_ime,
        "only_if_better": not bool(args.ime_ops_from_picks),
        "n_rows_derived_slower_than_rvv": n_loss,
        "note": ("conv winners carry rvv_cost/measured_speedup with source=ime_derived(...); "
                 "every other row is the rvv measurement copied verbatim so the solver keeps RVV"),
    }
    json.dump(prov, open(os.path.join(os.path.dirname(out_path), "PROVENANCE.json"), "w"),
              indent=1)
    print(f"{args.net}: wrote {out_path}")
    if n_ime == 0:
        print("  WARNING: no dispatch was given an IME cell -- the solver will see no IME "
              "option for this net at this width. Check the table/run pair.")
    print(f"  {n_ime}/{len(rows)} dispatches given a MEASURED IME cell "
          + (f"({n_ime - n_loss} faster than RVV, {n_loss} slower -- this build runs them "
             f"on the engine either way)" if args.ime_ops_from_picks else "(conv winners)")
          + "; the rest carry the rvv cost so the solver keeps RVV.")
    print(f"  provenance: {n_ime} DERIVED rows tagged source=ime_derived(...); "
          f"sidecar PROVENANCE.json written beside the CSV.")


if __name__ == "__main__":
    main()
