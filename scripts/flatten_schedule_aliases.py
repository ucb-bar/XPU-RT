#!/usr/bin/env python3
"""Flatten sweep rate-group ALIASES in a scheduled JSON onto the numeric
instance form that modelblaster's `ingest_xpurt_schedule` understands.

This is step 2 of the FPGA sweep (see runs/sweeps/*/RUNBOOK.md) and it is
REQUIRED, not optional: gen_random_workload.py names extra copies of a model
with a letter suffix -- sporadic ones as `yolov8_nano_a`, periodic ones as
`dronet_a0`, `dronet_a1`, ... -- while the ingest only knows base model names
and rejects the rest with

    ValueError: schedule entry 'dronet_a0_dispatch_0' references unknown network

Each alias is just another run of the same model, so every (alias, instance)
pair is flattened onto one contiguous instance range per base model:

    dronet_a0, dronet_a1, dronet_b0  ->  dronet0, dronet1, dronet2

Dispatch keys, `job_name`, and -- generically -- every string field whose
value is a dispatch key (dependencies, time_dependency, and anything added
later) are rewritten, so no reference is left dangling. A dangling reference
after the rewrite is a hard error: building on one produces a binary that
does not match the schedule it claims to run.

Base model names come from the model bank, never a hardcoded list. That list
used to be a literal ("mlp_control", "dronet", "yolov8_nano"), which silently
left fused_full_a/_b and vint_a/_b unflattened; any model added to the bank
is picked up automatically now.

Usage:
  scripts/flatten_schedule_aliases.py IN.json OUT.json [--bank PATH]

(This was runs/sweeps/fpga_20260829-195805/drivers/alias_fix.py, moved here so
a reproduction does not depend on one past run's output directory. The same
transform is also inlined in scripts/repro_fpga_sweep.sh step 2; keep the two
in sync if you change either.)
"""
from __future__ import annotations
import argparse, collections, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BANK = os.path.join(ROOT, "data", "banks", "model_bank.json")
FALLBACK_BASES = ("mlp_control", "dronet", "yolov8_nano")


def bank_bases(bank_path: str) -> tuple[str, ...]:
    if not os.path.exists(bank_path):
        print(f"  WARNING: no model bank at {bank_path}; falling back to "
              f"{FALLBACK_BASES}", file=sys.stderr)
        return FALLBACK_BASES
    names: set[str] = set()
    for plat in (json.load(open(bank_path)).get("platforms") or {}).values():
        names.update((plat.get("models") or {}).keys())
    return tuple(sorted(names)) or FALLBACK_BASES


def flatten(src: str, dst: str, bank_path: str) -> int:
    bases = bank_bases(bank_path)
    d = json.load(open(src))

    jobs: list[str] = []
    for k in d["dispatches"]:
        j = k.split("_dispatch")[0]
        if j not in jobs:
            jobs.append(j)

    def parse(j):
        for b in sorted(bases, key=len, reverse=True):
            m = re.match(rf"^{b}(?:_([a-z]))?(\d*)$", j)
            if m:
                return b, (m.group(1) or ""), (int(m.group(2)) if m.group(2) else 0)
        return None, None, None

    groups = collections.defaultdict(list)
    for j in jobs:
        b, a, i = parse(j)
        if b:
            groups[b].append((a, i, j))

    jm: dict[str, str] = {}
    for b, lst in groups.items():
        # sort by (alias, index) so the flattened range is contiguous and
        # deterministic regardless of dict order
        for n, (a, i, j) in enumerate(sorted(lst)):
            jm[j] = f"{b}{n}"
    for j in jobs:
        jm.setdefault(j, j)

    km = {k: jm[k.split("_dispatch")[0]] + k[len(k.split("_dispatch")[0]):]
          for k in d["dispatches"]}

    def remap(x):
        if isinstance(x, str):
            return km.get(x, jm.get(x, x))
        if isinstance(x, list):
            return [remap(i) for i in x]
        if isinstance(x, dict):
            return {k: remap(v) for k, v in x.items()}
        return x

    out = {}
    for k, v in d["dispatches"].items():
        v = remap(dict(v))
        v["job_name"] = jm[k.split("_dispatch")[0]]
        out[km[k]] = v
    d["dispatches"] = out
    json.dump(d, open(dst, "w"), indent=1)

    keys = set(out)
    bad = [x for v in out.values() for f in ("dependencies", "time_dependency")
           for x in (v.get(f) if isinstance(v.get(f), list) else [v.get(f)])
           if isinstance(x, str) and x not in keys]
    print(f"  {len(jobs)} jobs -> {len(set(jm.values()))} instances; "
          f"renamed {sum(1 for k, v in jm.items() if k != v)}; dangling {len(bad)}")
    if bad:
        print("  ERROR: dangling references after flatten -- refusing to build",
              file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bank", default=DEFAULT_BANK,
                    help="model bank JSON the base model names come from "
                         f"(default {os.path.relpath(DEFAULT_BANK, ROOT)})")
    a = ap.parse_args()
    return flatten(a.src, a.dst, a.bank)


if __name__ == "__main__":
    raise SystemExit(main())
