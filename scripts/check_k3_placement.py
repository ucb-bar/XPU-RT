#!/usr/bin/env python3
"""Did each dispatch of a K3 run execute where the schedule put it, with a
kernel that core can run?

WHY. On the K3 a wrong placement does not fail on its own. An A100 worker that
was never registered as an AI thread stays on an X100 and its timings are
reported as A100 timings; an X100 IME kernel on an A100 returns wrong numbers
rather than trapping (the tiles differ), and the A100 backends' zvl1024b code
is illegal on the X100. The walker places every worker itself (mb_placement);
this script is the after-the-fact evidence, from what the binary recorded.

INPUTS
  --placement     <sched>_placement.csv  (ModelBlaster run_xpurt_k3.sh with
                  MB_XPURT_PLACEMENT=1; columns entry_id, network, instance,
                  dispatch_id, core_kind, impl, hart, n_harts, harts,
                  cpu_before, cpu_after)
  --registry      ModelBlaster/cores/spacemit_k3.json
  --backends      the run's --core-kinds/--backends pairing, as kind=backend
                  pairs (x100=ime_k3_x100,a100=ime_k3_a100)
  --impl-backends the run's --impl-backends (x100:rvv=rvv_k3,...), if any

CHECKS (each violation is listed; exit 1 if any)
  1. every entry with a kernel ran (cpu_before != -1); dispatch_id < 0 rows
     are view/chunk ops the walker completes without a kernel, so they are skipped
  2. it ran on its hart, before and after the kernel (cpu_before == cpu_after
     == hart; a composite entry's master is its first hart)
  3. that cpu's registry kind is the entry's core_kind
  4. the backend the binary selects for the entry's (core_kind, impl) exists
     and is legal on that kind (LEGAL_BACKENDS)
  5. a composite entry's harts are all of the entry's kind

Not covered: which cpu each pool helper ran its slice on (the walker does not
record it; the profiling harness's pool slice trace does).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter

#: Backends each K3 core kind may execute. The A100 backends are built at
#: zvl1024b (illegal instruction stream on the X100, VLEN 256); the X100 IME
#: kernels use the 4x8x4 tile and give wrong answers on the A100 (8x16x8).
LEGAL_BACKENDS = {
    "x100": ("rvv_k3", "ime_k3_x100"),
    "a100": ("rvv_k3_a100", "ime_k3_a100"),
}


def load_registry(path: str) -> dict[int, dict]:
    """cpu -> {"name", "kind"} from a cores/*.json registry."""
    with open(path) as f:
        doc = json.load(f)
    by_cpu: dict[int, dict] = {}
    for core in doc["cores"]:
        for hart in core.get("harts", []):
            by_cpu[int(hart)] = {"name": core["name"], "kind": core["kind"]}
    return by_cpu


def parse_pairs(backends: str, impl_backends: str) -> dict[tuple[str, str], str]:
    """{(core_kind, impl): backend}. A kind's own pairing answers impl == kind."""
    out: dict[tuple[str, str], str] = {}
    for item in filter(None, (x.strip() for x in (backends or "").split(","))):
        kind, _, bs = item.partition("=")
        if not (kind and bs):
            raise ValueError(f"--backends: bad entry {item!r} (want kind=backend)")
        out[(kind, kind)] = bs
    for item in filter(None, (x.strip() for x in (impl_backends or "").split(","))):
        lhs, _, bs = item.partition("=")
        kind, _, impl = lhs.partition(":")
        if not (kind and impl and bs):
            raise ValueError(f"--impl-backends: bad entry {item!r} (want kind:impl=backend)")
        out[(kind, impl)] = bs
    return out


def check_rows(rows: list[dict], by_cpu: dict[int, dict],
               pairs: dict[tuple[str, str], str]) -> list[str]:
    bad: list[str] = []
    for r in rows:
        tag = f"entry {r['entry_id']} ({r['network']}#{r['instance']} d{r['dispatch_id']})"
        if int(r["dispatch_id"]) < 0:
            continue
        kind, impl = r["core_kind"], r["impl"]
        hart, before, after = int(r["hart"]), int(r["cpu_before"]), int(r["cpu_after"])
        backend = pairs.get((kind, impl))   # impl == kind: the kind's own pairing
        if backend is None:
            bad.append(f"{tag}: no backend for core_kind {kind} impl {impl} in this build")
        elif backend not in LEGAL_BACKENDS.get(kind, ()):
            bad.append(f"{tag}: impl {impl} runs backend {backend}, not legal on {kind} "
                       f"(legal: {list(LEGAL_BACKENDS.get(kind, ()))})")
        if before < 0:
            bad.append(f"{tag}: never ran")
            continue
        if not before == after == hart:
            bad.append(f"{tag}: scheduled on cpu {hart}, ran on {before} -> {after}")
        core = by_cpu.get(before)
        if core is None:
            bad.append(f"{tag}: ran on cpu {before}, which the registry does not list")
            continue
        if core["kind"] != kind:
            bad.append(f"{tag}: core_kind {kind} but cpu {before} is {core['kind']}")
        harts = [int(h) for h in str(r.get("harts") or hart).split()]
        kinds = {by_cpu.get(h, {}).get("kind") for h in harts}
        if kinds != {kind}:
            bad.append(f"{tag}: harts {harts} span kinds {sorted(map(str, kinds))}")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--placement", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--backends", required=True,
                    help="kind=backend pairs, e.g. x100=ime_k3_x100,a100=ime_k3_a100")
    ap.add_argument("--impl-backends", default="")
    args = ap.parse_args(argv)

    by_cpu = load_registry(args.registry)
    pairs = parse_pairs(args.backends, args.impl_backends)
    with open(args.placement) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"FAIL: {args.placement} has no rows")
        return 1
    bad = check_rows(rows, by_cpu, pairs)
    ran = Counter((r["core_kind"], r["impl"], r["cpu_before"]) for r in rows
                  if int(r["dispatch_id"]) >= 0)
    print(f"{len(rows)} entries")
    for (kind, impl, cpu), n in sorted(ran.items(), key=lambda kv: (kv[0][0], kv[0][1], int(kv[0][2]))):
        print(f"  {kind:5s} impl={impl:5s} backend={pairs.get((kind, impl), '?'):12s} cpu={cpu:>3s}  x{n}")
    if bad:
        print(f"FAIL: {len(bad)} placement violation(s)")
        for b in bad[:50]:
            print(f"  - {b}")
        return 1
    print("PASS: every entry ran on its scheduled cpu with a kernel that cpu can run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
