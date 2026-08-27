#!/usr/bin/env python3
"""Apply choose_implementation advice to a schedule, per dispatch.

The runner resolves a dispatch's VMFB from its cluster's ISA-variant directory,
which forces one implementation per cluster. But ResolveVmfbPath checks an
explicit `vmfb_path` in the schedule first, so writing that field per dispatch
lets a single run mix implementations at dispatch granularity -- which is what
the advice is expressed in.

Durations are rewritten from the chosen implementation's measured profile at the
same time. Leaving the old durations would make the schedule an assertion about
one binary while executing another.

Only choose_implementation is applied here. split / fuse_with_successor change
the dispatch graph itself and belong to the compiler front end, not to a
post-pass over a finished schedule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def profile_path(gen_root, impl, target, model, basename, topo="topo_0"):
    return os.path.join(gen_root, "profile", impl, target, model, basename,
                        topo, "profile.jsonl")


def vmfb_for(gen_root, model, target, impl, basename, module_name, elf_marker):
    stem = module_name
    pos = module_name.find(elf_marker)
    if pos != -1:
        stem = module_name[:pos + len(elf_marker)]
    fn = f"module_{stem}_benchmark.vmfb"
    return os.path.join(gen_root, "vmfb", model, target, impl, basename,
                        "benchmarks", "vmfb", fn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--advice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-root", default="gen")
    ap.add_argument("--remote-gen-root", default="/root/mb_k1/gen",
                    help="gen root as seen ON THE BOARD, for vmfb_path")
    ap.add_argument("--target", default="spacemit_x60")
    ap.add_argument("--models", default="mlp:mlp.q.int8,dronet:dronet.q.int8")
    ap.add_argument("--elf-marker", default="_embedded_elf_riscv_64")
    a = ap.parse_args()

    sched = json.load(open(a.schedule))
    advice = json.load(open(a.advice))
    basenames = dict(s.split(":") for s in a.models.split(","))

    # dispatch -> chosen implementation, from actionable advice only
    chosen = {}
    for item in advice["advice"]:
        if item["recommendation"] != "choose_implementation":
            continue
        ev = item["evidence"]
        chosen[(item["model"], int(item["dispatch_id"]))] = ev["proposed_impl"]
    if not chosen:
        print("no choose_implementation advice to apply", file=sys.stderr)
        return 1

    # measured medians for whichever implementations we are switching to
    prof_cache = {}
    def median_ms(model, impl, did):
        key = (model, impl)
        if key not in prof_cache:
            p = profile_path(a.gen_root, impl, a.target, model, basenames[model])
            recs = {}
            if os.path.exists(p):
                for line in open(p):
                    if line.strip():
                        r = json.loads(line)
                        recs[r["dispatch_id"]] = r
            prof_cache[key] = recs
        r = prof_cache[key].get(did)
        return r["median_ms"] if r else None

    applied = 0
    dur_before = dur_after = 0.0
    for key, node in sched["dispatches"].items():
        job = node.get("job_name", "")
        model = job.rstrip("0123456789") or job
        did = node.get("id")
        impl = chosen.get((model, did))
        if impl is None:
            continue
        mn = node.get("module_name", "")
        if not mn or model not in basenames:
            continue
        node["vmfb_path"] = vmfb_for(a.remote_gen_root, model, a.target, impl,
                                     basenames[model], mn, a.elf_marker)
        node["implementation"] = impl        # provenance, ignored by the runner
        m = median_ms(model, impl, did)
        if m is not None:
            dur_before += float(node["duration"])
            dur_after += m
            node["duration"] = m
        applied += 1

    meta = sched.setdefault("metadata", {})
    meta["applied_advice"] = {
        "source": os.path.basename(a.advice),
        "n_dispatches_retargeted": applied,
        "predicted_service_before_ms": round(dur_before, 3),
        "predicted_service_after_ms": round(dur_after, 3),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(sched, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    print(f"  retargeted {applied} dispatch instances")
    print(f"  predicted service over those dispatches: "
          f"{dur_before:.3f} ms -> {dur_after:.3f} ms "
          f"({100*(dur_before-dur_after)/dur_before:+.2f}% )" if dur_before else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
