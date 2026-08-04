"""Backfill the config-hash sidecars for fixtures solved before they existed.

`run.py --reuse-fixtures` proves a fixture corresponds to a config by comparing a
SHA256 of the config against a sidecar written at solve time. Fixtures produced
before that sidecar existed cannot be reused -- and re-solving them costs ~584 s
per B=4 cell, which is exactly the pressure that makes an evaluator fix feel too
expensive to apply everywhere.

This script backfills the sidecars, but it does NOT take the fixture's word for
it. For each (policy, burst, seed) cell it:

  1. re-derives the config with the CURRENT materialise(),
  2. compares it byte-for-byte against the config file left on disk by the run
     that produced the fixture,
  3. checks the fixture is NEWER than that config file.

Only when all three hold is the sidecar written. (1) fails if materialise has
changed since the solve; (2) fails if a later cell overwrote the stem with a
different workload -- the stem-reuse hazard the sidecar exists to catch; (3)
fails if the fixture predates the config, i.e. belongs to some earlier workload.

A cell that fails any check is reported and left alone, so it re-solves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)

from benchmarks.freshness_eval.run import (  # noqa: E402
    ALL_POLICIES,
    materialise,
    solver_tag,
)


# Keys that describe PROVENANCE rather than the workload, and are excluded from
# the equivalence check.
#
# `_materialised` is written by materialise() and read only by the sweep's own
# reporting (run.py takes `soft_instances_admitted` from it). It is never passed
# to the solver, so it cannot change a schedule. Excluding it is what makes the
# Gate A fixtures reusable: those configs predate the block entirely, and a
# byte-exact comparison refused all 15 cells over three provenance scalars
# (offered_burst, admitted_soft_instances, seed) while the networks, scheduler and
# epoch sections were identical.
#
# This is safe rather than convenient, and the reason is worth stating: every
# input that `_materialised` records is also expressed in the workload proper --
# `burst` sets the soft network's num_instances, `mutations` write into the
# network fields, and `seed` does not affect the workload at all. So any
# semantically meaningful difference still shows up outside the excluded block,
# and the exclusion cannot mask one.
#
# The sidecar still records the hash of the FULL current config, so
# `--reuse-fixtures` remains an exact match afterwards.
PROVENANCE_ONLY_KEYS = ("_materialised",)


def _workload_of(cfg: dict) -> str:
    """Canonical form of everything that can affect the schedule."""
    stripped = {k: v for k, v in cfg.items() if k not in PROVENANCE_ONLY_KEYS}
    return json.dumps(stripped, indent=2, sort_keys=True)


def backfill(base_path: str, policies, bursts, seeds, *, apply: bool):
    with open(base_path) as f:
        base = json.load(f)
    epoch_ms = float(base.get("epoch", {}).get("length_ms", 300.0))

    written, skipped = [], []
    for policy in policies:
        spec = ALL_POLICIES[policy]
        for burst in bursts:
            for seed in seeds:
                stem = f"_fx_{policy}_B{burst}_s{seed}"
                cfg_path = os.path.join(_REPO, "data", "toplevel", f"{stem}.json")
                fixture = os.path.join(
                    _REPO, "schedules",
                    f"scheduled_{stem}"
                    f"{solver_tag(spec['solver'], spec['scheduler'])}_profiled.json",
                )
                cell = f"{policy} B={burst} s={seed}"

                if not os.path.exists(fixture):
                    skipped.append((cell, "no fixture"))
                    continue
                if not os.path.exists(cfg_path):
                    skipped.append((cell, "no config on disk to verify against"))
                    continue

                cfg = materialise(base, burst=burst,
                                  mutations=spec.get("mutations"),
                                  epoch_ms=epoch_ms, seed=seed)
                derived = json.dumps(cfg, indent=2, sort_keys=True)
                with open(cfg_path) as f:
                    on_disk = json.load(f)
                if _workload_of(on_disk) != _workload_of(cfg):
                    skipped.append((cell, "on-disk config differs from re-derived"))
                    continue
                if os.path.getmtime(fixture) <= os.path.getmtime(cfg_path):
                    skipped.append((cell, "fixture older than its config"))
                    continue

                digest = hashlib.sha256(derived.encode()).hexdigest()
                if apply:
                    with open(fixture + ".cfgsha256", "w") as f:
                        f.write(digest + "\n")
                written.append(cell)
    return written, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--policies", required=True,
                    help="comma-separated, or ALL / PROBES")
    ap.add_argument("--bursts", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--apply", action="store_true",
                    help="write the sidecars (default is a dry run)")
    args = ap.parse_args()

    if args.policies.upper() == "ALL":
        policies = list(ALL_POLICIES)
    elif args.policies.upper() == "PROBES":
        policies = [p for p in ALL_POLICIES if p.startswith("probe_")]
        policies = ["static_nominal"] + policies
    else:
        policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    bursts = [int(b) for b in args.bursts.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    written, skipped = backfill(args.config, policies, bursts, seeds,
                                apply=args.apply)
    verb = "wrote" if args.apply else "would write"
    print(f"{verb} {len(written)} sidecar(s)")
    if skipped:
        print(f"\n{len(skipped)} cell(s) NOT backfilled (these will re-solve):")
        for cell, why in skipped:
            print(f"  {cell:<40} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
