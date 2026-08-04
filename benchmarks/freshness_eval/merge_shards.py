"""Merge sharded freshness-sweep output directories into one result set.

The sweep is serial and greedy scales badly with contention (measured: 6 s at
B=0 rising to 584 s at B=4 for a single policy), so a full policy grid is
sharded across concurrent workers by policy and merged here. Cells are
independent — each writes its own config stem and fixture — so sharding changes
wall-clock only, not results.

    python -m benchmarks.freshness_eval.merge_shards \
        --shards results/freshness_probe/shard_* \
        --output results/freshness_probe

ORACLE ROWS ARE RECOMPUTED, NOT CONCATENATED. Each shard's runner derived its
oracle as the best output_valid_rate among the policies IT ran, so a shard
oracle is an upper bound over a subset and is wrong for the merged set. They are
dropped on read and re-derived over everything.

The merged manifest keeps every shard's manifest verbatim under `shards` rather
than trying to reconcile them, and re-derives only the fields that are
meaningfully global (policies, failures, determinism, counts).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

ORACLE = "oracle"


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: List[Dict], lead: List[str] = ()) -> None:
    if not rows:
        return
    cols = sorted({k for r in rows for k in r})
    cols = [c for c in lead if c in cols] + [c for c in cols if c not in lead]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True,
                    help="shard output directories (globs accepted)")
    ap.add_argument("--output", required=True, help="merged output directory")
    args = ap.parse_args()

    shard_dirs: List[str] = []
    for pat in args.shards:
        hits = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        shard_dirs.extend(h for h in hits if os.path.isdir(h))
    shard_dirs = [d for d in shard_dirs if os.path.abspath(d) != os.path.abspath(args.output)]
    if not shard_dirs:
        raise SystemExit(f"no shard directories matched {args.shards}")

    os.makedirs(args.output, exist_ok=True)

    agg: List[Dict] = []
    per_inv: List[Dict] = []
    intervals: List[Dict] = []
    manifests: Dict[str, Dict] = {}
    missing: List[str] = []

    for d in shard_dirs:
        man_path = os.path.join(d, "manifest.json")
        if not os.path.exists(man_path):
            # A shard that never finished must be visible, not silently absent:
            # its policies would just be missing from the merged grid.
            missing.append(d)
            continue
        with open(man_path) as f:
            manifests[os.path.basename(d)] = json.load(f)
        # Drop shard-local oracle rows; they are upper bounds over a subset.
        agg.extend(r for r in _read_csv(os.path.join(d, "aggregate.csv"))
                   if r.get("policy") != ORACLE)
        per_inv.extend(_read_csv(os.path.join(d, "per_invocation.csv")))
        intervals.extend(_read_csv(os.path.join(d, "intervals.csv")))

    if missing:
        print(f"WARNING: {len(missing)} shard(s) have no manifest.json and were "
              f"skipped -- the merged grid is INCOMPLETE:")
        for d in missing:
            print(f"  {d}")

    if not agg:
        raise SystemExit("no aggregate rows found across shards")

    # Duplicate detection: the same (policy, B, seed, phi) appearing in two
    # shards means the shard split overlapped, which would double-weight it.
    seen: Dict[Tuple, int] = {}
    for r in agg:
        key = (r["policy"], r["contention_level"], r["seed"], r["freshness_window"])
        seen[key] = seen.get(key, 0) + 1
    dupes = {k: n for k, n in seen.items() if n > 1}
    if dupes:
        raise SystemExit(
            f"{len(dupes)} duplicated (policy, B, seed, phi) cell(s) across "
            f"shards -- the shard split overlapped. First few: "
            f"{sorted(dupes)[:5]}"
        )

    # Re-derive the oracle over the FULL merged policy set.
    best: Dict[Tuple[str, str], Dict] = {}
    for r in agg:
        key = (r["contention_level"], r["freshness_window"])
        cur = best.get(key)
        if cur is None or float(r["output_valid_rate"]) > float(cur["output_valid_rate"]):
            best[key] = r
    for (_burst, _phi), r in sorted(best.items()):
        o = dict(r)
        o["policy"] = ORACLE
        o["candidate_id"] = f"oracle<-{r['policy']}"
        agg.append(o)

    _write_csv(os.path.join(args.output, "aggregate.csv"), agg,
               lead=["policy", "candidate_id", "seed", "contention_level",
                     "freshness_window", "delta", "A0"])
    _write_csv(os.path.join(args.output, "per_invocation.csv"), per_inv)
    _write_csv(os.path.join(args.output, "intervals.csv"), intervals)

    policies = sorted({r["policy"] for r in agg if r["policy"] != ORACLE})
    failures = [f for m in manifests.values() for f in (m.get("failures") or [])]
    ref = next(iter(manifests.values())) if manifests else {}

    merged = {
        "schema": "xpurt.freshness_eval.merged/1",
        "merged_from": [os.path.relpath(d, _REPO) for d in shard_dirs],
        "incomplete_shards": [os.path.relpath(d, _REPO) for d in missing],
        "policies_merged": policies,
        "n_policies": len(policies),
        "n_aggregate_rows": len(agg),
        "n_per_invocation_rows": len(per_inv),
        "failures": failures,
        "oracle_note": (
            "Recomputed over the full merged policy set. Shard-local oracle rows "
            "were dropped on read because each was an upper bound over only that "
            "shard's policies."
        ),
        # Carried from one shard: identical by construction across shards, since
        # they all read the same config and the same profile tree.
        "config": ref.get("config"),
        "epoch_ms": ref.get("epoch_ms"),
        "A0": ref.get("A0"),
        "phis": ref.get("phis"),
        "post_passes": ref.get("post_passes"),
        "timing_provenance": ref.get("timing_provenance"),
        "producer_instance_provenance": ref.get("producer_instance_provenance"),
        "producer_reference_window_ms": ref.get("producer_reference_window_ms"),
        "mutation_vocabulary": ref.get("mutation_vocabulary"),
        "time_unit": ref.get("time_unit"),
        "shards": manifests,
    }

    # Cross-shard consistency: a differing A0 or config would make the merged
    # rows incomparable.
    for name, m in manifests.items():
        for field in ("config", "epoch_ms"):
            if m.get(field) != ref.get(field):
                raise SystemExit(
                    f"shard {name} has {field}={m.get(field)!r} but another has "
                    f"{ref.get(field)!r}; these results are not comparable"
                )
        a_ref = (ref.get("A0") or {}).get("A0_realized")
        a_m = (m.get("A0") or {}).get("A0_realized")
        if a_ref is not None and a_m is not None and abs(a_ref - a_m) > 1e-9:
            raise SystemExit(
                f"shard {name} has A0={a_m} but another has {a_ref}; the phi grid "
                f"differs and these results are not comparable"
            )

    with open(os.path.join(args.output, "manifest.json"), "w") as f:
        json.dump(merged, f, indent=2)

    print(f"merged {len(shard_dirs)} shard(s) -> {args.output}")
    print(f"  policies      {len(policies)}: {', '.join(policies)}")
    print(f"  aggregate     {len(agg)} rows (incl. recomputed oracle)")
    print(f"  per_invocation{len(per_inv):>6} rows")
    if failures:
        print(f"  FAILURES      {len(failures)}")
    return 1 if (failures or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
