"""Contention x freshness-window sweep for the freshness-validity evaluation.

Answers one question: can a dependent control pipeline keep meeting its own
deadlines while emitting invalid outputs because its inputs are stale?

    python -m benchmarks.freshness_eval.run \
        --config data/toplevel/freshness_canon_300ms.json \
        --output-dir results/freshness_eval --seeds 0,1,2,3,4

Sweep structure
---------------
  contention  B in {0..4}   YOLO instances released across the epoch
  window      phi = A0 + delta, delta in {5,10,20,30,50} ms
  policies    static_nominal, edf, heft, static_conservative (+ derived oracle)

phi is anchored on A0, the MEASURED uncontended input-age ceiling, not on the
DroNet period. On this grid A0 = 60.546 ms while the period is 50 ms, so
phi = period + delta would sit BELOW the uncontended ceiling for small delta and
the staleness it reported would come from the 50 ms sampling rate rather than
from contention. Every row records phi in absolute ms, the delta, and A0.

Scheduling is run once per (policy, B); phi is applied post-hoc, since the
freshness window changes the verdict but not the schedule. That makes the phi
axis free rather than a 5x cost.

Post-passes are forced OFF (XPURT_COMPACT / XPURT_AUTOMERGE unset, and their
force-off flags set): both rewrite the emitted fixture, so leaving them on
would make this a comparison of policy+post-passes, and would make the
per-instance intervals reflect the post-pass rather than the schedule. The
effective setting is recorded in the manifest.

The oracle is a post-hoc upper bound -- the best output_valid_rate available at
each (B, phi) among the deployable policies. It is not a deployable policy and
must not be reported as one.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))

from freshness import (  # noqa: E402
    CSV_COLUMNS,
    analytic_age_ceiling_realized,
    evaluate_freshness,
    freshness_edges_from_config,
    criticality_from_config,
)
from benchmarks.freshness_eval.trace import (  # noqa: E402
    invocations_from_fixture,
    load_fixture,
    soft_utility,
)

# --- policy definitions ----------------------------------------------------
#
# Each policy is a (solver, scheduler, config-mutation) triple. `preferred_hw`
# is a SOFT pin: profile_loader adds a large cost penalty to non-preferred
# combinations (PIN_PENALTY_MULT), it does not hard-exclude them. Describe it
# as a preference, not a reservation.

POLICIES: Dict[str, Dict] = {
    "static_nominal": {
        "solver": "greedy",
        "scheduler": "mosek",  # unused when solver != milp
        "preferred_hw": None,
        "intent": (
            "Contention-blind earliest-completion list schedule. Maximises soft "
            "throughput under low load; no protection for the perception chain."
        ),
    },
    "edf": {
        "solver": "milp",
        "scheduler": "edf",
        "preferred_hw": None,
        "intent": "Deadline-ordered list schedule; the conventional real-time baseline.",
    },
    "heft": {
        "solver": "milp",
        "scheduler": "heft",
        "preferred_hw": None,
        "intent": "Heterogeneous-DAG baseline (upward rank, earliest finish).",
    },
    "static_conservative": {
        "solver": "greedy",
        "scheduler": "mosek",
        # ONE mechanism: reserve the fast accelerator for the perception
        # producer. Applied statically at every contention level, which is
        # exactly why it should cost soft utility at low load.
        #
        # Deliberately NOT also forcing YOLO onto the vector unit. That is a
        # second, much blunter mechanism (yolov8_nano_64 is 1069 ms on rvv_opu
        # versus 67 ms on gemmini, so a single instance cannot finish inside a
        # 300 ms epoch and soft utility collapses to zero at every B). Mixing
        # the two would make it impossible to attribute the outcome to either.
        # It belongs to the degraded-safety candidate, not to this baseline.
        #
        # NOTE these are profile-hw names (hardware.profile_hw VALUES), not
        # cluster names: profile_loader compares preferred_hw against combo_hw,
        # which holds "gemmini"/"rvv_opu". Naming the cluster instead used to
        # penalise every combination silently; it now raises.
        "preferred_hw": {"dronet": "gemmini"},
        "intent": (
            "Protects the DroNet->control chain by reserving the fast "
            "accelerator for DroNet (soft cost penalty on other placements)."
        ),
    },
}

DEPLOYABLE = list(POLICIES)
ORACLE = "oracle"

DEFAULT_DELTAS = [5.0, 10.0, 20.0, 30.0, 50.0]
DEFAULT_BURSTS = [0, 1, 2, 3, 4]


def _git_info() -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name, repo in (
        ("xpu-rt", _REPO),
        ("ModelBlaster", "/scratch2/agustin/ModelBlaster"),
        ("zephyr-chipyard-sw", os.path.join(_REPO, "zephyr-chipyard-sw")),
    ):
        try:
            sha = subprocess.run(
                ["git", "-C", repo, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            dirty = subprocess.run(
                ["git", "-C", repo, "status", "--porcelain"],
                capture_output=True, text=True, timeout=30,
            )
            out[name] = {
                "sha": sha.stdout.strip() if sha.returncode == 0 else None,
                "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            }
        except Exception as exc:
            out[name] = {"error": str(exc)}
    return out


def materialise(base: Dict, *, burst: int, preferred_hw: Optional[Dict[str, str]],
                epoch_ms: float, seed: int) -> Dict:
    """Build a concrete workload config for one (burst, policy) cell."""
    cfg = copy.deepcopy(base)
    cfg.setdefault("scheduler", {})["random_seed"] = seed

    nets = cfg["networks"]
    soft = [n for n, i in nets.items() if str(i.get("criticality", "soft")) == "soft"]
    if len(soft) != 1:
        raise ValueError(
            f"expected exactly one soft network to parameterise the burst, "
            f"found {soft}"
        )
    soft_name = soft[0]

    if burst <= 0:
        # No soft work at all: remove the network rather than give it zero
        # instances, so the workload contains no vestigial entry.
        nets.pop(soft_name)
    else:
        info = nets[soft_name]
        info["num_instances"] = burst
        # Spread B releases evenly across the epoch. window_duration stays at
        # the epoch so the window is loose: soft work is interference to be
        # shed, not a hard constraint that would make the instance infeasible.
        info["period"] = epoch_ms / burst
        info["window_duration"] = epoch_ms

    if preferred_hw:
        for net, hw in preferred_hw.items():
            if net in nets:
                nets[net]["preferred_hw"] = hw

    return cfg


def solver_tag(solver: str, scheduler: str) -> str:
    """Reconstruct run_xpurt_schedule.py's output-path suffix."""
    if solver in ("greedy", "greedy_periodic", "decomposed"):
        return f"_{solver}"
    if solver == "milp" and scheduler != "mosek":
        return f"_{scheduler}"
    return ""


def run_schedule(cfg: Dict, *, stem: str, solver: str, scheduler: str,
                 time_limit: float, work_dir: str,
                 cell_timeout_s: float = 600.0) -> Tuple[Optional[str], float, str]:
    """Invoke run_xpurt_schedule.py on a materialised config."""
    cfg_path = os.path.join(_REPO, "data", "toplevel", f"{stem}.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    cmd = [
        sys.executable, os.path.join(_REPO, "scripts", "run_xpurt_schedule.py"),
        "--networks-json", os.path.relpath(cfg_path, _REPO),
        "--solver", solver,
        "--use-profiled",
        "--no-prune-periods",
        "--include-periodic-in-makespan",
        "--time-limit", str(time_limit),
    ]
    if solver == "milp":
        cmd += ["--scheduler", scheduler]

    env = dict(os.environ)
    # Force both fixture-rewriting post-passes off, and be explicit rather than
    # relying on the (now opt-in) defaults.
    env["XPURT_NO_COMPACT"] = "1"
    env["XPURT_NO_AUTOMERGE"] = "1"

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=_REPO, env=env, capture_output=True, text=True,
            timeout=cell_timeout_s,
        )
    except subprocess.TimeoutExpired:
        # A single pathological cell must not stall the whole sweep. Record it
        # as a failure so the manifest shows coverage was incomplete rather than
        # the grid silently having a hole.
        wall = time.time() - t0
        with open(os.path.join(work_dir, f"{stem}.timeout"), "w") as f:
            f.write(f"$ {' '.join(cmd)}\ntimed out after {cell_timeout_s}s\n")
        return None, wall, f"timeout after {cell_timeout_s:.0f}s"
    wall = time.time() - t0

    log_path = os.path.join(work_dir, f"{stem}{solver_tag(solver, scheduler)}.log")
    with open(log_path, "w") as f:
        f.write(f"$ {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")

    fixture = os.path.join(
        _REPO, "schedules",
        f"scheduled_{stem}{solver_tag(solver, scheduler)}_profiled.json",
    )
    if proc.returncode != 0:
        return None, wall, f"solver exit {proc.returncode} (see {log_path})"
    if not os.path.exists(fixture):
        return None, wall, f"fixture missing: {fixture} (see {log_path})"
    return fixture, wall, "ok"


def compute_a0(base: Dict, *, epoch_ms: float, edge) -> Dict[str, object]:
    """Closed-form uncontended input-age ceiling for the producer/consumer pair.

    Latencies are the per-network totals on the FAST cluster, read from the
    exported profile CSVs -- i.e. the intended uncontended placement.
    """
    from profile_loader import find_profile_csv, load_profiled_times

    hw = base["hardware"]["profile_hw"]["cpu_p"]
    target = base["hardware"]["profile"]["target"]
    topo = base["hardware"]["profile"].get("topo_tag", "topo_0")

    def latency(net: str) -> float:
        info = base["networks"][net]
        ident = info.get("identifier", net)
        basename = os.path.basename(
            info["dispatch_deps_path"]
        ).replace("_dispatch_graph.json", "")
        csv_path = find_profile_csv(
            _REPO, model=ident, target=target, hw=hw, basename=basename, topo_tag=topo
        )
        if not csv_path:
            raise FileNotFoundError(
                f"no profile CSV for {ident} on {hw}/{target}/{topo}; run "
                f"scripts/export_profile_db_to_results_csv.py"
            )
        return sum(v["time_ms"] for v in load_profiled_times(csv_path).values())

    p_net, c_net = edge.producer_task, edge.consumer_task
    lp, lc = latency(p_net), latency(c_net)
    tp = float(base["networks"][p_net]["period"])
    tc = float(base["networks"][c_net]["period"])

    res = analytic_age_ceiling_realized(
        producer_period=tp, producer_latency=lp,
        consumer_period=tc, consumer_latency=lc, horizon=epoch_ms,
    )
    res.update({
        "producer_task": p_net, "consumer_task": c_net,
        "producer_period_ms": tp, "producer_latency_ms": lp,
        "consumer_period_ms": tc, "consumer_latency_ms": lc,
        "fast_cluster_hw": hw,
    })
    return res


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="data/toplevel/freshness_canon_300ms.json")
    ap.add_argument("--output-dir", default="results/freshness_eval")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--bursts", default=",".join(str(b) for b in DEFAULT_BURSTS))
    ap.add_argument("--deltas", default=",".join(str(d) for d in DEFAULT_DELTAS),
                    help="phi = A0 + delta, in ms")
    ap.add_argument("--policies", default=",".join(DEPLOYABLE))
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--cell-timeout", type=float, default=600.0,
                    help="wall-clock cap per (policy, B, seed) cell; a cell that "
                         "exceeds it is recorded as a failure rather than "
                         "stalling the sweep")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO, args.config)
    with open(cfg_path) as f:
        base = json.load(f)

    epoch_ms = float(base.get("epoch", {}).get("length_ms", 300.0))
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    bursts = [int(b) for b in args.bursts.split(",") if b != ""]
    deltas = [float(d) for d in args.deltas.split(",") if d != ""]
    policies = [p for p in args.policies.split(",") if p]
    for p in policies:
        if p not in POLICIES:
            raise SystemExit(f"unknown policy {p!r}; have {sorted(POLICIES)}")

    out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(_REPO, args.output_dir)
    work_dir = os.path.join(out_dir, "work")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    edges0 = freshness_edges_from_config(base, freshness_window_override=1.0)
    if len(edges0) != 1:
        raise SystemExit(
            f"expected exactly one freshness edge, found {len(edges0)}"
        )
    edge = edges0[0]

    a0_info = compute_a0(base, epoch_ms=epoch_ms, edge=edge)
    A0 = float(a0_info["A0_realized"])
    phis = [(d, A0 + d) for d in deltas]

    crit = criticality_from_config(base)
    soft_tasks = [t for t, c in crit.items() if c == "soft"]

    print(f"config          {os.path.relpath(cfg_path, _REPO)}")
    print(f"epoch           {epoch_ms} ms")
    print(f"A0 (uncontended input-age ceiling)  {A0:.3f} ms")
    print(f"  closed-form supremum              {a0_info['A0_supremum']:.3f} ms")
    print(f"  distinct uncontended ages         "
          f"{[round(x, 2) for x in a0_info['distinct_ages']]}")
    print(f"  producer {edge.producer_task} T={a0_info['producer_period_ms']} "
          f"L={a0_info['producer_latency_ms']:.3f} ms on {a0_info['fast_cluster_hw']}")
    print(f"  consumer {edge.consumer_task} T={a0_info['consumer_period_ms']} "
          f"L={a0_info['consumer_latency_ms']:.3f} ms")
    print(f"phi             {[f'A0+{int(d)}={p:.1f}' for d, p in phis]}")
    print(f"bursts          {bursts}")
    print(f"policies        {policies}")
    print(f"seeds           {seeds}")
    print()

    per_inv_rows: List[Dict] = []
    interval_rows: List[Dict] = []
    agg_rows: List[Dict] = []
    schedule_digests: Dict[Tuple[str, int], Dict[int, str]] = {}
    failures: List[Dict] = []

    for policy in policies:
        spec = POLICIES[policy]
        for burst in bursts:
            for seed in seeds:
                stem = f"_fx_{policy}_B{burst}_s{seed}"
                cfg = materialise(
                    base, burst=burst, preferred_hw=spec["preferred_hw"],
                    epoch_ms=epoch_ms, seed=seed,
                )
                fixture_path, wall, status = run_schedule(
                    cfg, stem=stem, solver=spec["solver"],
                    scheduler=spec["scheduler"], time_limit=args.time_limit,
                    work_dir=work_dir, cell_timeout_s=args.cell_timeout,
                )
                if status != "ok":
                    print(f"  [FAIL] {policy:<20} B={burst} seed={seed}: {status}")
                    failures.append({
                        "policy": policy, "burst": burst, "seed": seed,
                        "status": status,
                    })
                    continue

                fx = load_fixture(fixture_path)
                invs = invocations_from_fixture(fx, cfg)
                makespan = float(fx["metadata"]["makespan"])

                # Determinism check: same (policy, B) across seeds must give the
                # same schedule for a deterministic scheduler. Recorded rather
                # than asserted, because a stochastic policy legitimately differs.
                digest = json.dumps(
                    sorted(
                        (i.task, i.instance, round(i.start_time, 6), round(i.end_time, 6))
                        for i in invs
                    )
                )
                schedule_digests.setdefault((policy, burst), {})[seed] = str(hash(digest))

                su = soft_utility(invs, soft_tasks, epoch_ms)

                # Every invocation interval, so the diagnostic timeline plot has
                # the interfering soft work too (per_invocation.csv only carries
                # the producer/consumer pair on the freshness edge).
                for i in invs:
                    interval_rows.append({
                        "policy": policy, "contention_level": burst, "seed": seed,
                        "task": i.task, "instance": i.instance,
                        "criticality": crit.get(i.task, "soft"),
                        "release_time": i.release_time,
                        "start_time": i.start_time,
                        "end_time": i.end_time,
                        "deadline": i.deadline,
                    })

                for delta, phi in phis:
                    ev = evaluate_freshness(
                        invs,
                        dependency_edges=freshness_edges_from_config(
                            base, freshness_window_override=phi
                        ),
                        experiment_id=f"{policy}_B{burst}_s{seed}_phi{phi:.1f}",
                        seed=seed, policy=policy, candidate_id=policy,
                        contention_level=float(burst), epoch_length=epoch_ms,
                        time_unit="ms",
                        provenance={
                            "timing_source": "firesim_measured",
                            "measured_or_derived": "measured_cycles_derived_ms",
                            "clock_mhz_assumed": 1000.0,
                            "pdb_hash": fx["metadata"].get("pdb_hash"),
                        },
                    )
                    per_inv_rows.extend(ev.rows())
                    a = dict(ev.aggregate)
                    a.update({
                        "policy": policy, "candidate_id": policy, "seed": seed,
                        "contention_level": burst, "freshness_window": phi,
                        "delta": delta, "A0": A0,
                        "makespan_ms": makespan,
                        "epoch_ms": epoch_ms,
                        "solver_wall_s": wall,
                        "fits_in_epoch": makespan <= epoch_ms,
                        "divergence": (
                            a["deadline_success_rate"] - a["output_valid_rate"]
                        ),
                        **su,
                    })
                    a.pop("soft_completed_by_task", None)
                    a.pop("soft_released_by_task", None)
                    agg_rows.append(a)

                print(f"  {policy:<20} B={burst} seed={seed} "
                      f"makespan={makespan:7.1f}ms  "
                      f"deadline={agg_rows[-1]['deadline_success_rate']:.3f} "
                      f"soft={su['soft_instances_completed']}/{max(burst,0)} "
                      f"({wall:.1f}s)")

    # --- oracle: post-hoc upper bound, NOT a deployable policy ---
    best: Dict[Tuple[int, float], Dict] = {}
    for r in agg_rows:
        key = (r["contention_level"], r["freshness_window"])
        cur = best.get(key)
        if cur is None or r["output_valid_rate"] > cur["output_valid_rate"]:
            best[key] = r
    for (burst, phi), r in sorted(best.items()):
        o = dict(r)
        o["policy"] = ORACLE
        o["candidate_id"] = f"oracle<-{r['policy']}"
        agg_rows.append(o)

    # --- write artifacts ---
    inv_csv = os.path.join(out_dir, "per_invocation.csv")
    with open(inv_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        w.writerows(per_inv_rows)

    if interval_rows:
        with open(os.path.join(out_dir, "intervals.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(interval_rows[0]))
            w.writeheader()
            w.writerows(interval_rows)

    agg_csv = os.path.join(out_dir, "aggregate.csv")
    if agg_rows:
        cols = sorted({k for r in agg_rows for k in r})
        lead = ["policy", "candidate_id", "seed", "contention_level",
                "freshness_window", "delta", "A0"]
        cols = lead + [c for c in cols if c not in lead]
        with open(agg_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(agg_rows)

    import schedulers as _sched
    import postprocessing as _pp

    manifest = {
        "schema": "xpurt.freshness_eval/1",
        "config": os.path.relpath(cfg_path, _REPO),
        "epoch_ms": epoch_ms,
        "bursts": bursts,
        "deltas": deltas,
        "phis": [{"delta": d, "phi_ms": p} for d, p in phis],
        "A0": a0_info,
        "policies": {p: POLICIES[p] for p in policies},
        "oracle_note": (
            "The oracle row is a post-hoc upper bound: the best output_valid_rate "
            "available at each (B, phi) among the deployable policies. It is not "
            "a deployable policy."
        ),
        "seeds": seeds,
        "post_passes": {
            "compaction_enabled": _sched.compaction_enabled(),
            "automerge_enabled": _pp.automerge_enabled(),
            "note": (
                "Both rewrite the emitted fixture and are forced off, so this is "
                "a comparison of policies rather than of policy+post-pass."
            ),
        },
        "timing_provenance": {
            "timing_source": "firesim_measured",
            "target": base["hardware"]["profile"]["target"],
            "backends": base["hardware"]["profile_hw"],
            "measured_or_derived": "measured_cycles_derived_ms",
            "clock_mhz_assumed": 1000.0,
            "scaling_factor": "mean_time_ms = cycles / 1e6",
            "clock_caveat": (
                "The assumed 1 GHz is NOT the Alveo U250 bitstream frequency "
                "(25-30 MHz). Raw cycles are preserved in the exported "
                "results.csv `cycles` column. Absolute millisecond claims must "
                "restate this assumption."
            ),
            "source": "ModelBlaster/benchmarks/profile_db (see gen/profile/**/_provenance.json)",
        },
        "producer_instance_provenance": (
            "inferred_from_schedule_timestamps -- no dataflow exists between "
            "the producer and consumer networks in either XPU-RT or "
            "ModelBlaster, so the consumed instance is inferred, never recorded"
        ),
        "time_unit": "ms",
        "schedule_digests_by_policy_burst": {
            f"{p}|B{b}": d for (p, b), d in schedule_digests.items()
        },
        "determinism": {
            f"{p}|B{b}": ("identical across seeds" if len(set(d.values())) == 1
                          else "DIFFERS across seeds")
            for (p, b), d in schedule_digests.items()
        },
        "failures": failures,
        "n_per_invocation_rows": len(per_inv_rows),
        "n_aggregate_rows": len(agg_rows),
        "git": _git_info(),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_dir, "git_commits.json"), "w") as f:
        json.dump(manifest["git"], f, indent=2)
    with open(os.path.join(out_dir, "command.txt"), "w") as f:
        f.write(" ".join([sys.executable, "-m", "benchmarks.freshness_eval.run",
                          *sys.argv[1:]]) + "\n")

    # --- report ---
    print()
    print("=" * 96)
    print(f"A0 = {A0:.3f} ms   (uncontended input-age ceiling)")
    print()
    print(f"{'policy':<21}{'B':>3}{'phi':>8}{'mkspn':>8}{'dl_ok':>8}"
          f"{'fresh_ok':>10}{'out_ok':>8}{'diverg':>8}{'maxage':>8}{'soft':>6}")
    for r in sorted(agg_rows, key=lambda r: (r["policy"], r["contention_level"],
                                             r["freshness_window"], r["seed"])):
        if r["seed"] != seeds[0] and r["policy"] != ORACLE:
            continue  # one seed in the console table; full data in the CSV
        print(f"{r['policy']:<21}{int(r['contention_level']):>3}"
              f"{r['freshness_window']:>8.1f}{r['makespan_ms']:>8.1f}"
              f"{r['deadline_success_rate']:>8.3f}{r['freshness_success_rate']:>10.3f}"
              f"{r['output_valid_rate']:>8.3f}{r['divergence']:>8.3f}"
              f"{(r['max_input_age'] or 0):>8.1f}{r['soft_instances_completed']:>6}")

    # Operating points where local deadline success hides invalid output.
    flagged = [
        r for r in agg_rows
        if r["policy"] != ORACLE
        and r["deadline_success_rate"] >= 0.95
        and r["output_valid_rate"] < r["deadline_success_rate"] - 0.10
    ]
    print()
    print(f"Operating points with deadline_success >= 0.95 and "
          f"output_valid < deadline_success - 0.10:  {len(flagged)}")
    for r in sorted(flagged, key=lambda r: -r["divergence"])[:12]:
        print(f"  {r['policy']:<21} B={int(r['contention_level'])} "
              f"phi={r['freshness_window']:.1f}  "
              f"deadline={r['deadline_success_rate']:.3f} "
              f"output_valid={r['output_valid_rate']:.3f} "
              f"divergence={r['divergence']:.3f}")
    if agg_rows:
        worst = max((r for r in agg_rows if r["policy"] != ORACLE),
                    key=lambda r: r["divergence"])
        print()
        print(f"Largest divergence: {worst['divergence']:.3f} at "
              f"{worst['policy']} B={int(worst['contention_level'])} "
              f"phi={worst['freshness_window']:.1f} "
              f"(deadline {worst['deadline_success_rate']:.3f}, "
              f"output_valid {worst['output_valid_rate']:.3f})")

    nondet = [k for k, v in manifest["determinism"].items() if v != "identical across seeds"]
    if nondet:
        print(f"\nNon-deterministic across seeds: {nondet}")
    else:
        print(f"\nAll {len(manifest['determinism'])} (policy, B) cells were "
              f"identical across {len(seeds)} seeds.")

    if failures:
        print(f"\n{len(failures)} cell(s) FAILED; see manifest.failures")

    print(f"\nwrote {out_dir}/")
    for name in ("manifest.json", "aggregate.csv", "per_invocation.csv",
                 "command.txt", "git_commits.json"):
        print(f"  {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
