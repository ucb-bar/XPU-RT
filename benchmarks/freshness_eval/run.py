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
import hashlib
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
    deadline_compliance,
    invocations_from_fixture,
    load_fixture,
    soft_utility,
)

# --- policy definitions ----------------------------------------------------
#
# Each policy is a (solver, scheduler, mutations) triple. `mutations` is a dict
# of config edits applied by `materialise`; see MUTATION_KEYS for the vocabulary
# and for what each one costs. Exactly one mutation per probe policy, so an
# outcome is attributable to a mechanism rather than to a bundle.

MUTATION_KEYS = {
    "preferred_hw": (
        "{net: profile_hw} SOFT pin. profile_loader multiplies non-preferred "
        "combinations by PIN_PENALTY_MULT; it does not hard-exclude them. A "
        "preference, never a reservation."
    ),
    "window_duration": (
        "{net: ms} tightens a network's own window, i.e. max_end_t = release + "
        "ms. greedy honours this via ALAP back-propagation and emergency "
        "promotion; the MILP enforces it outright. Applied to the PRODUCER this "
        "converts the consumer's freshness requirement into a producer "
        "deadline, which is the only mechanism here that acts on the quantity "
        "freshness actually depends on."
    ),
    "admit_cap": (
        "int cap on admitted soft instances: admitted = min(B, cap). Trades "
        "soft utility for headroom directly. `contention_level` still records "
        "the OFFERED B, so admission control is never free in the reporting."
    ),
    "soft_phase_ms": (
        "float start_time offset on the soft network, deferring its first "
        "release. Rate-limiting in time rather than in count."
    ),
}

POLICIES: Dict[str, Dict] = {
    "static_nominal": {
        "solver": "greedy",
        "scheduler": "mosek",  # unused when solver != milp
        "mutations": {},
        "intent": (
            "Contention-blind earliest-completion list schedule. Maximises soft "
            "throughput under low load; no protection for the perception chain."
        ),
    },
    "edf": {
        "solver": "milp",
        "scheduler": "edf",
        "mutations": {},
        "intent": "Deadline-ordered list schedule; the conventional real-time baseline.",
    },
    "heft": {
        "solver": "milp",
        "scheduler": "heft",
        "mutations": {},
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
        "mutations": {"preferred_hw": {"dronet": "gemmini"}},
        "intent": (
            "Protects the DroNet->control chain by reserving the fast "
            "accelerator for DroNet (soft cost penalty on other placements). "
            "MEASURED WORSE THAN static_nominal at every contention level -- see "
            "the mechanism probe below and results/freshness_eval/summary.md."
        ),
    },
}

# --- mechanism probe -------------------------------------------------------
#
# Gate A returned a negative finding: `static_conservative` -- a soft pin of the
# perception producer to the fast accelerator -- is WORSE than doing nothing at
# every contention level (0.146 vs 0.220 output-valid at B=3). A protection
# candidate cannot be ASSUMED to protect, so no selector is built on top of one
# until some mechanism is measured to beat the nominal baseline.
#
# The mechanistic reading of that failure: DroNet is 17.973 ms on gemmini vs
# 241.462 ms on rvv_opu, so it lands on gemmini regardless -- the pin cannot
# move a placement that was already optimal. All it does is perturb the costs
# the greedy picker orders by. It was never a reservation, and reserving a
# BACKEND was the wrong lever anyway: at B>=1 the contention is for gemmini
# TIME, which a preference does not allocate.
#
# Every probe below holds solver=greedy fixed -- the same solver as
# static_nominal -- so a difference is attributable to the mechanism and not to
# the scheduler. One mutation each.
PROBES: Dict[str, Dict] = {
    # M1: convert the freshness requirement into a producer deadline. The
    # hypothesis with an actual mechanism behind it: input age is bounded by
    # when the producer FINISHES, so bounding producer completion bounds age.
    # Swept, because the tightness that helps is an empirical question and too
    # tight should over-constrain and hurt.
    f"probe_prodwin{int(w)}": {
        "solver": "greedy", "scheduler": "mosek",
        "mutations": {"window_duration": {"dronet": float(w)}},
        "intent": f"Producer window tightened 50 -> {int(w)} ms.",
    }
    for w in (40, 30, 25, 20)
}
PROBES.update({
    # M2: admission control. Guaranteed to buy headroom; the question is the
    # exchange rate against soft utility, which is the whole point of the
    # adaptive comparison.
    "probe_admit1": {
        "solver": "greedy", "scheduler": "mosek",
        "mutations": {"admit_cap": 1},
        "intent": "Admit at most 1 soft instance regardless of offered B.",
    },
    "probe_admit2": {
        "solver": "greedy", "scheduler": "mosek",
        "mutations": {"admit_cap": 2},
        "intent": "Admit at most 2 soft instances regardless of offered B.",
    },
    # M3: rate-limit in time instead of in count -- same soft work, released
    # later. Separates "less soft work" from "soft work out of the way".
    #
    # MEASURED RESULT (phi = A0+20, clean region B<=2, seed 0): monotone in the
    # offset, every offset at least as good as no deferral, smallest offset best.
    #
    #     offset ms      0     10     25     40     50
    #     B=1        0.633  0.900  0.833  0.733  0.700
    #     B=2        0.400  0.867  0.733  0.533   (overruns epoch)
    #
    # Two assumptions in the original probe design were WRONG and are corrected
    # here rather than quietly dropped:
    #
    #  (a) "50 ms is the phase control -- a full producer period is the same
    #      alignment as no deferral." False. The offset is applied to the SOFT
    #      network, whose period is epoch/admitted (300 ms at B=1, 150 at B=2,
    #      100 at B=3), not to DroNet. A null control for a phase effect would
    #      need offset == the soft period, which is a different value at every B,
    #      so no single offset is a phase control. probe_defer50 is therefore a
    #      point on the curve, not a control.
    #  (b) The discriminator "only 25 works and 50 behaves like 0 -> resonance"
    #      never fired: the response is monotone, which rules out resonance at a
    #      single alignment. It does NOT yet establish the mechanism, because
    #      monotone-decreasing-in-offset is the opposite of what "get soft work
    #      out of the way" predicts (more deferral should help more). The offsets
    #      below 10 ms exist to find where the curve turns over.
    **{
        f"probe_defer{int(o)}": {
            "solver": "greedy", "scheduler": "mosek",
            "mutations": {"soft_phase_ms": float(o)},
            "intent": f"Defer the first soft release by {int(o)} ms.",
        }
        for o in (2, 5, 10, 15, 20, 25, 30, 40, 50)
    },
    # M4: directional control, expected to HURT.
    #
    # MEASURED RESULT: INERT -- bit-identical schedules to static_nominal at
    # every B (same invocation set, same makespan to 4 decimal places). So this
    # control did NOT function as a control, and must not be reported as a
    # passed falsification test.
    #
    # Cause: every network in this workload is periodic, so "deprioritise
    # periodic work" has nothing to deprioritise it RELATIVE TO -- greedy_periodic
    # and greedy order the same set the same way. The control was mis-specified,
    # not the metric. It is kept in the vocabulary as a documented null result;
    # probe_soft_first below is the replacement that can actually be worse.
    "probe_nonperiodic_priority": {
        "solver": "greedy_periodic", "scheduler": "mosek",
        "mutations": {},
        "intent": (
            "MIS-SPECIFIED CONTROL, measured INERT: intended to deprioritise the "
            "periodic producer, but all networks here are periodic so it reduces "
            "to plain greedy. Retained as a recorded null result."
        ),
    },
    # M4': the replacement directional control. If deferring soft work helps,
    # then ADVANCING it -- releasing the whole soft burst at t=0 with the
    # producer, which is what static_nominal already does -- should be the worst
    # case, and a NEGATIVE offset is not expressible. So instead invert the
    # mechanism the other way: tighten the SOFT window so soft work becomes a
    # hard constraint competing with the producer rather than sheddable
    # interference. Expected to be WORSE than static_nominal at B>=1.
    "probe_soft_first": {
        "solver": "greedy", "scheduler": "mosek",
        "mutations": {"window_duration": {"yolov8_nano_64": 75.0}},
        "intent": (
            "FALSIFICATION CONTROL, expected WORSE: give the soft network a tight "
            "window so it competes as a constraint instead of being sheddable."
        ),
    },
})

ALL_POLICIES: Dict[str, Dict] = {**POLICIES, **PROBES}
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


def materialise(base: Dict, *, burst: int, mutations: Optional[Dict],
                epoch_ms: float, seed: int) -> Dict:
    """Build a concrete workload config for one (burst, policy) cell.

    `burst` is the OFFERED soft load. A candidate may admit fewer via the
    `admit_cap` mutation; the offered value is what gets recorded as
    `contention_level`, so shedding work never looks free.
    """
    mutations = dict(mutations or {})
    unknown = set(mutations) - set(MUTATION_KEYS)
    if unknown:
        raise ValueError(
            f"unknown mutation key(s) {sorted(unknown)}; "
            f"vocabulary is {sorted(MUTATION_KEYS)}"
        )

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

    admitted = burst
    cap = mutations.get("admit_cap")
    if cap is not None:
        admitted = min(burst, int(cap))

    if admitted <= 0:
        # No soft work at all: remove the network rather than give it zero
        # instances, so the workload contains no vestigial entry.
        nets.pop(soft_name)
    else:
        info = nets[soft_name]
        info["num_instances"] = admitted
        # Spread the ADMITTED releases evenly across the epoch. window_duration
        # stays at the epoch so the window is loose: soft work is interference
        # to be shed, not a hard constraint that would make the instance
        # infeasible.
        info["period"] = epoch_ms / admitted
        info["window_duration"] = epoch_ms
        phase = mutations.get("soft_phase_ms")
        if phase:
            info["start_time"] = float(phase)

    for net, hw in (mutations.get("preferred_hw") or {}).items():
        if net in nets:
            nets[net]["preferred_hw"] = hw

    # Tightening a network's own window moves its max_end_t. For the producer
    # that is the point; it also moves that network's own deadline, which is
    # why deadline_compliance is additionally reported against a fixed
    # reference window (see trace.deadline_compliance).
    for net, win in (mutations.get("window_duration") or {}).items():
        if net not in nets:
            # Distinguish a typo from a network legitimately dropped by admission
            # control (the soft network is removed entirely at burst 0). Raising
            # on the latter would make any soft-side window mutation crash at
            # B=0; silently ignoring the former would hide a dead mutation.
            if net in base["networks"]:
                continue
            raise ValueError(
                f"window_duration mutation names {net!r}, which is not a network "
                f"in this workload ({sorted(base['networks'])})"
            )
        w = float(win)
        period = float(nets[net].get("period", w))
        if w > period:
            raise ValueError(
                f"{net}: window_duration {w} exceeds its period {period}; "
                f"overlapping instances are outside what this evaluation models"
            )
        nets[net]["window_duration"] = w

    cfg["_materialised"] = {
        "offered_burst": burst,
        "admitted_soft_instances": admitted,
        "mutations": mutations,
        "seed": seed,
    }
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
                 cell_timeout_s: float = 600.0,
                 reuse_fixtures: bool = False) -> Tuple[Optional[str], float, str]:
    """Invoke run_xpurt_schedule.py on a materialised config.

    With `reuse_fixtures`, skip the solve when a fixture for an IDENTICAL config
    already exists. Scheduling dominates the cost (measured: 6 s at B=0 rising to
    584 s at B=4 per cell) while evaluation is milliseconds, so a change to the
    EVALUATOR should not require re-solving. That is not a micro-optimisation: it
    is what makes it affordable to fix an evaluator bug and restate every number,
    rather than being tempted to leave results as they are.

    Reuse is gated on a content hash of the exact config, written as a sidecar
    when the fixture is produced. Matching on filename or mtime would silently
    reuse a fixture from a different workload -- and this sweep deliberately
    reuses stems across runs, so that risk is real rather than theoretical.
    """
    cfg_path = os.path.join(_REPO, "data", "toplevel", f"{stem}.json")
    cfg_blob = json.dumps(cfg, indent=2, sort_keys=True)
    cfg_digest = hashlib.sha256(cfg_blob.encode()).hexdigest()

    fixture_path = os.path.join(
        _REPO, "schedules",
        f"scheduled_{stem}{solver_tag(solver, scheduler)}_profiled.json",
    )
    sidecar = fixture_path + ".cfgsha256"
    if reuse_fixtures and os.path.exists(fixture_path) and os.path.exists(sidecar):
        with open(sidecar) as f:
            recorded = f.read().strip()
        if recorded == cfg_digest:
            return fixture_path, 0.0, "ok (reused fixture)"

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

    if proc.returncode != 0:
        return None, wall, f"solver exit {proc.returncode} (see {log_path})"
    if not os.path.exists(fixture_path):
        return None, wall, f"fixture missing: {fixture_path} (see {log_path})"
    # Record which config produced this fixture so a later --reuse-fixtures pass
    # can prove they correspond instead of trusting the filename.
    with open(sidecar, "w") as f:
        f.write(cfg_digest + "\n")
    return fixture_path, wall, "ok"


def compute_a0(base: Dict, *, epoch_ms: float, edge) -> Dict[str, object]:
    """Closed-form uncontended input-age ceiling for the producer/consumer pair.

    Latencies are the per-network totals on the FAST cluster, read from the
    exported profile CSVs -- i.e. the intended uncontended placement.
    """
    from profile_loader import find_profile_csv, load_profiled_times

    hw = base["hardware"]["profile_hw"]["cpu_p"]
    target = base["hardware"]["profile"]["target"]
    topo = base["hardware"]["profile"].get("topo_tag", "topo_0")
    # MUST match the tree the solver reads, or A0 -- and therefore the whole phi
    # grid -- is computed on a different timing basis than the schedules it is
    # used to judge. This silently happened once: gen_root was ignored
    # everywhere, so a 25 MHz control produced 25 MHz periods against 1 GHz
    # latencies and A0 came out as 2000.546 instead of 2421.84.
    gen_root = base["hardware"]["profile"].get("gen_root") or "gen"

    def latency(net: str) -> float:
        info = base["networks"][net]
        ident = info.get("identifier", net)
        basename = os.path.basename(
            info["dispatch_deps_path"]
        ).replace("_dispatch_graph.json", "")
        csv_path = find_profile_csv(
            _REPO, model=ident, target=target, hw=hw, basename=basename,
            topo_tag=topo, gen_root=gen_root,
        )
        if not csv_path:
            raise FileNotFoundError(
                f"no profile CSV for {ident} on {hw}/{target}/{topo} under "
                f"{gen_root}/profile; run "
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
    ap.add_argument("--reuse-fixtures", action="store_true",
                    help="skip the solve when a fixture for a byte-identical "
                         "config already exists (verified by content hash). Use "
                         "to re-evaluate after an EVALUATOR change without "
                         "re-solving; never use it to skip a workload change.")
    ap.add_argument("--cell-timeout", type=float, default=600.0,
                    help="wall-clock cap per (policy, B, seed) cell; a cell that "
                         "exceeds it is recorded as a failure rather than "
                         "stalling the sweep")
    ap.add_argument("--stem-tag", default="",
                    help="disambiguator inserted into every fixture stem. REQUIRED "
                         "when sweeping a config that is not the canonical one: "
                         "stems are (policy, B, seed) only, so two different "
                         "workloads swept with the same policy names overwrite each "
                         "other's fixtures. That has already happened once -- the "
                         "25 MHz clock-invariance control clobbered three "
                         "static_nominal fixtures. The content-hash sidecar makes "
                         "the collision safe (it forces a re-solve rather than "
                         "silently reusing the wrong schedule); this flag makes it "
                         "not happen.")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO, args.config)
    with open(cfg_path) as f:
        base = json.load(f)

    epoch_ms = float(base.get("epoch", {}).get("length_ms", 300.0))
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    bursts = [int(b) for b in args.bursts.split(",") if b != ""]
    deltas = [float(d) for d in args.deltas.split(",") if d != ""]
    policies = [p for p in args.policies.split(",") if p]
    if policies == ["PROBES"]:
        policies = ["static_nominal"] + list(PROBES)  # baseline always included
    for p in policies:
        if p not in ALL_POLICIES:
            raise SystemExit(
                f"unknown policy {p!r}; have {sorted(ALL_POLICIES)} "
                f"(or the literal 'PROBES' for baseline + every probe)"
            )

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

    # The producer's window as declared in the canonical config. Candidates may
    # tighten their own copy; this fixed value is the bar they are all compared
    # against.
    producer_ref_window = float(
        base["networks"][edge.producer_task]["window_duration"]
    )

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
        spec = ALL_POLICIES[policy]
        for burst in bursts:
            for seed in seeds:
                stem = f"_fx_{args.stem_tag + '_' if args.stem_tag else ''}" \
                       f"{policy}_B{burst}_s{seed}"
                cfg = materialise(
                    base, burst=burst, mutations=spec.get("mutations"),
                    epoch_ms=epoch_ms, seed=seed,
                )
                fixture_path, wall, status = run_schedule(
                    cfg, stem=stem, solver=spec["solver"],
                    scheduler=spec["scheduler"], time_limit=args.time_limit,
                    work_dir=work_dir, cell_timeout_s=args.cell_timeout,
                    reuse_fixtures=args.reuse_fixtures,
                )
                # run_schedule distinguishes a fresh solve ("ok") from a verified
                # reuse ("ok (reused fixture)"); both are successes. Comparing for
                # exact equality here silently turned all 57 reused cells into
                # "failures" with a passing status string, which is the worst kind
                # of bug -- it drops data while the manifest looks fine.
                if not status.startswith("ok"):
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
                su["soft_instances_offered"] = burst
                su["soft_instances_admitted"] = cfg["_materialised"][
                    "admitted_soft_instances"
                ]
                # Producer deadline behaviour, which the consumer-side freshness
                # records cannot show. At high contention the producer is what
                # is late, and that is the mechanism behind the staleness.
                #
                # `reference_window` is the BASELINE producer window, so a
                # candidate that tightens its own window is still scored against
                # the same fixed bar as every other candidate. Without it,
                # window-tightening would look like a deadline regression purely
                # because it moved its own goalposts.
                prod_dl = deadline_compliance(
                    invs, edge.producer_task, reference_window=producer_ref_window
                )

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
                        **prod_dl,
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

    # --- epoch comparability, stamped before anything is written -------------
    # A cell whose makespan exceeds the epoch is NOT comparable to one that fits.
    # When the schedule overruns, greedy's horizon search extends the horizon and
    # adds instances, so the rates are computed over a longer trace with a
    # different denominator: static_nominal at B=3 is scored over 41 consumer
    # invocations across 483 ms, while probe_defer10 at the same B is scored over
    # 30 across 297 ms. Ranking those two against each other compares a policy to
    # a different experiment, not to a different policy.
    #
    # Recorded as a column rather than filtered out: which B are comparable is a
    # property of the workload worth knowing, and an overrunning cell is still
    # valid evidence that the workload does not fit at that contention level.
    overrun_by_b: Dict[int, List[str]] = {}
    for r in agg_rows:
        if r["policy"] != ORACLE and not r["fits_in_epoch"]:
            overrun_by_b.setdefault(int(r["contention_level"]), []).append(r["policy"])
    comparable_b = sorted(
        b for b in {int(r["contention_level"]) for r in agg_rows}
        if b not in overrun_by_b
    )
    for r in agg_rows:
        r["epoch_comparable"] = int(r["contention_level"]) in comparable_b

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
        "policies": {p: ALL_POLICIES[p] for p in policies},
        "mutation_vocabulary": MUTATION_KEYS,
        "producer_reference_window_ms": producer_ref_window,
        "reuse_fixtures": bool(args.reuse_fixtures),
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
    prod_key = f"{edge.producer_task}_deadline_success_rate"
    print(f"{'policy':<21}{'B':>3}{'phi':>8}{'mkspn':>8}{'dl_ok':>8}"
          f"{'fresh_ok':>10}{'out_ok':>8}{'diverg':>8}{'maxage':>8}"
          f"{'prod_dl':>9}{'soft':>6}")
    print(f"{'':<21}{'':>3}{'':>8}{'':>8}{'(consumer)':>8}"
          f"{'':>10}{'':>8}{'':>8}{'':>8}{'(producer)':>9}{'':>6}")
    for r in sorted(agg_rows, key=lambda r: (r["policy"], r["contention_level"],
                                             r["freshness_window"], r["seed"])):
        if r["seed"] != seeds[0] and r["policy"] != ORACLE:
            continue  # one seed in the console table; full data in the CSV
        print(f"{r['policy']:<21}{int(r['contention_level']):>3}"
              f"{r['freshness_window']:>8.1f}{r['makespan_ms']:>8.1f}"
              f"{r['deadline_success_rate']:>8.3f}{r['freshness_success_rate']:>10.3f}"
              f"{r['output_valid_rate']:>8.3f}{r['divergence']:>8.3f}"
              f"{(r['max_input_age'] or 0):>8.1f}"
              f"{(r.get(prod_key) if r.get(prod_key) is not None else float('nan')):>9.3f}"
              f"{r['soft_instances_completed']:>6}")

    print()
    print("Epoch comparability (makespan <= epoch): cross-policy ranking is only "
          "well-posed where EVERY policy fits.")
    print(f"  comparable contention levels: {comparable_b}")
    for b in sorted(overrun_by_b):
        pols = sorted(set(overrun_by_b[b]))
        print(f"  B={b}: NOT comparable -- {len(pols)} policy/policies overrun the "
              f"epoch: {', '.join(pols[:6])}{' ...' if len(pols) > 6 else ''}")

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
    # The headline divergence is taken from cells that FIT the epoch. Divergence
    # is a within-cell claim, so an overrunning cell still exhibits it -- but its
    # rate is over a trace longer than the epoch the experiment specifies, so it
    # cannot be the number that gets quoted.
    def _worst(rows):
        return max(rows, key=lambda r: r["divergence"]) if rows else None

    deployable = [r for r in agg_rows if r["policy"] != ORACLE]
    worst_fit = _worst([r for r in deployable if r["fits_in_epoch"]])
    worst_any = _worst(deployable)
    print()
    if worst_fit:
        print(f"Largest divergence (schedule fits the epoch): "
              f"{worst_fit['divergence']:.3f} at "
              f"{worst_fit['policy']} B={int(worst_fit['contention_level'])} "
              f"phi={worst_fit['freshness_window']:.1f} "
              f"(deadline {worst_fit['deadline_success_rate']:.3f}, "
              f"output_valid {worst_fit['output_valid_rate']:.3f})")
    if worst_any and worst_any is not worst_fit:
        print(f"Largest divergence overall (epoch OVERRUN, rate is over a "
              f"{worst_any['makespan_ms']:.0f} ms trace, not the "
              f"{worst_any['epoch_ms']:.0f} ms epoch -- do not quote): "
              f"{worst_any['divergence']:.3f} at {worst_any['policy']} "
              f"B={int(worst_any['contention_level'])}")

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
