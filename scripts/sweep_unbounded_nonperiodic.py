#!/usr/bin/env python3
"""Sweep driver for UNBOUNDED non-periodic workloads.

Non-periodic tasks are emitted with no min_start_t/max_end_t, so they carry no
release window or deadline and the scheduler packs them as early as
dependencies allow. Periodic tasks stay period-bound. This is the variant that
avoids the f2opt_v1 degeneracy, where the cursor-based sporadic window layout
strands jobs hundreds of ms past the last periodic instance
(see experiments/sweep_fpga/GENERATOR_PROPOSALS.md).

Two knobs matter and interact:
  --max-ops        the op budget. Because the horizon is extended to cover the
                   estimated non-periodic completion, a heavy non-periodic model
                   (vint ~ 8.7 s single-backend) against a fast control period
                   (14 ms) needs a LARGE budget or the horizon is shrunk back to
                   the hyperperiod and the periodic groups stop ticking early.
                   At the 1200 default, seed 3 collapses to a 70 ms horizon;
                   at 8000 it reaches 5838 ms with uniform coverage.
  --horizon-periods / --no-horizon-covers-nonperiodic
                   turn off horizon extension to keep the old hyperperiod-only
                   behaviour.

Every generated workload is checked before it is scheduled, so a degenerate
point is rejected here rather than discovered after an FPGA run.

Usage:
  python scripts/sweep_unbounded_nonperiodic.py --seeds 0-11 --max-ops 8000 \
      --hardware f2_gemmini_q31_opt --out-dir runs/sweeps/unbounded_v1 [--schedule]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Named model mixes ("arms"). Each arm pins the draw to an explicit set so
# points are COMPARABLE across arms instead of depending on what the RNG
# happened to pick. `baseline` and `fused` are matched: same periodic slot,
# same non-periodic model, only the mid-size periodic net differs -- which is
# the dronet-vs-fused_full comparison, measured on FPGA at 40.82 vs 38.22 ms
# (see experiments/fused_fpga/). The `*_vint` arms add ViNT as a second
# non-periodic job; ViNT is 605 dispatches and ~2.7 s on its own, so those
# arms need a much larger --max-ops and will dominate any makespan they
# appear in.
ARMS: dict[str, list[str]] = {
    "baseline":    ["mlp_control", "dronet",     "yolov8_nano"],
    "fused":       ["mlp_control", "fused_full", "yolov8_nano"],
    "baseline_vint": ["mlp_control", "dronet",     "yolov8_nano", "vint"],
    "fused_vint":  ["mlp_control", "fused_full", "yolov8_nano", "vint"],
    "vint_only":   ["mlp_control", "vint"],
    "all":         ["mlp_control", "dronet", "fused_full", "yolov8_nano", "vint"],
}


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def validate(cfg: dict) -> list[str]:
    """Coherence predicates. Returns a list of problems; empty means good.

    These are the checks the generator's own validate() lacks -- it verifies
    graph shape only (paths, unique ids, acyclicity) and nothing about whether
    the horizon, the instance counts and the declared load agree.
    """
    problems: list[str] = []
    nets = cfg["networks"]
    horizon = float(cfg.get("_comment_horizon_ms") or cfg.get("horizon_ms") or 0.0)
    if not horizon:  # generator records it in the comment block
        for k, v in cfg.items():
            if isinstance(v, str) and "horizon" in k.lower():
                try: horizon = float(v)
                except ValueError: pass

    per = {k: v for k, v in nets.items() if v.get("period") is not None}
    non = {k: v for k, v in nets.items() if v.get("period") is None}

    # 1. no non-periodic task may carry a release window in this mode
    for k, v in non.items():
        if "min_start_t" in v or "max_end_t" in v:
            problems.append(f"{k}: non-periodic task still has a release window")

    # 2. every periodic group must cover the same span (uniform stop time)
    spans = {k: float(v["period"]) * int(v["num_instances"]) for k, v in per.items()}
    if spans:
        lo, hi = min(spans.values()), max(spans.values())
        if hi / max(lo, 1e-9) > 1.25:
            problems.append(
                f"periodic groups stop at different times (ratio {hi/lo:.2f} > 1.25): "
                + ", ".join(f"{k}={v:.0f}ms" for k, v in sorted(spans.items())))
    # 3. and that span must actually reach the horizon
    if horizon and spans:
        cov = min(spans.values()) / horizon
        if cov < 0.9:
            problems.append(
                f"periodic coverage {cov:.2f} of horizon {horizon:.0f} ms "
                f"(< 0.90): the control loop stops before the schedule ends")
    if not non:
        problems.append("no non-periodic work: nothing to pack against the periodic load")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-11")
    ap.add_argument("--arms", default="baseline,fused",
                    help="comma list of named model mixes to sweep, or "
                         "'all-arms' for every one. Available: "
                         + ",".join(ARMS) +
                         ". Each arm is generated for every seed, so arms are "
                         "directly comparable point-for-point.")
    ap.add_argument("--hardware", default="f2_gemmini_q31_opt")
    ap.add_argument("--max-ops", type=int, default=8000)
    ap.add_argument("--out-dir", default="runs/sweeps/unbounded_v1")
    ap.add_argument("--solver", default="greedy")
    ap.add_argument("--schedule", action="store_true",
                    help="also run the scheduler on each surviving workload")
    ap.add_argument("--no-horizon-covers-nonperiodic", action="store_true",
                    help="pass through to the generator: leave the horizon at "
                         "the periodic hyperperiod instead of extending it to "
                         "cover the unbounded non-periodic work. REQUIRED for "
                         "arms containing vint: extending the horizon to cover "
                         "ViNT's ~9 s at a 16 ms control period needs ~566 "
                         "mlp_control instances (~12k dispatches), which the "
                         "greedy solver does not finish in useful time "
                         "(measured: >1h39m of CPU on one point).")
    ap.add_argument("--keep-going", action="store_true",
                    help="schedule even workloads that failed validation")
    a = ap.parse_args()

    out = os.path.join(ROOT, a.out_dir)
    os.makedirs(os.path.join(out, "workloads"), exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)

    arms = list(ARMS) if a.arms == "all-arms" else [
        x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        sys.exit(f"unknown arm(s) {bad}; available: {sorted(ARMS)}")

    rows = []
    for arm in arms:
        print(f"\n=== arm {arm}: {'+'.join(ARMS[arm])} ===")
        for seed in parse_seeds(a.seeds):
            wl = os.path.join(out, "workloads", f"{arm}_seed{seed}.json")
            cmd = [sys.executable, os.path.join(HERE, "gen_random_workload.py"),
                   str(seed), "--hardware", a.hardware,
                   "--unbounded-nonperiodic",
                   "--include-models", ",".join(ARMS[arm]),
                   "--max-ops", str(a.max_ops), "--out", wl]
            if a.no_horizon_covers_nonperiodic:
                cmd.append("--no-horizon-covers-nonperiodic")
            g = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            if g.returncode != 0 or not os.path.exists(wl):
                print(f"  {arm}/seed{seed:<3} GEN FAILED")
                print("      " + g.stderr.strip()[-300:])
                rows.append(dict(arm=arm, seed=seed, status="gen_failed"))
                continue
            cfg = json.load(open(wl))
            problems = validate(cfg)
            nets = cfg["networks"]
            npd = sum(1 for v in nets.values() if v.get("period") is None)
            pd = len(nets) - npd
            status = "ok" if not problems else "REJECTED"
            print(f"  {arm}/seed{seed:<3} {status:<9} {pd} periodic + "
                  f"{npd} non-periodic")
            for prob in problems:
                print(f"      - {prob}")
            row = dict(arm=arm, seed=seed, status=status, periodic=pd,
                       nonperiodic=npd, problems=problems,
                       workload=os.path.relpath(wl, ROOT))
            if a.schedule and (not problems or a.keep_going):
                log = os.path.join(out, "logs", f"sched_{arm}_seed{seed}.log")
                sres = subprocess.run(
                    [sys.executable, os.path.join(HERE, "run_xpurt_schedule.py"),
                     "--networks-json", os.path.relpath(wl, ROOT),
                     "--solver", a.solver],
                    capture_output=True, text=True, cwd=ROOT)
                open(log, "w").write(sres.stdout + "\n" + sres.stderr)
                mk = [l for l in sres.stdout.splitlines() if "makespan" in l.lower()]
                row["sched_rc"] = sres.returncode
                row["makespan_line"] = mk[-1].strip() if mk else None
                print(f"      sched rc={sres.returncode}  {row['makespan_line'] or ''}")
            rows.append(row)

    json.dump(rows, open(os.path.join(out, "results.json"), "w"), indent=1)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\n  {ok}/{len(rows)} workloads passed validation "
          f"across {len(arms)} arm(s) -> {out}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
