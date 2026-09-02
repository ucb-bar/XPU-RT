#!/usr/bin/env python3
"""Sweep driver for UNBOUNDED non-periodic workloads — QRB5165 / Flow C port.

Ported from RoSE/soc/sw/xpu-rt/scripts/sweep_unbounded_nonperiodic.py (the
FPGA sweep fpga_20260829-195805 ran the original). The generator underneath
it, gen_random_workload.py, is copied beside this file with one change: the
banks it reads live in banks/ here rather than under <repo>/data/banks,
because XPU-RT has no data/banks tree and the qrb5165_flowc entries are
specific to this sweep.

What the port changes, and nothing else:

  * ARMS name this board's bank models. `fused_split` occupies the arm
    SETUP.md calls `fused_full`: SETUP.md's cell table for that arm lists
    vision_conv / depth_conv / tail, which are fused_split's tiles.
  * The hardware block is this board's three lanes (HTA=CPU_P, DSP=CPU_E,
    CPU=CPU_X) instead of the FPGA's two.
  * dispatch_deps_path points at the COARSE dispatch graphs
    flowc/artifacts.py emits under gen/qnn_vmfb/<net>/qrb5165_flowc/<HW>/,
    one dispatch per tile — the dispatch space the schedule is solved in
    and the one the emitted runtime executes.
  * validate() carries the FPGA's four predicates unchanged and adds
    predicate 5 from SETUP.md (every tile has at least one backend that
    COMPOSED, read out of the frozen cost model). Predicates 6 (context
    staged on the board) and 7 (no capability-excluded cell in the chosen
    placement) are post-solve / pre-run and live in drive.py.
  * Each surviving point also gets a Flow C workload spec written beside
    its xpu-rt workload, because the two halves of this flow read different
    files: run_xpurt_schedule.py reads the taskset, and flow_c.py reads
    which binding manifest each network name resolves to. The spec lists
    one entry per COPY (dronet_a, dronet_b, ...) so schedule ingest can
    look every scheduled job name up.

Scheduling is NOT done here (the original's --schedule): this target picks
its solver by size and falls back on timeout, which drive.py owns.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DEFAULT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

# Named model mixes ("arms"), matched point-for-point with the FPGA sweep's.
# baseline and fused are matched to each other: same rng stream, same bank
# order, identical period/window/count bands on the two mid-size models, so
# only which model occupies that slot differs.
ARMS: dict[str, list[str]] = {
    "baseline":   ["mlp_control", "dronet",      "yolov8n"],
    "fused":      ["mlp_control", "fused_split", "yolov8n"],
    "fused_vint": ["mlp_control", "fused_split", "yolov8n", "vint"],
}

# bank model -> Flow C binding manifest (relative to qnn_models/flow_c/)
BINDINGS: dict[str, str] = {
    "mlp_control": "bindings/mlp_control.json",
    "dronet":      "bindings/dronet.json",
    "fused_split": "bindings/fused_split.json",
    "yolov8n":     "bindings/yolov8n.json",
    "vint":        "bindings/vint.json",
}


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def model_of(key: str) -> str:
    """Copy key (dronet_a) -> bank model name (dronet)."""
    if key in BINDINGS:
        return key
    stem = key.rsplit("_", 1)[0]
    if stem in BINDINGS:
        return stem
    raise KeyError(f"cannot map workload network {key!r} to a bank model")


def validate(cfg: dict, cost_model: dict, flow_c_dir: str) -> list[str]:
    """Coherence predicates. Returns a list of problems; empty means good.

    1-4 are the FPGA sweep's, verbatim. 5 is SETUP.md's first target-specific
    predicate: a schedule can be feasible on paper and unbuildable in fact.
    """
    problems: list[str] = []
    nets = cfg["networks"]
    horizon = float(cfg.get("horizon_ms") or 0.0)

    per = {k: v for k, v in nets.items() if v.get("period") is not None}
    non = {k: v for k, v in nets.items() if v.get("period") is None}

    # 2. sporadic containment — no non-periodic task may carry a release window
    for k, v in non.items():
        if "min_start_t" in v or "max_end_t" in v:
            problems.append(f"{k}: non-periodic task still has a release window")

    # 3. uniform stop time — every periodic group must cover the same span
    spans = {k: float(v["period"]) * int(v["num_instances"]) for k, v in per.items()}
    if spans:
        lo, hi = min(spans.values()), max(spans.values())
        if hi / max(lo, 1e-9) > 1.25:
            problems.append(
                f"periodic groups stop at different times (ratio {hi/lo:.2f} > 1.25): "
                + ", ".join(f"{k}={v:.0f}ms" for k, v in sorted(spans.items())))
    # 1. periodic coverage — and that span must actually reach the horizon
    if horizon and spans:
        cov = min(spans.values()) / horizon
        if cov < 0.9:
            problems.append(
                f"periodic coverage {cov:.2f} of horizon {horizon:.0f} ms "
                f"(< 0.90): the control loop stops before the schedule ends")
    # 4. non-empty
    if not non:
        problems.append("no non-periodic work: nothing to pack against the periodic load")

    # 5. every tile has at least one backend that COMPOSED, so the solver is
    #    never offered a cell the board rejects. A measured cell IS a compose
    #    that succeeded; compose_failures is cross-checked so a cell that is
    #    both measured and listed as failing is caught rather than trusted.
    cells = cost_model.get("cells", {})
    fails = {(f.get("cell", ""), f.get("backend", ""))
             for f in cost_model.get("compose_failures", [])}
    for key in sorted({model_of(k) for k in nets}):
        doc = json.load(open(os.path.join(flow_c_dir, BINDINGS[key])))
        manifest_net = doc["network"]
        for b in doc["bindings"]:
            cell = f"{manifest_net}/{b['name']}"
            measured = {k: v for k, v in (cells.get(cell) or {}).items()
                        if v is not None and "@" not in k}
            if not measured:
                problems.append(
                    f"{cell}: no backend has a measured cell — nothing composed, "
                    f"so the solver would only ever see exclusion costs")
                continue
            contradicted = [k for k in measured if (cell, k) in fails]
            if contradicted:
                problems.append(
                    f"{cell}: backend(s) {contradicted} are measured AND listed in "
                    f"compose_failures — the cost model contradicts itself")
            declared = set(b.get("backends", {}))
            if not (set(measured) & declared):
                problems.append(
                    f"{cell}: measured on {sorted(measured)} but the manifest "
                    f"declares contexts for {sorted(declared)} — no overlap, so "
                    f"no schedulable placement has a context to run")
    return problems


def flow_c_spec(name: str, cfg: dict) -> dict:
    """The Flow C workload spec for one point: one entry per COPY."""
    nets = []
    for key, v in cfg["networks"].items():
        entry = {"name": key, "bindings": BINDINGS[model_of(key)]}
        if v.get("period") is not None:
            entry["period"] = v["period"]
        nets.append(entry)
    return {
        "name": name,
        "_comment": ("Generated by sweep_unbounded_nonperiodic.py for sweep point "
                     f"{name}. One entry per copy so schedule ingest resolves every "
                     "job name in the solved schedule; several entries may share a "
                     "binding manifest, which is how two copies of one model get "
                     "independent scheduling identities but the same tiles."),
        "target": "qrb5165_flowc",
        "board": "qrb5165_v66",
        "registry": "cores/qrb5165_qnn.json",
        "measurements": "sweeps/qrb5165_20260829-200620/cost_model.json",
        "slots": {"CPU_P": "hta", "CPU_E": "dsp", "CPU_X": "cpu"},
        "ctx_dir": "/root/qnn_runtime_ctx",
        "networks": nets,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--arms", default="baseline,fused",
                    help="comma list of named model mixes, or 'all-arms'. "
                         "Available: " + ",".join(ARMS))
    ap.add_argument("--hardware", default="qrb5165_flowc")
    ap.add_argument("--max-ops", type=int, default=800)
    ap.add_argument("--repo-root", default=REPO_DEFAULT)
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()

    out = a.out_dir
    os.makedirs(os.path.join(out, "workloads"), exist_ok=True)
    os.makedirs(os.path.join(out, "workloads", "rejected"), exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    flow_c_dir = os.path.abspath(os.path.join(HERE, "..", ".."))
    with open(os.path.join(HERE, "cost_model.json")) as f:
        cost_model = json.load(f)

    arms = list(ARMS) if a.arms == "all-arms" else [
        x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        sys.exit(f"unknown arm(s) {bad}; available: {sorted(ARMS)}")

    rows = []
    for arm in arms:
        print(f"\n=== arm {arm}: {'+'.join(ARMS[arm])} ===")
        for seed in parse_seeds(a.seeds):
            point = f"{arm}_seed{seed}"
            wl = os.path.join(out, "workloads", f"{point}.json")
            log = os.path.join(out, "logs", f"gen_{point}.log")
            cmd = [sys.executable, os.path.join(HERE, "gen_random_workload.py"),
                   str(seed), "--hardware", a.hardware,
                   "--repo-root", a.repo_root,
                   "--unbounded-nonperiodic",
                   "--include-models", ",".join(ARMS[arm]),
                   "--max-ops", str(a.max_ops), "--out", wl]
            g = subprocess.run(cmd, capture_output=True, text=True, cwd=a.repo_root)
            with open(log, "w") as f:
                f.write(" ".join(cmd) + "\n\n" + g.stdout + "\n" + g.stderr)
            if g.returncode != 0 or not os.path.exists(wl):
                print(f"  {point:<22} GEN FAILED")
                print("      " + g.stderr.strip()[-300:])
                rows.append(dict(arm=arm, seed=seed, point=point, status="gen_failed",
                                 problems=[g.stderr.strip()[-300:]]))
                continue
            cfg = json.load(open(wl))
            problems = validate(cfg, cost_model, flow_c_dir)
            nets = cfg["networks"]
            npd = sum(1 for v in nets.values() if v.get("period") is None)
            pd = len(nets) - npd
            ops = sum(1 for _ in ())  # filled below from the generator's note
            note = [l for l in cfg["_comment"] if "operations in total" in l]
            ops = int(note[0].split(",")[-1].split()[0]) if note else None
            status = "ok" if not problems else "REJECTED"
            print(f"  {point:<22} {status:<9} {pd} periodic + {npd} non-periodic, "
                  f"horizon {cfg['horizon_ms']:.0f} ms, {ops} ops")
            for prob in problems:
                print(f"      - {prob}")
            row = dict(arm=arm, seed=seed, point=point, status=status,
                       periodic=pd, nonperiodic=npd, problems=problems,
                       horizon_ms=cfg["horizon_ms"],
                       hyperperiod_ms=cfg["hyperperiod_ms"],
                       op_count=ops,
                       networks={k: {"period": v.get("period"),
                                     "window_duration": v.get("window_duration"),
                                     "num_instances": v.get("num_instances")}
                                 for k, v in nets.items()},
                       workload=os.path.relpath(wl, out))
            if problems:
                # A rejected point is never built: move it out of the way so
                # nothing downstream can pick it up by globbing workloads/.
                dead = os.path.join(out, "workloads", "rejected", f"{point}.json")
                os.replace(wl, dead)
                with open(os.path.join(out, "workloads", "rejected",
                                       f"{point}.reason.txt"), "w") as f:
                    f.write("\n".join(problems) + "\n")
                row["workload"] = os.path.relpath(dead, out)
            else:
                spec = flow_c_spec(point, cfg)
                sp = os.path.join(out, "workloads", f"{point}.flowc.json")
                with open(sp, "w") as f:
                    json.dump(spec, f, indent=2)
                row["flowc_spec"] = os.path.relpath(sp, out)
            rows.append(row)

    with open(os.path.join(out, "generated.json"), "w") as f:
        json.dump(rows, f, indent=1)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\n  {ok}/{len(rows)} workloads passed validation across "
          f"{len(arms)} arm(s) -> {out}/generated.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
